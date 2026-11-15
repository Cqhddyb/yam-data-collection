import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.utils import RateRecorder, override_log_level

DEFAULT_ROBOT_PORT = 11333
_MAX_DOFS = 16
"""Upper bound on robot DOF. The shared pos array is sized to this; the active
slice is exposed through an int Value the worker fills after constructing the robot."""

_TELEMETRY_COMPONENTS = 3
"""Position, velocity, and effort rows in ``TelemetrySharedState.joint_state``."""

_TELEMETRY_BUTTONS = 2
"""Two teaching-handle digital inputs exposed as diagnostic telemetry."""

_WORKER_LOOP_PERIOD_S = 0.002
"""~500 Hz pacing for worker loops whose reads hit cached state (get_joint_pos /
get_info return instantly). Without it these loops peg a CPU core; the actual YAM
hardware/control thread updates at ~250 Hz, well below this cap, so no data is lost."""

_WORKER_SHUTDOWN_TIMEOUT_S = 1.0
"""Grace period for cooperative worker shutdown before the kill fallback."""

_LEADER_INFO_WARN_S = 0.020
"""Warn when a leader state/encoder read exceeds one control-cycle budget."""

_PORTAL_RPC_TIMEOUT_S = 0.25
"""Maximum wait for one localhost Portal request or client shutdown."""

_LEADER_RPC_FAILURE_TIMEOUT_S = 1.0
"""Maximum continuous follower-RPC outage before leader mode stops safely."""

_LEADER_RPC_RETRY_PERIOD_S = 0.1
"""Pause between transient follower-RPC retries in leader mode."""

_PREPROGRAMMED_MAX_JOINT_SPEED_RAD_S = 1.0
"""Joint-speed ceiling for the optional leader move endpoint."""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class ClientRobot(Robot):
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._address = f"{host}:{port}"
        self._rpc_lock = threading.Lock()
        self._client = self._make_client()

    def _make_client(self) -> portal.Client:
        # Portal's default automatic reconnect causes ``Client.call`` to wait
        # indefinitely while a peer is unavailable.  We request connections
        # explicitly below so every wait is bounded by the RPC timeout.
        return portal.Client(self._address, autoconn=False)

    def _call(self, method: str, *args: Any) -> Any:
        """Make one bounded RPC and clear a timed-out in-flight request.

        Portal keeps timed-out futures in its in-flight map.  Replacing the
        client after a response timeout prevents that map from eventually
        reaching its limit and making a later ``call`` block forever.
        """
        with self._rpc_lock:
            if not self._client.connect(timeout=_PORTAL_RPC_TIMEOUT_S):
                raise TimeoutError(f"timed out connecting to Portal server {self._address}")
            try:
                return getattr(self._client, method)(*args).result(timeout=_PORTAL_RPC_TIMEOUT_S)
            except TimeoutError:
                # The server may still reply later, but this operation is no
                # longer useful.  Drop its pending Future before retrying.
                stale_client = self._client
                try:
                    stale_client.close(timeout=_PORTAL_RPC_TIMEOUT_S)
                except Exception:
                    logging.exception("Portal client reset failed for %s", self._address)
                self._client = self._make_client()
                raise

    def num_dofs(self) -> int:
        return self._call("num_dofs")

    def get_joint_pos(self) -> np.ndarray:
        return self._call("get_joint_pos")

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        # Portal limits the number of in-flight requests.  Waiting for the
        # acknowledgement here keeps an unavailable follower from filling that
        # limit with discarded command futures and blocking a later RPC forever.
        self._call("command_joint_pos", joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self._call("command_joint_state", joint_state)

    def enter_gravity_comp_idle(self) -> None:
        """Release position control while keeping gravity compensation active."""
        self._call("enter_gravity_comp_idle")

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._call("get_observations")

    def close(self) -> None:
        """Tear down the underlying portal client (background loop thread + socket)."""
        with self._rpc_lock:
            self._client.close(timeout=_PORTAL_RPC_TIMEOUT_S)

    def __enter__(self) -> "ClientRobot":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class YAMLeaderRobot:
    def __init__(
        self,
        robot: MotorChainRobot,
        *,
        gripper_command_closed: float = 0.0,
        gripper_command_open: float = 1.0,
    ):
        self._robot = robot
        self._motor_chain = robot.motor_chain
        if not (
            np.isfinite(gripper_command_closed)
            and np.isfinite(gripper_command_open)
            and 0.0 <= gripper_command_closed < gripper_command_open <= 1.0
        ):
            raise ValueError(
                "gripper command endpoints must satisfy 0 <= gripper_command_closed < gripper_command_open <= 1"
            )
        self._gripper_command_closed = float(gripper_command_closed)
        self._gripper_command_open = float(gripper_command_open)
        logging.info(
            "Leader gripper command endpoints: closed=%.6f, open=%.6f",
            self._gripper_command_closed,
            self._gripper_command_open,
        )

    def get_info(self) -> Tuple[np.ndarray, np.ndarray]:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        if encoder_obs is None:
            # Populated only after the first CAN read in the background driver thread.
            raise RuntimeError("CAN device states not ready yet")
        raw_gripper_cmd = 1 - encoder_obs[0].position
        gripper_cmd = np.clip(
            (raw_gripper_cmd - self._gripper_command_closed)
            / (self._gripper_command_open - self._gripper_command_closed),
            0.0,
            1.0,
        )
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


# ---------------------------------------------------------------------------
# Args + Resources
# ---------------------------------------------------------------------------


@dataclass
class Args:
    arm: Literal["yam", "yam_pro", "yam_ultra", "big_yam", "no_arm"] = "yam"
    version: int = 1
    """Arm hardware revision (the v<N> model/config version)."""
    gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "flexible_4310", "yam_teaching_handle", "no_gripper"
    ] = "yam_teaching_handle"
    mode: Literal["follower", "leader", "visualizer_local", "visualizer_remote"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    telemetry_port: Optional[int] = None
    """Optional read-only leader telemetry Portal port. Follower telemetry is exposed on ``server_port``."""
    enable_preprogrammed_motion: bool = False
    """Expose guarded leader motion RPCs on telemetry_port; disabled for existing launch commands."""
    can_channel: str = "can0"
    bilateral_kp: float = 0.0
    gripper_command_closed: float = 0.0
    """Raw teaching-handle command considered fully closed; default preserves the original mapping."""
    gripper_command_open: float = 1.0
    """Raw teaching-handle command considered fully open; default preserves the original mapping."""
    sim: bool = False
    """Use SimRobot instead of real hardware."""
    ee_mass: Optional[float] = None
    """Override end-effector (link_6) mass in kg for gravity compensation. Defaults to the value in the XML."""


@dataclass
class Resources:
    """Resources accumulated during mode setup; ``cleanup`` drains them on exit.

    Mirrors the ``server_processes`` accumulator pattern from xdof/envs/launch.py:
    every spawn helper appends here so a single ``cleanup(resources)`` call tears
    down the mode regardless of how it exits."""

    processes: List[portal.Process] = field(default_factory=list)
    servers: List[portal.Server] = field(default_factory=list)
    pos_shared: Optional[portal.SharedArray] = None
    telemetry_shared: Optional["TelemetrySharedState"] = None
    stop_event: Optional[Any] = None


@dataclass
class TelemetrySharedState:
    """A lock-protected, single-writer telemetry snapshot shared with Portal.

    The hardware worker is the only writer. Portal request threads copy all fields
    while holding the same lock, so a client never sees position, velocity, effort,
    timestamp, or sequence values from different hardware reads.
    """

    joint_state: portal.SharedArray
    n_dofs: Any
    timestamp_ns: Any
    sequence: Any
    lock: Any
    # These fields are diagnostics for the leader teaching handle.  They are
    # optional so small test doubles and older callers can still construct the
    # state object with only the original joint snapshot fields.
    button_state: Optional[portal.SharedArray] = None
    synchronized: Any = None


def _make_telemetry_shared_state() -> TelemetrySharedState:
    """Allocate the small shared-memory snapshot used by a read-only telemetry server."""
    return TelemetrySharedState(
        joint_state=portal.SharedArray(shape=(_TELEMETRY_COMPONENTS, _MAX_DOFS), dtype=np.float64),
        n_dofs=portal.mp.Value("i", 0),
        timestamp_ns=portal.mp.Value("q", 0),
        sequence=portal.mp.Value("q", 0),
        lock=portal.mp.Lock(),
        button_state=portal.SharedArray(shape=(_TELEMETRY_BUTTONS,), dtype=np.float64),
        synchronized=portal.mp.Value("b", False),
    )


def _publish_telemetry(
    telemetry_shared: Optional[TelemetrySharedState],
    position: np.ndarray,
    velocity: np.ndarray,
    effort: np.ndarray,
    *,
    button_state: Optional[np.ndarray] = None,
    synchronized: Optional[bool] = None,
) -> None:
    """Publish one coherent telemetry sample from the hardware-owning worker.

    ``button_state`` and ``synchronized`` are optional diagnostics.  They are
    intentionally kept outside the joint-state arrays so the Piper-compatible
    data writer can continue to consume the original three numeric components.
    """
    if telemetry_shared is None:
        return

    position = np.asarray(position, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    effort = np.asarray(effort, dtype=np.float64).reshape(-1)
    n_dofs = len(position)
    if len(velocity) != n_dofs or len(effort) != n_dofs:
        raise ValueError("telemetry position, velocity, and effort must have equal lengths")
    if n_dofs > _MAX_DOFS:
        raise ValueError(f"telemetry has {n_dofs} DOF, exceeding shared-state limit {_MAX_DOFS}")

    buttons = None
    if button_state is not None:
        buttons = np.asarray(button_state, dtype=np.float64).reshape(-1)
        if len(buttons) > _TELEMETRY_BUTTONS:
            raise ValueError(
                f"telemetry has {len(buttons)} buttons, exceeding shared-state limit {_TELEMETRY_BUTTONS}"
            )

    timestamp_ns = time.time_ns()
    with telemetry_shared.lock:
        telemetry_shared.joint_state.array[0, :n_dofs] = position
        telemetry_shared.joint_state.array[1, :n_dofs] = velocity
        telemetry_shared.joint_state.array[2, :n_dofs] = effort
        telemetry_shared.n_dofs.value = n_dofs
        telemetry_shared.timestamp_ns.value = timestamp_ns
        telemetry_shared.sequence.value += 1
        if telemetry_shared.button_state is not None and buttons is not None:
            telemetry_shared.button_state.array[:] = 0.0
            telemetry_shared.button_state.array[: len(buttons)] = buttons
        if telemetry_shared.synchronized is not None and synchronized is not None:
            telemetry_shared.synchronized.value = bool(synchronized)


def _get_telemetry_snapshot(telemetry_shared: TelemetrySharedState) -> Dict[str, Any]:
    """Copy the latest telemetry sample into a Portal-serializable dictionary."""
    with telemetry_shared.lock:
        n_dofs = int(telemetry_shared.n_dofs.value)
        state = telemetry_shared.joint_state.array[:, :n_dofs]
        snapshot: Dict[str, Any] = {
            "timestamp_ns": int(telemetry_shared.timestamp_ns.value),
            "sequence": int(telemetry_shared.sequence.value),
            "position": state[0].tolist(),
            "velocity": state[1].tolist(),
            "effort": state[2].tolist(),
        }
        if telemetry_shared.button_state is not None:
            snapshot["button_state"] = telemetry_shared.button_state.array[:_TELEMETRY_BUTTONS].tolist()
        if telemetry_shared.synchronized is not None:
            snapshot["synchronized"] = bool(telemetry_shared.synchronized.value)
        return snapshot


def _robot_observations_to_telemetry(observations: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a Robot observation dictionary into full joint-space telemetry.

    ``MotorChainRobot`` reports an actuated gripper separately from the six arm
    joints. Joining its three matching fields here keeps the snapshot in the same
    order as ``get_joint_pos`` and avoids a second, potentially different state read.
    """
    position = np.asarray(observations["joint_pos"], dtype=np.float64).reshape(-1)
    velocity = np.asarray(observations["joint_vel"], dtype=np.float64).reshape(-1)
    effort = np.asarray(observations["joint_eff"], dtype=np.float64).reshape(-1)
    if "gripper_pos" in observations:
        position = np.concatenate([position, np.asarray(observations["gripper_pos"], dtype=np.float64).reshape(-1)])
        velocity = np.concatenate([velocity, np.asarray(observations["gripper_vel"], dtype=np.float64).reshape(-1)])
        effort = np.concatenate([effort, np.asarray(observations["gripper_eff"], dtype=np.float64).reshape(-1)])
    return position, velocity, effort


def _close_robot_safely(robot: Any, name: str) -> None:
    """Close a hardware-owning robot without masking an earlier worker failure."""
    if robot is None:
        return
    try:
        robot.close()
    except Exception:
        logging.exception("[%s] robot close failed", name)


def _spawn_into(processes: List[portal.Process], proc: portal.Process) -> portal.Process:
    """Start a portal.Process and append it to the cleanup pool."""
    proc.start()
    processes.append(proc)
    return proc


def _require_clean_worker_exit(proc: portal.Process, name: str) -> None:
    """Turn an event-driven worker shutdown into a visible parent failure."""
    proc.join(timeout=_WORKER_SHUTDOWN_TIMEOUT_S)
    if proc.exitcode != 0:
        raise RuntimeError(f"{name} stopped unexpectedly (exitcode={proc.exitcode})")


# ---------------------------------------------------------------------------
# Background workers. Each runs in its own portal.Process so the foreground
# loop (web IO server, web IO client, or MuJoCo viewer) never blocks on
# hardware or RPC.
# ---------------------------------------------------------------------------


def _run_yam_polling_loop(
    yam: Robot,
    pos_shared: "portal.SharedArray",
    telemetry_shared: Optional[TelemetrySharedState],
    n_dofs_value: Any,
    cmd_queue: Any,
    stop_event: Any,
    rate_name: str,
) -> None:
    """Poll an already-created robot until the owner requests a stop."""
    n_dofs_value.value = yam.num_dofs()
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    while not stop_event.is_set():
        if not bool(getattr(yam, "is_running", True)):
            raise RuntimeError(f"{rate_name}: underlying motor control loop stopped")
        if cmd_queue is not None:
            # Drain to the newest command and drop any stale backlog. Commands arrive (over RPC)
            # at least as fast as this loop actuates them, so replaying every queued command in
            # FIFO order would make the follower lag further and further behind. Mirrors the
            # leader io loop's drain-to-newest (_run_leader_io_loop).
            cmd = None
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
            if cmd is not None:
                if isinstance(cmd, tuple) and cmd == ("enter_gravity_comp_idle",):
                    if not hasattr(yam, "enter_gravity_comp_idle"):
                        raise RuntimeError(f"{rate_name}: robot does not support gravity-comp idle")
                    yam.enter_gravity_comp_idle()
                else:
                    yam.command_joint_pos(np.asarray(cmd))
        try:
            pos, velocity, effort = _robot_observations_to_telemetry(yam.get_observations())
            pos_shared.array[: len(pos)] = pos
            _publish_telemetry(telemetry_shared, pos, velocity, effort)
            rate.track()
        except Exception as e:
            if not bool(getattr(yam, "is_running", True)):
                raise RuntimeError(f"{rate_name}: underlying motor control loop stopped") from e
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)
        time.sleep(_WORKER_LOOP_PERIOD_S)  # pace: get_joint_pos is a cached read, so this loop would otherwise spin


def _yam_polling_worker_impl(
    args: "Args",
    pos_shared: "portal.SharedArray",
    telemetry_shared: Optional[TelemetrySharedState],
    n_dofs_value: Any,
    cmd_queue: Any,
    stop_event: Any,
    rate_name: str,
) -> None:
    """Owns the YAM hardware. Streams joint_pos into ``pos_shared`` at hardware
    rate and, when configured, publishes coherent read-only telemetry. If
    ``cmd_queue`` is provided, drains pending commands onto the YAM. Publishes
    the actual DOF count via ``n_dofs_value`` once the YAM is up."""
    override_log_level()
    yam: Optional[Robot] = None
    try:
        arm_type = ArmType.from_string_name(args.arm)
        gripper_type = GripperType.from_string_name(args.gripper)
        yam = get_yam_robot(
            channel=args.can_channel,
            arm_type=arm_type,
            version=args.version,
            gripper_type=gripper_type,
            ee_mass=args.ee_mass,
            sim=args.sim,
        )
        _run_yam_polling_loop(yam, pos_shared, telemetry_shared, n_dofs_value, cmd_queue, stop_event, rate_name)
    finally:
        # Publish the stop signal before closing the robot.  A hardware close can
        # block briefly while its control thread unwinds; the owner must still
        # stop serving cached state and enter its bounded cleanup path promptly.
        if not stop_event.is_set():
            stop_event.set()
        _close_robot_safely(yam, rate_name)


def _yam_polling_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    telemetry_shared: Optional[TelemetrySharedState],
    n_dofs_value: Any,
    cmd_queue: Any,
    stop_event: Any,
    rate_name: str,
) -> None:
    """Run the hardware worker and propagate fatal CAN faults to its owner."""
    try:
        _yam_polling_worker_impl(
            args,
            pos_shared,
            telemetry_shared,
            n_dofs_value,
            cmd_queue,
            stop_event,
            rate_name,
        )
    except Exception:
        stop_event.set()
        logging.exception("[%s] hardware worker stopped; shutting down its server", rate_name)
        raise


def _rpc_polling_worker(
    server_port: int,
    server_host: str,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    rate_name: str,
) -> None:
    """Polls a remote follower over portal RPC and caches the latest joint_pos
    in ``pos_shared`` so the foreground viewer reads from memory."""
    override_log_level()
    client: Optional[ClientRobot] = None
    try:
        client = ClientRobot(server_port, host=server_host)
        rate = RateRecorder(name=rate_name, report_interval=1.0)
        rate.start()
        while not stop_event.is_set():
            try:
                pos = client.get_joint_pos()
                pos_shared.array[: len(pos)] = pos
                rate.track()
            except Exception as e:
                logging.error(f"[{rate_name}] error: {e}")
                time.sleep(0.1)
    finally:
        _close_robot_safely(client, rate_name)


def _run_leader_control_loop(
    yam: MotorChainRobot,
    pos_shared: "portal.SharedArray",
    telemetry_shared: Optional[TelemetrySharedState],
    cmd_queue: Any,
    stop_event: Any,
    bilateral_kp: float,
    gripper_command_closed: float = 0.0,
    gripper_command_open: float = 1.0,
    preprogrammed_move_queue: Any = None,
    preprogrammed_move_result_queue: Any = None,
    cancelled_move_id: Any = None,
) -> None:
    """Run leader encoder polling and bilateral command generation."""
    robot = YAMLeaderRobot(
        yam,
        gripper_command_closed=gripper_command_closed,
        gripper_command_open=gripper_command_open,
    )
    robot_current_kp = yam._kp
    robot_current_kd = yam._kd.copy() if preprogrammed_move_queue is not None else None
    last_info_warning = 0.0

    def read_info() -> Tuple[np.ndarray, np.ndarray]:
        """Read leader state and rate-limit diagnostics for slow encoder reads."""
        nonlocal last_info_warning
        started = time.monotonic()
        result = robot.get_info()
        elapsed = time.monotonic() - started
        now = time.monotonic()
        if elapsed > _LEADER_INFO_WARN_S and now - last_info_warning >= 1.0:
            logging.warning("yam-leader get_info took %.1f ms (passive encoder/state lock)", elapsed * 1000.0)
            last_info_warning = now
        return result

    def publish_info(
        joint_pos: np.ndarray,
        button_state: np.ndarray,
        is_synchronized: bool,
    ) -> None:
        _publish_telemetry(
            telemetry_shared,
            joint_pos,
            np.zeros_like(joint_pos),
            np.zeros_like(joint_pos),
            button_state=button_state,
            synchronized=is_synchronized,
        )

    control_rate = RateRecorder(name="yam-leader control", report_interval=1.0)
    control_rate.start()

    # The encoder device states are populated only after the first CAN read, so get_info() can
    # raise on startup. Retry until the first reading lands before entering the control loop.
    while not stop_event.is_set():
        try:
            if not bool(getattr(yam, "is_running", True)):
                raise RuntimeError("leader underlying motor control loop stopped")
            current_joint_pos, current_button = read_info()
            publish_info(current_joint_pos, current_button, False)
            break
        except RuntimeError:
            if not bool(getattr(yam, "is_running", True)):
                raise
            time.sleep(0.05)
    else:
        return  # stop requested before the encoder became ready
    current_follower_joint_pos = pos_shared.array.copy()
    logging.info(f"Current leader joint pos: {current_joint_pos}")
    logging.info(f"Current follower joint pos: {current_follower_joint_pos}")

    def slow_move(
        joint_pos: np.ndarray,
        is_synchronized: bool,
        duration: float = 3.0,
    ) -> None:
        start_pos = pos_shared.array.copy()
        steps = 100
        dt = duration / steps
        started = time.monotonic()
        for i in range(steps):
            if stop_event.is_set():
                return
            if not bool(getattr(yam, "is_running", True)):
                raise RuntimeError("leader underlying motor control loop stopped during synchronization")
            cmd = joint_pos * i / steps + start_pos * (1 - i / steps)
            # A setpoint stream is latest-wins.  A bounded, non-blocking put
            # prevents a Portal pipe stall from pausing the telemetry heartbeat.
            _put_latest_command(cmd_queue, cmd)
            # Keep telemetry tied to a real hardware read while the follower
            # catches up. The operator may still move a zero-Kp leader during
            # this interval, so replaying the initial target would be fresh but
            # factually stale data.
            live_joint_pos, live_button_state = read_info()
            publish_info(live_joint_pos, live_button_state, is_synchronized)
            time.sleep(dt)
        logging.info("yam-leader synchronization move completed in %.2f s", time.monotonic() - started)

    def execute_preprogrammed_move(request: tuple[Any, ...]) -> tuple[int, bool, str, list[float]]:
        request_id, expected_start, target, duration_s, tolerance = request
        control_changed = False
        previous_kp, previous_kd = yam._kp.copy(), yam._kd.copy()
        try:
            expected = np.asarray(expected_start, dtype=np.float64)
            destination = np.asarray(target, dtype=np.float64)
            if expected.shape != (6,) or destination.shape != (6,):
                raise ValueError("leader start and target must each contain six arm joints")
            if not np.isfinite(expected).all() or not np.isfinite(destination).all():
                raise ValueError("leader start and target must contain finite joint positions")
            if not math.isfinite(float(duration_s)) or not 0.5 <= float(duration_s) <= 120.0:
                raise ValueError("duration_s must be between 0.5 and 120 seconds")
            if not math.isfinite(float(tolerance)) or not 0.0 <= float(tolerance) <= 0.5:
                raise ValueError("position tolerance must be between 0 and 0.5 radians")
            if synchronized:
                raise RuntimeError("leader is synchronized; disable teleoperation before moving")
            current, buttons = read_info()
            if buttons[0] > 0.5:
                raise RuntimeError("leader button 0 is pressed; release it before preprogrammed motion")
            start_error = float(np.max(np.abs(current[:6] - expected)))
            if start_error > tolerance:
                raise RuntimeError(f"leader is not at the recorded start pose: max joint error={start_error:.4f} rad")
            max_joint_speed = float(np.max(np.abs(destination - current[:6]))) / float(duration_s)
            if max_joint_speed > _PREPROGRAMMED_MAX_JOINT_SPEED_RAD_S:
                raise RuntimeError(
                    f"requested joint speed {max_joint_speed:.3f} rad/s exceeds the "
                    f"{_PREPROGRAMMED_MAX_JOINT_SPEED_RAD_S:.2f} rad/s limit"
                )
            clipped = yam._clip_robot_joint_pos_command(destination.copy())
            if not np.allclose(clipped, destination, atol=1e-9, rtol=0.0):
                raise ValueError("leader target exceeds configured joint limits")
            if cancelled_move_id is not None and int(cancelled_move_id.value) == int(request_id):
                raise RuntimeError("preprogrammed move was cancelled before motion started")

            assert robot_current_kd is not None
            robot.update_kp_kd(kp=robot_current_kp.copy(), kd=robot_current_kd.copy())
            control_changed = True
            steps = max(1, math.ceil(float(duration_s) * 25.0))
            period_s = float(duration_s) / steps
            for step in range(1, steps + 1):
                if stop_event.is_set() or not bool(getattr(yam, "is_running", True)):
                    raise RuntimeError("leader controller stopped during preprogrammed motion")
                if cancelled_move_id is not None and int(cancelled_move_id.value) == int(request_id):
                    raise RuntimeError("preprogrammed move was cancelled")
                live, buttons = read_info()
                if buttons[0] > 0.5:
                    raise RuntimeError("leader button 0 was pressed during preprogrammed motion")
                alpha = step / steps
                robot.command_joint_pos((1.0 - alpha) * current[:6] + alpha * destination)
                publish_info(live, buttons, False)
                time.sleep(period_s)

            settle_deadline = time.monotonic() + min(5.0, max(1.0, float(duration_s) * 0.5))
            while time.monotonic() < settle_deadline:
                if stop_event.is_set() or not bool(getattr(yam, "is_running", True)):
                    raise RuntimeError("leader controller stopped while settling")
                if cancelled_move_id is not None and int(cancelled_move_id.value) == int(request_id):
                    raise RuntimeError("preprogrammed move was cancelled while settling")
                live, buttons = read_info()
                if buttons[0] > 0.5:
                    raise RuntimeError("leader button 0 was pressed before reaching the target")
                if float(np.max(np.abs(live[:6] - destination))) <= tolerance:
                    return int(request_id), True, "", live.tolist()
                robot.command_joint_pos(destination)
                publish_info(live, buttons, synchronized)
                time.sleep(0.02)
            raise RuntimeError("leader did not reach the target within the position tolerance")
        except Exception as exc:
            try:
                live, buttons = read_info()
                publish_info(live, buttons, synchronized)
                final_position = live.tolist()
            except Exception:
                final_position = []
            return int(request_id), False, str(exc), final_position
        finally:
            if control_changed:
                try:
                    yam.enter_gravity_comp_idle()
                except Exception as exc:
                    raise RuntimeError("could not release leader after preprogrammed motion") from exc
                finally:
                    robot.update_kp_kd(kp=previous_kp, kd=previous_kd)

    synchronized = False
    while not stop_event.is_set():
        if not bool(getattr(yam, "is_running", True)):
            raise RuntimeError("leader underlying motor control loop stopped")
        control_rate.track()
        current_joint_pos, current_button = read_info()
        publish_info(current_joint_pos, current_button, synchronized)
        if preprogrammed_move_queue is not None:
            try:
                request = preprogrammed_move_queue.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:
                if preprogrammed_move_result_queue is None:
                    raise RuntimeError("preprogrammed move result channel is unavailable")
                result = execute_preprogrammed_move(request)
                preprogrammed_move_result_queue.put(result, timeout=1.0)
                current_joint_pos, current_button = read_info()
                publish_info(current_joint_pos, current_button, synchronized)
                while current_button[0] > 0.5 and not stop_event.is_set():
                    time.sleep(0.03)
                    current_joint_pos, current_button = read_info()
                    publish_info(current_joint_pos, current_button, synchronized)
                continue
        if current_button[0] > 0.5:
            if not synchronized:
                robot.update_kp_kd(kp=robot_current_kp * bilateral_kp, kd=np.ones(6) * 0.0)
                robot.command_joint_pos(current_joint_pos[:6])
                slow_move(current_joint_pos, False)
            else:
                logging.info("clear bilateral pd")
                robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                current_follower_joint_pos = pos_shared.array.copy()
                robot.command_joint_pos(current_follower_joint_pos[:6])
            synchronized = not synchronized
            publish_info(current_joint_pos, current_button, synchronized)
            while current_button[0] > 0.5 and not stop_event.is_set():
                time.sleep(0.03)
                current_joint_pos, current_button = read_info()
                publish_info(current_joint_pos, current_button, synchronized)

            if stop_event.is_set():
                return

        current_follower_joint_pos = pos_shared.array.copy()

        if synchronized:
            _put_latest_command(cmd_queue, current_joint_pos)
            # bilateral feedback to the leader, proportional to bilateral_kp
            robot.command_joint_pos(current_follower_joint_pos[:6])

        # Pace the loop: get_info reads cached state, so without this the worker spins a core and
        # competes with the in-process robot_server control thread for the GIL, adding jitter.
        time.sleep(_WORKER_LOOP_PERIOD_S)


def _leader_control_worker_impl(
    args: "Args",
    pos_shared: "portal.SharedArray",
    telemetry_shared: Optional[TelemetrySharedState],
    cmd_queue: Any,
    stop_event: Any,
    preprogrammed_move_queue: Any = None,
    preprogrammed_move_result_queue: Any = None,
    cancelled_move_id: Any = None,
) -> None:
    """Owns the leader YAM hardware and runs the bilateral control
    loop. Reads the latest follower pos from ``pos_shared`` (written by main's
    foreground web-IO loop) and pushes commands for the follower into
    ``cmd_queue`` (drained by main and forwarded over RPC). Never touches the
    network itself."""
    override_log_level()
    yam: Optional[MotorChainRobot] = None
    try:
        arm_type = ArmType.from_string_name(args.arm)
        gripper_type = GripperType.from_string_name(args.gripper)
        yam = get_yam_robot(
            channel=args.can_channel,
            arm_type=arm_type,
            version=args.version,
            gripper_type=gripper_type,
            ee_mass=args.ee_mass,
            sim=args.sim,
        )
        _run_leader_control_loop(
            yam,
            pos_shared,
            telemetry_shared,
            cmd_queue,
            stop_event,
            args.bilateral_kp,
            args.gripper_command_closed,
            args.gripper_command_open,
            preprogrammed_move_queue,
            preprogrammed_move_result_queue,
            cancelled_move_id,
        )
    finally:
        if not stop_event.is_set():
            stop_event.set()
        _close_robot_safely(yam, "leader yam-hardware")


def _leader_control_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    telemetry_shared: Optional[TelemetrySharedState],
    cmd_queue: Any,
    stop_event: Any,
    preprogrammed_move_queue: Any = None,
    preprogrammed_move_result_queue: Any = None,
    cancelled_move_id: Any = None,
) -> None:
    """Run leader control and fail closed when its CAN worker faults."""
    try:
        _leader_control_worker_impl(
            args,
            pos_shared,
            telemetry_shared,
            cmd_queue,
            stop_event,
            preprogrammed_move_queue,
            preprogrammed_move_result_queue,
            cancelled_move_id,
        )
    except Exception:
        stop_event.set()
        logging.exception("leader hardware worker stopped; shutting down teleoperation")
        raise


# ---------------------------------------------------------------------------
# Foreground helpers and loops
# ---------------------------------------------------------------------------


def _wait_for_dof(n_dofs_value: Any, timeout_s: float = 30.0, stop_event: Any = None) -> int:
    """Block until the background worker publishes the active DOF count."""
    deadline = time.time() + timeout_s
    while n_dofs_value.value == 0:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("hardware worker stopped before publishing its DOF count")
        if time.time() > deadline:
            raise TimeoutError("worker did not publish DOF count within timeout")
        time.sleep(0.05)
    return int(n_dofs_value.value)


def _put_latest_command(cmd_queue: Any, item: np.ndarray) -> None:
    """Non-blocking latest-wins enqueue onto a (bounded) command queue.

    Never blocks the caller: if the queue is full (the hardware worker has not drained it
    yet), drop the stale item and replace it with the newest, so the follower always actuates
    the freshest target. A setpoint stream is latest-wins, not a work queue."""
    while True:
        try:
            cmd_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                cmd_queue.get_nowait()
            except queue.Empty:
                pass


def _build_follower_server(
    server_port: int,
    n_dofs_value: Any,
    pos_shared: "portal.SharedArray",
    telemetry_shared: TelemetrySharedState,
    cmd_queue: Any,
) -> portal.Server:
    """Build the portal.Server that exposes the follower YAM over RPC. Returned
    unstarted; caller does ``server.start()`` to block."""
    io_rate = RateRecorder(name=f"ServerRobot[{server_port}] io", report_interval=1.0)
    io_rate.start()
    io_lock = threading.Lock()

    def _tracked(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            with io_lock:
                io_rate.track()
            return fn(*a, **kw)

        return wrapper

    def _get_pos() -> np.ndarray:
        return pos_shared.array[: n_dofs_value.value].copy()

    def _cmd_pos(joint_pos: np.ndarray) -> None:
        _put_latest_command(cmd_queue, np.asarray(joint_pos))

    def _cmd_state(state: Dict[str, np.ndarray]) -> None:
        key = "position" if "position" in state else "pos"
        _put_latest_command(cmd_queue, np.asarray(state[key]))

    def _cmd_gravity_comp_idle() -> None:
        _put_latest_command(cmd_queue, ("enter_gravity_comp_idle",))

    def _get_obs() -> Dict[str, np.ndarray]:
        return {"joint_pos": _get_pos()}

    server = portal.Server(server_port)
    logging.info(f"Robot Server Binding to {server_port}")
    server.bind("num_dofs", _tracked(lambda: n_dofs_value.value))
    server.bind("get_joint_pos", _tracked(_get_pos))
    server.bind("command_joint_pos", _tracked(_cmd_pos))
    server.bind("command_joint_state", _tracked(_cmd_state))
    server.bind("enter_gravity_comp_idle", _tracked(_cmd_gravity_comp_idle))
    server.bind("get_observations", _tracked(_get_obs))
    server.bind("get_telemetry", _tracked(lambda: _get_telemetry_snapshot(telemetry_shared)))
    return server


def _build_telemetry_server(
    server_port: int,
    telemetry_shared: TelemetrySharedState,
    preprogrammed_move_queue: Any = None,
    preprogrammed_move_result_queue: Any = None,
    cancelled_move_id: Any = None,
) -> portal.Server:
    """Build a leader telemetry server with an optional guarded move endpoint.

    The caller starts this server with ``block=False`` so it cannot interfere
    with the leader's foreground follower-RPC loop.
    """
    motion_enabled = preprogrammed_move_queue is not None and preprogrammed_move_result_queue is not None
    server = portal.Server(server_port, errors=not motion_enabled)
    logging.info(f"Telemetry Server Binding to {server_port}")
    server.bind(
        "get_telemetry",
        lambda: {
            **_get_telemetry_snapshot(telemetry_shared),
            "preprogrammed_motion_available": motion_enabled,
        },
    )
    if motion_enabled:
        move_lock = threading.Lock()
        active_move_id = [0]
        server.bind("preprogrammed_motion_ready", lambda: {"version": 1, "available": True})

        def preprogrammed_move(
            request_id: int,
            expected_start: np.ndarray,
            target: np.ndarray,
            duration_s: float,
            position_tolerance_rad: float,
        ) -> dict[str, Any]:
            request_id = int(request_id)
            expected = np.asarray(expected_start, dtype=np.float64)
            destination = np.asarray(target, dtype=np.float64)
            if request_id <= 0:
                raise ValueError("request_id must be positive")
            if expected.shape != (6,) or destination.shape != (6,):
                raise ValueError("leader expected_start and target must each contain six values")
            if not np.isfinite(expected).all() or not np.isfinite(destination).all():
                raise ValueError("leader expected_start and target must be finite")
            if not 0.5 <= float(duration_s) <= 120.0:
                raise ValueError("duration_s must be between 0.5 and 120 seconds")
            if not 0.0 <= float(position_tolerance_rad) <= 0.5:
                raise ValueError("position_tolerance_rad must be between 0 and 0.5 radians")
            if not move_lock.acquire(blocking=False):
                raise RuntimeError("another leader preprogrammed move is already pending")
            try:
                active_move_id[0] = request_id
                try:
                    try:
                        preprogrammed_move_queue.put(
                            (
                                request_id,
                                expected.tolist(),
                                destination.tolist(),
                                float(duration_s),
                                float(position_tolerance_rad),
                            ),
                            timeout=0.5,
                        )
                    except queue.Full as exc:
                        raise RuntimeError("another leader preprogrammed move is already pending") from exc
                    deadline = time.monotonic() + float(duration_s) + 8.0
                    while True:
                        try:
                            completed_id, passed, error, final_position = preprogrammed_move_result_queue.get(
                                timeout=max(0.0, deadline - time.monotonic())
                            )
                        except queue.Empty as exc:
                            if cancelled_move_id is not None:
                                cancelled_move_id.value = request_id
                            raise TimeoutError("leader motion worker did not report completion in time") from exc
                        if int(completed_id) != request_id:
                            logging.warning("discarding stale leader motion result for request %s", completed_id)
                            continue
                        if not passed:
                            raise RuntimeError(str(error))
                        return {"request_id": request_id, "position": list(final_position)}
                finally:
                    if active_move_id[0] == request_id:
                        active_move_id[0] = 0
            finally:
                move_lock.release()

        server.bind("preprogrammed_move", preprogrammed_move, workers=2)
        if cancelled_move_id is not None:
            server.bind(
                "cancel_preprogrammed_move",
                lambda request_id: _cancel_preprogrammed_move(
                    cancelled_move_id, active_move_id, move_lock, int(request_id)
                ),
            )
    return server


def _cancel_preprogrammed_move(
    cancelled_move_id: Any,
    active_move_id: list[int],
    move_lock: threading.Lock,
    request_id: int,
) -> bool:
    """Set a request-specific cancellation flag consumed by the leader worker."""
    if request_id <= 0:
        return False
    if not move_lock.acquire(timeout=0.05):
        cancelled_move_id.value = request_id
        return True
    try:
        if int(active_move_id[0]) != request_id:
            logging.info("recorded cancellation for leader motion request %s before it was queued", request_id)
        cancelled_move_id.value = request_id
        return True
    finally:
        move_lock.release()


def _run_leader_io_loop(
    client_robot: ClientRobot,
    cmd_queue: Any,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    io_rate: RateRecorder,
    hardware_worker: Optional[portal.Process] = None,
) -> None:
    """Foreground loop for leader mode: drain commands from the bilateral
    control worker onto the follower over RPC, and pull the follower's latest
    joint_pos into ``pos_shared`` so the worker can read it for bilateral
    feedback."""
    failure_started: Optional[float] = None
    while not stop_event.is_set():
        if hardware_worker is not None and hardware_worker.exitcode is not None:
            stop_event.set()
            raise RuntimeError(f"leader hardware worker exited (exitcode={hardware_worker.exitcode})")
        # Drain to the newest command and drop any stale backlog: the control worker can enqueue
        # faster than each blocking RPC round-trip drains, so replaying every queued command would
        # make the follower lag further and further behind.
        cmd = None
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break
        try:
            if cmd is not None:
                client_robot.command_joint_pos(cmd)
            pos = client_robot.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            io_rate.track()
            failure_started = None
        except Exception as e:
            # A short outage can occur while the follower's RPC server restarts,
            # but retrying indefinitely leaves an energized leader running with
            # stale follower feedback.  Stop the shared hardware worker after a
            # bounded continuous failure period so its normal cleanup path runs.
            now = time.monotonic()
            if failure_started is None:
                failure_started = now
            elapsed = now - failure_started
            if elapsed >= _LEADER_RPC_FAILURE_TIMEOUT_S:
                stop_event.set()
                raise RuntimeError(
                    f"leader follower RPC unavailable for {elapsed:.2f}s; stopping teleoperation"
                ) from e
            logging.error(
                "[yam-leader web-port io] error (retrying for %.2fs more): %s",
                _LEADER_RPC_FAILURE_TIMEOUT_S - elapsed,
                e,
            )
            stop_event.wait(_LEADER_RPC_RETRY_PERIOD_S)


def _run_viewer_loop(
    arm_type: ArmType,
    gripper_type: GripperType,
    version: int,
    get_pos_fn: Callable[[], np.ndarray],
) -> None:
    """Foreground MuJoCo viewer. Pulls joint_pos from ``get_pos_fn`` each frame
    — typically a snapshot of a SharedArray populated by a background worker."""
    xml_path = combine_arm_and_gripper_xml(arm_type, gripper_type, version=version)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt: float = 0.01
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            joint_pos = get_pos_fn()
            nq = model.nq
            n = min(len(joint_pos), nq)
            data.qpos[:n] = joint_pos[:n]

            for j in range(model.njnt):
                adr = model.jnt_qposadr[j]
                if adr >= n:
                    continue
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                    lo, hi = model.jnt_range[j]
                    data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

            for i in range(model.neq):
                if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                adr1 = model.jnt_qposadr[model.eq_obj1id[i]]
                adr2 = model.jnt_qposadr[model.eq_obj2id[i]]
                coef = model.eq_data[i, :5]
                data.qpos[adr2] = np.polyval(coef[::-1], data.qpos[adr1])

            mujoco.mj_forward(model, data)
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _stop_process_safely(proc: portal.Process, timeout_s: float = 1.0) -> None:
    """Give a worker time to close CAN/Portal resources before killing it."""
    try:
        proc.join(timeout=timeout_s)
    except Exception:
        logging.exception("worker join failed")
    if getattr(proc, "exitcode", None) is None:
        try:
            proc.kill(timeout=timeout_s)
            proc.join(timeout=timeout_s)
        except Exception:
            logging.exception("worker forced stop failed")


def cleanup(resources: Resources) -> None:
    """Tear down accumulated resources with a bounded graceful phase."""
    logging.info("exiting")
    if resources.stop_event is not None:
        resources.stop_event.set()
    for server in resources.servers:
        try:
            if server.running:
                server.close(timeout=_WORKER_SHUTDOWN_TIMEOUT_S)
        except Exception:
            logging.exception("Portal server close failed")
    for proc in resources.processes:
        _stop_process_safely(proc)
    if resources.pos_shared is not None:
        resources.pos_shared.close()
    if resources.telemetry_shared is not None:
        resources.telemetry_shared.joint_state.close()
        button_state = getattr(resources.telemetry_shared, "button_state", None)
        if button_state is not None:
            button_state.close()
    logging.info("Processes terminated.")


# ---------------------------------------------------------------------------
# Per-mode runners. Each owns its own setup → foreground loop → cleanup, the
# same shape as ``main`` in xdof/envs/launch.py: build dependencies, call the
# control loop, wrap in try/finally with a single ``cleanup``.
# ---------------------------------------------------------------------------


def run_follower(args: Args) -> None:
    telemetry_shared = _make_telemetry_shared_state()
    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        telemetry_shared=telemetry_shared,
        stop_event=portal.mp.Event(),
    )
    spawn = partial(_spawn_into, resources.processes)

    n_dofs_value = portal.mp.Value("i", 0)
    # maxsize=1 so the command queue holds only the latest setpoint; _cmd_pos/_cmd_state drop
    # stale commands on overflow (see _build_follower_server). A setpoint stream is latest-wins,
    # not a work queue — bounding it prevents any backlog from accumulating on this hop.
    cmd_queue = portal.mp.Queue(maxsize=1)

    try:
        hardware_worker = spawn(
            portal.Process(
                _yam_polling_worker,
                args,
                resources.pos_shared,
                telemetry_shared,
                n_dofs_value,
                cmd_queue,
                resources.stop_event,
                "follower yam-hardware",
                name="follower-yam-control",
            )
        )
        n_dofs = _wait_for_dof(n_dofs_value, stop_event=resources.stop_event)
        logging.info(f"Follower YAM ready, num_dofs={n_dofs}")

        server = _build_follower_server(
            args.server_port, n_dofs_value, resources.pos_shared, telemetry_shared, cmd_queue
        )
        resources.servers.append(server)
    except BaseException:
        cleanup(resources)
        raise

    try:
        # Keep the owner process alive while the server serves requests, but also
        # watch the hardware worker's stop event. A CAN fault must close the RPC
        # endpoint instead of leaving a stale cached state available to clients.
        server.start(block=False)
        while not resources.stop_event.wait(0.1):
            if hardware_worker.exitcode is not None:
                raise RuntimeError(f"follower hardware worker exited (exitcode={hardware_worker.exitcode})")
        _require_clean_worker_exit(hardware_worker, "follower hardware worker")
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("follower mode failed")
        raise
    finally:
        cleanup(resources)


def run_leader(args: Args) -> None:
    if args.enable_preprogrammed_motion and args.telemetry_port is None:
        raise ValueError("--enable-preprogrammed-motion requires --telemetry-port")
    client_robot: Optional[ClientRobot] = None
    resources: Optional[Resources] = None
    try:
        client_robot = ClientRobot(args.server_port, host=args.server_host)
        initial_follower_pos = client_robot.get_joint_pos()
        logging.info(f"Initial follower joint pos: {initial_follower_pos}")

        telemetry_shared = _make_telemetry_shared_state() if args.telemetry_port is not None else None
        resources = Resources(
            pos_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
            telemetry_shared=telemetry_shared,
            stop_event=portal.mp.Event(),
        )
        resources.pos_shared.array[:] = initial_follower_pos
        # Setpoints are latest-wins; bounding this hop prevents a blocked
        # foreground RPC from accumulating an unbounded command backlog.
        cmd_queue = portal.mp.Queue(maxsize=1)
        preprogrammed_move_queue = portal.mp.Queue(maxsize=1) if args.enable_preprogrammed_motion else None
        preprogrammed_move_result_queue = portal.mp.Queue(maxsize=1) if args.enable_preprogrammed_motion else None
        cancelled_move_id = portal.mp.Value("q", 0) if args.enable_preprogrammed_motion else None

        spawn = partial(_spawn_into, resources.processes)
        hardware_worker = spawn(
            portal.Process(
                _leader_control_worker,
                args,
                resources.pos_shared,
                telemetry_shared,
                cmd_queue,
                resources.stop_event,
                preprogrammed_move_queue,
                preprogrammed_move_result_queue,
                cancelled_move_id,
                name="yam-leader-control",
            )
        )

        if args.telemetry_port is not None:
            assert telemetry_shared is not None
            telemetry_server = _build_telemetry_server(
                args.telemetry_port,
                telemetry_shared,
                preprogrammed_move_queue,
                preprogrammed_move_result_queue,
                cancelled_move_id,
            )
            resources.servers.append(telemetry_server)
            telemetry_server.start(block=False)

        io_rate = RateRecorder(name="yam-leader web-port io", report_interval=1.0)
        io_rate.start()
        _run_leader_io_loop(
            client_robot,
            cmd_queue,
            resources.pos_shared,
            resources.stop_event,
            io_rate,
            hardware_worker,
        )
        _require_clean_worker_exit(hardware_worker, "leader hardware worker")
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("leader mode failed")
        raise
    finally:
        try:
            if resources is not None:
                cleanup(resources)
        finally:
            _close_robot_safely(client_robot, "leader follower-client")


def run_visualizer_local(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    n_dofs_value = portal.mp.Value("i", 0)

    spawn = partial(_spawn_into, resources.processes)
    try:
        spawn(
            portal.Process(
                _yam_polling_worker,
                args,
                resources.pos_shared,
                None,
                n_dofs_value,
                None,  # no cmd_queue — viewer doesn't send commands
                resources.stop_event,
                "viz-local yam-read",
                name="viz-local-yam",
            )
        )
        n_dofs = _wait_for_dof(n_dofs_value, stop_event=resources.stop_event)
        logging.info(f"Viz YAM ready, num_dofs={n_dofs}")
    except BaseException:
        cleanup(resources)
        raise

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, args.version, get_pos)
    except Exception:
        logging.exception("visualizer_local mode failed")
        raise
    finally:
        cleanup(resources)


def run_visualizer_remote(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    with ClientRobot(args.server_port, host=args.server_host) as bootstrap:
        n_dofs = bootstrap.num_dofs()
    logging.info(f"Remote follower num_dofs={n_dofs}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )

    spawn = partial(_spawn_into, resources.processes)

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        spawn(
            portal.Process(
                _rpc_polling_worker,
                args.server_port,
                args.server_host,
                resources.pos_shared,
                resources.stop_event,
                "viz-remote rpc-poll",
                name="viz-remote-rpc",
            )
        )
        _run_viewer_loop(arm_type, gripper_type, args.version, get_pos)
    except Exception:
        logging.exception("visualizer_remote mode failed")
        raise
    finally:
        cleanup(resources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    # Make RateRecorder's logging.info output (and our own logging calls) visible.
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)

    if arm_type == ArmType.NO_ARM:
        # Every mode here needs an arm model (real YAM for follower/leader, an XML for the
        # visualizers); NO_ARM has no XML path and otherwise crashes deep inside the utils.
        raise ValueError("--arm 'no_arm' is not supported by minimum_gello; all modes require an arm")

    if args.mode == "leader" and args.sim:
        raise ValueError("Leader mode requires real hardware (--sim is not supported)")

    runners: Dict[str, Callable[[Args], None]] = {
        "follower": run_follower,
        "leader": run_leader,
        "visualizer_local": run_visualizer_local,
        "visualizer_remote": run_visualizer_remote,
    }
    runner = runners.get(args.mode)
    if runner is None:
        raise ValueError(f"Invalid mode: {args.mode}")
    runner(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
