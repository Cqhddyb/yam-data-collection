"""Fail-closed checks for cached MotorChainRobot state."""

import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from i2rt.robots.motor_chain_robot import MotorChainRobot


def _stopped_robot() -> Any:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = SimpleNamespace(running=False)
    robot._state_lock = threading.Lock()
    robot._joint_state = SimpleNamespace(
        pos=np.zeros(1),
        vel=np.zeros(1),
        eff=np.zeros(1),
        temp_mos=np.zeros(1),
        temp_rotor=np.zeros(1),
    )
    robot._gripper_index = None
    robot.temp_record_flag = False
    return robot


@pytest.mark.parametrize("getter", ["get_joint_pos", "get_observations"])
def test_state_getters_reject_stale_state_after_motor_fault(getter: str) -> None:
    robot = _stopped_robot()

    with pytest.raises(RuntimeError, match="underlying motor control loop is not running"):
        getattr(robot, getter)()


def test_is_running_reflects_motor_chain_health() -> None:
    robot = _stopped_robot()
    assert robot.is_running is False
    robot.motor_chain.running = True
    assert robot.is_running is True
