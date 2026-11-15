"""Read-only Portal clients for the telemetry added to i2rt minimum_gello."""

from __future__ import annotations

import math
import threading
from typing import Any, Mapping

from .config import TelemetryEndpoint
from .models import RobotSnapshot


class TelemetryError(RuntimeError):
    """A configured i2rt telemetry endpoint did not provide a valid snapshot."""


class PortalTelemetryClient:
    """Connect to one read-only i2rt `get_telemetry` Portal endpoint."""

    DEFAULT_READ_TIMEOUT_S = 0.05
    """Bound one localhost telemetry RPC so a stalled Portal cannot block recording."""

    def __init__(self, endpoint: TelemetryEndpoint, *, read_timeout_s: float = DEFAULT_READ_TIMEOUT_S) -> None:
        try:
            import portal
        except ImportError as exc:  # pragma: no cover - exercised on deployment setup
            raise TelemetryError("portal is not installed; run `uv sync`") from exc
        self.endpoint = endpoint
        self._address = f"{endpoint.host}:{endpoint.port}"
        self._read_timeout_s = self._validate_timeout(read_timeout_s)
        self._client_factory = portal.Client
        self._rpc_lock = threading.Lock()
        self._client = self._make_client()

    @staticmethod
    def _validate_timeout(timeout_s: float) -> float:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("telemetry read timeout must be a positive finite number of seconds")
        return timeout_s

    def _make_client(self) -> Any:
        # Portal's automatic reconnect can leave Client.call waiting forever
        # for a missing peer. Explicit connections keep both connect and
        # response waits bounded. The fallback supports older test doubles
        # that only accept the endpoint address.
        try:
            return self._client_factory(self._address, autoconn=False)
        except TypeError:
            return self._client_factory(self._address)

    @staticmethod
    def _call_with_timeout(method: Any, timeout_s: float) -> Any:
        """Use Portal's timeout API while accepting legacy no-argument fakes."""
        try:
            return method(timeout=timeout_s)
        except TypeError:
            return method()

    def _connect_locked(self, timeout_s: float) -> None:
        connect = getattr(self._client, "connect", None)
        if connect is None:
            # Minimal test doubles historically exposed only get_telemetry.
            return
        connected = self._call_with_timeout(connect, timeout_s)
        if connected is False:
            raise TimeoutError(f"timed out connecting to Portal server {self._address}")

    def _close_client_locked(self, client: Any, timeout_s: float) -> None:
        close = getattr(client, "close", None)
        if close is not None:
            self._call_with_timeout(close, timeout_s)

    def _reset_client_locked(self, timeout_s: float) -> None:
        """Discard a timed-out Future before it can fill Portal's in-flight map."""
        stale_client = self._client
        try:
            self._close_client_locked(stale_client, timeout_s)
        except Exception:
            # The timed-out request is already unusable. A fresh client is
            # still preferable even when the old socket cannot close cleanly.
            pass
        self._client = self._make_client()

    @staticmethod
    def _float_tuple(value: Any, field: str) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)):
            raise TelemetryError(f"{field} must be a list or tuple")
        return tuple(float(item) for item in value)

    def read(self, *, timeout_s: float | None = None) -> RobotSnapshot:
        """Return one snapshot, failing promptly and resetting after a timeout."""
        effective_timeout_s = self._read_timeout_s if timeout_s is None else self._validate_timeout(timeout_s)
        try:
            with self._rpc_lock:
                try:
                    self._connect_locked(effective_timeout_s)
                    future = self._client.get_telemetry()
                    payload = self._call_with_timeout(future.result, effective_timeout_s)
                except TimeoutError:
                    self._reset_client_locked(effective_timeout_s)
                    raise
        except TimeoutError as exc:
            raise TelemetryError(
                f"timed out reading {self.endpoint.role} telemetry from {self._address} after "
                f"{effective_timeout_s:.3f}s"
            ) from exc
        except Exception as exc:
            raise TelemetryError(
                f"cannot read {self.endpoint.role} telemetry from {self._address}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TelemetryError(f"{self.endpoint.role} telemetry must be a mapping")
        try:
            timestamp_ns = int(payload["timestamp_ns"])
            sequence = int(payload["sequence"])
            position = self._float_tuple(payload["position"], "position")
            velocity = self._float_tuple(payload.get("velocity", []), "velocity")
            effort = self._float_tuple(payload.get("effort", []), "effort")
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryError(f"{self.endpoint.role} returned malformed telemetry: {payload!r}") from exc
        if timestamp_ns <= 0 or sequence < 0 or not position:
            raise TelemetryError(f"{self.endpoint.role} returned an incomplete telemetry snapshot")
        return RobotSnapshot(
            role=self.endpoint.role,
            source_timestamp_ns=timestamp_ns,
            sequence=sequence,
            position=position,
            velocity=velocity,
            effort=effort,
        )

    def close(self) -> None:
        with self._rpc_lock:
            self._close_client_locked(self._client, self._read_timeout_s)

    def __enter__(self) -> "PortalTelemetryClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
