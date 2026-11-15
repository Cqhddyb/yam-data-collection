from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from yam_data_collection.config import TelemetryEndpoint
from yam_data_collection.telemetry import PortalTelemetryClient, TelemetryError


def _endpoint() -> TelemetryEndpoint:
    return TelemetryEndpoint(role="left_leader", host="127.0.0.1", port=1245)


def _payload(sequence: int = 1) -> dict[str, object]:
    return {
        "timestamp_ns": 123_456_789,
        "sequence": sequence,
        "position": [0.1, 0.2],
        "velocity": [0.0, 0.0],
        "effort": [0.0, 0.0],
    }


def test_read_uses_explicit_bounded_connect_and_response_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[object] = []

    class Future:
        def __init__(self) -> None:
            self.timeout: float | None = None

        def result(self, timeout: float | None = None) -> dict[str, object]:
            self.timeout = timeout
            return _payload(sequence=7)

    class Client:
        def __init__(self, address: str, *, autoconn: bool) -> None:
            self.address = address
            self.autoconn = autoconn
            self.connect_timeout: float | None = None
            self.future = Future()
            clients.append(self)

        def connect(self, timeout: float | None = None) -> bool:
            self.connect_timeout = timeout
            return True

        def get_telemetry(self) -> Future:
            return self.future

        def close(self, timeout: float | None = None) -> None:
            del timeout

    monkeypatch.setitem(sys.modules, "portal", SimpleNamespace(Client=Client))

    telemetry = PortalTelemetryClient(_endpoint(), read_timeout_s=0.02)
    snapshot = telemetry.read(timeout_s=0.012)

    assert snapshot.sequence == 7
    assert snapshot.position == (0.1, 0.2)
    assert len(clients) == 1
    client = clients[0]
    assert isinstance(client, Client)
    assert client.address == "127.0.0.1:1245"
    assert client.autoconn is False
    assert client.connect_timeout == pytest.approx(0.012)
    assert client.future.timeout == pytest.approx(0.012)


def test_timed_out_read_replaces_client_before_the_next_request(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[object] = []

    class TimedOutFuture:
        def result(self, timeout: float | None = None) -> object:
            assert timeout == pytest.approx(0.01)
            raise TimeoutError("synthetic delayed reply")

    class ReadyFuture:
        def result(self, timeout: float | None = None) -> dict[str, object]:
            assert timeout == pytest.approx(0.01)
            return _payload(sequence=2)

    class Client:
        def __init__(self, address: str, *, autoconn: bool) -> None:
            assert address == "127.0.0.1:1245"
            assert autoconn is False
            self.closed_timeout: float | None = None
            self.future: TimedOutFuture | ReadyFuture = TimedOutFuture() if not clients else ReadyFuture()
            clients.append(self)

        def connect(self, timeout: float | None = None) -> bool:
            assert timeout == pytest.approx(0.01)
            return True

        def get_telemetry(self) -> TimedOutFuture | ReadyFuture:
            return self.future

        def close(self, timeout: float | None = None) -> None:
            self.closed_timeout = timeout

    monkeypatch.setitem(sys.modules, "portal", SimpleNamespace(Client=Client))

    telemetry = PortalTelemetryClient(_endpoint(), read_timeout_s=0.01)
    with pytest.raises(TelemetryError, match="timed out reading left_leader telemetry"):
        telemetry.read()

    assert len(clients) == 2
    first = clients[0]
    assert isinstance(first, Client)
    assert first.closed_timeout == pytest.approx(0.01)

    snapshot = telemetry.read()
    assert snapshot.sequence == 2


def test_legacy_portal_test_double_without_timeout_parameters_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[object] = []

    class Future:
        def result(self) -> dict[str, object]:
            return _payload(sequence=3)

    class LegacyClient:
        def __init__(self, address: str) -> None:
            assert address == "127.0.0.1:1245"
            self.close_calls = 0
            clients.append(self)

        def get_telemetry(self) -> Future:
            return Future()

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setitem(sys.modules, "portal", SimpleNamespace(Client=LegacyClient))

    telemetry = PortalTelemetryClient(_endpoint())
    assert telemetry.read().sequence == 3
    telemetry.close()

    assert len(clients) == 1
    client = clients[0]
    assert isinstance(client, LegacyClient)
    assert client.close_calls == 1


@pytest.mark.parametrize("timeout_s", [0.0, -0.01, float("inf"), float("nan")])
def test_invalid_read_timeout_is_rejected(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        PortalTelemetryClient._validate_timeout(timeout_s)
