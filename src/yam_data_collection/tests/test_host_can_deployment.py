from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]


def _run_launcher_can_guard(link: dict) -> subprocess.CompletedProcess[str]:
    launcher = (DEPLOYMENT_ROOT / "yam.sh").read_text()
    guard = "import json, sys\n" + launcher.split("import json, sys\n", 1)[1].split("\n' \"$link_json\"", 1)[0]
    return subprocess.run(
        [sys.executable, "-c", guard, json.dumps([link])],
        capture_output=True,
        text=True,
        check=False,
    )


def _healthy_link() -> dict:
    return {
        "flags": ["UP", "LOWER_UP", "ECHO"],
        "txqlen": 2000,
        "linkinfo": {
            "info_kind": "can",
            "info_data": {
                "state": "ERROR-ACTIVE",
                "bittiming": {"bitrate": 1000000},
                "restart_ms": 0,
            },
        },
    }


def test_host_service_disables_unsupported_automatic_bus_off_restart() -> None:
    service = (DEPLOYMENT_ROOT / "deploy" / "yam-can@.service").read_text()
    assert "type can bitrate 1000000 restart-ms 0\n" in service
    assert "restart-ms 100" not in service


def test_launcher_accepts_host_compatible_can_configuration() -> None:
    assert _run_launcher_can_guard(_healthy_link()).returncode == 0


@pytest.mark.parametrize("state", ["BUS-OFF", "STOPPED", "ERROR-PASSIVE", "ERROR-WARNING"])
def test_launcher_rejects_unhealthy_can_state(state: str) -> None:
    link = copy.deepcopy(_healthy_link())
    link["linkinfo"]["info_data"]["state"] = state
    result = _run_launcher_can_guard(link)
    assert result.returncode != 0
    assert "ERROR-ACTIVE" in result.stderr


@pytest.mark.parametrize("invalid_setting", ["down", "bitrate", "restart_ms", "txqlen"])
def test_launcher_rejects_other_incorrect_can_settings(invalid_setting: str) -> None:
    link = copy.deepcopy(_healthy_link())
    if invalid_setting == "down":
        link["flags"].remove("UP")
    elif invalid_setting == "bitrate":
        link["linkinfo"]["info_data"]["bittiming"]["bitrate"] = 500000
    elif invalid_setting == "restart_ms":
        link["linkinfo"]["info_data"]["restart_ms"] = 100
    else:
        link["txqlen"] = 10
    assert _run_launcher_can_guard(link).returncode != 0
