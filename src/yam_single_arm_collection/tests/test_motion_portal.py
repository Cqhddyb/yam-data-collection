from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import portal
import pytest

from yam_single_arm_collection import motion


@pytest.mark.parametrize("from_current", [False, True])
def test_paired_preposition_over_portal_previews_executes_and_releases(
    tmp_path: Path, config_factory: Any, monkeypatch: pytest.MonkeyPatch, from_current: bool
) -> None:
    follower_port, leader_port = portal.free_port(), portal.free_port()
    config = config_factory("right")
    config = replace(
        config,
        telemetry={
            role: replace(endpoint, port=follower_port if role.endswith("follower") else leader_port)
            for role, endpoint in config.telemetry.items()
        },
    )
    monkeypatch.setattr(motion, "_PORTS", {"right": {"follower": follower_port, "leader": leader_port}})
    positions = {role: np.asarray([0.0] * 6 + [0.5]) for role in ("follower", "leader")}
    if from_current:
        positions["follower"][:6] = 0.15
        positions["leader"][:6] = -0.05
    initial_positions = {role: position.tolist() for role, position in positions.items()}
    commands: list[np.ndarray] = []
    releases: list[bool] = []
    sequences = {"follower": 0, "leader": 0}

    def telemetry(role: str) -> dict[str, Any]:
        sequences[role] += 1
        return {
            "position": positions[role].tolist(),
            "timestamp_ns": time.time_ns(),
            "sequence": sequences[role],
            "synchronized": False,
            "button_state": [0.0, 0.0],
            "preprogrammed_motion_available": role == "leader",
        }

    def command_follower(target: np.ndarray) -> None:
        positions["follower"] = np.asarray(target).copy()
        commands.append(positions["follower"].copy())

    def move_leader(
        request_id: int, start: np.ndarray, target: np.ndarray, duration_s: float, tolerance: float
    ) -> dict[str, Any]:
        assert positions["leader"][:6] == pytest.approx(start)
        assert tolerance > 0
        time.sleep(float(duration_s))
        positions["leader"][:6] = target
        return {"request_id": int(request_id), "position": positions["leader"].tolist()}

    follower = portal.Server(follower_port, errors=False)
    follower.bind("get_telemetry", lambda: telemetry("follower"))
    follower.bind("command_joint_pos", command_follower)
    follower.bind("enter_gravity_comp_idle", lambda: releases.append(True))
    leader = portal.Server(leader_port, workers=2, errors=False)
    leader.bind("get_telemetry", lambda: telemetry("leader"))
    leader.bind("preprogrammed_motion_ready", lambda: {"version": 1, "available": True})
    leader.bind("preprogrammed_move", move_leader)
    leader.bind("cancel_preprogrammed_move", lambda request_id: True)
    plan = tmp_path / "pair.json"
    plan.write_text(
        json.dumps(
            {
                "side": "right",
                "follower_start": [0.0] * 6 + [0.5],
                "leader_start": [0.0] * 6 + [0.5],
                "target": [0.45] * 6 + [0.5],
                "duration_s": 0.5,
            }
        ),
        encoding="utf-8",
    )
    follower.start(block=False)
    leader.start(block=False)
    try:
        poses = motion.read_pair_poses(config)
        assert json.loads(json.dumps(poses)) == poses
        assert poses["leader"]["button_state"] == [0.0, 0.0]
        assert not commands and not releases

        preview = motion.preposition_pair(config, plan, execute=False, from_current=from_current)
        assert not preview["executed"]
        assert preview["start_mode"] == ("current" if from_current else "recorded")
        assert preview["follower_start"] == initial_positions["follower"]
        assert preview["leader_start"] == initial_positions["leader"]
        assert not commands and not releases

        result = motion.preposition_pair(config, plan, from_current=from_current)

        assert result["executed"]
        assert result["verified_follower_final"] == pytest.approx([0.45] * 6 + [0.5])
        assert result["verified_leader_final"] == pytest.approx(result["verified_follower_final"])
        assert result["follower_before"] == initial_positions["follower"]
        assert result["leader_before"] == initial_positions["leader"]
        assert np.max(np.abs(commands[0][:6] - np.asarray(initial_positions["follower"][:6]))) <= 1.0 / 25
        assert len(commands) >= 13
        assert releases == [True]
    finally:
        leader.close(timeout=2.0)
        follower.close(timeout=2.0)
