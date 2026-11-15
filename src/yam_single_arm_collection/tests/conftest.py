from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from yam_data_collection.config import DatasetConfig

from yam_single_arm_collection.collector import SingleArmCollector
from yam_single_arm_collection.config import load_config


@pytest.fixture
def config_factory(tmp_path: Path) -> Callable[..., DatasetConfig]:
    def make_config(side: str = "right", setup: str = "wrist", mode: str = "60hz") -> DatasetConfig:
        config = load_config(side=side, camera_setup=setup, capture_mode=mode)
        dataset = tmp_path / "data" / side / setup / mode
        raw = deepcopy(dict(config.raw))
        raw["collection_config"]["dataset_dir"] = str(dataset)
        raw["collection_config"]["recorder_cpu_set"] = None
        raw["collection_config"]["live_video_cpu_set"] = None
        raw["camera_test_config"]["output_dir"] = str(tmp_path / "camera_tests" / side / setup / mode)
        return replace(
            config,
            raw=raw,
            collection=replace(config.collection, dataset_dir=dataset, recorder_cpu_set=None, live_video_cpu_set=None),
        )

    return make_config


@pytest.fixture
def collector_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., SingleArmCollector]:
    class ReadOnlyTelemetry:
        def __init__(self, endpoint: object) -> None:
            self.endpoint = endpoint

        def close(self) -> None:
            pass

    class UnopenedCamera:
        def __init__(self, config: object) -> None:
            self.config = config
            self.role = config.role

        def stop(self) -> None:
            pass

    monkeypatch.setattr("yam_data_collection.collector.PortalTelemetryClient", ReadOnlyTelemetry)
    monkeypatch.setattr("yam_data_collection.collector.RealSenseCamera", UnopenedCamera)
    collectors: list[SingleArmCollector] = []

    def make_collector(config: DatasetConfig, **kwargs: object) -> SingleArmCollector:
        collector = SingleArmCollector(config, enable_runtime_diagnostics=False, **kwargs)
        collectors.append(collector)
        return collector

    yield make_collector
    for collector in collectors:
        collector.close()
