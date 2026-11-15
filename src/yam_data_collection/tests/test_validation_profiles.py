from __future__ import annotations

import pytest

from yam_data_collection.validation_profiles import (
    YAM_30HZ_PROFILE,
    YAM_60HZ_PROFILE,
    YAM_60HZ_STRICT_PROFILE,
    get_validation_profile,
    normalize_validation_profile,
    validation_profile_cli_choices,
)


def test_yam_capture_profiles_match_the_two_acceptance_standards() -> None:
    profile_30 = get_validation_profile(YAM_30HZ_PROFILE)
    assert profile_30.expected_camera_fps == 30.0
    assert profile_30.camera_min_average_fps == 29.5
    assert profile_30.camera_max_average_fps == 30.5
    assert profile_30.camera_max_gap_ms == 69.0
    assert profile_30.camera_sync_ms == 34.0
    assert profile_30.joint_camera_sync_ms == 34.0
    assert not profile_30.require_exact_camera_resolution

    profile_60 = get_validation_profile(YAM_60HZ_PROFILE)
    assert profile_60.expected_camera_fps == 60.0
    assert profile_60.camera_min_average_fps == 59.0
    assert profile_60.camera_max_average_fps == 60.5
    assert profile_60.camera_max_gap_ms == 69.0
    assert profile_60.camera_sync_ms == 17.0
    assert profile_60.joint_camera_sync_ms == 17.0
    assert not profile_60.require_exact_camera_resolution


def test_legacy_strict_profile_remains_distinct() -> None:
    profile = get_validation_profile(YAM_60HZ_STRICT_PROFILE)
    assert profile.camera_min_average_fps == 59.5
    assert profile.camera_max_average_fps == 60.5
    assert profile.camera_max_gap_ms == 35.0
    assert profile.camera_sync_ms == 17.0
    assert profile.require_exact_camera_resolution
    assert normalize_validation_profile("yam-60hz") == YAM_60HZ_PROFILE
    assert normalize_validation_profile("yam-60hz-strict") == YAM_60HZ_STRICT_PROFILE


def test_cli_exposes_both_yam_capture_modes() -> None:
    assert "yam-30hz" in validation_profile_cli_choices()
    assert "yam-60hz" in validation_profile_cli_choices()


def test_unknown_profile_lists_the_new_names() -> None:
    with pytest.raises(ValueError, match="yam_30hz.*yam_60hz"):
        normalize_validation_profile("not-a-profile")
