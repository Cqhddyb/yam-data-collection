"""Named validation standards shared by capture and offline qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CUSTOM_PROFILE = "custom"
PIPER_COMPATIBLE_PROFILE = "piper_compatible"
YAM_30HZ_PROFILE = "yam_30hz"
YAM_60HZ_PROFILE = "yam_60hz"
YAM_60HZ_STRICT_PROFILE = "yam_60hz_strict"


@dataclass(frozen=True)
class ValidationProfile:
    """Immutable quality thresholds for one capture-data standard."""

    name: str
    expected_camera_fps: float | None
    auto_detect_camera_fps: bool
    camera_min_average_fps: float | None
    camera_max_average_fps: float | None
    camera_max_gap_ms: float
    camera_sync_ms: float
    joint_camera_sync_ms: float
    require_exact_camera_resolution: bool


PIPER_COMPATIBLE = ValidationProfile(
    name=PIPER_COMPATIBLE_PROFILE,
    # The source Piper validator detects the nearest of 15, 30, and 60 Hz.
    # Most existing Piper episodes are 30 Hz, but preserving that behavior lets
    # the compatibility profile inspect all historical Piper recordings.
    expected_camera_fps=None,
    auto_detect_camera_fps=True,
    camera_min_average_fps=None,
    camera_max_average_fps=None,
    camera_max_gap_ms=69.0,
    camera_sync_ms=34.0,
    joint_camera_sync_ms=34.0,
    require_exact_camera_resolution=False,
)

YAM_30HZ = ValidationProfile(
    name=YAM_30HZ_PROFILE,
    expected_camera_fps=30.0,
    auto_detect_camera_fps=False,
    camera_min_average_fps=29.5,
    camera_max_average_fps=30.5,
    # The YAM 30 Hz acceptance standard intentionally keeps Piper's temporal
    # thresholds, including its two-frame camera-gap allowance.
    camera_max_gap_ms=69.0,
    camera_sync_ms=34.0,
    joint_camera_sync_ms=34.0,
    require_exact_camera_resolution=False,
)

YAM_60HZ = ValidationProfile(
    name=YAM_60HZ_PROFILE,
    expected_camera_fps=60.0,
    auto_detect_camera_fps=False,
    # The production 60 Hz capture accepts source averages down to 59 Hz.
    # Keep the existing upper bound: this is a lower-bound relaxation, not
    # a symmetric +/- 1 Hz contract.
    camera_min_average_fps=59.0,
    camera_max_average_fps=60.5,
    # The new 60 Hz standard retains Piper's gap allowance. Only the
    # synchronization budgets tighten to one 60 Hz frame period.
    camera_max_gap_ms=69.0,
    camera_sync_ms=17.0,
    joint_camera_sync_ms=17.0,
    require_exact_camera_resolution=False,
)

YAM_60HZ_STRICT = ValidationProfile(
    name=YAM_60HZ_STRICT_PROFILE,
    expected_camera_fps=60.0,
    auto_detect_camera_fps=False,
    camera_min_average_fps=59.5,
    camera_max_average_fps=60.5,
    # Permit at most slightly more than two 60 Hz frame periods. This matches
    # the live preflight and standalone camera qualification gate.
    camera_max_gap_ms=35.0,
    # Piper's 34 ms is approximately one 30 Hz frame period. At 60 Hz the
    # corresponding one-frame synchronization budget is 17 ms.
    camera_sync_ms=17.0,
    joint_camera_sync_ms=17.0,
    require_exact_camera_resolution=True,
)


_PROFILES = {
    PIPER_COMPATIBLE_PROFILE: PIPER_COMPATIBLE,
    YAM_30HZ_PROFILE: YAM_30HZ,
    YAM_60HZ_PROFILE: YAM_60HZ,
    YAM_60HZ_STRICT_PROFILE: YAM_60HZ_STRICT,
}

_ALIASES = {
    "piper": PIPER_COMPATIBLE_PROFILE,
    "piper_compat": PIPER_COMPATIBLE_PROFILE,
    "piper_compatible": PIPER_COMPATIBLE_PROFILE,
    "piper_30hz": PIPER_COMPATIBLE_PROFILE,
    "yam_30": YAM_30HZ_PROFILE,
    "yam_30hz": YAM_30HZ_PROFILE,
    "yam_60": YAM_60HZ_PROFILE,
    "yam_60hz": YAM_60HZ_PROFILE,
    "yam_60hz_strict": YAM_60HZ_STRICT_PROFILE,
}


def normalize_validation_profile(value: str | None) -> str:
    """Normalize a human-friendly profile name or reject an unknown one."""
    if value is None:
        return PIPER_COMPATIBLE_PROFILE
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == CUSTOM_PROFILE:
        return CUSTOM_PROFILE
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        valid = ", ".join(
            (
                PIPER_COMPATIBLE_PROFILE,
                YAM_30HZ_PROFILE,
                YAM_60HZ_PROFILE,
                YAM_60HZ_STRICT_PROFILE,
                CUSTOM_PROFILE,
            )
        )
        raise ValueError(f"unknown validation profile {value!r}; expected one of: {valid}") from exc


def validation_profile_cli_choices() -> tuple[str, ...]:
    """Return concise user-facing names accepted by the command-line override."""
    return ("piper-compatible", "yam-30hz", "yam-60hz", "yam-60hz-strict")


def get_validation_profile(value: str) -> ValidationProfile:
    """Return the named standard. ``custom`` has no fixed standard."""
    normalized = normalize_validation_profile(value)
    if normalized == CUSTOM_PROFILE:
        raise ValueError("the custom validation profile has no fixed thresholds")
    return _PROFILES[normalized]


def validate_profile_camera_fps(profile_name: str, cameras: Mapping[str, object]) -> None:
    """Ensure a fixed-rate profile cannot silently record another camera mode."""
    normalized = normalize_validation_profile(profile_name)
    if normalized == CUSTOM_PROFILE:
        return
    profile = get_validation_profile(normalized)
    if profile.expected_camera_fps is None:
        return
    incorrect = [
        f"{role}={getattr(camera, 'fps', None)}"
        for role, camera in cameras.items()
        if float(getattr(camera, "fps", 0.0)) != profile.expected_camera_fps
    ]
    if incorrect:
        details = ", ".join(incorrect)
        raise ValueError(
            f"validation profile {profile.name} requires every camera at "
            f"{profile.expected_camera_fps:g} Hz; found {details}"
        )
