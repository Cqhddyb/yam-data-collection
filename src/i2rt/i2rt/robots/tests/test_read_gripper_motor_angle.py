import pytest

from scripts.read_gripper_motor_angle import decode_position_rad


def _feedback_frame(position: int, error_code: int = 1) -> bytes:
    return bytes([(error_code << 4), (position >> 8) & 0xFF, position & 0xFF, 0, 0, 0, 0, 0])


def test_decode_position_rad_maps_dm4310_feedback_limits() -> None:
    assert decode_position_rad(_feedback_frame(0), "DM4310") == pytest.approx(-12.5)
    assert decode_position_rad(_feedback_frame(65535), "DM4310") == pytest.approx(12.5)


def test_decode_position_rad_rejects_motor_error_frame() -> None:
    with pytest.raises(RuntimeError, match="error code"):
        decode_position_rad(_feedback_frame(32768, error_code=2), "DM4310")


def test_decode_position_rad_requires_complete_can_frame() -> None:
    with pytest.raises(ValueError, match="8-byte"):
        decode_position_rad(b"\x10\x00", "DM4310")
