"""Regression tests for teaching-handle passive encoder reports."""

import struct

import can
import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import PASSIVE_ENCODER_COUNTS_PER_REV, PassiveEncoderReader


def _parse_position(position_counts: int) -> float:
    reader = PassiveEncoderReader.__new__(PassiveEncoderReader)
    message = can.Message(
        arbitration_id=0x50F,
        data=struct.pack("!B h h B", 0, position_counts, 0, 0),
        is_extended_id=False,
    )
    position_rad, _, _ = reader._parse_encoder_message(message)
    return position_rad


@pytest.mark.parametrize(
    ("reported_counts", "centered_counts"),
    [
        (4118, 22),
        (3672, -424),
        (-424, -424),
    ],
)
def test_passive_encoder_centers_periodic_position_counts(
    reported_counts: int,
    centered_counts: int,
) -> None:
    expected_rad = centered_counts * 2 * np.pi / PASSIVE_ENCODER_COUNTS_PER_REV

    assert _parse_position(reported_counts) == pytest.approx(expected_rad)
