# -*- coding: utf-8 -*-
"""TEST FIXTURE ONLY -- a deliberately buggy GeoMathUtil double.

    !!! THIS IS NOT GeoMathUtil. !!!
    Never import it from student/ hooks, never copy it into the Release tree,
    never let it shadow the real module. The real GeoMathUtil.py lives in
    AIP_LIB/DogFightEnv/Release/ , is platform-owned, is outside version
    control, and must never be edited or replaced (CLAUDE.md
    "Hard constraints" #1). This file exists so the degeneracy can be
    reproduced on a machine that has no Release tree at all.

What it reproduces
------------------
The real helpers compose their signed result as `sign * magnitude` with
`sign = np.sign(p_unit_t[2])`. `np.sign(0.0)` is `0.0`, so a target at exactly
the same altitude collapses the entire angle to 0.0 -- an enemy at our six
reads as nose-on. This double reproduces that exact failure:

    * magnitude: correct, arccos-based (matches the real helper)
    * sign:      np.sign of the vertical LOS component -> 0.0 when co-altitude

Fidelity limits (accepted, because the fixture only has to pin the degeneracy)
-----------------------------------------------------------------------------
1. The left/right sign convention of the non-degenerate branch is keyed off the
   vertical component, exactly as the defect description states, but the real
   module derives `p_unit_t` in the body frame. For a level aircraft the two
   agree on *when* the sign is zero -- which is the property under test -- but
   the +/- assignment of a non-degenerate angle is NOT claimed to match the
   real module. No test here asserts on that sign.
2. Only the three helpers the student hooks actually call are modelled.

Verification against the real module is the job of
student/tests/check_against_release.py, which loads GeoMathUtil by path and
skips (exit 2) outside a Release root.
"""
from __future__ import annotations

import numpy as np


IDX_NORTH, IDX_EAST, IDX_DOWN = 0, 1, 2
IDX_PITCH, IDX_YAW = 4, 5


def _nose_unit(state):
    pitch = np.radians(float(state[IDX_PITCH]))
    yaw = np.radians(float(state[IDX_YAW]))
    return np.array(
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
        dtype=np.float64,
    )


def _position(state):
    return np.array(
        [float(state[IDX_NORTH]), float(state[IDX_EAST]), float(state[IDX_DOWN])],
        dtype=np.float64,
    )


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


def _magnitude_deg(u, v):
    """Correct unsigned angle -- this part of the real helper is not buggy."""
    u_unit, v_unit = _unit(u), _unit(v)
    if not u_unit.any() or not v_unit.any():
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u_unit, v_unit)), -1.0, 1.0))))


class GeoMathUtilStub:
    """Stand-in for the per-env `geo_info` object passed to the student hooks."""

    def _get_distance(self, ownship_state, target_state):
        """Meters (GeoMathUtil.py:21)."""
        return float(np.linalg.norm(_position(target_state) - _position(ownship_state)))

    def _get_antenna_train_angle(self, ownship_state, target_state, radian=False):
        """Signed ATA. 0 = nose-on. Reproduces the sign-collapse defect."""
        los = _position(target_state) - _position(ownship_state)
        magnitude = _magnitude_deg(_nose_unit(ownship_state), los)
        return self._apply_defective_sign(magnitude, los, radian)

    def _get_aspect_angle(self, ownship_state, target_state, radian=False):
        """Signed AA. 0 = at the target's six. Reproduces the same defect."""
        los = _position(ownship_state) - _position(target_state)
        magnitude = 180.0 - _magnitude_deg(_nose_unit(target_state), los)
        # The real module signs the aspect angle off the ownship-relative
        # vertical, so use the same LOS orientation the ATA path uses.
        return self._apply_defective_sign(magnitude, -los, radian)

    @staticmethod
    def _apply_defective_sign(magnitude_deg, los, radian):
        # THE BUG, verbatim in spirit: np.sign(0.0) == 0.0 wipes the magnitude.
        p_unit_t = _unit(los)
        sign = float(np.sign(p_unit_t[IDX_DOWN]))
        signed = sign * magnitude_deg
        return float(np.radians(signed)) if radian else float(signed)


__all__ = ["GeoMathUtilStub"]
