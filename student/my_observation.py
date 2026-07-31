# -*- coding: utf-8 -*-
"""Custom 9-D observation.

This module is loaded with:

  python train_rllib.py --observation-mode custom --observation-module student.my_observation

Keep OBSERVATION_SIZE synchronized with the vector returned by build_observation().

Dimension history -- a change here invalidates every existing checkpoint and
bundle, so it needs a new output-tag (see Docs/troubleshooting.md §2):
  student8 -> student9 : added obs[8] closure rate.

build_observation() is a PURE function of (ownship_state, target_state).
No caching, no module globals, no reset-detection heuristics. With an action
provider, _build_action_context() calls it step_ratio (=6) extra times per step
and the target-provider path passes ownship/target SWAPPED, so a hook that
remembered anything would return different values in training than in local
verification -- silently. Closure is therefore computed from the two velocity
vectors, never from a previous-step distance.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Same bootstrap my_reward.py / my_submission.py use. Without it the dogfight
# imports below fail first when this module is loaded by file path rather than
# as "student.my_observation" -- before the geometry_utils fallback is reached.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402

from dogfight.envs.observation import normalize  # noqa: E402
from dogfight.sim.state_schema import StateIndex  # noqa: E402

try:  # normal case: loaded as "student.my_observation" by train_rllib.py
    from student.geometry_utils import _repair_degenerate_angle, closure_rate_mps
except ImportError:  # loaded by file path (some action-provider paths do this)
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from geometry_utils import _repair_degenerate_angle, closure_rate_mps


OBSERVATION_MODE = "student9"
OBSERVATION_SIZE = 9
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0

# Normalization ranges, one named constant per feature -- no magic numbers at
# the assignment sites, and describe_observation() reports them.
ROLL_RANGE_DEG = (-180.0, 180.0)
PITCH_RANGE_DEG = (-90.0, 90.0)
YAW_RANGE_DEG = (0.0, 360.0)
# StateIndex.KCAS is METER/SEC despite the name (FighterSim.py:186).
KCAS_RANGE_MPS = (0.0, 600.0)
# StateIndex.ALT is METERS, not feet (FighterSim.py:244).
ALT_RANGE_M = (0.0, 15000.0)
DISTANCE_RANGE_M = (0.0, 20000.0)
ANGLE_RANGE_DEG = (-180.0, 180.0)
# Two fighters merging head-on at ~450 m/s each is the physical worst case.
CLOSURE_RANGE_MPS = (-900.0, 900.0)


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _norm(value, value_range) -> float:
    """Clip into `value_range`, then normalize to [-1, 1].

    The clip is this hook's own guarantee, not the platform's:
    dogfight.envs.observation.normalize is an affine map whose clipping
    behaviour is not part of any documented contract (a Release-tree probe in
    check_against_release.py reports which one you have). Without the clip, a
    separation beyond DISTANCE_RANGE_M -- or any KCAS/ALT excursion -- would
    push a feature outside OBSERVATION_LOW/HIGH and silently break the
    observation-space contract the policy was trained against.
    """
    low, high = value_range
    return float(normalize(_clip(float(value), low, high), low, high))


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    """Return a custom 9-D observation vector as float32."""
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)

    # GeoMathUtil signs both angles with np.sign(p_unit_t[2]), which is 0.0 when
    # the target is exactly co-altitude -- collapsing the angle to 0 (nose-on)
    # even when the target is at our six. Both aircraft share
    # initial_scenario.altitude_m, so reset() can hit this. Repair before the
    # values reach obs[6]/obs[7]; non-degenerate geometry passes through
    # untouched. Stateless, per observation gotcha #3.
    ata = _repair_degenerate_angle(ata, ownship_state, target_state, geo_info, "ata")
    aa = _repair_degenerate_angle(aa, ownship_state, target_state, geo_info, "aa")

    # Positive = closing. Derived from both velocity vectors, so swapping the
    # arguments yields the correct value for the swapped pair rather than a
    # stale one.
    closure = closure_rate_mps(ownship_state, target_state)

    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    obs[0] = _norm(ownship_state[StateIndex.ROLL], ROLL_RANGE_DEG)
    obs[1] = _norm(ownship_state[StateIndex.PITCH], PITCH_RANGE_DEG)
    obs[2] = _norm(ownship_state[StateIndex.YAW], YAW_RANGE_DEG)
    obs[3] = _norm(ownship_state[StateIndex.KCAS], KCAS_RANGE_MPS)
    obs[4] = _norm(ownship_state[StateIndex.ALT], ALT_RANGE_M)
    obs[5] = _norm(distance, DISTANCE_RANGE_M)
    obs[6] = _norm(ata, ANGLE_RANGE_DEG)
    obs[7] = _norm(aa, ANGLE_RANGE_DEG)
    obs[8] = _norm(closure, CLOSURE_RANGE_MPS)
    return obs


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "ownship_roll_norm",
            "ownship_pitch_norm",
            "ownship_yaw_norm",
            "ownship_kcas_norm",
            "ownship_alt_norm",
            "distance_norm",
            "ata_norm",
            "aa_norm",
            "closure_norm",
        ],
        "ranges": [
            ROLL_RANGE_DEG,
            PITCH_RANGE_DEG,
            YAW_RANGE_DEG,
            KCAS_RANGE_MPS,
            ALT_RANGE_M,
            DISTANCE_RANGE_M,
            ANGLE_RANGE_DEG,
            ANGLE_RANGE_DEG,
            CLOSURE_RANGE_MPS,
        ],
        "description": "Student custom 9-D observation (stateless, closure from velocity).",
    }
