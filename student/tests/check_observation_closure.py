# -*- coding: utf-8 -*-
"""Stateless 9-D observation + velocity-derived closure. Runs anywhere.

    python student\tests\check_observation_closure.py

Why statelessness is a correctness requirement, not a style preference
----------------------------------------------------------------------
_build_action_context() calls _build_observation_for() step_ratio (=6) extra
times when an action provider is present, and the target-provider path passes
ownship/target SWAPPED (single_agent_env.py L454, L900-905):

    training  (train_rllib.py, BT opponent via step_behavior): no provider
              -> 1 call per step
    local     (run_local_dogfight.py --target-backend bt):     provider
              -> 7 calls per step, some with swapped arguments

So a hook that caches the previous step's distance produces different numbers in
training than in local verification, with no error anywhere. Closure is
therefore computed from both velocity vectors instead of a distance delta.

(The reward hook is safe: _reward_fn runs only inside step(), once. Its WEZ
hysteresis state is deliberately left alone.)

Exit 0 = all passed, 1 = at least one failed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Release root

import numpy as np  # noqa: E402

from student.tests._stubs import Checker, close, install_dogfight_stubs, make_state  # noqa: E402

install_dogfight_stubs()

from student import my_observation  # noqa: E402
from student.geometry_utils import (  # noqa: E402
    _nose_vector_ned,
    body_to_ned_velocity,
    closure_rate_mps,
)
from student.tests.geomath_stub import GeoMathUtilStub  # noqa: E402

C = Checker("stateless 9-D observation with velocity-derived closure")
geo = GeoMathUtilStub()
ALT = 7000.0

# ==========================================================================
# 1. Dimension contract (observation consistency chain, troubleshooting §2)
# ==========================================================================
own = make_state(0.0, 0.0, ALT, yaw_deg=0.0, u=200.0)
target = make_state(1000.0, 0.0, ALT, yaw_deg=180.0, u=200.0)
obs = my_observation.build_observation(own, target, geo)

C.check("1.1 OBSERVATION_SIZE is 9", my_observation.OBSERVATION_SIZE == 9)
C.check("1.2 returned vector has shape (9,)", obs.shape == (9,), f"got {obs.shape}")
C.check("1.3 dtype is float32", obs.dtype == np.float32, f"got {obs.dtype}")
C.check("1.4 all values finite", bool(np.all(np.isfinite(obs))))
C.check("1.5 all values within [-1, 1]",
        bool(np.all(obs >= -1.0) and np.all(obs <= 1.0)), f"got {obs}")

# The [-1, 1] contract must hold on out-of-range inputs too. The fixture's
# normalize() is affine WITHOUT clipping (the real one's behaviour is not a
# documented contract), so this only passes if the hook clips every feature
# itself -- not just closure.
EXTREMES = [
    ("distance far beyond DISTANCE_RANGE_M",
     make_state(0.0, 0.0, ALT, yaw_deg=0.0, u=200.0),
     make_state(80000.0, 0.0, ALT, yaw_deg=180.0, u=200.0)),
    ("KCAS and ALT above their ranges",
     make_state(0.0, 0.0, 22000.0, yaw_deg=0.0, u=200.0, kcas=900.0),
     make_state(1000.0, 0.0, 22000.0, yaw_deg=180.0, u=200.0)),
    ("negative altitude and huge closure",
     make_state(0.0, 0.0, -500.0, yaw_deg=0.0, u=4000.0, kcas=-30.0),
     make_state(1000.0, 0.0, -500.0, yaw_deg=180.0, u=4000.0)),
]
for label, o_ext, t_ext in EXTREMES:
    ext = my_observation.build_observation(o_ext, t_ext, geo)
    C.check(f"1.x [-1,1] contract holds: {label}",
            bool(np.all(ext >= -1.0) and np.all(ext <= 1.0) and np.all(np.isfinite(ext))),
            f"got {ext}")

described = my_observation.describe_observation()
C.check("1.6 describe_observation size matches OBSERVATION_SIZE",
        described["size"] == my_observation.OBSERVATION_SIZE)
C.check("1.7 describe_observation lists exactly 9 features",
        len(described["features"]) == 9, f"got {len(described['features'])}")
C.check("1.8 obs[8] is documented as the closure feature",
        described["features"][8] == "closure_norm", f"got {described['features'][8]}")
C.check("1.8b describe_observation reports one range per feature",
        len(described.get("ranges", [])) == 9
        and all(len(r) == 2 and r[0] < r[1] for r in described["ranges"]),
        f"got {described.get('ranges')}")
C.check("1.9 mode string reflects the new dimension",
        my_observation.OBSERVATION_MODE == "student9",
        f"got {my_observation.OBSERVATION_MODE}")

# ==========================================================================
# 2. body -> NED rotation, 3-2-1 (Rz(yaw) Ry(pitch) Rx(roll)), hand-checked
# ==========================================================================
CASES = [
    # (label, roll, pitch, yaw, u, v, w, expected N/E/D)
    ("level north, u=100", 0, 0, 0, 100, 0, 0, (100.0, 0.0, 0.0)),
    ("level east (yaw 90)", 0, 0, 90, 100, 0, 0, (0.0, 100.0, 0.0)),
    ("level south (yaw 180)", 0, 0, 180, 100, 0, 0, (-100.0, 0.0, 0.0)),
    # pitch +90 = nose up -> Down component is negative (climbing)
    ("vertical climb (pitch 90)", 0, 90, 0, 100, 0, 0, (0.0, 0.0, -100.0)),
    ("dive (pitch -90)", 0, -90, 0, 100, 0, 0, (0.0, 0.0, 100.0)),
    # roll 90 puts the body y-axis (right wing) straight down
    ("rolled 90, sideslip v=100", 90, 0, 0, 0, 100, 0, (0.0, 0.0, 100.0)),
    ("level, w=100 (belly down)", 0, 0, 0, 0, 0, 100, (0.0, 0.0, 100.0)),
    ("yaw 45, u=100", 0, 0, 45, 100, 0, 0,
     (100 * math.sqrt(0.5), 100 * math.sqrt(0.5), 0.0)),
]
for label, roll, pitch, yaw, u, v, w, expected in CASES:
    got = body_to_ned_velocity(
        make_state(0.0, 0.0, ALT, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                   u=u, v=v, w=w))
    C.check(f"2.x NED rotation: {label}",
            all(close(g, e, 1e-9) for g, e in zip(got, expected)),
            f"got {tuple(round(x, 6) for x in got)}, expected {expected}")

# Unit body-x through the rotation must equal the nose vector the angle repair
# uses -- one convention, not two.
_attitude = make_state(0, 0, ALT, yaw_deg=37.0, pitch_deg=11.0, roll_deg=63.0,
                       u=1.0, v=0.0, w=0.0)
C.check("2.9 rotation's first column equals _nose_vector_ned",
        all(close(x, y, 1e-12) for x, y in zip(
            body_to_ned_velocity(_attitude), _nose_vector_ned(_attitude))),
        "body_to_ned_velocity and _nose_vector_ned disagree on convention")

# ==========================================================================
# 3. Closure sign and magnitude
# ==========================================================================
def closure(own_kw, tgt_kw, tgt_pos=(1000.0, 0.0)):
    o = make_state(0.0, 0.0, ALT, **own_kw)
    t = make_state(tgt_pos[0], tgt_pos[1], ALT, **tgt_kw)
    return closure_rate_mps(o, t)

# Head-on merge: both at 200 m/s straight at each other.
C.check("3.1 head-on merge closes at the sum of speeds",
        close(closure({"yaw_deg": 0.0, "u": 200.0}, {"yaw_deg": 180.0, "u": 200.0}),
              400.0, 1e-9),
        f"got {closure({'yaw_deg': 0.0, 'u': 200.0}, {'yaw_deg': 180.0, 'u': 200.0})}")

# Back-to-back separation.
C.check("3.2 mutual separation is negative",
        close(closure({"yaw_deg": 180.0, "u": 200.0}, {"yaw_deg": 0.0, "u": 200.0}),
              -400.0, 1e-9))

# Line-abreast, same heading and speed -> pure formation, no range rate.
C.check("3.3 parallel at equal speed gives ~0",
        close(closure({"yaw_deg": 0.0, "u": 200.0}, {"yaw_deg": 0.0, "u": 200.0}),
              0.0, 1e-9))

# Tail chase, overtaking by 50 m/s.
C.check("3.4 overtaking tail chase closes at the speed difference",
        close(closure({"yaw_deg": 0.0, "u": 250.0}, {"yaw_deg": 0.0, "u": 200.0}),
              50.0, 1e-9))

# Bandit crossing perpendicular contributes no radial component.
C.check("3.5 perpendicular bandit motion adds no closure",
        close(closure({"yaw_deg": 0.0, "u": 200.0}, {"yaw_deg": 90.0, "u": 200.0}),
              200.0, 1e-9))

# Pure vertical geometry: bandit 1000 m above, ownship climbing at 100.
own_climb = make_state(0.0, 0.0, ALT, yaw_deg=0.0, pitch_deg=90.0, u=100.0)
tgt_high = make_state(0.0, 0.0, ALT + 1000.0, yaw_deg=0.0, u=0.0)
C.check("3.6 climbing straight at a bandit overhead closes at 100",
        close(closure_rate_mps(own_climb, tgt_high), 100.0, 1e-9),
        f"got {closure_rate_mps(own_climb, tgt_high)}")

# Range rate belongs to the PAIR, so it must be symmetric.
o = make_state(0.0, 0.0, ALT, yaw_deg=17.0, pitch_deg=5.0, roll_deg=30.0, u=210.0, v=3.0)
t = make_state(800.0, -400.0, ALT + 120.0, yaw_deg=200.0, pitch_deg=-8.0, u=190.0)
C.check("3.7 closure(a,b) == closure(b,a)",
        close(closure_rate_mps(o, t), closure_rate_mps(t, o), 1e-9))

C.check("3.8 co-located aircraft guard returns 0.0",
        closure_rate_mps(make_state(0, 0, ALT, yaw_deg=0.0, u=200.0),
                         make_state(0, 0, ALT, yaw_deg=90.0, u=200.0)) == 0.0)

# Clipping keeps obs[8] in range even at absurd speeds.
fast_o = make_state(0.0, 0.0, ALT, yaw_deg=0.0, u=5000.0)
fast_t = make_state(1000.0, 0.0, ALT, yaw_deg=180.0, u=5000.0)
fast_obs = my_observation.build_observation(fast_o, fast_t, geo)
C.check("3.9 extreme closure is clipped into [-1, 1]",
        -1.0 <= float(fast_obs[8]) <= 1.0 and close(float(fast_obs[8]), 1.0, 1e-6),
        f"got {fast_obs[8]}")

# ==========================================================================
# 4. Statelessness -- the whole point
# ==========================================================================
a = make_state(0.0, 0.0, ALT, yaw_deg=0.0, pitch_deg=3.0, roll_deg=-12.0, u=220.0)
b = make_state(1500.0, 600.0, ALT + 300.0, yaw_deg=200.0, pitch_deg=-4.0, u=190.0)

first = my_observation.build_observation(a, b, geo)
repeats = [my_observation.build_observation(a, b, geo) for _ in range(5)]
C.check("4.1 five identical calls return identical vectors",
        all(np.array_equal(first, r) for r in repeats))

# The 7-calls-per-step provider pattern, including the swapped target path.
my_observation.build_observation(b, a, geo)
my_observation.build_observation(a, b, geo)
my_observation.build_observation(b, a, geo)
for _ in range(4):
    my_observation.build_observation(a, b, geo)
after_history = my_observation.build_observation(a, b, geo)
C.check("4.2 value survives a swapped/interleaved call history unchanged",
        np.array_equal(first, after_history),
        "build_observation depends on call history -- it is not stateless")

swapped = my_observation.build_observation(b, a, geo)
C.check("4.3 the swapped call describes the swapped ownship, not a stale value",
        not np.array_equal(first, swapped)
        and close(float(swapped[0]),
                  my_observation.build_observation(b, a, geo)[0], 0.0))
C.check("4.4 closure is identical either way (it is a property of the pair)",
        close(float(first[8]), float(swapped[8]), 1e-6),
        f"{first[8]} vs {swapped[8]}")

C.check("4.5 hook holds no module-level mutable cache",
        not any(
            isinstance(getattr(my_observation, n), (dict, list, set))
            and not n.startswith("__")
            for n in dir(my_observation)
            if n.isupper()
        ),
        "an uppercase module-level container looks like a cache")

# A fresh interpreter-order check: reversing the very first call must not matter.
reversed_first = my_observation.build_observation(b, a, geo)
back = my_observation.build_observation(a, b, geo)
C.check("4.6 alternating order converges to the same two vectors",
        np.array_equal(back, first) and np.array_equal(reversed_first, swapped))

sys.exit(C.finish())
