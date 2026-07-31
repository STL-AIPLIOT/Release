# -*- coding: utf-8 -*-
"""Degenerate-angle repair checks. Runs anywhere (numpy only, no Ray, no env).

    python student\tests\check_degenerate_angle.py

Stubs `dogfight.*` so the student hooks import without a Release tree. Geometry
comes from student/tests/geomath_stub.py -- a FIXTURE that reproduces the
GeoMathUtil sign-collapse defect, never a replacement for the real module.

Covers:
  1. reproduction   -- the defect is real and the fixture reproduces it
  2. repair         -- co-altitude geometry recovers the true magnitude
  3. continuity     -- +/-1e-9 m altitude offset does not make values jump
  4. regression     -- non-degenerate geometry is bit-identical before/after
  5. hooks          -- obs[6]/obs[7] are truthful; reward contract intact
  6. source guard   -- no hook may call the raw angle helpers unrepaired
  7. statelessness  -- build_observation is call-order independent (gotcha #3)

Section 6 keeps the repair from being dropped later: it fails the moment a hook
reads the angle helpers in live code without wrapping them.

Exit 0 = all passed, 1 = at least one failed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

# The dogfight.* stubs, the state builder and the assert helpers come from the
# ONE shared fixture. This file used to carry a private copy whose normalize()
# clipped to [-1, 1]; _stubs.normalize deliberately does NOT clip
# (Docs/troubleshooting.md §8), so the two fixtures disagreed on whether the
# hook has to clip for itself. The permissive one has to win, or an
# out-of-range bug passes here and only fails in check_observation_closure.py.
from student.tests._stubs import (  # noqa: E402
    Checker,
    close,
    install_dogfight_stubs,
    make_state,
    raises,
)

install_dogfight_stubs()

from student import my_observation, my_reward  # noqa: E402
from student.geometry_utils import (  # noqa: E402
    repair_degenerate_angle,
    true_angle_magnitude_deg,
)
from student.tests.geomath_stub import GeoMathUtilStub  # noqa: E402

C = Checker("degenerate-angle repair verified")
check = C.check

geo = GeoMathUtilStub()
ALT = 7000.0  # experiments/*.yaml initial_scenario.altitude_m

# ==========================================================================
# 1. Reproduction: the fixture shows the defect
# ==========================================================================
# Ownship at origin heading north; enemy 1000 m due SOUTH = dead astern.
own = make_state(0.0, 0.0, ALT, yaw_deg=0.0)
bandit_six = make_state(-1000.0, 0.0, ALT, yaw_deg=0.0)

raw_ata = geo._get_antenna_train_angle(own, bandit_six, False)
check("1.1 co-altitude dead-astern ATA collapses to 0", close(raw_ata, 0.0),
      f"got {raw_ata}")
check("1.2 distance is unaffected by the defect",
      close(geo._get_distance(own, bandit_six), 1000.0, 1e-6))

# Head-on: enemy 1000 m north, pointing back at us -> AA should be 180.
bandit_headon = make_state(1000.0, 0.0, ALT, yaw_deg=180.0)
raw_aa = geo._get_aspect_angle(own, bandit_headon, False)
check("1.3 co-altitude head-on AA collapses to 0", close(raw_aa, 0.0),
      f"got {raw_aa}")

# A 1e-9 m altitude offset is enough to flip the sign back on.
bandit_six_hi = make_state(-1000.0, 0.0, ALT + 1e-9, yaw_deg=0.0)
bandit_six_lo = make_state(-1000.0, 0.0, ALT - 1e-9, yaw_deg=0.0)
check("1.4 +1e-9 m offset restores |ATA| = 180",
      close(abs(geo._get_antenna_train_angle(own, bandit_six_hi, False)), 180.0, 1e-6))
check("1.5 -1e-9 m offset restores |ATA| = 180",
      close(abs(geo._get_antenna_train_angle(own, bandit_six_lo, False)), 180.0, 1e-6))
check("1.6 the 1e-9 flip is a pure sign change (the defect, not a real motion)",
      geo._get_antenna_train_angle(own, bandit_six_hi, False)
      * geo._get_antenna_train_angle(own, bandit_six_lo, False) < 0.0)

# ==========================================================================
# 2. Repair recovers the true magnitude
# ==========================================================================
rep_ata = repair_degenerate_angle(raw_ata, own, bandit_six, geo, "ata")
check("2.1 repaired dead-astern ATA = 180", close(rep_ata, 180.0, 1e-6),
      f"got {rep_ata}")
check("2.2 repair is deterministic (+magnitude)", rep_ata > 0.0)

rep_aa = repair_degenerate_angle(raw_aa, own, bandit_headon, geo, "aa")
check("2.3 repaired head-on AA = 180", close(rep_aa, 180.0, 1e-6), f"got {rep_aa}")

# Ownship at the bandit's six, co-altitude: ATA and AA are genuinely ~0.
bandit_ahead = make_state(1000.0, 0.0, ALT, yaw_deg=0.0)
check("2.4 genuine nose-on stays 0 (no false repair)",
      close(repair_degenerate_angle(
          geo._get_antenna_train_angle(own, bandit_ahead, False),
          own, bandit_ahead, geo, "ata"), 0.0, 1e-9))
check("2.5 genuine at-six AA stays 0 (no false repair)",
      close(repair_degenerate_angle(
          geo._get_aspect_angle(own, bandit_ahead, False),
          own, bandit_ahead, geo, "aa"), 0.0, 1e-9))

# Off-boresight co-altitude: bandit 90 deg to the right, still degenerate.
bandit_right = make_state(0.0, 1000.0, ALT, yaw_deg=0.0)
check("2.6 co-altitude 90 deg beam repairs to 90",
      close(repair_degenerate_angle(
          geo._get_antenna_train_angle(own, bandit_right, False),
          own, bandit_right, geo, "ata"), 90.0, 1e-6))

# Pathological inputs must not raise and must not invent a value.
same_place = make_state(0.0, 0.0, ALT, yaw_deg=0.0)
check("2.7 co-located aircraft -> reported returned unchanged (no true angle exists)",
      repair_degenerate_angle(0.0, own, same_place, geo, "ata") == 0.0)
check("2.8 NaN reported stays NaN",
      math.isnan(repair_degenerate_angle(float("nan"), own, bandit_six, geo, "ata")))
check("2.9 unknown kind raises ValueError",
      raises(ValueError, lambda: true_angle_magnitude_deg(own, bandit_six, "hca")))

# ==========================================================================
# 3. Continuity across the 1e-9 m altitude flip
# ==========================================================================
mags = []
for bandit in (bandit_six, bandit_six_hi, bandit_six_lo):
    raw = geo._get_antenna_train_angle(own, bandit, False)
    mags.append(abs(repair_degenerate_angle(raw, own, bandit, geo, "ata")))
check("3.1 |ATA| identical at dz = 0, +1e-9, -1e-9",
      max(mags) - min(mags) <= 1e-6, f"got {mags}")
check("3.2 and all three equal 180", all(close(m, 180.0, 1e-6) for m in mags))

# ==========================================================================
# 4. Regression: non-degenerate geometry is untouched
# ==========================================================================
rng = np.random.default_rng(20260731)
untouched = 0
for _ in range(400):
    o = make_state(
        rng.uniform(-5000, 5000), rng.uniform(-5000, 5000),
        ALT + rng.uniform(-500, 500),
        yaw_deg=rng.uniform(0, 360), pitch_deg=rng.uniform(-40, 40),
        roll_deg=rng.uniform(-90, 90))
    # Force a non-zero altitude separation so the platform sign is well defined.
    dz = float(rng.choice([-1.0, 1.0])) * rng.uniform(5.0, 900.0)
    t = make_state(
        o[0] + rng.uniform(-4000, 4000), o[1] + rng.uniform(-4000, 4000),
        o[44] + dz,
        yaw_deg=rng.uniform(0, 360), pitch_deg=rng.uniform(-40, 40))

    for kind, fn in (("ata", geo._get_antenna_train_angle),
                     ("aa", geo._get_aspect_angle)):
        reported = fn(o, t, False)
        repaired = repair_degenerate_angle(reported, o, t, geo, kind)
        if repaired == reported:
            untouched += 1
        else:
            C.fail(f"4.x non-degenerate {kind} was modified: {reported} -> {repaired}")
check("4.1 800 non-degenerate samples pass through bit-identical",
      untouched == 800, f"untouched={untouched}/800")

# Magnitudes must also agree with the fixture in the non-degenerate band --
# proves the repair's arccos path is the same geometry, not a different metric.
worst = 0.0
for _ in range(200):
    o = make_state(0.0, 0.0, ALT, yaw_deg=rng.uniform(0, 360),
                   pitch_deg=rng.uniform(-30, 30))
    t = make_state(rng.uniform(-3000, 3000), rng.uniform(-3000, 3000),
                   ALT + float(rng.choice([-1.0, 1.0])) * rng.uniform(10.0, 800.0),
                   yaw_deg=rng.uniform(0, 360))
    worst = max(worst, abs(abs(geo._get_antenna_train_angle(o, t, False))
                           - true_angle_magnitude_deg(o, t, "ata")))
check("4.2 true-magnitude matches the fixture magnitude off-degeneracy",
      worst <= 1e-9, f"max |diff| = {worst}")

# ==========================================================================
# 5. Hook level
# ==========================================================================
obs_bad_geo = my_observation.build_observation(own, bandit_six, geo)
check("5.1 obs dtype/shape unchanged",
      obs_bad_geo.dtype == np.float32
      and obs_bad_geo.shape == (my_observation.OBSERVATION_SIZE,))
check("5.2 obs is finite", bool(np.all(np.isfinite(obs_bad_geo))))
check("5.3 obs[6] no longer reads nose-on with the bandit at six",
      close(float(obs_bad_geo[6]), 1.0, 1e-6),
      f"got {obs_bad_geo[6]} (normalize(180, -180, 180) = 1.0)")
obs_headon = my_observation.build_observation(own, bandit_headon, geo)
check("5.4 obs[7] reports AA = 180 head-on", close(float(obs_headon[7]), 1.0, 1e-6),
      f"got {obs_headon[7]}")
obs_good = my_observation.build_observation(own, bandit_ahead, geo)
check("5.5 genuine nose-on still normalizes to 0.0", close(float(obs_good[6]), 0.0, 1e-9))

# Reward hook. The WEZ shaping reads ATA, so the degeneracy is live here.
wez_cfg = {"angle_deg": 2.0, "min_range_m": 152.4, "max_range_m": 914.4}
my_reward.reset_reward_state()
bandit_six_close = make_state(-500.0, 0.0, ALT, yaw_deg=0.0)  # 500 m astern
total, comps = my_reward.compute_reward(
    own, bandit_six_close, 0.0, 0.0, geo, wez_cfg, my_reward.MY_REWARD_CONFIG,
    False, False, "")
check("5.6 reward returns (float, dict)",
      isinstance(total, float) and isinstance(comps, dict))
check("5.7 no WEZ entry bonus for a bandit at our six",
      close(comps["wez_entry"], 0.0), f"got {comps['wez_entry']}")
check("5.8 angle_align ~ 0 at ATA 180 (was full credit before the repair)",
      abs(comps["angle_align"]) <= 1e-9, f"got {comps['angle_align']}")
check("5.9 all reward components finite",
      all(math.isfinite(float(v)) for v in comps.values()) and math.isfinite(total))

my_reward.reset_reward_state()
bandit_front_close = make_state(500.0, 0.0, ALT, yaw_deg=0.0)  # 500 m ahead, nose-on
_, comps_good = my_reward.compute_reward(
    own, bandit_front_close, 0.0, 0.0, geo, wez_cfg, my_reward.MY_REWARD_CONFIG,
    False, False, "")
check("5.10 genuine nose-on inside the half-angle cone earns the entry bonus",
      comps_good["wez_entry"] > 0.0, f"got {comps_good['wez_entry']}")
check("5.11 genuine nose-on keeps full angle_align",
      close(comps_good["angle_align"],
            my_reward.MY_REWARD_CONFIG["angle_shaping_max"], 1e-9))

dead_bandit = make_state(-1000.0, 0.0, ALT, yaw_deg=0.0, health=0.0)
dead_own = make_state(0.0, 0.0, ALT, yaw_deg=0.0, health=0.0)
my_reward.reset_reward_state()
win, _ = my_reward.compute_reward(own, dead_bandit, 0.0, 0.0, geo, wez_cfg,
                                  my_reward.MY_REWARD_CONFIG, True, False, "")
loss, _ = my_reward.compute_reward(dead_own, bandit_six, 0.0, 0.0, geo, wez_cfg,
                                   my_reward.MY_REWARD_CONFIG, True, False, "")
draw, _ = my_reward.compute_reward(own, bandit_six, 0.0, 0.0, geo, wez_cfg,
                                   my_reward.MY_REWARD_CONFIG, False, True, "")
check("5.12 terminal win/loss/draw are ordered win > draw > loss",
      win > draw > loss, f"{win} / {draw} / {loss}")

# ==========================================================================
# 6. Source guard -- a hook that reads angles must also repair them
# ==========================================================================
ANGLE_CALLS = ("_get_antenna_train_angle", "_get_aspect_angle")


def live_source(module) -> str:
    """Module source with comment tails stripped.

    Naive on '#' inside string literals; the student hooks have none on the
    lines that matter, and a false positive here only makes the guard stricter.
    """
    text = Path(module.__file__).read_text(encoding="utf-8")
    body = text.split('"""')[-1] if text.count('"""') >= 2 else text
    return "\n".join(line.split("#", 1)[0] for line in body.splitlines())


for module in (my_observation, my_reward):
    src = live_source(module)
    name = Path(module.__file__).name
    calls_raw = any(call in src for call in ANGLE_CALLS)
    repairs = "_repair_degenerate_angle" in src
    check(f"6.x {name} does not read angles unrepaired",
          repairs or not calls_raw,
          "calls _get_*_angle in live code without _repair_degenerate_angle")

# ==========================================================================
# 7. Statelessness (observation gotcha #3)
# ==========================================================================
first = my_observation.build_observation(own, bandit_six, geo)
my_observation.build_observation(bandit_six, own, geo)  # swapped target-provider call
my_observation.build_observation(own, bandit_ahead, geo)  # extra step_ratio call
again = my_observation.build_observation(own, bandit_six, geo)
check("7.1 build_observation is call-order independent",
      bool(np.array_equal(first, again)))
check("7.2 repair itself holds no state",
      close(repair_degenerate_angle(raw_ata, own, bandit_six, geo, "ata"), 180.0, 1e-6))


# --------------------------------------------------------------------------
sys.exit(C.finish())
