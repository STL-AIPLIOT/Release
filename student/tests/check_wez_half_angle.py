# -*- coding: utf-8 -*-
"""WEZ angle gate must match the platform damage cone. Runs anywhere.

    python student\tests\check_wez_half_angle.py

Oracle -- single_agent_env.update_damage(), quoted as the standard to match,
never edited:

    half_wez_angle_deg = self._wez["angle_deg"] / 2.0     # 2.0 -> 1.0
    if half_wez_angle_deg >= abs(ownship_ata_deg): ...

So the real damage cone is HALF of wez.angle_deg. Gating the reward on the full
angle_deg pays the agent for shots that cannot score (the 07/29 defect).

This encodes the rule as an oracle and sweeps it against the student gate.
On a machine with the Release tree, check_against_release.py compares against
the actual update_damage() instead.

Exit 0 = all passed, 1 = at least one failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Release root

from student.tests._stubs import Checker, close, install_dogfight_stubs, make_state  # noqa: E402

install_dogfight_stubs()

from student import my_reward  # noqa: E402
from student.tests.geomath_stub import GeoMathUtilStub  # noqa: E402

C = Checker("WEZ angle gate matches update_damage()")
geo = GeoMathUtilStub()
ALT = 7000.0


def oracle_damage_allowed(ata_deg: float, angle_deg: float) -> bool:
    """update_damage()'s angle test, verbatim."""
    half_wez_angle_deg = angle_deg / 2.0
    return half_wez_angle_deg >= abs(ata_deg)


def wez_cfg(angle_deg: float) -> dict:
    return {"angle_deg": angle_deg, "min_range_m": 152.4, "max_range_m": 914.4}


# ==========================================================================
# 1. 41-point sweep, 0.0 .. 4.0 deg in 0.1 steps, against the oracle
# ==========================================================================
SWEEP = [i / 10.0 for i in range(41)]  # i/10 is correctly rounded; i*0.1 is not
C.check("1.0 sweep has 41 points at 0.1 deg spacing",
        len(SWEEP) == 41 and close(SWEEP[-1], 4.0, 1e-12))

for angle_deg in (1.0, 2.0, 3.0):
    mismatches = []
    for ata in SWEEP:
        expected = oracle_damage_allowed(ata, angle_deg)
        actual = my_reward.in_wez_angle_strict(ata, wez_cfg(angle_deg))
        if actual != expected:
            mismatches.append(f"ATA={ata} expected={expected} got={actual}")
    C.check(f"1.x angle_deg={angle_deg}: 41/41 points match update_damage",
            not mismatches, "; ".join(mismatches[:4]))

# Negative ATA must behave identically (the platform takes abs()).
neg_mismatches = [
    ata for ata in SWEEP
    if my_reward.in_wez_angle_strict(-ata, wez_cfg(2.0))
    != oracle_damage_allowed(-ata, 2.0)
]
C.check("1.4 negative ATA sweep matches too", not neg_mismatches,
        f"first mismatches: {neg_mismatches[:4]}")

# ==========================================================================
# 2. The boundary sits exactly at half, for several angle_deg values
# ==========================================================================
for angle_deg in (1.0, 2.0, 2.5, 3.0, 4.0):
    half = angle_deg / 2.0
    cfg = wez_cfg(angle_deg)
    C.check(f"2.x angle_deg={angle_deg}: half-angle reported as {half}",
            close(my_reward.wez_damage_half_angle_deg(cfg), half, 1e-12))
    C.check(f"2.x angle_deg={angle_deg}: ATA exactly at half is INSIDE",
            my_reward.in_wez_angle_strict(half, cfg))
    C.check(f"2.x angle_deg={angle_deg}: ATA just past half is OUTSIDE",
            not my_reward.in_wez_angle_strict(half + 1e-9, cfg))
    C.check(f"2.x angle_deg={angle_deg}: the FULL angle is OUTSIDE (the fix)",
            not my_reward.in_wez_angle_strict(angle_deg, cfg) or angle_deg == 0.0,
            "full angle_deg still accepted -- half-angle not applied")

C.check("2.9 angle_deg=0 admits only ATA 0",
        my_reward.in_wez_angle_strict(0.0, wez_cfg(0.0))
        and not my_reward.in_wez_angle_strict(0.1, wez_cfg(0.0)))

# ==========================================================================
# 3. compute_reward() actually uses that threshold (wiring, not just the helper)
# ==========================================================================
def entry_at(ata_deg: float, angle_deg: float, distance_m: float = 500.0) -> float:
    """wez_entry component for a target at `ata_deg` off the nose."""
    import math

    own = make_state(0.0, 0.0, ALT, yaw_deg=0.0)
    bearing = math.radians(ata_deg)
    target = make_state(distance_m * math.cos(bearing), distance_m * math.sin(bearing),
                        ALT, yaw_deg=0.0)
    my_reward.reset_reward_state()
    _, comps = my_reward.compute_reward(
        own, target, 0.0, 0.0, geo, wez_cfg(angle_deg), my_reward.MY_REWARD_CONFIG,
        False, False, "")
    return comps["wez_entry"]


# angle_deg=2.0 -> cone is |ATA| <= 1.0. Probed away from the exact boundary so
# the arccos reconstruction's 1e-13 slop cannot decide the outcome.
C.check("3.1 ATA 0.5 deg (inside the 1.0 deg cone) grants entry",
        entry_at(0.5, 2.0) > 0.0, f"got {entry_at(0.5, 2.0)}")
C.check("3.2 ATA 0.95 deg (just inside) grants entry",
        entry_at(0.95, 2.0) > 0.0, f"got {entry_at(0.95, 2.0)}")
C.check("3.3 ATA 1.05 deg (just outside) grants NOTHING",
        entry_at(1.05, 2.0) == 0.0, f"got {entry_at(1.05, 2.0)}")
C.check("3.4 ATA 1.5 deg -- the old full-angle gate would have paid here",
        entry_at(1.5, 2.0) == 0.0, f"got {entry_at(1.5, 2.0)}")
C.check("3.5 ATA 1.9 deg (inside full 2.0, outside half) grants NOTHING",
        entry_at(1.9, 2.0) == 0.0, f"got {entry_at(1.9, 2.0)}")

# The gate must track angle_deg, not a hard-coded number.
C.check("3.6 angle_deg=4.0 widens the cone to 2.0 deg", entry_at(1.9, 4.0) > 0.0,
        f"got {entry_at(1.9, 4.0)}")
C.check("3.7 angle_deg=1.0 narrows the cone to 0.5 deg", entry_at(0.7, 1.0) == 0.0,
        f"got {entry_at(0.7, 1.0)}")

# ==========================================================================
# 4. Hysteresis is defined on top of the half-angle, and only relaxes holding
# ==========================================================================
half = my_reward.wez_damage_half_angle_deg(wez_cfg(2.0))
hold = half * (1.0 + my_reward.WEZ_HOLD_ANGLE_MARGIN_RATIO)
C.check("4.1 hold angle is strictly wider than the entry half-angle", hold > half)
C.check("4.2 hold margin is a ratio of the half-angle, not an absolute constant",
        close(hold, 1.25, 1e-12), f"got {hold}")
C.check("4.3 hold angle scales with angle_deg",
        close(my_reward.wez_damage_half_angle_deg(wez_cfg(4.0))
              * (1.0 + my_reward.WEZ_HOLD_ANGLE_MARGIN_RATIO), 2.5, 1e-12))
C.check("4.4 entry itself has no angle slack (bit-identical to update_damage)",
        not hasattr(my_reward, "WEZ_ANGLE_EPS_DEG"),
        "an angle epsilon would make the gate disagree with update_damage on the boundary")

# Holding: enter at 0.5 deg, then drift to 1.1 deg -- outside entry, inside hold.
import math  # noqa: E402

own = make_state(0.0, 0.0, ALT, yaw_deg=0.0)


def target_at(ata_deg, distance_m=500.0):
    bearing = math.radians(ata_deg)
    return make_state(distance_m * math.cos(bearing), distance_m * math.sin(bearing),
                      ALT, yaw_deg=0.0)


my_reward.reset_reward_state()
_, c_in = my_reward.compute_reward(own, target_at(0.5), 0.0, 0.0, geo, wez_cfg(2.0),
                                   my_reward.MY_REWARD_CONFIG, False, False, "")
_, c_drift = my_reward.compute_reward(own, target_at(1.1), 0.0, 0.0, geo, wez_cfg(2.0),
                                      my_reward.MY_REWARD_CONFIG, False, False, "")
C.check("4.5 entry then drift to 1.1 deg keeps the hold reward",
        c_in["wez_entry"] > 0.0 and c_drift["wez_hold"] > 0.0,
        f"entry={c_in['wez_entry']}, hold={c_drift['wez_hold']}")

my_reward.reset_reward_state()
_, c_cold = my_reward.compute_reward(own, target_at(1.1), 0.0, 0.0, geo, wez_cfg(2.0),
                                     my_reward.MY_REWARD_CONFIG, False, False, "")
C.check("4.6 but 1.1 deg from cold grants nothing (hold never gates entry)",
        c_cold["wez_entry"] == 0.0 and c_cold["wez_hold"] == 0.0,
        f"entry={c_cold['wez_entry']}, hold={c_cold['wez_hold']}")

# ==========================================================================
# 5. Distance logic untouched
# ==========================================================================
C.check("5.1 range thresholds still 500 ft / 3000 ft",
        close(my_reward._wez_thresholds_ft(wez_cfg(2.0))[1], 152.4 * 3.28084, 1e-6)
        and close(my_reward._wez_thresholds_ft(wez_cfg(2.0))[2], 914.4 * 3.28084, 1e-6))
C.check("5.2 nose-on but too far grants nothing", entry_at(0.0, 2.0, 5000.0) == 0.0)
C.check("5.3 nose-on but too close is penalized, not rewarded",
        entry_at(0.0, 2.0, 100.0) == 0.0)
C.check("5.4 range epsilon retained (500.000016 ft boundary)",
        close(my_reward.WEZ_RANGE_EPS_FT, 1e-3, 0.0))

sys.exit(C.finish())
