# -*- coding: utf-8 -*-
"""Prove the student hooks' assumptions against the REAL platform source.

    python student\tests\check_against_release.py

The other check_*.py scripts run anywhere because they stub `dogfight.*` and use
a hand-written GeoMathUtil double. That makes them fast, but it also means they
can only prove the hooks are self-consistent -- never that the assumptions the
hooks are BUILT on are true. This script closes that gap: it loads the real
GeoMathUtil, state_schema, observation.normalize, FighterSim and
single_agent_env by path and checks each assumption against them.

Exit codes:
    0  every assumption verified
    1  at least one assumption is WRONG -- a hook is built on a false premise
    2  skipped: no Release tree here (this is the normal result on a dev box)

Nothing in this file writes to the platform tree. It only reads and imports.
"""
from __future__ import annotations

import importlib
import inspect
import math
import re
import sys
from pathlib import Path

EXIT_OK, EXIT_FAIL, EXIT_SKIP = 0, 1, 2

# Assumptions under test, as recorded in Docs/troubleshooting.md and claude.md.
EXPECTED_STATE_INDEX = {"ROLL": 3, "PITCH": 4, "YAW": 5, "KCAS": 12, "ALT": 44}
EXPECTED_WEZ = {"angle_deg": 2.0, "min_range_m": 152.4, "max_range_m": 914.4}
FIXTURE_HEALTH_INDEX = 45  # what student/tests/_stubs.py assumes


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.unverified: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        self.passed.append(f"{label}{(': ' + detail) if detail else ''}")

    def bad(self, label: str, detail: str = "") -> None:
        self.failed.append(f"{label}{(': ' + detail) if detail else ''}")

    def unknown(self, label: str, reason: str) -> None:
        """Could not be checked here -- not a failure, but not proof either."""
        self.unverified.append(f"{label}: {reason}")

    def expect(self, label: str, condition: bool, detail: str = "") -> None:
        """`detail` describes what went WRONG, so it is printed only on failure."""
        if condition:
            self.ok(label)
        else:
            self.bad(label, detail)

    def finish(self) -> int:
        print(f"verified   : {len(self.passed)}")
        for item in self.passed:
            print(f"  OK   {item}")
        if self.unverified:
            print(f"unverified : {len(self.unverified)}")
            for item in self.unverified:
                print(f"  ??   {item}")
        if self.failed:
            print(f"FAILED     : {len(self.failed)}")
            for item in self.failed:
                print(f"  FAIL {item}")
            print("\nA hook is built on a false assumption. Fix the hook, not the platform.")
            return EXIT_FAIL
        print("\nOK - all checkable assumptions hold against the real Release tree")
        return EXIT_OK


# ---------------------------------------------------------------------------
# Locating and importing the real tree
# ---------------------------------------------------------------------------
def find_release_root() -> Path | None:
    """Walk up looking for a directory that owns both train_rllib.py and src/dogfight."""
    for candidate in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if candidate.is_dir() and (candidate / "train_rllib.py").is_file() \
                and (candidate / "src" / "dogfight").is_dir():
            return candidate
    return None


def import_by_filename(root: Path, filename: str):
    """Import <root>/src/**/<filename> under its real dotted package name.

    Importing by dotted name (rather than spec_from_file_location) keeps the
    module's own relative imports working.
    """
    matches = sorted((root / "src").rglob(filename))
    if not matches:
        return None, f"{filename} not found under {root / 'src'}"
    rel = matches[0].relative_to(root / "src").with_suffix("")
    dotted = ".".join(rel.parts)
    try:
        return importlib.import_module(dotted), dotted
    except Exception as exc:  # noqa: BLE001 - report, never crash the run
        return None, f"import {dotted} failed: {exc!r}"


def make_state(size, north, east, alt_m, yaw_deg, pitch_deg=0.0, roll_deg=0.0,
               u=200.0, v=0.0, w=0.0):
    import numpy as np

    state = np.zeros(size, dtype=np.float64)
    state[0], state[1], state[2] = north, east, -alt_m
    state[3], state[4], state[5] = roll_deg, pitch_deg, yaw_deg
    state[6], state[7], state[8] = u, v, w
    return state


def main() -> int:
    root = find_release_root()
    if root is None:
        print("SKIP - not inside a Release tree (need train_rllib.py + src/dogfight).")
        print("       Run this from AIP_LIB\\DogFightEnv\\Release after copying student/.")
        return EXIT_SKIP

    print(f"Release root: {root}\n")
    for path in (str(root), str(root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    R = Report()

    # -----------------------------------------------------------------------
    # A. State layout -- every literal index in geometry_utils.py depends on it
    # -----------------------------------------------------------------------
    try:
        from dogfight.sim.state_schema import StateIndex  # type: ignore
    except Exception as exc:  # noqa: BLE001
        R.bad("A. StateIndex import", repr(exc))
        StateIndex = None  # type: ignore

    if StateIndex is not None:
        for name, expected in EXPECTED_STATE_INDEX.items():
            actual = getattr(StateIndex, name, None)
            actual = getattr(actual, "value", actual)  # tolerate IntEnum
            R.expect(f"A. StateIndex.{name} == {expected}", actual == expected,
                     f"actual {actual}")
        health = getattr(StateIndex, "HEALTH", None)
        health = getattr(health, "value", health)
        if health is None:
            R.bad("A. StateIndex.HEALTH exists")
        elif health != FIXTURE_HEALTH_INDEX:
            R.unknown(
                "A. StateIndex.HEALTH",
                f"real index {health}, student/tests/_stubs.py uses "
                f"{FIXTURE_HEALTH_INDEX}. Hooks use StateIndex so they are fine; "
                f"update the FIXTURE so local tests exercise the real slot.")
        else:
            R.ok(f"A. StateIndex.HEALTH == {health}")

    fighter_sim, note = import_by_filename(root, "FighterSim.py")
    if fighter_sim is None:
        R.unknown("A. state[0:3]/[3:6]/[6:9] raw layout", note)
    else:
        try:
            src = inspect.getsource(fighter_sim)
        except Exception as exc:  # noqa: BLE001
            src = ""
            R.unknown("A. FighterSim source", repr(exc))
        if src:
            # Look for "state[6] = ..." style assignments feeding body velocity.
            body_vel = re.search(r"state\[\s*6\s*\][^\n]*", src)
            R.expect("A. state[6:9] is assigned (body u/v/w)", bool(body_vel),
                     "no state[6] assignment found -- re-read FighterSim before"
                     " trusting closure_rate_mps")
            R.unknown(
                "A. units of state[6:9]",
                "source only shows the assignment; confirm m/s (not ft/s) by"
                " reading the surrounding lines: " +
                (body_vel.group(0).strip()[:90] if body_vel else "n/a"))

    # -----------------------------------------------------------------------
    # B. GeoMathUtil -- the P1 degeneracy and the sign convention
    # -----------------------------------------------------------------------
    geo_mod, note = import_by_filename(root, "GeoMathUtil.py")
    geo = None
    if geo_mod is None:
        R.bad("B. GeoMathUtil import", note)
    else:
        for _, obj in inspect.getmembers(geo_mod, inspect.isclass):
            if hasattr(obj, "_get_antenna_train_angle") and hasattr(obj, "_get_distance"):
                try:
                    geo = obj()
                    break
                except Exception:  # noqa: BLE001 - needs ctor args; try next
                    continue
        if geo is None and hasattr(geo_mod, "_get_antenna_train_angle"):
            geo = geo_mod  # module-level functions
        if geo is None:
            R.bad("B. GeoMathUtil usable instance",
                  "no zero-arg class or module-level helpers found")

    if geo is not None:
        size = max(EXPECTED_STATE_INDEX.values()) + 20
        ALT = 7000.0
        own = make_state(size, 0.0, 0.0, ALT, yaw_deg=0.0)
        six_flat = make_state(size, -1000.0, 0.0, ALT, yaw_deg=0.0)
        six_hi = make_state(size, -1000.0, 0.0, ALT + 1e-9, yaw_deg=0.0)
        six_lo = make_state(size, -1000.0, 0.0, ALT - 1e-9, yaw_deg=0.0)

        try:
            flat = float(geo._get_antenna_train_angle(own, six_flat, False))
            hi = float(geo._get_antenna_train_angle(own, six_hi, False))
            lo = float(geo._get_antenna_train_angle(own, six_lo, False))
            dist = float(geo._get_distance(own, six_flat))
        except Exception as exc:  # noqa: BLE001
            R.bad("B. GeoMathUtil call", repr(exc))
            flat = hi = lo = dist = math.nan

        R.expect("B. _get_distance returns meters", abs(dist - 1000.0) < 1e-3,
                 f"expected 1000.0 m, got {dist}")
        R.expect("B. co-altitude dead-astern ATA collapses to 0 (the P1 defect)",
                 abs(flat) < 1e-6,
                 f"got {flat}. If this is +/-180 the defect is FIXED upstream and "
                 f"_repair_degenerate_angle is now dead code -- verify before removing")
        R.expect("B. +/-1e-9 m altitude offset restores |ATA| = 180",
                 abs(abs(hi) - 180.0) < 1e-6 and abs(abs(lo) - 180.0) < 1e-6,
                 f"hi={hi}, lo={lo}")
        R.expect("B. the 1e-9 flip is a pure sign change", hi * lo < 0.0,
                 f"hi={hi}, lo={lo}")

        # THE key cross-check: our sign-independent magnitude must equal the
        # platform's |value| wherever the platform is well defined.
        from student.geometry_utils import true_angle_magnitude_deg  # noqa: E402

        worst_ata = worst_aa = 0.0
        try:
            import numpy as np

            rng = np.random.default_rng(20260731)
            for _ in range(200):
                o = make_state(size, 0.0, 0.0, ALT, yaw_deg=float(rng.uniform(0, 360)),
                               pitch_deg=float(rng.uniform(-30, 30)),
                               roll_deg=float(rng.uniform(-90, 90)))
                dz = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(10.0, 800.0))
                t = make_state(size, float(rng.uniform(-3000, 3000)),
                               float(rng.uniform(-3000, 3000)), ALT + dz,
                               yaw_deg=float(rng.uniform(0, 360)))
                worst_ata = max(worst_ata, abs(
                    abs(float(geo._get_antenna_train_angle(o, t, False)))
                    - true_angle_magnitude_deg(o, t, "ata")))
                worst_aa = max(worst_aa, abs(
                    abs(float(geo._get_aspect_angle(o, t, False)))
                    - true_angle_magnitude_deg(o, t, "aa")))
        except Exception as exc:  # noqa: BLE001
            R.bad("B. magnitude cross-check", repr(exc))
        else:
            R.expect("B. our ATA magnitude matches the real helper (200 samples)",
                     worst_ata < 1e-6, f"max |diff| = {worst_ata} deg")
            R.expect("B. our AA convention matches the real helper (200 samples)",
                     worst_aa < 1e-6,
                     f"max |diff| = {worst_aa} deg -- if this fails, AA is NOT "
                     f"'180 - angle(target nose, target->ownship LOS)'. Fix "
                     f"true_angle_magnitude_deg(kind='aa').")

    # -----------------------------------------------------------------------
    # C. update_damage -- the P2 half-angle gate
    # -----------------------------------------------------------------------
    env_mod, note = import_by_filename(root, "single_agent_env.py")
    if env_mod is None:
        R.bad("C. single_agent_env import", note)
    else:
        update_damage = None
        for _, obj in inspect.getmembers(env_mod, inspect.isclass):
            if hasattr(obj, "update_damage"):
                update_damage = obj.update_damage
                break
        if update_damage is None:
            R.bad("C. update_damage found")
        else:
            try:
                src = inspect.getsource(update_damage)
            except Exception as exc:  # noqa: BLE001
                src = ""
                R.bad("C. update_damage source", repr(exc))
            if src:
                halves = re.search(
                    r"angle_deg[\"']\s*\]\s*/\s*2(\.0)?|/\s*2(\.0)?\s*#.*half", src)
                R.expect("C. damage cone is angle_deg / 2.0", bool(halves),
                         "no '/ 2.0' on angle_deg -- the half-angle premise of "
                         "my_reward.wez_damage_half_angle_deg() is WRONG")
                inclusive = re.search(r"half_wez_angle_deg\s*>=|>=\s*abs\(", src)
                R.expect("C. the boundary is inclusive (>=)", bool(inclusive),
                         "gate is not '>=' -- my_reward's boundary handling is off "
                         "by one float at exactly the half-angle")
                R.expect("C. the gate compares abs(ATA)", "abs(" in src,
                         "no abs() -- sign handling differs from in_wez_angle_strict")

                # Sweep our gate against the rule as written in the real source.
                sys.path.insert(0, str(root))
                from student import my_reward  # noqa: E402

                mismatches = []
                for angle_deg in (1.0, 2.0, 3.0):
                    cfg = dict(EXPECTED_WEZ, angle_deg=angle_deg)
                    for i in range(41):
                        ata = i / 10.0
                        expected = (angle_deg / 2.0) >= abs(ata)
                        if my_reward.in_wez_angle_strict(ata, cfg) != expected:
                            mismatches.append(f"angle_deg={angle_deg}, ATA={ata}")
                R.expect("C. student gate matches the real rule at 41x3 points",
                         not mismatches, f"first: {mismatches[:3]}")

    # -----------------------------------------------------------------------
    # D. normalize() -- does it clip? obs[] range contract depends on the answer
    # -----------------------------------------------------------------------
    try:
        from dogfight.envs.observation import normalize  # type: ignore

        over = float(normalize(30000.0, 0.0, 20000.0))
        under = float(normalize(-5000.0, 0.0, 20000.0))
        clips = abs(over) <= 1.0 and abs(under) <= 1.0
        if clips:
            R.ok("D. normalize clips to [-1, 1]", f"30000 -> {over}")
        else:
            R.ok("D. normalize does NOT clip (affine only)",
                 f"30000 -> {over}; my_observation clips every feature itself, "
                 f"so the [-1, 1] contract still holds")
        R.expect("D. normalize maps the midpoint to 0",
                 abs(float(normalize(10000.0, 0.0, 20000.0))) < 1e-9,
                 f"got {float(normalize(10000.0, 0.0, 20000.0))}")
    except Exception as exc:  # noqa: BLE001
        R.bad("D. normalize import/probe", repr(exc))

    # -----------------------------------------------------------------------
    # E. WEZ defaults and step_ratio
    # -----------------------------------------------------------------------
    cfg_mod, note = import_by_filename(root, "config.py")
    if cfg_mod is None:
        R.unknown("E. wez defaults", note)
    else:
        try:
            src = inspect.getsource(cfg_mod)
        except Exception as exc:  # noqa: BLE001
            src = ""
            R.unknown("E. config source", repr(exc))
        for key, expected in EXPECTED_WEZ.items():
            # Matches both `angle_deg = 2.0` and `"angle_deg": 2.0`.
            found = re.search(rf"{key}[\"']?\s*[:=]\s*([0-9.]+)", src) if src else None
            if found is None:
                R.unknown(f"E. wez.{key}", "not found in config.py")
            else:
                R.expect(f"E. wez.{key} default == {expected}",
                         abs(float(found.group(1)) - expected) < 1e-9,
                         f"config says {found.group(1)}")

    # -----------------------------------------------------------------------
    # F. Observation shape actually produced by the hook
    # -----------------------------------------------------------------------
    if geo is not None:
        try:
            sys.path.insert(0, str(root))
            from student import my_observation  # noqa: E402

            size = max(EXPECTED_STATE_INDEX.values()) + 20
            obs = my_observation.build_observation(
                make_state(size, 0.0, 0.0, 7000.0, yaw_deg=0.0),
                make_state(size, 1000.0, 0.0, 7000.0, yaw_deg=180.0), geo)
            R.expect("F. build_observation returns OBSERVATION_SIZE values",
                     obs.shape == (my_observation.OBSERVATION_SIZE,),
                     f"shape {obs.shape} vs OBSERVATION_SIZE "
                     f"{my_observation.OBSERVATION_SIZE}")
            R.expect("F. every observation value is within [-1, 1]",
                     bool((obs >= -1.0).all() and (obs <= 1.0).all()), f"{obs}")
            R.expect("F. every observation value is finite",
                     bool(__import__("numpy").isfinite(obs).all()), f"{obs}")
        except Exception as exc:  # noqa: BLE001
            R.bad("F. build_observation against the real geo_info", repr(exc))

    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
