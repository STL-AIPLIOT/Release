# -*- coding: utf-8 -*-
"""Shared test scaffolding: dogfight.* stubs, state builder, tiny assert helpers.

TEST FIXTURE ONLY. The stubs mirror just enough of the platform surface for the
student hooks to import on a machine with no Release tree. They are not a
substitute for running against the real package -- that is
check_against_release.py's job.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

STATE_SIZE = 64
I_HEALTH = 45


def install_dogfight_stubs() -> None:
    """Register minimal dogfight.* modules unless the real ones are importable."""
    if "dogfight" in sys.modules:
        return

    class StateIndex:
        ROLL, PITCH, YAW = 3, 4, 5
        KCAS = 12
        ALT = 44
        HEALTH = I_HEALTH

    def normalize(value, low, high):
        """Affine map, deliberately WITHOUT clipping.

        Whether the real dogfight.envs.observation.normalize clips is not
        documented, and AIP_LIB is not present on every dev box. Modelling the
        permissive case here is the safe choice: it forces the student hook to
        own its own [-1, 1] guarantee instead of inheriting one that may not
        exist. check_against_release.py probes which variant the real tree has.
        """
        span = high - low
        if span == 0:
            return 0.0
        return float(2.0 * (value - low) / span - 1.0)

    dogfight = types.ModuleType("dogfight")
    envs = types.ModuleType("dogfight.envs")
    observation = types.ModuleType("dogfight.envs.observation")
    sim = types.ModuleType("dogfight.sim")
    state_schema = types.ModuleType("dogfight.sim.state_schema")

    observation.normalize = normalize
    state_schema.StateIndex = StateIndex
    envs.observation = observation
    sim.state_schema = state_schema
    dogfight.envs = envs
    dogfight.sim = sim

    sys.modules.update(
        {
            "dogfight": dogfight,
            "dogfight.envs": envs,
            "dogfight.envs.observation": observation,
            "dogfight.sim": sim,
            "dogfight.sim.state_schema": state_schema,
        }
    )


def make_state(north, east, alt_m, yaw_deg, pitch_deg=0.0, roll_deg=0.0,
               u=180.0, v=0.0, w=0.0, kcas=180.0, health=100.0):
    """Build a state vector.

    N/E/D meters, roll/pitch/yaw degrees, u/v/w body velocity m/s
    (FighterSim.py:170-180).
    """
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[0] = north
    state[1] = east
    state[2] = -alt_m  # Down is negative altitude
    state[3] = roll_deg
    state[4] = pitch_deg
    state[5] = yaw_deg
    state[6] = u
    state[7] = v
    state[8] = w
    state[12] = kcas
    state[44] = alt_m
    state[I_HEALTH] = health
    return state


class Checker:
    """Counts passes, collects failures, prints a summary, returns an exit code."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.passed = 0
        self.failures: list[str] = []

    def check(self, label, condition, detail: str = "") -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(f"{label}{(' -- ' + detail) if detail else ''}")

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def finish(self) -> int:
        print(f"checks passed: {self.passed}")
        if self.failures:
            print(f"checks FAILED: {len(self.failures)}")
            for item in self.failures:
                print(f"  - {item}")
            return 1
        print(f"OK - {self.title}")
        return 0


def close(a, b, tol=1e-6) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


__all__ = [
    "install_dogfight_stubs",
    "make_state",
    "Checker",
    "close",
    "raises",
    "STATE_SIZE",
    "I_HEALTH",
    "RELEASE_ROOT",
]
