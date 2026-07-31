# -*- coding: utf-8 -*-
"""Student reward: WEZ-shaped, aligned with the platform damage cone.

Ported from branch `main` and corrected -- see "WEZ angle" below.

Required contract:
  - MY_REWARD_CONFIG must be a dict.
  - compute_reward(...) must return (total_reward: float, components: dict).
  - Each item in components is recorded as ep_reward_<name> by the callbacks.
    Keep the component names stable across experiments; they become
    ep_reward_<name> columns in training_log.csv.

Units (verified, do not guess):
  - geo_info._get_distance(...) returns METERS.
    experiments/*.yaml set wez.min_range_m=152.4 / max_range_m=914.4, which are
    exactly 500 ft / 3000 ft. Conversion happens once, right before WEZ checks.
  - geo_info._get_antenna_train_angle(..., False) returns signed DEGREES in
    [-180, 180]. No radian conversion is needed.

WEZ angle -- the damage cone is HALF of wez.angle_deg:
  Platform single_agent_env.update_damage() gates damage on

      half_wez_angle_deg = self._wez["angle_deg"] / 2.0     # 2.0 -> 1.0
      if half_wez_angle_deg >= abs(ownship_ata_deg): ...

  so with the shipped angle_deg=2.0 the real cone is |ATA| <= 1.0 deg. Gating
  the reward on the full 2.0 accepts a cone twice as wide as the one that
  actually scores -- the agent gets paid for shots that do nothing. Entry here
  therefore uses angle_deg / 2.0, computed once in _wez_thresholds_ft().

Angle degeneracy -- read before touching any angle:
  GeoMathUtil signs both angle helpers with np.sign(p_unit_t[2]), and
  np.sign(0.0) == 0.0. At exactly co-altitude -- where every episode starts,
  since initial_scenario.altitude_m is shared -- the angle collapses to 0
  ("nose-on") with the enemy anywhere, including our six. Every angle read here
  goes through student.geometry_utils._repair_degenerate_angle first.
  Docs/troubleshooting.md §6.
"""
from __future__ import annotations

import math
import sys
import weakref
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex

try:  # normal case: loaded as "student.my_reward" by train_rllib.py
    from student.geometry_utils import _repair_degenerate_angle
except ImportError:  # loaded by file path
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from geometry_utils import _repair_degenerate_angle


M_TO_FT = 3.28084

# WEZ fallbacks, used only when wez_config does not provide them.
# They match env_config.wez in experiments/student_sac_mlp.yaml.
DEFAULT_WEZ_ANGLE_DEG = 2.0
DEFAULT_WEZ_MIN_RANGE_M = 152.4  # 500 ft
DEFAULT_WEZ_MAX_RANGE_M = 914.4  # 3000 ft

# update_damage() divides angle_deg by this to get the damage cone half-angle.
# Defined once; never write the literal 2.0 at a comparison site.
WEZ_DAMAGE_HALF_ANGLE_DIVISOR = 2.0

# Hysteresis. Entry uses the strict half-angle with no slack, so it matches
# update_damage() exactly; only STAYING inside is relaxed. Expressed as a ratio
# of the half-angle rather than an absolute degree margin, so it keeps its
# proportion if angle_deg changes (half-angle 1.0 -> hold 1.25).
WEZ_HOLD_ANGLE_MARGIN_RATIO = 0.25
WEZ_HOLD_MIN_MARGIN_FT = 50.0
WEZ_HOLD_MAX_MARGIN_FT = 200.0

# Range boundary tolerance. 152.4 m converts to 500.000016 ft, so an
# exactly-500 ft range would otherwise read as below the floor. Physically
# negligible. There is deliberately NO angle counterpart: the angle comparison
# must be bit-identical to update_damage()'s.
WEZ_RANGE_EPS_FT = 1e-3

# Distance at which approach shaping starts (it saturates at the WEZ max range).
APPROACH_SHAPING_FAR_FT = 12000.0
# Angle shaping decays linearly over the full [0, 180] deg ATA range.
ANGLE_SHAPING_SPAN_DEG = 180.0


M_TO_FT = 3.28084

# WEZ fallbacks, used only when wez_config does not provide them.
# They match env_config.wez in experiments/student_sac_mlp.yaml.
DEFAULT_WEZ_ANGLE_DEG = 2.0
DEFAULT_WEZ_MIN_RANGE_M = 152.4  # 500 ft
DEFAULT_WEZ_MAX_RANGE_M = 914.4  # 3000 ft

# Hysteresis margins applied on top of the strict WEZ gate. Entry always uses
# the strict gate; only staying inside uses the relaxed gate. With the defaults
# above this yields |ata| <= 2.5 deg and 450 ft <= d <= 3200 ft.
WEZ_HOLD_ANGLE_MARGIN_DEG = 0.5
WEZ_HOLD_MIN_MARGIN_FT = 50.0
WEZ_HOLD_MAX_MARGIN_FT = 200.0

# Boundary tolerance. 152.4 m converts to 500.000016 ft, so an exactly-500 ft
# range would otherwise read as below the floor. These are physically negligible.
WEZ_RANGE_EPS_FT = 1e-3
WEZ_ANGLE_EPS_DEG = 1e-9

# Distance at which approach shaping starts (it saturates at the WEZ max range).
APPROACH_SHAPING_FAR_FT = 12000.0
# Angle shaping decays linearly over the full [0, 180] deg ATA range.
ANGLE_SHAPING_SPAN_DEG = 180.0


MY_REWARD_CONFIG = {
    "step_penalty": -0.01,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -10.0,
    # --- WEZ shaping (all tunable, never hard-coded below) ---
    "wez_entry_bonus": 3.0,
    # Repeated entries within one episode are worth less, so leaving and
    # re-entering the WEZ cannot be farmed.
    "wez_entry_bonus_decay": 0.5,
    "wez_hold_reward": 0.05,
    "approach_shaping_max": 0.15,
    "angle_shaping_max": 0.15,
    "overclose_boundary_penalty": 1.0,
    "overclose_max_penalty": 6.0,
}


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _finite(value, fallback: float = 0.0) -> float:
    """Return float(value) if it is finite, otherwise fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _tunable(cfg: dict, key: str) -> float:
    """Read one tunable from `cfg`, falling back to MY_REWARD_CONFIG.

    MY_REWARD_CONFIG is the single source of truth for every default. Writing
    the fallback as a literal at the call site instead (`cfg.get(k, -0.01)`)
    creates two numbers that must be kept equal by hand -- retuning the dict
    then silently leaves the fallback behind, and the two disagree exactly in
    the case the fallback exists for: a reward_config that is missing the key
    (a curriculum stage's reward_overrides, or env_config.reward under
    `reward.mode: default`).
    """
    default = float(MY_REWARD_CONFIG[key])
    return _finite(cfg.get(key, default), default)


class WezRewardState:
    """Per-episode WEZ tracking state.

    compute_reward() is a stateless module function in this template, so the
    previous-step WEZ flag lives here instead of on a reward class.

    Unlike build_observation(), the reward hook IS called exactly once per step
    (_reward_fn runs only inside step(); get_reward() is not used externally),
    so this state is safe. Do not copy the pattern into the observation hook.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._was_in_wez = False
        self._entry_count = 0
        self._episode_done = False

    @property
    def was_in_wez(self) -> bool:
        return self._was_in_wez


# One state object per environment instance. geo_info is the per-env geometry
# helper, so it is a stable key; the weak map keeps this leak-free with multiple
# env runners. Falls back to a single shared state if geo_info is not weakref-able.
_WEZ_STATES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_FALLBACK_WEZ_STATE = WezRewardState()


def _get_wez_state(geo_info) -> WezRewardState:
    try:
        state = _WEZ_STATES.get(geo_info)
        if state is None:
            state = WezRewardState()
            _WEZ_STATES[geo_info] = state
        return state
    except TypeError:  # unhashable / not weakref-able
        return _FALLBACK_WEZ_STATE


def reset_reward_state(geo_info=None) -> None:
    """Clear the stored WEZ state. Safe to call at episode reset."""
    if geo_info is None:
        _FALLBACK_WEZ_STATE.reset()
        _WEZ_STATES.clear()
        return
    _get_wez_state(geo_info).reset()


def wez_damage_half_angle_deg(wez_config: dict) -> float:
    """The ONLY place the damage-cone half-angle is derived.

    Mirrors update_damage():  half_wez_angle_deg = wez["angle_deg"] / 2.0
    """
    cfg = wez_config if isinstance(wez_config, dict) else {}
    angle_deg = abs(
        _finite(cfg.get("angle_deg", DEFAULT_WEZ_ANGLE_DEG), DEFAULT_WEZ_ANGLE_DEG)
    )
    return angle_deg / WEZ_DAMAGE_HALF_ANGLE_DIVISOR


def _wez_thresholds_ft(wez_config: dict) -> tuple[float, float, float]:
    """Return (strict_half_angle_deg, strict_min_ft, strict_max_ft).

    The angle is the damage-cone HALF-angle, not wez.angle_deg.
    """
    cfg = wez_config if isinstance(wez_config, dict) else {}
    half_angle_deg = wez_damage_half_angle_deg(cfg)
    min_m = _finite(cfg.get("min_range_m", DEFAULT_WEZ_MIN_RANGE_M), DEFAULT_WEZ_MIN_RANGE_M)
    max_m = _finite(cfg.get("max_range_m", DEFAULT_WEZ_MAX_RANGE_M), DEFAULT_WEZ_MAX_RANGE_M)
    min_ft = max(0.0, min_m) * M_TO_FT
    max_ft = max(min_ft, max_m * M_TO_FT)
    return half_angle_deg, min_ft, max_ft


def in_wez_angle_strict(abs_ata_deg: float, wez_config: dict) -> bool:
    """Strict WEZ angle gate -- identical to update_damage()'s comparison.

    Exposed so tests can sweep it against the platform rule without
    reconstructing the reward's internals.
    """
    return wez_damage_half_angle_deg(wez_config) >= abs(abs_ata_deg)


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Return a WEZ-shaped reward.

    The arguments expose aircraft state, damage, geometry, WEZ settings, and
    termination status. Add your own tactical components here.
    """
    cfg = reward_config if isinstance(reward_config, dict) else {}

    components: dict[str, float] = {
        "step": _tunable(cfg, "step_penalty"),
    }

    state = _get_wez_state(geo_info)
    if state._episode_done:
        # Previous call ended the episode -> this call is a fresh episode.
        state.reset()

    entry_bonus = _tunable(cfg, "wez_entry_bonus")
    entry_decay = _clip(_tunable(cfg, "wez_entry_bonus_decay"), 0.0, 1.0)
    hold_reward = _tunable(cfg, "wez_hold_reward")
    approach_max = _tunable(cfg, "approach_shaping_max")
    angle_max = _tunable(cfg, "angle_shaping_max")
    boundary_penalty = abs(_tunable(cfg, "overclose_boundary_penalty"))
    max_penalty = abs(_tunable(cfg, "overclose_max_penalty"))
    max_penalty = max(max_penalty, boundary_penalty)

    # half_angle_deg is the damage-cone half-angle (angle_deg / 2), derived in
    # one place. Hold relaxes it; entry does not.
    half_angle_deg, wez_min_ft, wez_max_ft = _wez_thresholds_ft(wez_config)
    hold_angle_deg = half_angle_deg * (1.0 + WEZ_HOLD_ANGLE_MARGIN_RATIO)
    hold_min_ft = max(0.0, wez_min_ft - WEZ_HOLD_MIN_MARGIN_FT)
    hold_max_ft = wez_max_ft + WEZ_HOLD_MAX_MARGIN_FT

    # --- geometry (units converted exactly once, right here) ---
    distance_m = _finite(geo_info._get_distance(ownship_state, target_state), math.nan)
    # Repaired before use: at exactly co-altitude the platform reports 0 deg for
    # any geometry, which would hand out full angle_align credit -- and open the
    # WEZ gate -- with the enemy sitting at our six. Only that degenerate band is
    # touched; every other reading passes through unchanged. abs() below makes
    # the repaired value's (positive) sign irrelevant.
    ata_deg = _finite(
        _repair_degenerate_angle(
            geo_info._get_antenna_train_angle(ownship_state, target_state, False),
            ownship_state,
            target_state,
            geo_info,
            "ata",
        ),
        math.nan,
    )
    geometry_ok = math.isfinite(distance_m) and math.isfinite(ata_deg)

    wez_entry = 0.0
    wez_hold = 0.0
    overclose = 0.0
    approach = 0.0
    angle_align = 0.0

    if geometry_ok:
        distance_ft = max(0.0, distance_m) * M_TO_FT
        abs_ata_deg = abs(ata_deg)

        # No epsilon on the angle: this comparison must agree with
        # update_damage() at every float, including exactly on the boundary.
        strict_in_wez = (
            half_angle_deg >= abs_ata_deg
            and wez_min_ft - WEZ_RANGE_EPS_FT <= distance_ft <= wez_max_ft + WEZ_RANGE_EPS_FT
        )
        relaxed_in_wez = (
            hold_angle_deg >= abs_ata_deg
            and hold_min_ft - WEZ_RANGE_EPS_FT <= distance_ft <= hold_max_ft + WEZ_RANGE_EPS_FT
        )
        # Entering uses the strict gate; only holding uses the relaxed gate.
        in_wez = relaxed_in_wez if state._was_in_wez else strict_in_wez
        too_close = distance_ft < wez_min_ft - WEZ_RANGE_EPS_FT

        if too_close:
            # Continuous, monotonically growing penalty below the WEZ floor.
            severity = _clip((wez_min_ft - distance_ft) / max(wez_min_ft, 1e-6), 0.0, 1.0)
            overclose = -(
                boundary_penalty + (max_penalty - boundary_penalty) * severity ** 2
            )
        else:
            # Entry bonus only on a strict, non-overclose first entry.
            if in_wez and not state._was_in_wez:
                wez_entry = entry_bonus * (entry_decay ** state._entry_count)
                state._entry_count += 1
            elif in_wez:
                wez_hold = hold_reward

        state._was_in_wez = in_wez

        # Approach shaping: gradient only outside the WEZ max range, then a flat
        # plateau inside it, so "closer is always better" never holds under 3000 ft.
        span = max(APPROACH_SHAPING_FAR_FT - wez_max_ft, 1e-6)
        approach = approach_max * _clip(
            (APPROACH_SHAPING_FAR_FT - distance_ft) / span, 0.0, 1.0
        )

        # Bounded angle alignment shaping.
        angle_align = angle_max * _clip(
            1.0 - abs_ata_deg / ANGLE_SHAPING_SPAN_DEG, 0.0, 1.0
        )

    components["wez_entry"] = wez_entry
    components["wez_hold"] = wez_hold
    components["overclose"] = overclose
    components["approach"] = approach
    components["angle_align"] = angle_align

    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = _finite(ownship_state[StateIndex.HEALTH])
        target_health = _finite(target_state[StateIndex.HEALTH])
        if target_health <= 0.0 < ownship_health:
            terminal_reward = _tunable(cfg, "win_reward")
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = _tunable(cfg, "loss_reward")
        else:
            terminal_reward = _tunable(cfg, "draw_reward")
        # Mark the episode boundary so the next call starts from a clean state.
        state._episode_done = True
    components["terminal"] = terminal_reward

    for key, value in components.items():
        components[key] = _finite(value)

    return _finite(sum(components.values())), components


__all__ = [
    "MY_REWARD_CONFIG",
    "compute_reward",
    "WezRewardState",
    "reset_reward_state",
    "wez_damage_half_angle_deg",
    "in_wez_angle_strict",
]
