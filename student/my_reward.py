# -*- coding: utf-8 -*-
"""Minimal student reward template.

Edit this file when you want to change what the agent learns.

Required contract:
  - MY_REWARD_CONFIG must be a dict.
  - compute_reward(...) must return (total_reward: float, components: dict).
  - Each item in components is recorded as ep_reward_<name> by the callbacks.

Units (verified, do not guess):
  - geo_info._get_distance(...) returns METERS.
    experiments/*.yaml set wez.min_range_m=152.4 / max_range_m=914.4, which are
    exactly 500 ft / 3000 ft. Conversion happens once, right before WEZ checks.
  - geo_info._get_antenna_train_angle(..., False) returns signed DEGREES in
    [-180, 180] (student/my_observation.py normalizes it with -180/180, and the
    WEZ config uses wez.angle_deg). No radian conversion is needed.
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


class WezRewardState:
    """Per-episode WEZ tracking state.

    compute_reward() is a stateless module function in this template, so the
    previous-step WEZ flag lives here instead of on a reward class.
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


def _wez_thresholds_ft(wez_config: dict) -> tuple[float, float, float]:
    """Return (strict_angle_deg, strict_min_ft, strict_max_ft)."""
    cfg = wez_config if isinstance(wez_config, dict) else {}
    angle_deg = abs(_finite(cfg.get("angle_deg", DEFAULT_WEZ_ANGLE_DEG), DEFAULT_WEZ_ANGLE_DEG))
    min_m = _finite(cfg.get("min_range_m", DEFAULT_WEZ_MIN_RANGE_M), DEFAULT_WEZ_MIN_RANGE_M)
    max_m = _finite(cfg.get("max_range_m", DEFAULT_WEZ_MAX_RANGE_M), DEFAULT_WEZ_MAX_RANGE_M)
    min_ft = max(0.0, min_m) * M_TO_FT
    max_ft = max(min_ft, max_m * M_TO_FT)
    return angle_deg, min_ft, max_ft


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
        "step": _finite(cfg.get("step_penalty", -0.01), -0.01),
    }

    state = _get_wez_state(geo_info)
    if state._episode_done:
        # Previous call ended the episode -> this call is a fresh episode.
        state.reset()

    entry_bonus = _finite(cfg.get("wez_entry_bonus", 3.0), 3.0)
    entry_decay = _clip(_finite(cfg.get("wez_entry_bonus_decay", 0.5), 0.5), 0.0, 1.0)
    hold_reward = _finite(cfg.get("wez_hold_reward", 0.05), 0.05)
    approach_max = _finite(cfg.get("approach_shaping_max", 0.15), 0.15)
    angle_max = _finite(cfg.get("angle_shaping_max", 0.15), 0.15)
    boundary_penalty = abs(_finite(cfg.get("overclose_boundary_penalty", 1.0), 1.0))
    max_penalty = abs(_finite(cfg.get("overclose_max_penalty", 6.0), 6.0))
    max_penalty = max(max_penalty, boundary_penalty)

    wez_angle_deg, wez_min_ft, wez_max_ft = _wez_thresholds_ft(wez_config)
    hold_angle_deg = wez_angle_deg + WEZ_HOLD_ANGLE_MARGIN_DEG
    hold_min_ft = max(0.0, wez_min_ft - WEZ_HOLD_MIN_MARGIN_FT)
    hold_max_ft = wez_max_ft + WEZ_HOLD_MAX_MARGIN_FT

    # --- geometry (units converted exactly once, right here) ---
    distance_m = _finite(geo_info._get_distance(ownship_state, target_state), math.nan)
    ata_deg = _finite(
        geo_info._get_antenna_train_angle(ownship_state, target_state, False), math.nan
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

        strict_in_wez = (
            abs_ata_deg <= wez_angle_deg + WEZ_ANGLE_EPS_DEG
            and wez_min_ft - WEZ_RANGE_EPS_FT <= distance_ft <= wez_max_ft + WEZ_RANGE_EPS_FT
        )
        relaxed_in_wez = (
            abs_ata_deg <= hold_angle_deg + WEZ_ANGLE_EPS_DEG
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
            terminal_reward = _finite(cfg.get("win_reward", 100.0), 100.0)
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = _finite(cfg.get("loss_reward", -100.0), -100.0)
        else:
            terminal_reward = _finite(cfg.get("draw_reward", -10.0), -10.0)
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
]
