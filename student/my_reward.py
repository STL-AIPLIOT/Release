# -*- coding: utf-8 -*-
"""STIL reward v2 (Gate 2).

v1 구조:
    w1*cos(ATA) + w2*cos(AA) + WEZ bonus + dEnergy(소가중치) + 추락 대패널티

v2에서 추가된 것 (W2 "보상 함수 v1 -> v2 조정"):
    - 거리 shaping 안전 밴드: WEZ 500~3000 ft 진입 보너스 / 500 ft 미만 패널티
    - ATA:AA 가중치 비율 고정 (상쇄 방지, 아래 참조)
    - 모든 컴포넌트에 0 division / NaN 가드

컴포넌트 (training_log.csv에 ep_reward_<name>으로 기록. 이름을 바꾸면 실험 간
비교가 끊기므로 유지할 것):
    step      매 step 소액 페널티 ("우물쭈물하지 마라")
    pursuit   +w_pursuit  * cos(ATA)   "적을 향해 기수를 두라"
    position  +w_position * cos(AA)    "적의 6시로 가라"
    wez_entry WEZ 최초 진입 보너스 (에피소드 내 반복 진입은 체감)
    wez_hold  WEZ 유지 보너스
    overclose WEZ 최소 사거리(500 ft) 미만 패널티
    energy    비에너지 우위의 step 변화량 * 소가중치
    crash     추락(최저고도) / FDM 종료 대패널티
    terminal  승/패/무 종료 보상

상쇄 위험 점검 (Docs/05 §13):
    헤드온(ATA=0, AA=180)에서 pursuit=+w1, position=-w2로 서로 반대가 된다.
    w_pursuit > w_position 을 강제해 "적을 향해 기수를 드는 행동"의 합이 항상
    양수가 되게 했다 (기본값 0.6 / 0.3 -> 헤드온 순합 +0.3).
    두 항을 분리 기록하므로 대시보드에서 상쇄 구간을 직접 볼 수 있다.

단위 (Release 본체 소스에서 확인, 추측 아님):
    GeoMathUtil.py  _get_distance                  -> METER
                    _get_antenna_train_angle(F)    -> signed DEGREE [-180, 180]
                    _get_aspect_angle(F)           -> signed DEGREE [-180, 180], 0 = 적의 6시
    FighterSim.py L186  StateIndex.KCAS            -> **METER/SEC** (knot 아님)
    FighterSim.py L244  StateIndex.ALT             -> **METER** (feet 아님)
    src/dogfight/config.py 의 wez.min_range_m / max_range_m 는 500 ft / 3000 ft를
    미터로 환산한 값이다. ft 변환은 WEZ 판정 직전 한 번만 수행한다.
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
G = 9.80665
DEG_TO_RAD = math.pi / 180.0

# WEZ fallback. wez_config가 값을 주지 않을 때만 쓰이며
# src/dogfight/config.py 의 wez 기본값과 같다.
DEFAULT_WEZ_ANGLE_DEG = 2.0
DEFAULT_WEZ_MIN_RANGE_M = 152.4  # 500 ft
DEFAULT_WEZ_MAX_RANGE_M = 914.4  # 3000 ft

# 엄격 게이트 위에 얹는 히스테리시스 여유. 진입은 항상 엄격 게이트로 판정하고,
# 머무는 판정에만 완화 게이트를 쓴다. 기본값에서 |ata| <= 2.5 deg,
# 450 ft <= d <= 3200 ft.
WEZ_HOLD_ANGLE_MARGIN_DEG = 0.5
WEZ_HOLD_MIN_MARGIN_FT = 50.0
WEZ_HOLD_MAX_MARGIN_FT = 200.0

# 경계 허용오차. 152.4 m는 500.000016 ft라 정확히 500 ft인 거리가 하한 미만으로
# 읽히는 것을 막는다. 물리적으로는 무시할 수 있는 크기.
WEZ_RANGE_EPS_FT = 1e-3
WEZ_ANGLE_EPS_DEG = 1e-9

# 비에너지 변화량 정규화 폭 [m/step]. policy step 하나(step_ratio=6) 동안
# F-16이 낼 수 있는 비행경로 에너지 변화 규모.
ENERGY_DELTA_SPAN_M = 20.0

# 한 step에 이 이상 에너지가 튀면 에피소드 reset으로 보고 energy 항을 0으로 둔다.
ENERGY_RESET_JUMP_M = 1000.0

# 추락으로 간주할 end_condition 부분 문자열 (소문자 비교).
# 실제 문자열은 src/dogfight/envs/termination.py 와 single_agent_env.py 가 만든다:
#   "FDM Update Fail" / "ownship altitude below min" / "target altitude below min"
#   "ownship destroyed" / "target destroyed" / "fuel fail"
#   "two circle headon guard fail" / "max time out" / "episode step limit"
#   "Ownship FDM output Fall" / "Target FDM output Fall"
# Docs/03에 적힌 ownship_alt / max_time 같은 토큰은 실제 값이 아니다.
CRASH_END_CONDITION_KEYS = ("altitude below min", "fdm", "crash", "ground")

# "target ..." 으로 시작하는 종료는 적기 쪽 사건이다. 적기가 지면에 박은 것을
# 우리 추락으로 세면 승리 직전에 -150을 맞는다.
TARGET_SIDE_PREFIX = "target"

# GeoMathUtil의 sign=0 축퇴를 감지하는 임계값 (_repair_degenerate_angle 참조).
DEGENERATE_ANGLE_EPS_DEG = 1e-9


MY_REWARD_CONFIG = {
    "step_penalty": -0.005,
    # --- 각도 shaping (상쇄 방지를 위해 pursuit > position 유지) ---
    "pursuit_weight": 0.6,     # cos(ATA)
    "position_weight": 0.3,    # cos(AA)
    # --- WEZ 안전 밴드 ---
    "wez_entry_bonus": 3.0,
    # 에피소드 내 반복 진입은 가치가 줄어 나갔다 들어오기를 farming할 수 없다.
    "wez_entry_bonus_decay": 0.5,
    "wez_hold_reward": 0.05,
    "overclose_boundary_penalty": 1.0,
    "overclose_max_penalty": 6.0,
    # --- 에너지 (소가중치. 크게 주면 회피 정책으로 붕괴한다) ---
    "energy_weight": 0.02,
    # --- 종료 ---
    "crash_penalty": -150.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -10.0,
}


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _finite(value, fallback: float = 0.0) -> float:
    """Return float(value) if it is finite, otherwise fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError, IndexError):
        return fallback
    return out if math.isfinite(out) else fallback


def _state_value(state, index, fallback: float = 0.0) -> float:
    try:
        return _finite(state[index], fallback)
    except (TypeError, IndexError, KeyError):
        return fallback


def _geo_call(fn, *args) -> float:
    """geo_info 호출의 예외와 NaN을 흡수한다."""
    try:
        return _finite(fn(*args), math.nan)
    except Exception:
        return math.nan


def _nose_component(state, px: float, py: float, pz: float) -> float:
    """LOS 단위벡터를 기체 전방축(body x)에 투영한 성분.

    GeoMathUtil이 쓰는 회전행렬 T = Tx(roll)·Ty(pitch)·Tz(yaw)의 첫 행과 같다.
    roll은 x 성분에 기여하지 않는다.
    """
    pitch = math.radians(_state_value(state, StateIndex.PITCH, math.nan))
    yaw = math.radians(_state_value(state, StateIndex.YAW, math.nan))
    if not (math.isfinite(pitch) and math.isfinite(yaw)):
        return math.nan
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return cp * cy * px + cp * sy * py - sp * pz


def _unit_los(from_state, to_state) -> tuple[float, float, float]:
    """from -> to 방향 단위벡터 (NED). 계산 불가면 (nan, nan, nan)."""
    coords = []
    for state in (from_state, to_state):
        coords.append(
            (
                _state_value(state, StateIndex.N, math.nan),
                _state_value(state, StateIndex.E, math.nan),
                _state_value(state, StateIndex.D, math.nan),
            )
        )
    (fn, fe, fd), (tn, te, td) = coords
    if not all(math.isfinite(x) for x in (fn, fe, fd, tn, te, td)):
        return (math.nan, math.nan, math.nan)
    rx, ry, rz = tn - fn, te - fe, td - fd
    norm = math.sqrt(rx * rx + ry * ry + rz * rz)
    if not math.isfinite(norm) or norm < 1e-9:
        return (math.nan, math.nan, math.nan)
    return (rx / norm, ry / norm, rz / norm)


def _ata_magnitude_deg(ownship_state, target_state) -> float:
    """부호 없는 ATA 크기 [deg]. 계산 불가면 nan."""
    ux, uy, uz = _unit_los(ownship_state, target_state)
    if not math.isfinite(ux):
        return math.nan
    comp = _nose_component(ownship_state, ux, uy, uz)
    if not math.isfinite(comp):
        return math.nan
    return math.degrees(math.acos(_clip(comp, -1.0, 1.0)))


def _aa_magnitude_deg(ownship_state, target_state) -> float:
    """부호 없는 AA 크기 [deg]. 0 = 적의 6시, 180 = 헤드온. 계산 불가면 nan."""
    ux, uy, uz = _unit_los(target_state, ownship_state)
    if not math.isfinite(ux):
        return math.nan
    comp = _nose_component(target_state, ux, uy, uz)
    if not math.isfinite(comp):
        return math.nan
    return math.degrees(math.acos(_clip(-comp, -1.0, 1.0)))


def _repair_degenerate_angle(reported: float, magnitude_fn, *args) -> float:
    """GeoMathUtil의 sign=0 붕괴를 보정한다.

    GeoMathUtil._get_antenna_train_angle / _get_aspect_angle 은 3D 부호를
    `sign = np.sign(p_unit_t[2])` 로 정하는데, 상대가 정확히 같은 고도의
    정면/정후방에 있으면 `np.sign(0.0) == 0.0` 이라 각도가 통째로 0이 된다.
    즉 "적이 내 6시에 붙어 있음"(ATA 180)이 "정조준"(ATA 0)으로 보고되고,
    보정하지 않으면 pursuit 항이 -0.6이어야 할 자리에서 +0.6이 된다.

    initial_scenario.altitude_m 이 양쪽 공통이라 에피소드 reset 시점에 걸릴 수 있다.
    보고값이 0인데 실제 크기가 0이 아니면 실제 크기로 바꾼다. cos()만 쓰므로
    부호가 사라져도 보상 값은 정확하다.
    """
    if not math.isfinite(reported) or abs(reported) > DEGENERATE_ANGLE_EPS_DEG:
        return reported
    magnitude = magnitude_fn(*args)
    if math.isfinite(magnitude) and magnitude > DEGENERATE_ANGLE_EPS_DEG:
        return magnitude
    return reported


def _specific_energy_m(state) -> float:
    """비에너지 Es = h + V^2 / 2g [m]. 유한하지 않으면 nan.

    ALT는 meter, KCAS는 meter/sec이므로 단위 변환이 필요 없다 (FighterSim.py 확인).
    """
    alt_m = _state_value(state, StateIndex.ALT, math.nan)
    speed_ms = _state_value(state, StateIndex.KCAS, math.nan)
    if not (math.isfinite(alt_m) and math.isfinite(speed_ms)):
        return math.nan
    speed_ms = max(0.0, speed_ms)
    return alt_m + (speed_ms * speed_ms) / (2.0 * G)


def _is_crash(end_condition) -> bool:
    """end_condition이 **아군** 추락/FDM 계열인지 판정."""
    if not end_condition:
        return False
    text = str(end_condition).strip().lower()
    if text.startswith(TARGET_SIDE_PREFIX):
        return False
    return any(key in text for key in CRASH_END_CONDITION_KEYS)


class WezRewardState:
    """에피소드 단위 상태 (WEZ 점유 + 직전 에너지 우위).

    이 템플릿의 compute_reward()는 상태 없는 module 함수라서, 직전 step 값을
    보상 클래스 대신 여기에 둔다.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._was_in_wez = False
        self._entry_count = 0
        self._episode_done = False
        self._prev_energy_adv_m = math.nan

    @property
    def was_in_wez(self) -> bool:
        return self._was_in_wez


# env 인스턴스마다 상태 하나. geo_info는 env별 기하 helper라서 안정적인 키이고,
# weak map이라 env runner가 여러 개여도 누수가 없다. weakref가 불가능한
# geo_info면 공유 상태로 대체한다.
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
    """저장된 에피소드 상태를 지운다. 에피소드 reset에서 호출해도 안전."""
    if geo_info is None:
        _FALLBACK_WEZ_STATE.reset()
        _WEZ_STATES.clear()
        return
    _get_wez_state(geo_info).reset()


def _wez_thresholds_ft(wez_config: dict) -> tuple[float, float, float]:
    """Return (strict_half_angle_deg, strict_min_ft, strict_max_ft).

    half_angle = angle_deg / 2.0 — single_agent_env.update_damage()가 쓰는
    `half_wez_angle_deg = angle_deg / 2.0` 와 같은 기준이어야 보상이 실제
    대미지 판정과 어긋나지 않는다.
    """
    cfg = wez_config if isinstance(wez_config, dict) else {}
    angle_deg = abs(_finite(cfg.get("angle_deg", DEFAULT_WEZ_ANGLE_DEG), DEFAULT_WEZ_ANGLE_DEG)) / 2.0
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
    """(total, components)를 반환한다. 반환값은 항상 finite."""
    cfg = reward_config if isinstance(reward_config, dict) else {}

    components: dict[str, float] = {
        "step": _finite(cfg.get("step_penalty", -0.005), -0.005),
    }

    state = _get_wez_state(geo_info)
    if state._episode_done:
        # 직전 호출이 에피소드를 끝냈다 -> 이번 호출은 새 에피소드다.
        state.reset()

    # --- 가중치 (상쇄 방지: pursuit > position 을 강제) ---
    pursuit_w = abs(_finite(cfg.get("pursuit_weight", 0.6), 0.6))
    position_w = abs(_finite(cfg.get("position_weight", 0.3), 0.3))
    if position_w >= pursuit_w:
        position_w = pursuit_w * 0.5

    entry_bonus = _finite(cfg.get("wez_entry_bonus", 3.0), 3.0)
    entry_decay = _clip(_finite(cfg.get("wez_entry_bonus_decay", 0.5), 0.5), 0.0, 1.0)
    hold_reward = _finite(cfg.get("wez_hold_reward", 0.05), 0.05)
    boundary_penalty = abs(_finite(cfg.get("overclose_boundary_penalty", 1.0), 1.0))
    max_penalty = abs(_finite(cfg.get("overclose_max_penalty", 6.0), 6.0))
    max_penalty = max(max_penalty, boundary_penalty)
    energy_w = _finite(cfg.get("energy_weight", 0.02), 0.02)

    wez_angle_deg, wez_min_ft, wez_max_ft = _wez_thresholds_ft(wez_config)
    hold_angle_deg = wez_angle_deg + WEZ_HOLD_ANGLE_MARGIN_DEG
    hold_min_ft = max(0.0, wez_min_ft - WEZ_HOLD_MIN_MARGIN_FT)
    hold_max_ft = wez_max_ft + WEZ_HOLD_MAX_MARGIN_FT

    # --- 기하 (단위 변환은 정확히 여기서 한 번) ---
    distance_m = _geo_call(geo_info._get_distance, ownship_state, target_state)
    ata_deg = _geo_call(
        geo_info._get_antenna_train_angle, ownship_state, target_state, False
    )
    aa_deg = _geo_call(geo_info._get_aspect_angle, ownship_state, target_state, False)

    # GeoMathUtil의 sign=0 축퇴 보정 (같은 고도 정면/정후방)
    ata_deg = _repair_degenerate_angle(
        ata_deg, _ata_magnitude_deg, ownship_state, target_state
    )
    aa_deg = _repair_degenerate_angle(
        aa_deg, _aa_magnitude_deg, ownship_state, target_state
    )

    # --- 각도 shaping: pursuit(ATA) / position(AA) ---
    # ATA=0 -> 내 기수가 적을 향함 -> +. AA=0 -> 적의 6시 -> +. AA=180(헤드온) -> -.
    pursuit = pursuit_w * math.cos(ata_deg * DEG_TO_RAD) if math.isfinite(ata_deg) else 0.0
    position = position_w * math.cos(aa_deg * DEG_TO_RAD) if math.isfinite(aa_deg) else 0.0

    # --- WEZ 안전 밴드 ---
    wez_entry = 0.0
    wez_hold = 0.0
    overclose = 0.0

    if math.isfinite(distance_m) and math.isfinite(ata_deg):
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
        # 진입은 엄격 게이트, 유지만 완화 게이트.
        in_wez = relaxed_in_wez if state._was_in_wez else strict_in_wez
        too_close = distance_ft < wez_min_ft - WEZ_RANGE_EPS_FT

        if too_close:
            # 최소 사거리 아래로 파고들수록 연속적으로 커지는 패널티.
            severity = _clip((wez_min_ft - distance_ft) / max(wez_min_ft, 1e-6), 0.0, 1.0)
            overclose = -(
                boundary_penalty + (max_penalty - boundary_penalty) * severity ** 2
            )
        else:
            # 진입 보너스는 엄격 게이트 + 과근접이 아닐 때의 최초 진입에만.
            if in_wez and not state._was_in_wez:
                wez_entry = entry_bonus * (entry_decay ** state._entry_count)
                state._entry_count += 1
            elif in_wez:
                wez_hold = hold_reward

        state._was_in_wez = in_wez

    # --- 에너지 우위의 step 변화량 (소가중치) ---
    energy = 0.0
    own_es = _specific_energy_m(ownship_state)
    tgt_es = _specific_energy_m(target_state)
    if math.isfinite(own_es) and math.isfinite(tgt_es):
        energy_adv = own_es - tgt_es
        prev_adv = state._prev_energy_adv_m
        if math.isfinite(prev_adv):
            delta = energy_adv - prev_adv
            # reset 직후의 점프는 에너지 획득이 아니다.
            if abs(delta) <= ENERGY_RESET_JUMP_M:
                energy = energy_w * _clip(delta / ENERGY_DELTA_SPAN_M, -1.0, 1.0)
        state._prev_energy_adv_m = energy_adv
    else:
        state._prev_energy_adv_m = math.nan

    components["pursuit"] = pursuit
    components["position"] = position
    components["wez_entry"] = wez_entry
    components["wez_hold"] = wez_hold
    components["overclose"] = overclose
    components["energy"] = energy

    # --- 종료 ---
    crash = 0.0
    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = _finite(_state_value(ownship_state, StateIndex.HEALTH))
        target_health = _finite(_state_value(target_state, StateIndex.HEALTH))
        if target_health <= 0.0 < ownship_health:
            terminal_reward = float(reward_config.get("win_reward", 100.0))
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = float(reward_config.get("loss_reward", -100.0))
        else:
            terminal_reward = _finite(cfg.get("draw_reward", -10.0), -10.0)

        # 추락은 무승부보다도 확실히 나쁘게. terminal과 별도 컴포넌트로 기록해
        # crash_rate와 보상을 따로 읽을 수 있게 한다.
        if _is_crash(end_condition):
            crash = -abs(_finite(cfg.get("crash_penalty", -150.0), -150.0))

        # 에피소드 경계 표시 -> 다음 호출은 깨끗한 상태에서 시작한다.
        state._episode_done = True

    components["crash"] = crash
    components["terminal"] = terminal_reward

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
