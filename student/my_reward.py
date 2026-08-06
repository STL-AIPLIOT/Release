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
    pursuit   +w_pursuit  * cos(ATA) * f(거리)  "적을 향해 기수를 두라"
    position  +w_position * cos(AA)  * f(거리)  "적의 6시로 가라"
    closure   거리 변화량 * w         "멀어지지 마라" (위 f(거리)가 0인 구간의 유일한 기울기)
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
# 한 step 만에 이만큼 거리가 변하면 기동이 아니라 에피소드 reset 이다.
# 실측 최대 접근 속도가 ~250 m/s, step_ratio=6 이면 1 step 이 0.1초라 20~30 m 다.
# 2000 m 는 거기서 두 자릿수 여유가 있으면서 개막 배치 점프(수 km)는 확실히 거른다.
CLOSURE_RESET_JUMP_M = 2000.0

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
    # --- 각도 shaping 의 거리 게이트 (2026-08-04 추가) ---
    # v3까지 pursuit/position 은 거리와 무관하게 지급됐다. 그래서 8 km 밖에서
    # 기수만 적 쪽으로 두면 WEZ 안에 들어간 것과 비슷한 보상이 계속 나왔다.
    #
    # BT 상대 10판(22,921 프레임) 실측:
    #     8 km 초과   53.2% 프레임, |ATA| 중앙 6.3도  -> pursuit +0.520/step
    #     4~8 km      33.9% 프레임, |ATA| 중앙 7.9도  -> pursuit +0.513/step
    #     WEZ 내       0.1% 프레임
    #   pursuit 누적 11,715 중 10,320(88%)이 4 km 밖에서 나왔다.
    #
    # 즉 "붙지 않고 정조준만 유지"가 가장 이득인 구조였다. 교전이 성립하지 않고
    # 거리가 중앙값 8.3 km 로 유지된 직접 원인이다.
    #
    # full_range 안에서는 1.0, zero_range 밖에서는 0.0, 그 사이는 선형으로 준다.
    # 하드 컷오프가 아니라 선형인 이유: 컷오프는 밖에서 기울기가 0이라 붙을 이유를
    # 만들지 못한다. 선형이면 가까워질수록 계수가 커져 접근 자체가 이득이 된다.
    #
    # full_range 기본값은 WEZ 최대 사거리(914.4 m = 3000 ft)다. 그 안에서만
    # 각도가 실제 피해로 이어지므로 감쇠 없이 전액 지급한다.
    # zero_range 4000 m 는 BFM 진입 게이트(SCISSORS 800 m / DBFM 1500 m)보다
    # 넉넉히 잡아, 붙는 도중에도 신호가 끊기지 않게 한 값이다.
    "angle_full_range_m": 914.4,   # 이 거리 이내면 계수 1.0 (WEZ max_range_m)
    "angle_zero_range_m": 4000.0,  # 이 거리 이상이면 계수 0.0
    # --- 접근 보상 (거리 게이트와 반드시 짝으로 쓴다, 2026-08-04 추가) ---
    # 위 게이트만 넣으면 zero_range 밖이 **무신호 구간**이 된다. 실측 중앙 교전거리가
    # 8 km 인데 4 km 밖에서 pursuit/position 이 0 이면, 거기서는 step_penalty 말고
    # 아무 기울기가 없어서 "붙을 이유"가 사라진다. 이탈을 막으려면 이탈 자체에
    # 값을 매겨야 한다.
    #
    # closure = w * clip((직전거리 - 현재거리) / span, -1, +1)
    #
    # 거리 **변화량**에 비례하므로 에피소드 전체 합은 (시작거리 - 끝거리) 에만
    # 의존한다. 나갔다 들어오기를 반복해도 합이 0 이라 farming 이 원리적으로
    # 불가능하다. "가까이 있으면 매 step 지급" 같은 형태를 쓰지 않은 이유다.
    # 멀어지면 그대로 음수라 이탈에 값이 매겨진다.
    #
    # 크기: span 1000 m / w 0.5 면 11 km -> 914 m 완주 시 누적 약 +5.0 이다.
    # wez_entry_bonus(3.0)와 같은 자릿수로, 접근이 진입 보너스를 압도하지 않는다.
    #
    # angle_full_range_m 안쪽에서는 끈다. 그 안은 pursuit 와 overclose 가 맡는
    # 구간이고, 켜두면 최소 사거리 아래로 파고드는 것까지 보상하게 되어
    # overclose 패널티와 정면으로 싸운다.
    "closure_weight": 0.5,
    "closure_span_m": 1000.0,
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
    # --- 고도 안전 마진 (v3에서 추가) ---
    # v2까지 추락에 대한 신호는 crash_penalty 하나뿐이었다. 그건 에피소드가 끝난 뒤
    # 한 번 주는 값이라, 7000 m에서 300 m까지 내려가는 400 step 내내 정책은
    # "낮아지고 있다"는 정보를 받지 못했다. 400 iteration 실측에서 92개 에피소드 중
    # 82개가 추락했고(crash_rate 0.891), 살아남은 10개는 pitch 평균이 -0.17로
    # 추락 그룹(-0.036)보다 5배 컸다. 즉 기수를 드는 방향이 정답인데 그쪽으로
    # 밀어주는 dense 신호가 없었다.
    #
    # altitude_safe_band_m 아래로 내려가면 매 step 패널티가 들어오고,
    # min_altitude 에 가까워질수록 제곱으로 커진다.
    "altitude_weight": 0.4,           # pursuit_weight(0.6)와 같은 자릿수. 크면 교전을 안 한다.
    "altitude_safe_band_m": 2000.0,   # 이 높이 아래부터 신호가 들어온다
    "min_altitude_m": 300.0,          # env min_altitude 기본값 (config.py)
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
        self._prev_distance_m = math.nan

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

    # --- 각도 shaping 의 거리 계수 ---
    # full_range 이내 1.0, zero_range 이상 0.0, 그 사이 선형.
    # 거리를 못 읽으면 1.0 으로 둔다(계수를 만들어내 신호를 지우지 않는다).
    full_range_m = abs(_finite(cfg.get("angle_full_range_m", 914.4), 914.4))
    zero_range_m = abs(_finite(cfg.get("angle_zero_range_m", 4000.0), 4000.0))
    if math.isfinite(distance_m):
        if distance_m <= full_range_m:
            range_factor = 1.0
        elif zero_range_m <= full_range_m or distance_m >= zero_range_m:
            # zero_range 를 full_range 이하로 잘못 설정하면 하드 컷오프로 동작한다.
            range_factor = 0.0
        else:
            range_factor = (zero_range_m - distance_m) / (zero_range_m - full_range_m)
        range_factor = _clip(range_factor, 0.0, 1.0)
    else:
        range_factor = 1.0

    # --- 각도 shaping: pursuit(ATA) / position(AA) ---
    # ATA=0 -> 내 기수가 적을 향함 -> +. AA=0 -> 적의 6시 -> +. AA=180(헤드온) -> -.
    # 거리 계수를 곱하므로, 멀리서 기수만 맞추는 것으로는 보상이 나오지 않는다.
    # 두 항에 같은 계수를 쓰므로 아래 상쇄 가드(pursuit > position)의 부호 관계는 그대로다.
    pursuit = (pursuit_w * math.cos(ata_deg * DEG_TO_RAD) * range_factor
               if math.isfinite(ata_deg) else 0.0)
    position = (position_w * math.cos(aa_deg * DEG_TO_RAD) * range_factor
                if math.isfinite(aa_deg) else 0.0)

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

    # --- 접근(이탈) 보상 ---
    # 각도 게이트가 죽어 있는 구간(full_range 밖)에서 유일하게 남는 기울기다.
    # 거리 변화량에 비례하므로 왕복해도 합이 0 이라 farming 이 안 된다.
    closure = 0.0
    closure_w = abs(_finite(cfg.get("closure_weight", 0.5), 0.5))
    closure_span_m = abs(_finite(cfg.get("closure_span_m", 1000.0), 1000.0))
    if math.isfinite(distance_m):
        prev_d = state._prev_distance_m
        if (closure_w > 0.0 and closure_span_m > 0.0
                and math.isfinite(prev_d) and distance_m > full_range_m):
            delta = prev_d - distance_m          # 가까워지면 양수
            # reset 직후의 점프는 기동의 결과가 아니다(에너지 항과 같은 처리).
            if abs(delta) <= CLOSURE_RESET_JUMP_M:
                closure = closure_w * _clip(delta / closure_span_m, -1.0, 1.0)
        state._prev_distance_m = distance_m
    else:
        state._prev_distance_m = math.nan

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

    # --- 고도 안전 마진 (dense) ---
    # margin = (alt - min_alt) / band 를 [0, 1]로 자르고, (1 - margin)^2 에 비례해
    # 패널티를 준다. band 위에서는 정확히 0이라 정상 교전 고도에서는 아무 영향이 없다.
    # 고도를 못 읽으면 0으로 둔다(패널티를 만들어내지 않는다).
    altitude = 0.0
    altitude_w = abs(_finite(cfg.get("altitude_weight", 0.4), 0.4))
    band_m = _finite(cfg.get("altitude_safe_band_m", 2000.0), 2000.0)
    min_alt_m = _finite(cfg.get("min_altitude_m", 300.0), 300.0)
    alt_m = _state_value(ownship_state, StateIndex.ALT, math.nan)
    if altitude_w > 0.0 and band_m > 0.0 and math.isfinite(alt_m):
        margin = _clip((alt_m - min_alt_m) / band_m, 0.0, 1.0)
        shortfall = 1.0 - margin
        altitude = -altitude_w * shortfall * shortfall

    components["pursuit"] = pursuit
    components["position"] = position
    components["wez_entry"] = wez_entry
    components["wez_hold"] = wez_hold
    components["overclose"] = overclose
    components["closure"] = closure
    components["energy"] = energy
    components["altitude"] = altitude

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
