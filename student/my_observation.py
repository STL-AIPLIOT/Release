# -*- coding: utf-8 -*-
"""STIL custom observation (Gate 2 확정본).

W1 과제 "observation space 확정 + NaN 안전 가드"의 결과물.

관측 벡터 (8차원, 모두 [-1, 1] 정규화):

    0 own_alt          고도
    1 own_speed        속도 (KCAS)
    2 ata              Antenna Train Angle, 부호 유지
    3 aa               Aspect Angle, 부호 유지
    4 distance         상대와의 거리
    5 energy_advantage 비에너지 우위 (내 Es - 적 Es)
    6 closure_rate     접근률 (양수 = 접근 중)
    7 in_wez           WEZ 안이면 +1, 밖이면 -1

실행:
    python train_rllib.py --observation-mode custom \
        --observation-module student.my_observation

주의 (관측 일관성 체인, Docs/09 §2):
    이 module path는 YAML / train CLI / metadata.json / run_local_dogfight.py /
    my_submission.py 다섯 곳에 동일하게 적혀 있어야 한다. OBSERVATION_SIZE나
    feature 순서를 바꾸면 기존 checkpoint·bundle은 재사용할 수 없다.

NaN 안전 가드:
    build_observation()은 어떤 입력에서도 finite한 float32 벡터를 반환한다.
    상태 배열에 NaN/inf가 들어오거나 geo_info 호출이 실패해도 해당 feature는
    0.0으로 대체된다.

**이 함수는 상태를 갖지 않는다.** 같은 입력이면 몇 번을 불러도 같은 값이 나온다.
env가 build_observation을 step당 한 번만 부르지 않기 때문이다:
single_agent_env._build_action_context()가 action provider가 있을 때
step_ratio 횟수만큼 추가로 부르고, target provider 쪽은 ownship/target 인자를
**바꿔서** 부른다. 이전 step 값을 캐시해 closure를 만들면 학습(provider 없음)과
로컬 교전(provider 있음)에서 값이 달라진다 — 에러 없이 성능만 무너지는
전형적인 silent failure다. 그래서 closure는 캐시 대신 속도 벡터에서 직접 계산한다.

단위 (Release 본체 소스에서 확인, 추측 아님):
    FighterSim.py L170-172  state[0:3] N/E/D            -> meter
    FighterSim.py L174-176  state[3:6] roll/pitch/yaw   -> degree
    FighterSim.py L178-180  state[6:9] u/v/w body vel   -> meter/sec
    FighterSim.py L186      state[12] KCAS              -> **meter/sec** (knot 아님)
    FighterSim.py L244      state[44] ALT               -> **meter** (feet 아님)
    GeoMathUtil.py          _get_distance               -> meter
                            _get_antenna_train_angle    -> signed degree
                            _get_aspect_angle           -> signed degree (0=적의 6시)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dogfight.sim.state_schema import StateIndex


OBSERVATION_MODE = "stil8"
OBSERVATION_SIZE = 8
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0


# --- StateIndex에 없는 인덱스 (FighterSim.py 기준) ---------------------------
IDX_VEL_U = 6   # body x 속도 [m/s]
IDX_VEL_V = 7   # body y 속도 [m/s]
IDX_VEL_W = 8   # body z 속도 [m/s]


# --- 정규화 범위 -------------------------------------------------------------
# 고도 인코딩 (2026-08-05 변경: 선형 -> sqrt)
#
# 종전: _norm(alt, 0, 15000) 선형. 대회 초기 배치(약 610~914 m)와 추락선(300 m)이
# 전체 범위 2.0 중 **0.062 (3%)** 밖에 차지하지 못했다. 정책이 고도 변화를 사실상
# 보지 못했고, 실측에서 30/30 판이 마지막 5초에 1,449 m 를 잃고(≈290 m/s 수직)
# 추락했다 (analysis/bt30_after/판정_20260805.md §3).
#
# 변경: 추락선을 원점으로 잡고 제곱근으로 압축한다.
#     f(alt) = 2 * sqrt(clip((alt - ALT_FLOOR_M) / (ALT_TOP_M - ALT_FLOOR_M), 0, 1)) - 1
#
#     300 m(추락선) -1.000     762 m -0.645     914 m -0.591
#     2300 m        -0.262     7000 m +0.350   15000 m +1.000
#
# 왜 sqrt 인가 — 선형 300~2300 이 저고도 해상도는 더 좋지만(0.462) 2300 m 위가 전부
# 포화해 고고도를 구분하지 못한다. 대회 초기 고도는 "정확한 수치 별도 공지" 라 아직
# 확정이 아니므로, 저고도를 5.8배 키우면서 전 구간을 살려 두는 쪽을 택했다.
#
#     추락선~시작(300->762 m) 해상도:  선형 0~15000  0.062
#                                      sqrt         0.355  (5.8배)
#                                      선형 300~2300 0.462  (단 2300 m 위 포화)
ALT_FLOOR_M = 300.0     # env min_altitude. 여기서 -1.0
ALT_TOP_M = 15000.0     # 여기서 +1.0
# 하위 호환용 별칭. 외부에서 이 상수를 참조하는 코드가 있을 수 있다.
ALT_MAX_M = ALT_TOP_M
# KCAS는 m/s. 코너속도 부근이 ~180 m/s, 실전 범위 80~350 m/s라 400에서 포화시킨다.
KCAS_MAX_MS = 400.0
DISTANCE_MAX_M = 20000.0
ANGLE_MAX_DEG = 180.0

# 비에너지(specific energy) 차이의 정규화 폭 [m].
# Es = h + V^2/2g. 고도 3000 m 차이 또는 그에 준하는 속도 차이에서 포화한다.
ENERGY_SPAN_M = 3000.0

# 접근률 정규화 폭 [m/s]. 정면 교차에서 양쪽 300 m/s면 600 m/s까지 나오지만,
# 유의미한 구간을 넓게 쓰려고 400에서 포화시킨다.
CLOSURE_MAX_MS = 400.0

G = 9.80665

# WEZ 기본값 (env_config.wez가 없을 때만 사용). src/dogfight/config.py와 동일.
# angle_deg는 원뿔의 **전체** 각이다. single_agent_env.update_damage()가
# half_wez_angle_deg = angle_deg / 2.0 로 쓰므로 여기서도 반으로 나눠 쓴다.
DEFAULT_WEZ_ANGLE_DEG = 2.0
DEFAULT_WEZ_MIN_RANGE_M = 152.4   # 500 ft
DEFAULT_WEZ_MAX_RANGE_M = 914.4   # 3000 ft

# GeoMathUtil의 sign=0 축퇴를 감지하는 임계값 (아래 _repair_degenerate_angle 참조).
DEGENERATE_ANGLE_EPS_DEG = 1e-9


def _finite(value, fallback: float = 0.0) -> float:
    """float(value)가 유한하면 그 값, 아니면 fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError, IndexError):
        return fallback
    return out if math.isfinite(out) else fallback


def _state_value(state, index, fallback: float = 0.0) -> float:
    """상태 배열에서 안전하게 스칼라를 읽는다."""
    try:
        return _finite(state[index], fallback)
    except (TypeError, IndexError, KeyError):
        return fallback


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _norm(value: float, low: float, high: float) -> float:
    """[low, high]를 [-1, 1]로 선형 정규화하고 범위를 넘으면 자른다.

    dogfight.envs.observation.normalize 대신 쓰는 이유는 NaN 입력과
    low == high 인 축퇴 구간까지 여기서 함께 막기 위해서다.
    """
    span = high - low
    if not math.isfinite(span) or abs(span) < 1e-12:
        return 0.0
    scaled = (_finite(value) - low) / span * 2.0 - 1.0
    return _clip(scaled, -1.0, 1.0)


def _alt_norm(value: float) -> float:
    """고도를 [-1, 1]로 인코딩한다. 추락선 근처의 해상도를 우선한다.

    선형 정규화는 추락선(300 m)과 교전 시작 고도 사이를 전체 범위의 3%로만 표현해,
    정책이 "지면에 가까워지고 있다"를 볼 수 없었다. 제곱근은 낮은 쪽에 해상도를
    몰아주면서 상한까지의 단조성을 유지한다.

    무상태이고 NaN/inf 를 흡수한다(build_observation 계약, defect 3 참조).
    """
    span = ALT_TOP_M - ALT_FLOOR_M
    if not math.isfinite(span) or span <= 0.0:
        return 0.0
    ratio = _clip((_finite(value) - ALT_FLOOR_M) / span, 0.0, 1.0)
    return _clip(2.0 * math.sqrt(ratio) - 1.0, -1.0, 1.0)


def _symmetric_norm(value: float, half_span: float) -> float:
    """부호를 유지한 채 [-half_span, half_span]을 [-1, 1]로 정규화."""
    if not math.isfinite(half_span) or half_span <= 0.0:
        return 0.0
    return _clip(_finite(value) / half_span, -1.0, 1.0)


def _geo_call(fn, *args) -> float:
    """geo_info 메서드를 호출하되 예외·NaN을 nan으로 흡수한다."""
    try:
        return _finite(fn(*args), math.nan)
    except Exception:  # geo_info 구현 예외까지 학습을 죽이지 않는다
        return math.nan


def _nose_component(state, px: float, py: float, pz: float) -> float:
    """LOS 단위벡터를 기체 전방축(body x)에 투영한 성분.

    GeoMathUtil이 쓰는 회전행렬 T = Tx(roll)·Ty(pitch)·Tz(yaw)의 첫 행과 같다.
    roll은 x 성분에 기여하지 않으므로 pitch/yaw만 있으면 된다.
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
    fp = _position_ned(from_state)
    tp = _position_ned(to_state)
    if not all(math.isfinite(x) for x in fp + tp):
        return (math.nan, math.nan, math.nan)
    rx, ry, rz = tp[0] - fp[0], tp[1] - fp[1], tp[2] - fp[2]
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
    정면/정후방에 있으면 `np.sign(0.0) == 0.0` 이라 각도 자체가 0으로 뭉개진다.
    즉 "적이 내 6시에 정확히 붙어 있음"(ATA 180)이 "정조준"(ATA 0)으로 보고된다.
    고도차가 1e-9 m만 있어도 ±180으로 돌아온다.

    initial_scenario.altitude_m 이 양쪽 공통이라 **에피소드 reset 시점에 걸릴 수
    있다**. 보고값이 0인데 실제 크기가 0이 아니면 실제 크기로 바꾼다.
    부호는 이 축퇴 상황에서 원래 정의되지 않으므로 양수로 둔다.
    """
    if not math.isfinite(reported) or abs(reported) > DEGENERATE_ANGLE_EPS_DEG:
        return reported
    magnitude = magnitude_fn(*args)
    if math.isfinite(magnitude) and magnitude > DEGENERATE_ANGLE_EPS_DEG:
        return magnitude
    return reported


def _position_ned(state) -> tuple[float, float, float]:
    return (
        _state_value(state, StateIndex.N, math.nan),
        _state_value(state, StateIndex.E, math.nan),
        _state_value(state, StateIndex.D, math.nan),
    )


def _velocity_ned(state) -> tuple[float, float, float]:
    """body 속도(u, v, w)를 NED 속도로 변환한다. 값이 없으면 (nan, nan, nan).

    회전은 표준 body->NED DCM: R = Rz(yaw) Ry(pitch) Rx(roll).
    """
    u = _state_value(state, IDX_VEL_U, math.nan)
    v = _state_value(state, IDX_VEL_V, math.nan)
    w = _state_value(state, IDX_VEL_W, math.nan)
    roll = _state_value(state, StateIndex.ROLL, math.nan)
    pitch = _state_value(state, StateIndex.PITCH, math.nan)
    yaw = _state_value(state, StateIndex.YAW, math.nan)
    if not all(math.isfinite(x) for x in (u, v, w, roll, pitch, yaw)):
        return (math.nan, math.nan, math.nan)

    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))

    vn = u * cp * cy + v * (sr * sp * cy - cr * sy) + w * (cr * sp * cy + sr * sy)
    ve = u * cp * sy + v * (sr * sp * sy + cr * cy) + w * (cr * sp * sy - sr * cy)
    vd = -u * sp + v * sr * cp + w * cr * cp
    return (vn, ve, vd)


def _closure_rate_ms(ownship_state, target_state) -> float:
    """LOS 방향 접근률 [m/s]. 양수 = 가까워지는 중. 계산 불가면 nan.

    closure = -d|r|/dt = -(v_target - v_ownship) · r_hat,  r = p_target - p_ownship
    이전 step 값을 쓰지 않으므로 호출 횟수·인자 순서에 영향받지 않는다.
    """
    own_p = _position_ned(ownship_state)
    tgt_p = _position_ned(target_state)
    own_v = _velocity_ned(ownship_state)
    tgt_v = _velocity_ned(target_state)
    if not all(math.isfinite(x) for x in own_p + tgt_p + own_v + tgt_v):
        return math.nan

    rx, ry, rz = (tgt_p[0] - own_p[0], tgt_p[1] - own_p[1], tgt_p[2] - own_p[2])
    norm = math.sqrt(rx * rx + ry * ry + rz * rz)
    if not math.isfinite(norm) or norm < 1e-6:
        return 0.0  # 같은 위치 -> 접근률 정의 불가

    rel = (tgt_v[0] - own_v[0], tgt_v[1] - own_v[1], tgt_v[2] - own_v[2])
    range_rate = (rel[0] * rx + rel[1] * ry + rel[2] * rz) / norm
    return -range_rate


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


def _wez_thresholds(wez_config) -> tuple[float, float, float]:
    """(half_angle_deg, min_range_m, max_range_m). 값이 없으면 config.py 기본값.

    half_angle_deg = angle_deg / 2.0 — 실제 대미지 판정(update_damage)과 같은 기준.
    """
    cfg = wez_config if isinstance(wez_config, dict) else {}
    angle = abs(_finite(cfg.get("angle_deg", DEFAULT_WEZ_ANGLE_DEG), DEFAULT_WEZ_ANGLE_DEG))
    min_m = max(0.0, _finite(cfg.get("min_range_m", DEFAULT_WEZ_MIN_RANGE_M), DEFAULT_WEZ_MIN_RANGE_M))
    max_m = max(min_m, _finite(cfg.get("max_range_m", DEFAULT_WEZ_MAX_RANGE_M), DEFAULT_WEZ_MAX_RANGE_M))
    return angle / 2.0, min_m, max_m


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    """8차원 float32 관측 벡터를 반환한다. 반환값은 항상 finite."""
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # --- 기하 (한 번만 호출하고 이후 재사용) ---
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

    # --- 0, 1: 자기 상태 ---
    obs[0] = _alt_norm(_state_value(ownship_state, StateIndex.ALT))
    obs[1] = _norm(_state_value(ownship_state, StateIndex.KCAS), 0.0, KCAS_MAX_MS)

    # --- 2, 3: 각도 (부호 유지) ---
    obs[2] = _symmetric_norm(ata_deg, ANGLE_MAX_DEG) if math.isfinite(ata_deg) else 0.0
    obs[3] = _symmetric_norm(aa_deg, ANGLE_MAX_DEG) if math.isfinite(aa_deg) else 0.0

    # --- 4: 거리 ---
    distance_ok = math.isfinite(distance_m)
    obs[4] = _norm(max(0.0, distance_m), 0.0, DISTANCE_MAX_M) if distance_ok else 1.0

    # --- 5: 에너지 우위 ---
    own_es = _specific_energy_m(ownship_state)
    tgt_es = _specific_energy_m(target_state)
    if math.isfinite(own_es) and math.isfinite(tgt_es):
        obs[5] = _symmetric_norm(own_es - tgt_es, ENERGY_SPAN_M)
    else:
        obs[5] = 0.0

    # --- 6: 접근률 (상태 없이 속도 벡터에서 직접) ---
    closure_ms = _closure_rate_ms(ownship_state, target_state)
    obs[6] = _symmetric_norm(closure_ms, CLOSURE_MAX_MS) if math.isfinite(closure_ms) else 0.0

    # --- 7: WEZ 안 여부 (update_damage와 같은 기준: |ATA| <= angle_deg/2) ---
    wez_half_angle_deg, wez_min_m, wez_max_m = _wez_thresholds(wez_config)
    in_wez = (
        distance_ok
        and math.isfinite(ata_deg)
        and abs(ata_deg) <= wez_half_angle_deg
        and wez_min_m <= distance_m <= wez_max_m
    )
    obs[7] = 1.0 if in_wez else -1.0

    # 마지막 방어선: 위 경로 중 하나라도 NaN을 흘리면 여기서 잡는다.
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "own_alt_norm",
            "own_kcas_norm",
            "ata_norm",
            "aa_norm",
            "distance_norm",
            "energy_advantage_norm",
            "closure_rate_norm",
            "in_wez_flag",
        ],
        "description": (
            "STIL Gate 2 observation: altitude, speed, ATA, AA, distance, "
            "specific-energy advantage, LOS closure rate, WEZ occupancy flag. "
            "Stateless: identical inputs always produce identical output."
        ),
    }


__all__ = [
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "OBSERVATION_LOW",
    "OBSERVATION_HIGH",
    "build_observation",
    "describe_observation",
]
