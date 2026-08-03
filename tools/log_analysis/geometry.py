# -*- coding: utf-8 -*-
"""Tacview 궤적 두 개에서 ATA / AA / WEZ 를 파생시킨다.

왜 파생이 필요한가
------------------
2026-08-04 조사 기준, 저장된 어떤 로그에도 ATA / AA / WEZ 컬럼이 없다.

    Tacview CSV        Time, Longitude, Latitude, Altitude, Roll, Pitch, Yaw, Health
    <ts>_summary.json  end_condition, outcome, ownship_health, target_health
    replay_index.jsonl schemas.REPLAY_INDEX_KEYS
    training_log.csv   iteration 단위 집계 (프레임 단위 각도 없음)

그래서 두 기체의 위치·자세에서 다시 계산한다. 여기서 나온 값은 전부
**derived** 이며 원본 필드가 아니다. 산출물에서는 `derived_` 접두사를 붙인다.

계산은 GeoMathUtil.GeometryInfo 와 **같은 식**을 쓴다(호스트 트리에서 확인).

    _get_antenna_train_angle : GeoMathUtil.py:107-139
    _get_aspect_angle        : GeoMathUtil.py:40-74
    _get_distance            : GeoMathUtil.py:21-25

플랫폼 결함 1(각도 0 붕괴)도 그대로 재현한다. sign 이 np.sign(p_unit_t[2]) 인
구간에서 값이 0 이 되면 실제로 학습/판정이 본 값이 그것이기 때문이다.
붕괴가 일어난 프레임은 `sign_degenerate` 로 표시해 두어, 보고서에서
"0 이 진짜 nose-on 인지 결함인지"를 구분할 수 있게 한다.

좌표 변환
---------
Tacview 는 위경도/고도(WGS84)를, GeoMathUtil 은 NED(meter)를 쓴다.
첫 ownship 샘플을 원점으로 하는 국소 평면 근사로 변환한다.

    N = (lat - lat0) * R
    E = (lon - lon0) * R * cos(lat0)
    D = -alt

교전 반경이 수 km 이므로 이 근사의 오차는 각도 계산에서 무시할 수준이다.
근사를 썼다는 사실은 산출물 메타데이터에 남긴다.

단위: 각도 degree, 거리 meter, 시간 second.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import Track

EARTH_R = 6_378_137.0  # WGS84 장반경 [m]

# config.py:36-39 의 기본값 (500 ft / 3000 ft). CLI 로 덮어쓸 수 있어야 한다.
DEFAULT_WEZ_ANGLE_DEG = 2.0
DEFAULT_WEZ_MIN_M = 152.4
DEFAULT_WEZ_MAX_M = 914.4


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def _body_transform(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[list[float]]:
    """GeoMathUtil 의 Tx*Ty*Tz (NED -> body). 부호 규약까지 동일하다."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    tx = [[1, 0, 0], [0, math.cos(r), math.sin(r)], [0, -math.sin(r), math.cos(r)]]
    ty = [[math.cos(p), 0, -math.sin(p)], [0, 1, 0], [math.sin(p), 0, math.cos(p)]]
    tz = [[math.cos(y), math.sin(y), 0], [-math.sin(y), math.cos(y), 0], [0, 0, 1]]
    return _matmul(tx, _matmul(ty, tz))


# Tz_pi: z 축 180도 회전. _get_aspect_angle 이 앞에 곱한다.
_TZ_PI = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]


def _signed_from_body(p_unit_t: list[float]) -> tuple[float, bool]:
    """GeoMathUtil 의 부호 규칙을 그대로 옮긴다.

    반환: (부호 있는 각도[deg], 부호가 np.sign(z) 분기에서 나왔는지)

    두 번째 값이 True 이고 각도가 0 이면 플랫폼 결함 1 이 발동한 프레임이다
    (np.sign(0.0) == 0.0 이라 각도 전체가 0 으로 붕괴한다).
    """
    base = math.degrees(math.acos(max(-1.0, min(1.0, p_unit_t[0]))))
    sign = 1.0
    degenerate = False
    if p_unit_t[1] < -0.10:
        sign = -1.0
    elif -0.01 < p_unit_t[1] < 0.01:
        degenerate = True
        z = p_unit_t[2]
        sign = 0.0 if z == 0.0 else (1.0 if z > 0 else -1.0)
    return sign * base, degenerate


@dataclass
class GeoSample:
    """한 시점의 파생 기하값. 계산 불가한 값은 None 이다."""

    time_sec: float
    distance_m: float | None
    own_ata_deg: float | None
    target_ata_deg: float | None
    target_aa_deg: float | None
    own_in_wez: bool | None
    target_in_wez: bool | None
    ata_sign_degenerate: bool = False
    aa_sign_degenerate: bool = False


def lla_to_ned(lat_deg: float, lon_deg: float, alt_m: float,
               lat0_deg: float, lon0_deg: float) -> tuple[float, float, float]:
    """국소 평면 근사로 위경도/고도를 NED [m] 로 바꾼다."""
    n = math.radians(lat_deg - lat0_deg) * EARTH_R
    e = math.radians(lon_deg - lon0_deg) * EARTH_R * math.cos(math.radians(lat0_deg))
    d = -alt_m
    return n, e, d


def _state(track: Track, i: int, lat0: float, lon0: float) -> list[float] | None:
    """GeoMathUtil 이 기대하는 [N, E, D, roll, pitch, yaw] 를 만든다."""
    try:
        vals = (track.lat[i], track.lon[i], track.alt[i],
                track.roll[i], track.pitch[i], track.yaw[i])
    except IndexError:
        return None
    if any(v is None or math.isnan(v) for v in vals):
        return None
    n, e, d = lla_to_ned(vals[0], vals[1], vals[2], lat0, lon0)
    return [n, e, d, vals[3], vals[4], vals[5]]


def antenna_train_angle_deg(own: list[float], tgt: list[float]) -> tuple[float, bool]:
    """부호 있는 ATA [deg]. 0 = 정조준. GeoMathUtil._get_antenna_train_angle 과 동일."""
    p = [tgt[0] - own[0], tgt[1] - own[1], tgt[2] - own[2]]
    norm = math.sqrt(sum(c * c for c in p))
    p_unit = [c / norm for c in p] if norm != 0 else p
    t = _body_transform(own[3], own[4], own[5])
    return _signed_from_body(_matvec(t, p_unit))


def aspect_angle_deg(own: list[float], tgt: list[float]) -> tuple[float, bool]:
    """부호 있는 AA [deg]. **0 = 표적의 6시.** GeoMathUtil._get_aspect_angle 과 동일.

    주의: BT 의 BB->MyAspectAngle_Degree 는 반대 규약(0 = 표적 기수 앞)이고
    부호도 없다. 두 값을 섞지 말 것.
    """
    p = [own[0] - tgt[0], own[1] - tgt[1], own[2] - tgt[2]]
    norm = math.sqrt(sum(c * c for c in p))
    p_unit = [c / norm for c in p] if norm != 0 else p
    t = _matmul(_TZ_PI, _body_transform(tgt[3], tgt[4], tgt[5]))
    return _signed_from_body(_matvec(t, p_unit))


def in_wez(distance_m: float, ata_deg: float,
           wez_angle_deg: float = DEFAULT_WEZ_ANGLE_DEG,
           wez_min_m: float = DEFAULT_WEZ_MIN_M,
           wez_max_m: float = DEFAULT_WEZ_MAX_M) -> bool:
    """update_damage 와 **비트 단위로 같은** WEZ 판정.

    src/dogfight/envs/single_agent_env.py:572, 577-579

        min_range_m <= dis_m <= max_range_m
        angle_deg / 2.0 >= abs(ata_deg)

    각도 게이트는 angle_deg 가 아니라 **angle_deg/2** 다(플랫폼 결함 2).
    epsilon 을 넣지 않는다 — 넣으면 실제로는 피해가 없는 프레임이 WEZ 로 잡힌다.
    """
    if not (wez_min_m <= distance_m <= wez_max_m):
        return False
    return (wez_angle_deg / 2.0) >= abs(ata_deg)


def derive_series(own: Track, tgt: Track,
                  wez_angle_deg: float = DEFAULT_WEZ_ANGLE_DEG,
                  wez_min_m: float = DEFAULT_WEZ_MIN_M,
                  wez_max_m: float = DEFAULT_WEZ_MAX_M) -> list[GeoSample]:
    """두 궤적을 시간 인덱스로 짝지어 파생 기하 시계열을 만든다.

    두 CSV 는 같은 스텝에서 기록되므로 인덱스로 짝짓는다. 길이가 다르면 짧은 쪽에
    맞추고, 그 사실은 호출부가 len 으로 알 수 있다. 값을 만들어 채우지 않는다.
    """
    n = min(len(own), len(tgt))
    if n == 0 or not own.lat or not tgt.lat:
        return []

    lat0, lon0 = own.lat[0], own.lon[0]
    if lat0 is None or math.isnan(lat0) or lon0 is None or math.isnan(lon0):
        return []

    out: list[GeoSample] = []
    for i in range(n):
        os_ = _state(own, i, lat0, lon0)
        ts_ = _state(tgt, i, lat0, lon0)
        if os_ is None or ts_ is None:
            out.append(GeoSample(time_sec=own.time[i], distance_m=None,
                                 own_ata_deg=None, target_ata_deg=None,
                                 target_aa_deg=None, own_in_wez=None,
                                 target_in_wez=None))
            continue

        dist = math.sqrt(sum((ts_[k] - os_[k]) ** 2 for k in range(3)))
        own_ata, ata_degen = antenna_train_angle_deg(os_, ts_)
        tgt_ata, _ = antenna_train_angle_deg(ts_, os_)
        tgt_aa, aa_degen = aspect_angle_deg(os_, ts_)

        out.append(GeoSample(
            time_sec=own.time[i],
            distance_m=dist,
            own_ata_deg=own_ata,
            target_ata_deg=tgt_ata,
            target_aa_deg=tgt_aa,
            own_in_wez=in_wez(dist, own_ata, wez_angle_deg, wez_min_m, wez_max_m),
            target_in_wez=in_wez(dist, tgt_ata, wez_angle_deg, wez_min_m, wez_max_m),
            ata_sign_degenerate=ata_degen,
            aa_sign_degenerate=aa_degen,
        ))
    return out


def wez_intervals(samples: list[GeoSample], side: str = "own") -> list[tuple[float, float | None]]:
    """WEZ 활성 구간 [(진입시각, 이탈시각)] 목록.

    마지막까지 WEZ 안이면 이탈 시각을 None 으로 둔다(0 이나 마지막 시각으로
    위장하지 않는다).
    """
    key = "own_in_wez" if side == "own" else "target_in_wez"
    out: list[tuple[float, float | None]] = []
    start: float | None = None
    for s in samples:
        active = getattr(s, key)
        if active and start is None:
            start = s.time_sec
        elif not active and start is not None:
            out.append((start, s.time_sec))
            start = None
    if start is not None:
        out.append((start, None))
    return out
