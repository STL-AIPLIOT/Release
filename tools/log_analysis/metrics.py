# -*- coding: utf-8 -*-
"""Tacview 궤적에서 파생 지표를 계산한다.

Tacview CSV 에는 속도 컬럼이 없다(Time/Lon/Lat/Altitude/Roll/Pitch/Yaw/Health 뿐).
따라서 속도는 위치 차분으로 추정한다. 질량 정보가 없으므로 절대 에너지 대신
specific energy 근사를 쓴다:

    specific_energy = g * altitude + 0.5 * speed^2      [J/kg]

이 근사를 쓴다는 사실은 산출물에도 함께 기록한다.
"""
from __future__ import annotations

import math

G = 9.80665
EARTH_R = 6_378_137.0  # WGS84 장반경 [m]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이 수평거리 [m]."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def speed_series(time: list[float], lat: list[float], lon: list[float],
                 alt: list[float]) -> list[float]:
    """위치 차분으로 대지속도 [m/s] 를 추정한다.

    첫 샘플은 두 번째 값을 복사한다(차분 불가). 계산 불가 구간은 nan.
    """
    n = min(len(time), len(lat), len(lon), len(alt))
    if n < 2:
        return [math.nan] * n
    out = [math.nan] * n
    for i in range(1, n):
        dt = time[i] - time[i - 1]
        if dt <= 0:
            continue
        vals = (lat[i], lon[i], lat[i - 1], lon[i - 1], alt[i], alt[i - 1])
        if any(math.isnan(v) for v in vals):
            continue
        horiz = _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
        vert = alt[i] - alt[i - 1]
        out[i] = math.hypot(horiz, vert) / dt
    out[0] = out[1] if n >= 2 else math.nan
    return out


def specific_energy_series(alt: list[float], speed: list[float]) -> list[float]:
    """specific energy = g*h + 0.5*v^2 [J/kg]. 질량 미상이라 절대 에너지 대신 쓴다."""
    n = min(len(alt), len(speed))
    out: list[float] = []
    for i in range(n):
        if math.isnan(alt[i]) or math.isnan(speed[i]):
            out.append(math.nan)
        else:
            out.append(G * alt[i] + 0.5 * speed[i] * speed[i])
    return out


def descent_rate_series(time: list[float], alt: list[float]) -> list[float]:
    """하강률 [m/s]. 양수 = 하강 중. 계산 불가 구간은 nan."""
    n = min(len(time), len(alt))
    if n < 2:
        return [math.nan] * n
    out = [math.nan] * n
    for i in range(1, n):
        dt = time[i] - time[i - 1]
        if dt <= 0 or math.isnan(alt[i]) or math.isnan(alt[i - 1]):
            continue
        out[i] = -(alt[i] - alt[i - 1]) / dt
    out[0] = out[1]
    return out


def window_indices(time: list[float], window_sec: float,
                   anchor: str = "end") -> tuple[int, int]:
    """마지막(또는 처음) window_sec 구간의 [start, end) 인덱스.

    구간보다 짧은 경기는 전체를 돌려주고, 호출부가 그 사실을 알 수 있도록
    실제 길이를 함께 확인할 수 있게 한다.
    """
    if not time:
        return (0, 0)
    if anchor == "end":
        cutoff = time[-1] - window_sec
        start = 0
        for i, t in enumerate(time):
            if t >= cutoff:
                start = i
                break
        return (start, len(time))
    cutoff = time[0] + window_sec
    end = len(time)
    for i, t in enumerate(time):
        if t > cutoff:
            end = i
            break
    return (0, end)


def mean_ignoring_nan(values: list[float]) -> float | None:
    """nan 을 뺀 평균. 전부 nan 이면 None(0 으로 대체하지 않는다)."""
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None
