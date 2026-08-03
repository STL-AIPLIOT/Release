# -*- coding: utf-8 -*-
"""각도 wrap-around 공통 함수.

C++ 쪽 BT_Content/AngleUtil.h 와 **같은 규약**을 쓴다. 두 구현이 갈라지면
BT 가 기록한 CSV 를 분석 도구가 다르게 해석하게 되므로, 규약을 여기 한 번만
적어 두고 양쪽 테스트가 같은 표를 검증한다.

단위
----
이 저장소의 각도는 전부 **degree** 다.

    FighterSim.py:174-176   state[3:6] roll/pitch/yaw  -> degree
    GeoMathUtil.py          ATA / AA / HCA             -> signed degree
    CPPBlackBoard.h         EulerAngle, *_Degree       -> degree

radian 로그를 다루게 될 경우를 위해 *_rad 대응 함수를 함께 둔다. 분석 보고서는
사람이 읽기 쉽도록 degree 로 환산해 쓰되, 원본 단위를 산출물에 함께 기록한다.

반환 범위
---------
    wrap_angle_deg / signed_angle_delta_deg  ->  [-180, 180)
    wrap_angle_rad / signed_angle_delta_rad  ->  [-pi, pi)

경계 정책
---------
정확히 180도(=pi) 차이는 좌/우 회전량이 같아 부호를 정할 수 없다.
반열린 구간이므로 **항상 -180 (-pi) 으로 접는다**.

    signed_angle_delta_deg(180, 0) == signed_angle_delta_deg(0, 180) == -180.0

입력 제약
---------
0~360 으로 정규화되어 있을 필요가 없다. 음수, 360 초과, 여러 바퀴 돈 값 모두
허용한다. NaN / inf 는 보정하지 않고 그대로 돌려준다(0 으로 위장하면 이상
프레임을 정상 프레임과 구분할 수 없게 된다).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = [
    "circular_mean_deg",
    "mean_signed_delta_deg",
    "signed_angle_delta_deg",
    "signed_angle_delta_rad",
    "signed_delta_series_deg",
    "wrap_angle_deg",
    "wrap_angle_rad",
]


def wrap_angle_deg(angle: float) -> float:
    """각도를 [-180, 180) 으로 접는다. NaN/inf 는 그대로 통과시킨다."""
    if not math.isfinite(angle):
        return angle
    return (angle + 180.0) % 360.0 - 180.0


def signed_angle_delta_deg(current: float, previous: float) -> float:
    """두 각도(degree) 사이의 최소 부호 회전량. 범위 [-180, 180).

    양수 = current 가 previous 보다 각도가 증가한 방향.
    """
    if not (math.isfinite(current) and math.isfinite(previous)):
        return math.nan
    return wrap_angle_deg(current - previous)


def wrap_angle_rad(angle: float) -> float:
    """각도를 [-pi, pi) 로 접는다. NaN/inf 는 그대로 통과시킨다."""
    if not math.isfinite(angle):
        return angle
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def signed_angle_delta_rad(current: float, previous: float) -> float:
    """두 각도(radian) 사이의 최소 부호 회전량. 범위 [-pi, pi)."""
    if not (math.isfinite(current) and math.isfinite(previous)):
        return math.nan
    return wrap_angle_rad(current - previous)


def signed_delta_series_deg(angles: Sequence[float]) -> list[float]:
    """연속된 각도열의 인접 차이(wrap 보정) 리스트. 길이는 len(angles)-1.

    첫 샘플은 이전 값이 없어 차이를 만들 수 없다. 0 으로 채우지 않고 아예
    원소를 만들지 않는다(호출부가 개수 차이를 인지하도록).
    """
    return [signed_angle_delta_deg(angles[i], angles[i - 1]) for i in range(1, len(angles))]


def mean_signed_delta_deg(deltas: Iterable[float]) -> float | None:
    """이미 wrap 보정된 차이들의 산술 평균. 유효값이 없으면 None.

    **circular mean 을 쓰지 않는다.** 각도 '차이'는 이미 [-180,180) 안의 작은
    부호값이라 산술 평균이 회전 방향을 그대로 보존한다. circular mean 은 여기서
    부호 정보를 뭉개므로 오히려 틀리다.
    (heading 처럼 '각도 자체'를 평균낼 때만 circular_mean_deg 를 쓴다.)
    """
    clean = [d for d in deltas if math.isfinite(d)]
    return sum(clean) / len(clean) if clean else None


def circular_mean_deg(angles: Iterable[float]) -> float | None:
    """각도 '자체'의 원형 평균 [-180, 180). 유효값이 없으면 None.

    heading 같이 주기성이 있는 값을 평균낼 때 쓴다. 차이(delta) 평균에는
    쓰지 말 것 — mean_signed_delta_deg 참조.
    """
    xs, ys = 0.0, 0.0
    n = 0
    for a in angles:
        if not math.isfinite(a):
            continue
        r = math.radians(a)
        xs += math.cos(r)
        ys += math.sin(r)
        n += 1
    if n == 0 or (xs == 0.0 and ys == 0.0):
        return None
    return wrap_angle_deg(math.degrees(math.atan2(ys, xs)))
