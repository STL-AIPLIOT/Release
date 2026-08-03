# -*- coding: utf-8 -*-
"""패배 직전 구간에서 이벤트를 뽑아 시퀀스로 만든다.

현재 로그로 계산 가능한 축과 불가능한 축을 명확히 나눈다.

  계산 가능 : 에너지 역전, 저고도 진입, 급하강, 속도 손실, 고도체크 실패
  계산 불가 : BFM 전환 실패 계열 — Tacview CSV / summary.json / training_log.csv
              어디에도 BFM 모드가 없다(2026-08-04 조사). BFM 로깅이 추가되면
              detect_bfm_events() 가 활성화된다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import descent_rate_series, specific_energy_series, speed_series
from .schemas import Track

# 이벤트 코드
ENERGY_REVERSAL = "ENERGY_REVERSAL"
ENERGY_DEFICIT = "ENERGY_DEFICIT"
SPEED_LOSS = "SPEED_LOSS"
LOW_ALTITUDE = "LOW_ALTITUDE"
HIGH_DESCENT = "HIGH_DESCENT"
ALTITUDE_CHECK_FAILED = "ALTITUDE_CHECK_FAILED"
BFM_STUCK = "BFM_STUCK"
BFM_THRASH = "BFM_THRASH"


@dataclass
class Event:
    """구간 안에서 관측된 사건 하나."""

    code: str
    time: float
    detail: str = ""


@dataclass
class Thresholds:
    """판정 임계값. 전부 CLI 로 조정 가능해야 한다(코드 고정 금지)."""

    min_altitude_m: float = 300.0
    low_altitude_margin_m: float = 700.0   # min_altitude + margin 이하이면 저고도
    descent_rate_ms: float = 40.0          # 이 값 이상이면 급하강
    speed_loss_ms: float = 30.0            # 구간 내 속도 감소량이 이 값 이상이면 손실
    bfm_stuck_sec: float = 10.0
    bfm_thrash_count: int = 4
    # Tacview Time 을 실제 경과 시간으로 바꾸는 배율.
    # 호스트는 env step 1회마다 1행을 쓰면서 Time 은 내부 스텝 1개분만 올린다
    # (log_analysis/metrics.py 설명 참조). 학습 env_config.step_ratio 와 같게 준다.
    time_scale: float = 1.0


def detect_energy_events(time: list[float], own: Track, tgt: Track,
                         lo: int, hi: int, th: Thresholds) -> list[Event]:
    """에너지 역전 / 열세 / 속도 손실. specific energy 근사를 쓴다."""
    events: list[Event] = []
    own_speed = speed_series(own.time, own.lat, own.lon, own.alt, th.time_scale)
    own_se = specific_energy_series(own.alt, own_speed)

    tgt_se: list[float] = []
    if len(tgt) >= 2:
        tgt_speed = speed_series(tgt.time, tgt.lat, tgt.lon, tgt.alt, th.time_scale)
        tgt_se = specific_energy_series(tgt.alt, tgt_speed)

    # 에너지 우위 부호 변화 (양 -> 음)
    if tgt_se:
        n = min(len(own_se), len(tgt_se), len(time))
        prev_sign: int | None = None
        for i in range(max(0, lo), min(hi, n)):
            if math.isnan(own_se[i]) or math.isnan(tgt_se[i]):
                continue
            diff = own_se[i] - tgt_se[i]
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if prev_sign is not None and prev_sign > 0 and sign < 0:
                events.append(Event(ENERGY_REVERSAL, time[i],
                                    f"SE차 {diff:+.0f} J/kg 로 우위 상실"))
            prev_sign = sign
        # 구간 내내 열세면 별도 이벤트
        window = [own_se[i] - tgt_se[i] for i in range(max(0, lo), min(hi, n))
                  if not (math.isnan(own_se[i]) or math.isnan(tgt_se[i]))]
        if window and all(d < 0 for d in window):
            events.append(Event(ENERGY_DEFICIT, time[min(hi, n) - 1],
                                f"구간 내내 SE 열세 (평균 {sum(window)/len(window):+.0f})"))

    # 속도 손실
    seg = [own_speed[i] for i in range(max(0, lo), min(hi, len(own_speed)))
           if not math.isnan(own_speed[i])]
    if len(seg) >= 2 and (seg[0] - seg[-1]) >= th.speed_loss_ms:
        events.append(Event(SPEED_LOSS, time[min(hi, len(time)) - 1],
                            f"{seg[0]:.0f} -> {seg[-1]:.0f} m/s"))
    return events


def detect_altitude_events(time: list[float], own: Track,
                           lo: int, hi: int, th: Thresholds) -> list[Event]:
    """저고도 진입 / 급하강."""
    events: list[Event] = []
    descent = descent_rate_series(own.time, own.alt, th.time_scale)
    low_line = th.min_altitude_m + th.low_altitude_margin_m

    entered_low = False
    flagged_descent = False
    for i in range(max(0, lo), min(hi, len(own.alt), len(time))):
        alt = own.alt[i]
        if not math.isnan(alt) and alt <= low_line and not entered_low:
            entered_low = True
            events.append(Event(LOW_ALTITUDE, time[i], f"고도 {alt:.0f} m (<= {low_line:.0f})"))
        if i < len(descent) and not math.isnan(descent[i]) \
                and descent[i] >= th.descent_rate_ms and not flagged_descent:
            flagged_descent = True
            events.append(Event(HIGH_DESCENT, time[i], f"하강률 {descent[i]:.0f} m/s"))
    return events


def detect_bfm_events(time: list[float], own: Track,
                      lo: int, hi: int, th: Thresholds) -> tuple[list[Event], str | None]:
    """BFM 고착 / 잦은 전환.

    반환값의 두 번째 원소는 계산 불가 사유다. None 이면 계산됐다는 뜻이다.
    """
    if not own.has_bfm:
        return [], "BFM 모드가 로그에 없다 (Tacview CSV 에 BFM 컬럼 없음)"

    seq = own.bfm[max(0, lo):hi]
    stamps = time[max(0, lo):hi]
    if not seq:
        return [], "구간에 샘플이 없다"

    events: list[Event] = []
    transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    if transitions >= th.bfm_thrash_count:
        events.append(Event(BFM_THRASH, stamps[-1], f"구간 내 전환 {transitions}회"))

    run_start, longest, longest_mode = 0, 0.0, seq[0]
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[run_start]:
            span = stamps[i - 1] - stamps[run_start]
            if span > longest:
                longest, longest_mode = span, seq[run_start]
            run_start = i
    if longest >= th.bfm_stuck_sec:
        events.append(Event(BFM_STUCK, stamps[-1], f"{longest_mode} 연속 {longest:.1f}s"))
    return events, None


def build_sequence(events: list[Event], end_condition_type: str) -> list[str]:
    """시간순 이벤트 코드 시퀀스. 같은 코드가 연속되면 하나로 접는다."""
    ordered = sorted(events, key=lambda e: e.time)
    codes: list[str] = []
    for ev in ordered:
        if not codes or codes[-1] != ev.code:
            codes.append(ev.code)
    if end_condition_type == "altitude_check_failed":
        if not codes or codes[-1] != ALTITUDE_CHECK_FAILED:
            codes.append(ALTITUDE_CHECK_FAILED)
    return codes
