# -*- coding: utf-8 -*-
"""로그 분석 공통 모듈.

스크립트마다 CSV/JSON 파싱을 중복하지 않도록 로더·정규화·지표를 모아 둔다.
구조는 2026-08-04 에 실제 로그를 조사해 확정했다. 자세한 내용은 schemas.py 참조.

하위 모듈
    loaders           replay_index / Tacview CSV / training_log 로더
    normalization     end_condition / outcome / BFM 모드명 정규화
    metrics           속도·에너지·하강률 등 궤적 파생 지표
    events            패배 직전 이벤트 추출
    angles            각도 wrap-around 공통 함수 (C++ AngleUtil.h 와 같은 규약)
    geometry          Tacview 궤적에서 ATA/AA/WEZ 파생 (GeoMathUtil 과 같은 식)
    predict_maneuver  PredictManeuver CSV 로더와 avgDelta / SCISSORS 집계
"""
from .angles import (
    circular_mean_deg,
    mean_signed_delta_deg,
    signed_angle_delta_deg,
    signed_angle_delta_rad,
    signed_delta_series_deg,
    wrap_angle_deg,
    wrap_angle_rad,
)
from .loaders import (
    find_replay_indexes,
    find_summaries,
    find_training_logs,
    iter_training_log,
    load_episodes,
    load_summary,
    load_track,
    warn,
)
from .metrics import (
    descent_rate_series,
    mean_ignoring_nan,
    specific_energy_series,
    speed_series,
    window_indices,
)
from .normalization import (
    END_CONDITION_ALIASES,
    END_CONDITION_TYPES,
    normalize_bfm,
    normalize_end_condition,
    normalize_outcome,
)
from .schemas import BFM_MODES, DASHBOARD_METRICS, Episode, Track

__all__ = [
    "BFM_MODES", "DASHBOARD_METRICS", "END_CONDITION_ALIASES", "END_CONDITION_TYPES",
    "Episode", "Track",
    "circular_mean_deg", "descent_rate_series", "find_replay_indexes",
    "find_summaries", "find_training_logs", "iter_training_log", "load_episodes",
    "load_summary", "load_track", "mean_ignoring_nan", "mean_signed_delta_deg",
    "normalize_bfm", "normalize_end_condition", "normalize_outcome",
    "signed_angle_delta_deg", "signed_angle_delta_rad", "signed_delta_series_deg",
    "specific_energy_series", "speed_series", "warn", "window_indices",
    "wrap_angle_deg", "wrap_angle_rad",
]
