# -*- coding: utf-8 -*-
"""로그 분석 공통 모듈.

스크립트마다 CSV/JSON 파싱을 중복하지 않도록 로더·정규화·지표를 모아 둔다.
구조는 2026-08-04 에 실제 로그를 조사해 확정했다. 자세한 내용은 schemas.py 참조.
"""
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
    "descent_rate_series", "find_replay_indexes", "find_summaries",
    "find_training_logs", "iter_training_log", "load_episodes", "load_summary",
    "load_track", "mean_ignoring_nan", "normalize_bfm", "normalize_end_condition",
    "normalize_outcome", "specific_energy_series", "speed_series", "warn",
    "window_indices",
]
