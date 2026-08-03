# -*- coding: utf-8 -*-
"""로그 스키마 정의와 컬럼 alias.

실제 로그에서 확인한 구조만 담는다(2026-08-04 조사 기준).

  replay_index.jsonl   경기 단위 인덱스. 아래 REPLAY_INDEX_KEYS 참조.
  <ts>_summary.json    키 4개뿐: end_condition, outcome, ownship_health, target_health
  Tacview CSV          Time, Longitude, Latitude, Altitude, Roll (deg), Pitch (deg),
                       Yaw (deg), Health
  training_log.csv     iteration 단위 49컬럼

실험 버전마다 컬럼명이 달라질 수 있으므로 alias 로 흡수한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Tacview CSV 컬럼 alias --------------------------------------------------
# 키는 정규화된 이름, 값은 실제 로그에서 나타날 수 있는 후보들.
TACVIEW_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("Time", "time", "t", "Time (s)"),
    "lon": ("Longitude", "longitude", "lon", "Lon"),
    "lat": ("Latitude", "latitude", "lat", "Lat"),
    "alt": ("Altitude", "altitude", "alt", "Altitude (m)", "ALT"),
    "roll": ("Roll (deg)", "Roll", "roll", "roll_deg"),
    "pitch": ("Pitch (deg)", "Pitch", "pitch", "pitch_deg"),
    "yaw": ("Yaw (deg)", "Yaw", "yaw", "yaw_deg", "Heading"),
    "health": ("Health", "health", "HP", "hp"),
    # 아래는 현재 로그에 없다. BFM 로깅이 추가되면 여기로 흡수된다.
    "bfm": ("BFM", "bfm", "BFM_Mode", "bfm_mode", "Mode"),
    "speed": ("Speed", "speed", "KCAS", "TAS", "speed_ms"),
}

# --- replay_index.jsonl 키 ---------------------------------------------------
REPLAY_INDEX_KEYS = (
    "iteration", "episode", "steps", "total_reward",
    "terminated", "truncated", "outcome", "end_condition",
    "ownship_health", "target_health", "ep_min_distance",
    "replay_dir", "ownship_log", "target_log", "summary_json",
    "sampled_steps", "stage",
)

# --- training_log.csv 에서 자주 쓰는 지표 ------------------------------------
DASHBOARD_METRICS = (
    "reward_mean", "crash_rate", "ep_min_distance", "ep_wez_steps", "win_rate",
)

BFM_MODES = ("OBFM", "DBFM", "HABFM", "SCISSORS", "UNKNOWN")


@dataclass
class Episode:
    """경기 하나. replay_index.jsonl 한 행 + 연결된 파일 경로."""

    iteration: int
    episode: int
    steps: int
    total_reward: float
    outcome: str                  # 정규화된 값 (win/loss/draw/crash/timeout/unknown)
    outcome_raw: str
    end_condition: str            # 정규화된 유형
    end_condition_raw: str
    ownship_health: float | None
    target_health: float | None
    ep_min_distance: float | None
    ownship_log: Path | None
    target_log: Path | None
    summary_json: Path | None
    source_index: Path | None = None
    run: str = ""                 # 실험 태그. 여러 실험을 함께 읽을 때 ID 충돌을 막는다.

    @property
    def match_id(self) -> str:
        """경기 식별자.

        iteration/episode 만으로는 실험이 다르면 겹치므로 실험 태그를 앞에 붙인다.
        """
        stem = f"iter{self.iteration:06d}_ep{self.episode:02d}"
        return f"{self.run}/{stem}" if self.run else stem


@dataclass
class Track:
    """Tacview CSV 한 기체의 시계열. 값이 없는 컬럼은 빈 리스트로 둔다."""

    path: Path
    time: list[float] = field(default_factory=list)
    lat: list[float] = field(default_factory=list)
    lon: list[float] = field(default_factory=list)
    alt: list[float] = field(default_factory=list)
    roll: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)
    health: list[float] = field(default_factory=list)
    bfm: list[str] = field(default_factory=list)
    missing: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.time)

    @property
    def duration_sec(self) -> float:
        return (self.time[-1] - self.time[0]) if len(self.time) >= 2 else 0.0

    @property
    def has_bfm(self) -> bool:
        return bool(self.bfm)
