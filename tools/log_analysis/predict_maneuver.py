# -*- coding: utf-8 -*-
"""PredictManeuver CSV 로더와 집계.

입력 파일
---------
BT(C++) 의 PredictManeuverCsvLogger 가 쓴 CSV 다. 환경변수 PM_CSV_LOG 가
지정된 실행에서만 만들어진다.

    Behaviortree/BT_Content/Service/PredictManeuverCsvLogger.h

컬럼 (v2, 2026-08-04)
    runType,episode,frame,time,prevAngle,currAngle,rawDelta,normalizedDelta,
    avgDelta,predictedTurn,bfmMode,scissorsEntered,distance_m,ownAta_deg,
    targetAa_deg,angleOff_deg,enemyInSight

컬럼 (v1, episode/geometry 컬럼이 없던 초기 버전)
    runType,frame,time,prevAngle,currAngle,rawDelta,normalizedDelta,
    avgDelta,bfmMode,scissorsEntered

두 버전을 모두 읽는다. v1 에는 episode 컬럼이 없으므로 time(RunningTime) 이
되감기는 지점을 경기 경계로 보아 파생시킨다(tools/extract_bfm_log.py 와 같은 규칙).
없는 컬럼은 None 으로 두고 0 으로 채우지 않는다.

단위
----
각도는 전부 degree, 거리는 meter, 시간은 second 다. CSV 자체에는 단위가
적혀 있지 않으므로 이 사실을 산출물 메타데이터에 함께 기록한다.

SCISSORS 집계 규칙
------------------
**진입 횟수는 상태 행 수가 아니라 transition 수다.** 비-SCISSORS -> SCISSORS 로
바뀐 프레임만 1회로 센다. CSV 의 scissorsEntered 컬럼도 같은 규칙으로 기록되지만,
v1 로그나 다른 경로에서 온 데이터를 위해 여기서 bfmMode 시퀀스로부터 다시 계산한다.
(두 값이 다르면 report 에 함께 남긴다.)
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .angles import wrap_angle_deg
from .loaders import warn
from .normalization import normalize_bfm

SCISSORS = "SCISSORS"

# CSV 헤더 -> 정규화 이름. 실제 로거가 쓰는 이름만 담는다(추측 금지).
PM_COLUMNS = (
    "runType", "episode", "frame", "time", "prevAngle", "currAngle",
    "rawDelta", "normalizedDelta", "avgDelta", "predictedTurn", "bfmMode",
    "scissorsEntered", "distance_m", "ownAta_deg", "targetAa_deg",
    "angleOff_deg", "enemyInSight",
)

# v1 에는 없던 컬럼. 없으면 None 으로 두고 그 사실을 보고한다.
PM_OPTIONAL_COLUMNS = (
    "episode", "predictedTurn", "distance_m", "ownAta_deg", "targetAa_deg",
    "angleOff_deg", "enemyInSight",
)


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out  # NaN 은 그대로 둔다. 이상 프레임 표시로 의미가 있다.


def _to_int(value: object) -> int | None:
    f = _to_float(value)
    if f is None or math.isnan(f):
        return None
    return int(f)


@dataclass
class PredictFrame:
    """PredictManeuver CSV 한 행."""

    run_type: str
    episode: int
    frame: int | None
    time_sec: float | None
    prev_angle_deg: float | None
    curr_angle_deg: float | None
    raw_delta_deg: float | None
    normalized_delta_deg: float | None
    avg_delta_deg: float | None
    predicted_turn: str | None
    bfm_mode: str
    scissors_entered_logged: int | None
    distance_m: float | None
    own_ata_deg: float | None
    target_aa_deg: float | None
    angle_off_deg: float | None
    enemy_in_sight: int | None
    source: Path | None = None
    episode_derived: bool = False

    @property
    def match_id(self) -> str:
        """경기 식별자. runType 이 다르면 같은 episode 번호라도 다른 경기다."""
        return f"{self.run_type}/ep{self.episode:03d}"


@dataclass
class PredictLog:
    """한 그룹(before 또는 after)의 프레임 전체."""

    label: str
    frames: list[PredictFrame] = field(default_factory=list)
    sources: list[Path] = field(default_factory=list)
    missing_columns: tuple[str, ...] = ()
    episode_derived: bool = False

    @property
    def match_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for f in self.frames:
            seen.setdefault(f.match_id, None)
        return list(seen)

    def by_match(self) -> dict[str, list[PredictFrame]]:
        out: dict[str, list[PredictFrame]] = {}
        for f in self.frames:
            out.setdefault(f.match_id, []).append(f)
        return out


def find_predict_csvs(path: Path) -> list[Path]:
    """파일이면 그 파일, 디렉터리면 아래의 *.csv 를 모두 찾는다."""
    if not path.exists():
        warn(f"경로가 없다: {path}")
        return []
    if path.is_file():
        return [path]
    found = sorted(p for p in path.rglob("*.csv"))
    if not found:
        warn(f"CSV 를 찾지 못했다: {path}")
    return found


def _looks_like_predict_csv(fieldnames: list[str] | None) -> bool:
    """PredictManeuver CSV 인지 헤더로 판별한다. 필수 컬럼 3개로 본다."""
    if not fieldnames:
        return False
    names = {n.strip() for n in fieldnames}
    return {"avgDelta", "rawDelta", "bfmMode"}.issubset(names)


def load_predict_log(path: Path, label: str) -> PredictLog:
    """디렉터리 또는 파일에서 PredictManeuver 프레임을 읽는다."""
    log = PredictLog(label=label)
    missing: set[str] = set()

    for csv_path in find_predict_csvs(path):
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                if not _looks_like_predict_csv(reader.fieldnames):
                    # 같은 디렉터리에 다른 CSV 가 섞여 있을 수 있다. 조용히 건너뛰지 않는다.
                    warn(f"{csv_path.name}: PredictManeuver CSV 형식이 아니다 (건너뜀)")
                    continue
                available = {n.strip() for n in (reader.fieldnames or [])}
                missing |= {c for c in PM_OPTIONAL_COLUMNS if c not in available}
                has_episode = "episode" in available
                if not has_episode:
                    log.episode_derived = True

                prev_time: float | None = None
                derived_episode = 0

                for row in reader:
                    time_sec = _to_float(row.get("time"))
                    if not has_episode:
                        # RunningTime 이 되감기면 새 경기.
                        if (prev_time is not None and time_sec is not None
                                and time_sec < prev_time - 1e-6):
                            derived_episode += 1
                        if time_sec is not None:
                            prev_time = time_sec
                        episode = derived_episode
                    else:
                        episode = _to_int(row.get("episode")) or 0

                    log.frames.append(PredictFrame(
                        run_type=str(row.get("runType", "") or label).strip(),
                        episode=episode,
                        frame=_to_int(row.get("frame")),
                        time_sec=time_sec,
                        prev_angle_deg=_to_float(row.get("prevAngle")),
                        curr_angle_deg=_to_float(row.get("currAngle")),
                        raw_delta_deg=_to_float(row.get("rawDelta")),
                        normalized_delta_deg=_to_float(row.get("normalizedDelta")),
                        avg_delta_deg=_to_float(row.get("avgDelta")),
                        predicted_turn=(row.get("predictedTurn") or None),
                        bfm_mode=normalize_bfm(row.get("bfmMode")),
                        scissors_entered_logged=_to_int(row.get("scissorsEntered")),
                        distance_m=_to_float(row.get("distance_m")),
                        own_ata_deg=_to_float(row.get("ownAta_deg")),
                        target_aa_deg=_to_float(row.get("targetAa_deg")),
                        angle_off_deg=_to_float(row.get("angleOff_deg")),
                        enemy_in_sight=_to_int(row.get("enemyInSight")),
                        source=csv_path,
                        episode_derived=not has_episode,
                    ))
                log.sources.append(csv_path)
        except OSError as exc:
            warn(f"{csv_path} 읽기 실패: {exc}")

    log.missing_columns = tuple(sorted(missing))
    return log


# --------------------------------------------------------------------------- avgDelta
@dataclass
class AvgDeltaStats:
    """avgDelta 분포. 계산 불가능한 값은 None 으로 둔다(0 으로 위장하지 않는다)."""

    sample_count: int = 0
    nan_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None
    p95: float | None = None
    p99: float | None = None
    abs_max: float | None = None
    out_of_range_count: int = 0          # |avgDelta| > 180 : 정의상 나올 수 없는 값
    outlier_counts: dict[int, int] = field(default_factory=dict)   # 임계값 -> 개수
    spike_count: int = 0                 # 프레임 간 급변 횟수

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "nan_count": self.nan_count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "p95": self.p95,
            "p99": self.p99,
            "abs_max": self.abs_max,
            "out_of_range_count": self.out_of_range_count,
            "outlier_counts": {str(k): v for k, v in sorted(self.outlier_counts.items())},
            "spike_count": self.spike_count,
        }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """선형보간 분위수. 표본이 없으면 None."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def avg_delta_stats(frames: list[PredictFrame],
                    outlier_thresholds_deg: tuple[float, ...] = (170.0, 175.0, 179.0),
                    spike_threshold_deg: float = 90.0) -> AvgDeltaStats:
    """avgDelta 분포와 이상값 개수를 센다.

    spike 는 **같은 경기 안에서** 연속 프레임의 avgDelta 차이가 임계값 이상인
    경우다. 경기 경계를 넘는 차이는 세지 않는다.
    """
    st = AvgDeltaStats()
    values: list[float] = []

    prev_by_match: dict[str, float] = {}
    for f in frames:
        v = f.avg_delta_deg
        if v is None:
            continue
        st.sample_count += 1
        if math.isnan(v):
            st.nan_count += 1
            continue
        values.append(v)
        if abs(v) > 180.0 + 1e-6:
            st.out_of_range_count += 1
        prev = prev_by_match.get(f.match_id)
        if prev is not None and abs(v - prev) >= spike_threshold_deg:
            st.spike_count += 1
        prev_by_match[f.match_id] = v

    for th in outlier_thresholds_deg:
        st.outlier_counts[int(th)] = sum(1 for v in values if abs(v) >= th)

    if not values:
        return st

    ordered = sorted(values)
    st.minimum = ordered[0]
    st.maximum = ordered[-1]
    st.mean = statistics.fmean(values)
    st.median = statistics.median(values)
    st.stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    st.p95 = _percentile(ordered, 0.95)
    st.p99 = _percentile(ordered, 0.99)
    st.abs_max = max(abs(v) for v in values)
    return st


@dataclass
class Outlier:
    """이상값 한 건과 그 전후 문맥."""

    match_id: str
    run_type: str
    episode: int
    frame: int | None
    time_sec: float | None
    avg_delta_deg: float
    raw_delta_deg: float | None
    normalized_delta_deg: float | None
    kind: str                     # "magnitude" | "spike" | "out_of_range"
    bfm_before: str | None
    bfm_at: str
    bfm_after: str | None
    scissors_before: bool | None
    scissors_at: bool
    scissors_after: bool | None
    own_ata_deg: float | None
    target_aa_deg: float | None
    distance_m: float | None
    derived_in_wez: bool | None
    wez_reason: str


def _wez_state(distance_m: float | None, own_ata_deg: float | None,
               wez_angle_deg: float, wez_min_m: float, wez_max_m: float,
               ) -> tuple[bool | None, str]:
    """derived WEZ 판정.

    src/dogfight/envs/single_agent_env.py:572,577-579 의 update_damage 와
    **같은 비교식**을 쓴다. 각도 epsilon 을 두지 않는다.

        min_range_m <= dis_m <= max_range_m  and  angle_deg/2 >= abs(ATA)

    BT 의 ownAta_deg(=Los_Degree) 는 부호가 없는 0~180 이라 abs() 가 항등이다.
    값이 없으면 None 을 돌려주고 사유를 함께 남긴다(0/False 로 위장하지 않는다).
    """
    if distance_m is None or own_ata_deg is None:
        return None, "distance_m 또는 ownAta_deg 컬럼이 없다 (v1 로그)"
    if math.isnan(distance_m) or math.isnan(own_ata_deg):
        return None, "distance_m 또는 ownAta_deg 가 NaN"
    in_range = wez_min_m <= distance_m <= wez_max_m
    in_cone = (wez_angle_deg / 2.0) >= abs(own_ata_deg)
    return (in_range and in_cone), "update_damage 와 동일한 비교식"


def detect_outliers(frames: list[PredictFrame],
                    outlier_threshold_deg: float = 170.0,
                    spike_threshold_deg: float = 90.0,
                    wez_angle_deg: float = 2.0,
                    wez_min_m: float = 152.4,
                    wez_max_m: float = 914.4) -> list[Outlier]:
    """이상값 프레임을 전후 문맥과 함께 뽑는다.

    kind
        magnitude    |avgDelta| >= outlier_threshold_deg
        spike        같은 경기 안에서 직전 프레임 대비 변화량 >= spike_threshold_deg
        out_of_range |avgDelta| > 180 (wrap 보정이 되지 않았다는 직접 증거)
    """
    out: list[Outlier] = []
    by_match: dict[str, list[PredictFrame]] = {}
    for f in frames:
        by_match.setdefault(f.match_id, []).append(f)

    for match_id, seq in by_match.items():
        for i, f in enumerate(seq):
            v = f.avg_delta_deg
            if v is None or math.isnan(v):
                continue
            kinds: list[str] = []
            if abs(v) > 180.0 + 1e-6:
                kinds.append("out_of_range")
            if abs(v) >= outlier_threshold_deg:
                kinds.append("magnitude")
            if i > 0:
                prev = seq[i - 1].avg_delta_deg
                if (prev is not None and not math.isnan(prev)
                        and abs(v - prev) >= spike_threshold_deg):
                    kinds.append("spike")
            if not kinds:
                continue

            before = seq[i - 1] if i > 0 else None
            after = seq[i + 1] if i + 1 < len(seq) else None
            in_wez, reason = _wez_state(f.distance_m, f.own_ata_deg,
                                        wez_angle_deg, wez_min_m, wez_max_m)
            out.append(Outlier(
                match_id=match_id,
                run_type=f.run_type,
                episode=f.episode,
                frame=f.frame,
                time_sec=f.time_sec,
                avg_delta_deg=v,
                raw_delta_deg=f.raw_delta_deg,
                normalized_delta_deg=f.normalized_delta_deg,
                kind="+".join(kinds),
                bfm_before=before.bfm_mode if before else None,
                bfm_at=f.bfm_mode,
                bfm_after=after.bfm_mode if after else None,
                scissors_before=(before.bfm_mode == SCISSORS) if before else None,
                scissors_at=f.bfm_mode == SCISSORS,
                scissors_after=(after.bfm_mode == SCISSORS) if after else None,
                own_ata_deg=f.own_ata_deg,
                target_aa_deg=f.target_aa_deg,
                distance_m=f.distance_m,
                derived_in_wez=in_wez,
                wez_reason=reason,
            ))
    return out


def wraparound_evidence(frames: list[PredictFrame]) -> dict[str, object]:
    """wrap 보정이 실제로 적용됐는지 직접 확인한다.

    rawDelta 와 normalizedDelta 를 함께 기록하므로, 두 값이 다른 프레임이
    'wrap 보정이 실제로 일어난' 프레임이다. normalizedDelta 가 범위를 벗어나면
    보정이 되지 않았다는 뜻이다.
    """
    raw_beyond_180 = 0
    normalized_beyond_180 = 0
    corrected = 0
    checked = 0
    max_abs_normalized: float | None = None

    for f in frames:
        raw = f.raw_delta_deg
        norm = f.normalized_delta_deg
        if raw is None or norm is None:
            continue
        if math.isnan(raw) or math.isnan(norm):
            continue
        checked += 1
        if abs(raw) > 180.0:
            raw_beyond_180 += 1
        if abs(norm) > 180.0 + 1e-3:
            normalized_beyond_180 += 1
        if abs(raw - norm) > 1e-3:
            corrected += 1
        a = abs(norm)
        if max_abs_normalized is None or a > max_abs_normalized:
            max_abs_normalized = a

    # 보정이 맞다면 normalizedDelta 는 wrap_angle_deg(rawDelta) 와 같아야 한다.
    mismatches = 0
    for f in frames:
        raw, norm = f.raw_delta_deg, f.normalized_delta_deg
        if raw is None or norm is None or math.isnan(raw) or math.isnan(norm):
            continue
        if abs(wrap_angle_deg(raw) - norm) > 1e-2:
            mismatches += 1

    return {
        "checked_frames": checked,
        "raw_delta_beyond_180": raw_beyond_180,
        "normalized_delta_beyond_180": normalized_beyond_180,
        "wrap_corrected_frames": corrected,
        "max_abs_normalized_delta": max_abs_normalized,
        "normalized_vs_expected_mismatch": mismatches,
    }


# --------------------------------------------------------------------------- SCISSORS
@dataclass
class ScissorsStats:
    """SCISSORS 진입/체류 집계. 진입은 transition 기준이다."""

    match_count: int = 0
    matches_with_entry: int = 0
    entry_count: int = 0
    entries_per_match_mean: float | None = None
    entries_per_match_median: float | None = None
    total_dwell_sec: float | None = None
    dwell_per_match_mean: float | None = None
    dwell_per_entry_mean: float | None = None
    longest_dwell_sec: float | None = None
    reentry_count: int = 0
    open_segment_count: int = 0            # 경기 끝까지 SCISSORS 였던 구간 수
    entered_from: dict[str, int] = field(default_factory=dict)
    exited_to: dict[str, int] = field(default_factory=dict)
    logged_entry_count: int | None = None  # CSV scissorsEntered 합계 (대조용)

    @property
    def entry_match_ratio(self) -> float | None:
        if self.match_count == 0:
            return None
        return self.matches_with_entry / self.match_count

    def as_dict(self) -> dict[str, object]:
        return {
            "match_count": self.match_count,
            "matches_with_entry": self.matches_with_entry,
            "entry_match_ratio": self.entry_match_ratio,
            "entry_count": self.entry_count,
            "entries_per_match_mean": self.entries_per_match_mean,
            "entries_per_match_median": self.entries_per_match_median,
            "total_dwell_sec": self.total_dwell_sec,
            "dwell_per_match_mean": self.dwell_per_match_mean,
            "dwell_per_entry_mean": self.dwell_per_entry_mean,
            "longest_dwell_sec": self.longest_dwell_sec,
            "reentry_count": self.reentry_count,
            "open_segment_count": self.open_segment_count,
            "entered_from": dict(sorted(self.entered_from.items())),
            "exited_to": dict(sorted(self.exited_to.items())),
            "logged_entry_count": self.logged_entry_count,
        }


@dataclass
class ScissorsSegment:
    """SCISSORS 체류 구간 하나."""

    match_id: str
    start_sec: float | None
    end_sec: float | None
    duration_sec: float | None
    entered_from: str | None
    exited_to: str | None
    open_ended: bool
    entry_index: int


def scissors_segments(frames: list[PredictFrame]) -> list[ScissorsSegment]:
    """경기별 SCISSORS 체류 구간 목록.

    - 진입은 비-SCISSORS -> SCISSORS transition 1회다. 체류 프레임은 세지 않는다.
    - 첫 프레임이 이미 SCISSORS 면 직전 상태를 알 수 없으므로 진입으로 세지 않고
      entered_from=None 으로 남긴다(없는 정보를 지어내지 않는다).
    - 경기 마지막까지 SCISSORS 였던 구간은 종료 시각을 알 수 없어
      duration_sec=None, open_ended=True 로 둔다. 평균 체류 계산에서 제외된다.
    - timestamp 간격이 일정하지 않아도 duration 은 end-start 로 계산되므로 영향이 없다.
      time 컬럼이 비어 있으면 duration 은 None 이다.
    """
    segments: list[ScissorsSegment] = []
    by_match: dict[str, list[PredictFrame]] = {}
    for f in frames:
        by_match.setdefault(f.match_id, []).append(f)

    for match_id, seq in by_match.items():
        entry_index = 0
        i = 0
        while i < len(seq):
            if seq[i].bfm_mode != SCISSORS:
                i += 1
                continue
            start_i = i
            while i < len(seq) and seq[i].bfm_mode == SCISSORS:
                i += 1
            end_i = i                     # 첫 비-SCISSORS 인덱스 (없으면 len)
            is_entry = start_i > 0        # 첫 프레임부터 SCISSORS 면 진입 미상
            open_ended = end_i >= len(seq)

            start_t = seq[start_i].time_sec
            end_t = seq[end_i].time_sec if not open_ended else None
            duration = None
            if start_t is not None and end_t is not None:
                duration = end_t - start_t

            segments.append(ScissorsSegment(
                match_id=match_id,
                start_sec=start_t,
                end_sec=end_t,
                duration_sec=duration,
                entered_from=seq[start_i - 1].bfm_mode if is_entry else None,
                exited_to=seq[end_i].bfm_mode if not open_ended else None,
                open_ended=open_ended,
                entry_index=entry_index,
            ))
            entry_index += 1
    return segments


def scissors_stats(frames: list[PredictFrame]) -> tuple[ScissorsStats, list[ScissorsSegment]]:
    """SCISSORS 진입/체류 집계와 구간 목록을 함께 돌려준다."""
    st = ScissorsStats()
    segments = scissors_segments(frames)

    by_match: dict[str, list[PredictFrame]] = {}
    for f in frames:
        by_match.setdefault(f.match_id, []).append(f)
    st.match_count = len(by_match)

    entries_per_match: list[int] = []
    dwell_per_match: dict[str, float] = {}
    closed_durations: list[float] = []

    for match_id in by_match:
        segs = [s for s in segments if s.match_id == match_id]
        # entered_from 이 있는 구간만 '진입'이다(첫 프레임부터 SCISSORS 인 구간 제외).
        entries = [s for s in segs if s.entered_from is not None]
        entries_per_match.append(len(entries))
        if entries:
            st.matches_with_entry += 1
        # 2회차 이후 진입 = 재진입
        st.reentry_count += max(0, len(entries) - 1)
        for s in entries:
            st.entered_from[s.entered_from or "UNKNOWN"] = \
                st.entered_from.get(s.entered_from or "UNKNOWN", 0) + 1
        for s in segs:
            if s.exited_to is not None:
                st.exited_to[s.exited_to] = st.exited_to.get(s.exited_to, 0) + 1
            if s.open_ended:
                st.open_segment_count += 1
            if s.duration_sec is not None:
                closed_durations.append(s.duration_sec)
                dwell_per_match[match_id] = dwell_per_match.get(match_id, 0.0) + s.duration_sec

    st.entry_count = sum(entries_per_match)
    if entries_per_match:
        st.entries_per_match_mean = statistics.fmean(entries_per_match)
        st.entries_per_match_median = statistics.median(entries_per_match)
    if closed_durations:
        st.total_dwell_sec = sum(closed_durations)
        st.dwell_per_entry_mean = statistics.fmean(closed_durations)
        st.longest_dwell_sec = max(closed_durations)
    if dwell_per_match and st.match_count:
        # 체류 0초인 경기도 분모에 넣는다(진입한 경기만 평균내면 과대평가된다).
        st.dwell_per_match_mean = sum(dwell_per_match.values()) / st.match_count

    logged = [f.scissors_entered_logged for f in frames
              if f.scissors_entered_logged is not None]
    st.logged_entry_count = sum(logged) if logged else None

    return st, segments


def scissors_after_outliers(frames: list[PredictFrame], outliers: list[Outlier],
                            window_frames: int = 20) -> int:
    """이상값 발생 직후 window_frames 안에서 SCISSORS 로 진입한 횟수."""
    by_match: dict[str, list[PredictFrame]] = {}
    for f in frames:
        by_match.setdefault(f.match_id, []).append(f)

    count = 0
    for o in outliers:
        seq = by_match.get(o.match_id, [])
        idx = next((i for i, f in enumerate(seq) if f.frame == o.frame), None)
        if idx is None:
            continue
        window = seq[idx: idx + window_frames + 1]
        for i in range(1, len(window)):
            if window[i].bfm_mode == SCISSORS and window[i - 1].bfm_mode != SCISSORS:
                count += 1
                break
    return count
