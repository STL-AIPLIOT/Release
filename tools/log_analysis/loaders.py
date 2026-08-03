# -*- coding: utf-8 -*-
"""로그 파일 로더.

- replay_index.jsonl  경기 인덱스 (경기 단위 진입점)
- Tacview CSV         기체별 시계열
- training_log.csv    iteration 단위 지표

대용량 로그를 통째로 메모리에 올리지 않도록 training_log 는 필요한 컬럼만 읽는
경로를 따로 둔다. 오류는 조용히 넘기지 않고 파일명과 원인을 warn 으로 남긴다.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

from .normalization import normalize_bfm, normalize_end_condition, normalize_outcome
from .schemas import TACVIEW_ALIASES, Episode, Track


def warn(message: str) -> None:
    """경고를 stderr 로 남긴다. 호출부가 조용히 무시하지 못하게 한다."""
    print(f"[log_analysis] 경고: {message}", file=sys.stderr)


def _to_float(value: object) -> float | None:
    """float 로 바꾸되 실패하거나 NaN 이면 None. 0 으로 대체하지 않는다."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _resolve(base: Path, raw: object) -> Path | None:
    """로그에 적힌 경로를 Path 로. 절대경로가 깨졌으면 base 기준으로 재시도."""
    if not raw:
        return None
    p = Path(str(raw))
    if p.exists():
        return p
    candidate = base / p.name
    return candidate if candidate.exists() else None


def find_replay_indexes(logdir: Path) -> list[Path]:
    """logdir 아래 replay_index.jsonl 을 모두 찾는다."""
    if not logdir.exists():
        warn(f"logdir 이 없다: {logdir}")
        return []
    direct = logdir / "replay_index.jsonl"
    found = [direct] if direct.exists() else []
    found += sorted(p for p in logdir.rglob("replay_index.jsonl") if p != direct)
    if not found:
        warn(f"replay_index.jsonl 을 찾지 못했다: {logdir}")
    return found


def load_episodes(logdir: Path) -> list[Episode]:
    """logdir 아래 모든 경기를 읽어 iteration/episode 순으로 돌려준다."""
    episodes: list[Episode] = []
    for index_path in find_replay_indexes(logdir):
        base = index_path.parent
        # .../artifacts/logs/<name>/<tag>/engagement_replays/replay_index.jsonl
        # -> 실험 태그는 engagement_replays 의 부모 디렉터리 이름이다.
        run = base.parent.name if base.name == "engagement_replays" else base.name
        for lineno, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warn(f"{index_path}:{lineno} JSON 파싱 실패: {exc}")
                continue
            outcome_raw = str(row.get("outcome", ""))
            end_raw = str(row.get("end_condition", ""))
            episodes.append(
                Episode(
                    iteration=int(row.get("iteration", -1)),
                    episode=int(row.get("episode", -1)),
                    steps=int(row.get("steps", 0)),
                    total_reward=_to_float(row.get("total_reward")) or 0.0,
                    outcome=normalize_outcome(outcome_raw),
                    outcome_raw=outcome_raw,
                    end_condition=normalize_end_condition(end_raw),
                    end_condition_raw=end_raw,
                    ownship_health=_to_float(row.get("ownship_health")),
                    target_health=_to_float(row.get("target_health")),
                    ep_min_distance=_to_float(row.get("ep_min_distance")),
                    ownship_log=_resolve(base, row.get("ownship_log")),
                    target_log=_resolve(base, row.get("target_log")),
                    summary_json=_resolve(base, row.get("summary_json")),
                    source_index=index_path,
                    run=run,
                )
            )
    episodes.sort(key=lambda e: (e.run, e.iteration, e.episode))
    return episodes


def load_summary(path: Path) -> dict[str, object]:
    """<ts>_summary.json 을 읽는다. 실제 키는 4개뿐이다."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"{path} 읽기 실패: {exc}")
        return {}


def find_summaries(logdir: Path) -> list[Path]:
    """`<timestamp>_summary.json` 을 모두 찾는다. 이름이 summary.json 이 아니다."""
    return sorted(logdir.rglob("*_summary.json"))


def _column_map(fieldnames: list[str] | None) -> dict[str, str]:
    """정규화 이름 -> 실제 컬럼명. 없는 것은 넣지 않는다."""
    if not fieldnames:
        return {}
    lookup = {name.strip(): name for name in fieldnames}
    lowered = {name.strip().lower(): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, candidates in TACVIEW_ALIASES.items():
        for cand in candidates:
            if cand in lookup:
                resolved[canonical] = lookup[cand]
                break
            if cand.lower() in lowered:
                resolved[canonical] = lowered[cand.lower()]
                break
    return resolved


def load_track(path: Path) -> Track:
    """Tacview CSV 한 개를 Track 으로 읽는다.

    컬럼이 없으면 해당 리스트를 비워 두고 missing 에 이름을 남긴다.
    0 으로 채우지 않는다.
    """
    track = Track(path=path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cmap = _column_map(reader.fieldnames)
            wanted = ("time", "lat", "lon", "alt", "roll", "pitch", "yaw", "health")
            missing = tuple(k for k in wanted if k not in cmap)
            track.missing = missing
            if missing:
                warn(f"{path.name}: 컬럼 없음 {missing}")
            for row in reader:
                if "time" in cmap:
                    t = _to_float(row.get(cmap["time"]))
                    if t is None:
                        continue
                    track.time.append(t)
                for key, target in (
                    ("lat", track.lat), ("lon", track.lon), ("alt", track.alt),
                    ("roll", track.roll), ("pitch", track.pitch), ("yaw", track.yaw),
                    ("health", track.health),
                ):
                    if key in cmap:
                        val = _to_float(row.get(cmap[key]))
                        target.append(val if val is not None else math.nan)
                if "bfm" in cmap:
                    track.bfm.append(normalize_bfm(row.get(cmap["bfm"])))
    except OSError as exc:
        warn(f"{path} 읽기 실패: {exc}")
        return track

    if len(track.time) >= 2 and any(
        track.time[i] > track.time[i + 1] for i in range(len(track.time) - 1)
    ):
        warn(f"{path.name}: Time 이 단조 증가가 아니다. 정렬해서 사용한다.")
        order = sorted(range(len(track.time)), key=lambda i: track.time[i])
        for seq in (track.time, track.lat, track.lon, track.alt,
                    track.roll, track.pitch, track.yaw, track.health):
            if len(seq) == len(order):
                seq[:] = [seq[i] for i in order]
    return track


def iter_training_log(path: Path, columns: tuple[str, ...] | None = None) -> Iterator[dict[str, float | None]]:
    """training_log.csv 를 한 행씩 흘려보낸다. 전부 메모리에 올리지 않는다.

    columns 를 주면 그 컬럼만 담는다. 없는 컬럼은 키를 만들지 않는다
    (임의로 0 을 넣지 않는다).
    """
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            available = set(reader.fieldnames or ())
            if columns:
                absent = [c for c in columns if c not in available]
                if absent:
                    warn(f"{path.name}: 없는 컬럼 {absent}")
                keys = [c for c in columns if c in available]
            else:
                keys = sorted(available)
            for row in reader:
                yield {k: _to_float(row.get(k)) for k in keys}
    except OSError as exc:
        warn(f"{path} 읽기 실패: {exc}")


def find_training_logs(logdir: Path) -> list[Path]:
    """logdir 아래 training_log.csv 를 모두 찾는다."""
    direct = logdir / "training_log.csv"
    if direct.exists():
        return [direct]
    return sorted(logdir.rglob("training_log.csv"))
