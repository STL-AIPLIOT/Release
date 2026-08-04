# -*- coding: utf-8 -*-
"""run_local_dogfight.py --save-log 산출물에 replay_index.jsonl 을 만들어 준다.

왜 필요한가
-----------
학습 실행은 `artifacts/logs/<name>/<tag>/engagement_replays/replay_index.jsonl` 을 남기고,
분석 도구(analyze_loss_last5s.py, export_playback_cases.py 등)는 그 인덱스를 진입점으로 쓴다.

그런데 `run_local_dogfight.py --save-log` 는 인덱스를 만들지 않는다.
`artifacts/logs/` 에 평면으로 아래 세 파일만 떨군다(2026-08-04 실측).

    <ts>_ownship_(F-16)[Blue].csv
    <ts>_target_(F-16)[Red].csv
    <ts>_summary.json          키 4개: end_condition, outcome, ownship_health, target_health

이 스크립트가 그 세 쌍을 모아 학습 쪽과 같은 스키마의 replay_index.jsonl 을 만든다.
그러면 기존 분석 도구를 **한 줄도 고치지 않고** 로컬 교전에 그대로 쓸 수 있다.

인덱스에 넣는 값은 전부 실제 파일에서 읽은 것이다. summary.json 에 없는 필드
(total_reward, steps 등)는 계산 가능한 것만 채우고 나머지는 null 로 둔다. 지어내지 않는다.

    steps            ownship CSV 의 데이터 행 수 (env step 1회당 1행)
    ep_min_distance  두 궤적에서 계산 (GeoMathUtil 과 같은 식, geometry.py)
    total_reward     로컬 교전 로그에 없다 -> null
    iteration        로컬 교전에는 없다 -> 파일 순서(0부터)

실행
----
    python tools/index_local_dogfights.py --logdir artifacts/logs --run local_bt
    python tools/index_local_dogfights.py --logdir artifacts/logs --run local_bt \
        --since "2026-08-04 21:00"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import load_summary, load_track, warn  # noqa: E402
from log_analysis.geometry import derive_series  # noqa: E402

# 2026_8_4_21_29_27_summary.json
TS_RE = re.compile(r"^(?P<ts>\d{4}_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2})_summary\.json$")


def parse_timestamp(ts: str) -> datetime | None:
    """파일명 timestamp 를 datetime 으로. 형식이 다르면 None."""
    try:
        return datetime.strptime(ts, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return None


def find_engagements(logdir: Path) -> list[tuple[str, Path, Path, Path]]:
    """(timestamp, summary, ownship_csv, target_csv) 쌍을 모은다.

    세 파일이 다 있어야 한 경기로 친다. 하나라도 없으면 건너뛰고 경고한다.
    """
    out: list[tuple[str, Path, Path, Path]] = []
    for summary in sorted(logdir.glob("*_summary.json")):
        m = TS_RE.match(summary.name)
        if not m:
            continue
        ts = m.group("ts")
        own = sorted(logdir.glob(f"{ts}_ownship_*.csv"))
        tgt = sorted(logdir.glob(f"{ts}_target_*.csv"))
        if not own or not tgt:
            warn(f"{ts}: ownship/target CSV 를 찾지 못해 건너뛴다")
            continue
        out.append((ts, summary, own[0], tgt[0]))
    return out


def min_distance(own_csv: Path, tgt_csv: Path) -> float | None:
    """두 궤적의 최근접 거리 [m]. 계산 불가면 None."""
    own, tgt = load_track(own_csv), load_track(tgt_csv)
    if len(own) == 0 or len(tgt) == 0:
        return None
    samples = derive_series(own, tgt)
    dists = [s.distance_m for s in samples if s.distance_m is not None]
    return min(dists) if dists else None


def count_steps(csv_path: Path) -> int:
    """데이터 행 수(헤더 제외). env step 1회당 1행이다."""
    try:
        with csv_path.open(encoding="utf-8-sig") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError as exc:
        warn(f"{csv_path} 읽기 실패: {exc}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="로컬 교전 로그에 replay_index.jsonl 생성")
    ap.add_argument("--logdir", required=True, type=Path,
                    help="--save-log 산출물이 있는 디렉터리 (보통 artifacts/logs)")
    ap.add_argument("--run", default="local",
                    help="경기 ID 앞에 붙는 실험 태그 (기본 local)")
    ap.add_argument("--since", help='이 시각 이후 파일만 (예: "2026-08-04 21:00")')
    ap.add_argument("--output", type=Path,
                    help="인덱스 경로 (기본: <logdir>/replay_index.jsonl)")
    args = ap.parse_args()

    if not args.logdir.exists():
        warn(f"logdir 이 없다: {args.logdir}")
        return 2

    since = None
    if args.since:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                since = datetime.strptime(args.since, fmt)
                break
            except ValueError:
                continue
        if since is None:
            warn(f"--since 형식을 해석하지 못했다: {args.since}")
            return 2

    engagements = find_engagements(args.logdir)
    if since is not None:
        kept = []
        for ts, s, o, t in engagements:
            dt = parse_timestamp(ts)
            if dt is None or dt >= since:
                kept.append((ts, s, o, t))
        engagements = kept

    if not engagements:
        warn(f"교전 로그를 찾지 못했다: {args.logdir}")
        return 2

    out_path = args.output or (args.logdir / "replay_index.jsonl")
    rows: list[str] = []
    for i, (ts, summary_path, own_csv, tgt_csv) in enumerate(engagements):
        summary = load_summary(summary_path)
        dmin = min_distance(own_csv, tgt_csv)
        row = {
            "iteration": i,
            "episode": 0,
            "steps": count_steps(own_csv),
            # 로컬 교전 로그에는 보상이 없다. 0 으로 위장하지 않는다.
            "total_reward": None,
            "terminated": None,
            "truncated": None,
            "outcome": summary.get("outcome", ""),
            "end_condition": summary.get("end_condition", ""),
            "ownship_health": summary.get("ownship_health"),
            "target_health": summary.get("target_health"),
            "ep_min_distance": dmin if (dmin is None or math.isfinite(dmin)) else None,
            "replay_dir": str(args.logdir.resolve()),
            "ownship_log": str(own_csv.resolve()),
            "target_log": str(tgt_csv.resolve()),
            "summary_json": str(summary_path.resolve()),
            "sampled_steps": None,
            "stage": None,
            "source_timestamp": ts,
        }
        rows.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"교전 {len(rows)}건 -> {out_path}")
    outcomes: dict[str, int] = {}
    for r in rows:
        o = json.loads(r)["outcome"] or "(빈값)"
        outcomes[o] = outcomes.get(o, 0) + 1
    print(f"outcome 분포: {outcomes}")
    print()
    print("이제 기존 도구를 그대로 쓸 수 있다:")
    print(f"  python tools/analyze_loss_last5s.py --logdir {args.logdir} --release-root .")
    print(f"  python tools/export_playback_cases.py --logdir {args.logdir} "
          f"--output analysis/playback_cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
