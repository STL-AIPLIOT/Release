# -*- coding: utf-8 -*-
"""종료 원인(end_condition) 집계.

replay_index.jsonl 과 <timestamp>_summary.json 을 읽어 종료 원인을 정규화 유형별로
센다. 매핑되지 않은 원본 값은 임의로 other 에 묻지 않고 별도 파일로 남긴다.

실행:
    python tools/count_end_conditions.py --logdir <디렉터리> --output analysis/end_conditions
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import (  # noqa: E402
    END_CONDITION_TYPES,
    find_summaries,
    load_episodes,
    load_summary,
    normalize_end_condition,
    normalize_outcome,
    warn,
)


def collect(logdir: Path) -> tuple[list[dict[str, str]], list[str]]:
    """(레코드 목록, 사용한 소스 설명). replay_index 를 우선 쓰고 없으면 summary 를 훑는다."""
    records: list[dict[str, str]] = []
    sources: list[str] = []

    episodes = load_episodes(logdir)
    for ep in episodes:
        records.append({
            "match_id": ep.match_id,
            "iteration": str(ep.iteration),
            "episode": str(ep.episode),
            "end_condition_raw": ep.end_condition_raw,
            "end_condition_type": ep.end_condition,
            "outcome_raw": ep.outcome_raw,
            "outcome": ep.outcome,
            "source": "replay_index.jsonl",
        })
    if episodes:
        sources.append(f"replay_index.jsonl ({len(episodes)}경기)")

    known = {(r["iteration"], r["episode"]) for r in records}
    extra = 0
    for path in find_summaries(logdir):
        data = load_summary(path)
        if not data:
            continue
        raw = str(data.get("end_condition", ""))
        # replay_index 로 이미 잡힌 경기는 중복 계산하지 않는다.
        if any(r["end_condition_raw"] == raw for r in records) and known:
            continue
        extra += 1
        records.append({
            "match_id": path.parent.name or path.stem,
            "iteration": "", "episode": "",
            "end_condition_raw": raw,
            "end_condition_type": normalize_end_condition(raw),
            "outcome_raw": str(data.get("outcome", "")),
            "outcome": normalize_outcome(data.get("outcome")),
            "source": str(path),
        })
    if extra:
        sources.append(f"*_summary.json ({extra}건, replay_index 에 없던 것)")
    return records, sources


def build_report(records: list[dict[str, str]], logdir: Path,
                 sources: list[str]) -> dict[str, object]:
    """집계 결과를 dict 로 만든다."""
    raw_counts = Counter(r["end_condition_raw"] for r in records)
    type_counts = Counter(r["end_condition_type"] for r in records)
    total = len(records)

    cross: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        cross[r["outcome"]][r["end_condition_type"]] += 1

    unmapped = sorted({
        r["end_condition_raw"] for r in records
        if r["end_condition_type"] == "unknown" and r["end_condition_raw"]
    })

    return {
        "source_logdir": str(logdir),
        "sources": sources,
        "total_matches": total,
        "raw_counts": dict(sorted(raw_counts.items(), key=lambda kv: -kv[1])),
        "type_counts": {k: type_counts.get(k, 0) for k in END_CONDITION_TYPES},
        "type_ratio": {
            k: (type_counts.get(k, 0) / total if total else None)
            for k in END_CONDITION_TYPES
        },
        "outcome_by_end_condition": {k: dict(v) for k, v in sorted(cross.items())},
        "unmapped_values": unmapped,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_outputs(report: dict[str, object], records: list[dict[str, str]],
                  outdir: Path) -> None:
    """JSON / CSV / unmapped 텍스트를 쓴다."""
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "end_condition_counts.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (outdir / "end_condition_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["end_condition_type", "count", "ratio"])
        total = report["total_matches"]
        for kind in END_CONDITION_TYPES:
            cnt = report["type_counts"][kind]
            ratio = f"{cnt / total:.4f}" if total else "N/A"
            writer.writerow([kind, cnt, ratio])
        writer.writerow([])
        writer.writerow(["end_condition_raw", "count", "ratio"])
        for raw, cnt in report["raw_counts"].items():
            ratio = f"{cnt / total:.4f}" if total else "N/A"
            writer.writerow([raw, cnt, ratio])

    unmapped = report["unmapped_values"]
    text = ("\n".join(unmapped) + "\n") if unmapped else "(매핑되지 않은 값 없음)\n"
    (outdir / "unmapped_end_conditions.txt").write_text(text, encoding="utf-8")

    with (outdir / "end_condition_matches.csv").open("w", newline="", encoding="utf-8") as fh:
        if records:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def main() -> int:
    ap = argparse.ArgumentParser(description="종료 원인 집계")
    ap.add_argument("--logdir", required=True, type=Path, help="로그 디렉터리")
    ap.add_argument("--output", type=Path, default=Path("analysis/end_conditions"))
    args = ap.parse_args()

    records, sources = collect(args.logdir)
    if not records:
        warn(f"집계할 경기를 찾지 못했다: {args.logdir}")
        return 1

    report = build_report(records, args.logdir, sources)
    write_outputs(report, records, args.output)

    total = report["total_matches"]
    print(f"총 경기 수: {total}")
    print(f"사용한 소스: {', '.join(sources)}")
    print("\n정규화 유형별:")
    for kind in END_CONDITION_TYPES:
        cnt = report["type_counts"][kind]
        if cnt:
            print(f"  {kind:<24} {cnt:>4}  ({cnt / total:.1%})")
    print("\n원본 값별:")
    for raw, cnt in report["raw_counts"].items():
        print(f"  {raw:<34} {cnt:>4}")
    print("\n승패 x 종료원인:")
    for outcome, kinds in report["outcome_by_end_condition"].items():
        print(f"  {outcome:<10} {kinds}")
    if report["unmapped_values"]:
        print(f"\n매핑되지 않은 값 {len(report['unmapped_values'])}종 -> "
              f"{args.output / 'unmapped_end_conditions.txt'}")
    else:
        print("\n매핑되지 않은 값 없음")
    print(f"\n출력: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
