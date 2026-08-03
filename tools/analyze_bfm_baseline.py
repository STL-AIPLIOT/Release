# -*- coding: utf-8 -*-
"""BFM 모드 체류 비율 baseline 측정.

입력은 extract_bfm_log.py 가 만든 bfm_timeline.csv 다. BFM 모드는 Python 로그에
없고 BT stdout 에만 있으므로, 먼저 stdout 을 받아 추출해야 한다.

    python run_local_dogfight.py ... > bt_stdout.log 2>&1
    python tools/extract_bfm_log.py --stdout bt_stdout.log --output analysis/bfm
    python tools/analyze_bfm_baseline.py --logdir analysis/bfm --episodes 20

체류시간 계산 규칙
------------------
- 한 구간의 체류시간 = 다음 Enter 로그까지의 간격.
- 각 에피소드의 **마지막 구간은 종료 시각을 알 수 없어** duration 이 비어 있다.
  기본값(--last-segment drop)은 그 구간을 집계에서 제외한다.
  --last-segment median 을 주면 같은 에피소드 다른 구간들의 중앙값으로 대체하고,
  그 사실을 산출물에 기록한다. 0 으로 채우지 않는다.
- 샘플 개수 기준 비율(구간 진입 횟수)과 시간 기준 비율을 모두 낸다.

baseline 과 비교:
    python tools/analyze_bfm_baseline.py --logdir analysis/bfm_new \
        --compare-baseline analysis/baseline/bfm_baseline_20.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import BFM_MODES, normalize_bfm, warn  # noqa: E402


def load_timeline(logdir: Path) -> list[dict[str, object]]:
    """bfm_timeline.csv 를 읽는다. logdir 아래를 재귀 탐색한다."""
    direct = logdir / "bfm_timeline.csv"
    paths = [direct] if direct.exists() else sorted(logdir.rglob("bfm_timeline.csv"))
    if not paths:
        warn(f"bfm_timeline.csv 를 찾지 못했다: {logdir}")
        return []
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                dur = r.get("duration_sec")
                rows.append({
                    "source": str(path),
                    "episode": f"{path.parent.name}#{r.get('episode', '')}",
                    "mode": normalize_bfm(r.get("mode")),
                    "circle": r.get("circle") or None,
                    "start_sec": float(r["start_sec"]) if r.get("start_sec") else None,
                    "duration_sec": float(dur) if dur not in (None, "", "None") else None,
                    "last_segment": str(r.get("last_segment", "")).lower() == "true",
                })
    return rows


def aggregate(rows: list[dict[str, object]], episodes: int,
              last_segment: str) -> dict[str, object]:
    """경기별/누적 체류시간과 비율을 계산한다."""
    by_ep: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_ep[str(r["episode"])].append(r)

    ep_ids = sorted(by_ep)[:episodes]
    if len(ep_ids) < episodes:
        warn(f"요청 {episodes}경기이나 {len(ep_ids)}경기만 있다. 있는 만큼만 집계한다.")

    substituted = 0
    per_episode: list[dict[str, object]] = []
    total_dur: dict[str, float] = {m: 0.0 for m in BFM_MODES}
    total_cnt: dict[str, int] = {m: 0 for m in BFM_MODES}

    for ep in ep_ids:
        segs = by_ep[ep]
        known = [s["duration_sec"] for s in segs
                 if s["duration_sec"] is not None]
        median = statistics.median(known) if known else None

        ep_dur: dict[str, float] = {m: 0.0 for m in BFM_MODES}
        ep_cnt: dict[str, int] = {m: 0 for m in BFM_MODES}
        for s in segs:
            mode = str(s["mode"])
            ep_cnt[mode] = ep_cnt.get(mode, 0) + 1
            total_cnt[mode] = total_cnt.get(mode, 0) + 1
            dur = s["duration_sec"]
            if dur is None:
                if last_segment == "median" and median is not None:
                    dur = median
                    substituted += 1
                else:
                    continue
            ep_dur[mode] = ep_dur.get(mode, 0.0) + float(dur)
            total_dur[mode] = total_dur.get(mode, 0.0) + float(dur)

        ep_total = sum(ep_dur.values())
        per_episode.append({
            "episode_id": ep,
            "total_duration_sec": round(ep_total, 4),
            "segment_count": len(segs),
            "mode_duration_sec": {m: round(ep_dur[m], 4) for m in BFM_MODES},
            "mode_ratio": {m: (ep_dur[m] / ep_total if ep_total else None) for m in BFM_MODES},
            "mode_segment_count": {m: ep_cnt[m] for m in BFM_MODES},
        })

    grand = sum(total_dur.values())
    grand_cnt = sum(total_cnt.values())
    return {
        "baseline_name": "post_bugfix_first_20",
        "episode_count": len(ep_ids),
        "episode_ids": ep_ids,
        "total_duration_sec": round(grand, 4),
        "mode_duration_sec": {m: round(total_dur[m], 4) for m in BFM_MODES},
        "mode_ratio": {m: (total_dur[m] / grand if grand else None) for m in BFM_MODES},
        "mode_segment_count": {m: total_cnt[m] for m in BFM_MODES},
        "mode_segment_ratio": {m: (total_cnt[m] / grand_cnt if grand_cnt else None)
                               for m in BFM_MODES},
        "last_segment_policy": last_segment,
        "last_segment_substituted": substituted,
        "per_episode": per_episode,
    }


def compare(current: dict[str, object], baseline_path: Path) -> dict[str, object] | None:
    """baseline JSON 과 비교한다."""
    try:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"baseline 읽기 실패 {baseline_path}: {exc}")
        return None

    diffs = {}
    for mode in BFM_MODES:
        b = (base.get("mode_ratio") or {}).get(mode)
        c = (current.get("mode_ratio") or {}).get(mode)
        if b is None or c is None:
            diffs[mode] = {"baseline": b, "current": c, "pp_diff": None, "rel_change": None}
            continue
        diffs[mode] = {
            "baseline": b, "current": c,
            "pp_diff": (c - b) * 100.0,
            "rel_change": ((c - b) / b) if b else None,
        }
    scored = [(m, d["pp_diff"]) for m, d in diffs.items() if d["pp_diff"] is not None]
    habfm = diffs.get("HABFM", {}).get("pp_diff")
    return {
        "baseline_file": str(baseline_path),
        "baseline_name": base.get("baseline_name"),
        "per_mode": diffs,
        "most_increased": max(scored, key=lambda kv: kv[1])[0] if scored else None,
        "most_decreased": min(scored, key=lambda kv: kv[1])[0] if scored else None,
        "habfm_bias_vs_baseline": (
            None if habfm is None
            else ("증가" if habfm > 1.0 else ("감소" if habfm < -1.0 else "변화 미미"))
        ),
    }


def write_outputs(result: dict[str, object], outdir: Path, logdir: Path,
                  cmp_result: dict[str, object] | None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["source_logdir"] = str(logdir)
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    if cmp_result:
        payload["comparison"] = cmp_result

    (outdir / "bfm_baseline_20.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (outdir / "bfm_baseline_20.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "mode", "duration_sec", "time_ratio",
                    "segment_count", "segment_ratio"])
        for m in BFM_MODES:
            ratio = result["mode_ratio"][m]
            sratio = result["mode_segment_ratio"][m]
            w.writerow(["TOTAL", m, result["mode_duration_sec"][m],
                        "N/A" if ratio is None else f"{ratio:.6f}",
                        result["mode_segment_count"][m],
                        "N/A" if sratio is None else f"{sratio:.6f}"])
        for ep in result["per_episode"]:
            for m in BFM_MODES:
                r = ep["mode_ratio"][m]
                w.writerow([ep["episode_id"], m, ep["mode_duration_sec"][m],
                            "N/A" if r is None else f"{r:.6f}",
                            ep["mode_segment_count"][m], ""])

    lines = [
        "# BFM 모드 baseline",
        "",
        f"- 소스: `{logdir}`",
        f"- 경기 수: {result['episode_count']}",
        f"- 총 체류시간: {result['total_duration_sec']} s",
        f"- 마지막 구간 처리: `{result['last_segment_policy']}` "
        f"(대체 {result['last_segment_substituted']}건)",
        "",
        "## 모드별",
        "",
        "| 모드 | 체류시간(s) | 시간 비율 | 진입 횟수 | 횟수 비율 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m in BFM_MODES:
        r = result["mode_ratio"][m]
        sr = result["mode_segment_ratio"][m]
        lines.append(
            f"| {m} | {result['mode_duration_sec'][m]:.3f} | "
            f"{'N/A' if r is None else f'{r:.1%}'} | "
            f"{result['mode_segment_count'][m]} | "
            f"{'N/A' if sr is None else f'{sr:.1%}'} |")

    if cmp_result:
        lines += ["", "## baseline 대비", "",
                  "| 모드 | baseline | 현재 | pp 차이 | 상대 변화 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for m in BFM_MODES:
            d = cmp_result["per_mode"][m]
            def f(v, pct=True):
                if v is None:
                    return "N/A"
                return f"{v:.1%}" if pct else f"{v:+.2f}"
            lines.append(f"| {m} | {f(d['baseline'])} | {f(d['current'])} | "
                         f"{f(d['pp_diff'], False)} | {f(d['rel_change'])} |")
        lines += ["",
                  f"- 가장 증가: {cmp_result['most_increased']}",
                  f"- 가장 감소: {cmp_result['most_decreased']}",
                  f"- HABFM 편향: {cmp_result['habfm_bias_vs_baseline']}"]

    lines += ["", "## 사용한 경기 ID", ""] + [f"- `{e}`" for e in result["episode_ids"]]
    (outdir / "bfm_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="BFM 모드 baseline 측정")
    ap.add_argument("--logdir", required=True, type=Path,
                    help="extract_bfm_log.py 산출물(bfm_timeline.csv)이 있는 디렉터리")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--last-segment", choices=["drop", "median"], default="drop",
                    help="에피소드 마지막 구간의 체류시간 처리 방식")
    ap.add_argument("--compare-baseline", type=Path)
    ap.add_argument("--output", type=Path, default=Path("analysis/baseline"))
    args = ap.parse_args()

    rows = load_timeline(args.logdir)
    if not rows:
        print("\nBFM 타임라인이 없다. 먼저 stdout 을 받아 추출해야 한다:")
        print("  python run_local_dogfight.py ... > bt_stdout.log 2>&1")
        print("  python tools/extract_bfm_log.py --stdout bt_stdout.log --output analysis/bfm")
        return 1

    result = aggregate(rows, args.episodes, args.last_segment)
    cmp_result = compare(result, args.compare_baseline) if args.compare_baseline else None
    write_outputs(result, args.output, args.logdir, cmp_result)

    print(f"경기 {result['episode_count']}개, 총 체류 {result['total_duration_sec']:.1f}s")
    print(f"마지막 구간 처리: {result['last_segment_policy']} "
          f"(대체 {result['last_segment_substituted']}건)\n")
    for m in BFM_MODES:
        r = result["mode_ratio"][m]
        print(f"  {m:<10} {result['mode_duration_sec'][m]:>9.3f}s  "
              f"{'N/A' if r is None else f'{r:>6.1%}'}  "
              f"진입 {result['mode_segment_count'][m]:>4}회")
    if cmp_result:
        print(f"\nbaseline 대비 HABFM: {cmp_result['habfm_bias_vs_baseline']}")
        print(f"  가장 증가 {cmp_result['most_increased']} / "
              f"가장 감소 {cmp_result['most_decreased']}")
    print(f"\n출력: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
