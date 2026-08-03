# -*- coding: utf-8 -*-
"""타임아웃 무승부 경기의 종료 직전 구간 BFM 분포 분석과 수정 전후 비교.

입력은 extract_bfm_log.py 가 만든 bfm_timeline.csv 와 replay_index.jsonl 이다.
BFM 모드는 Python 로그에 없으므로 stdout 추출이 선행되어야 한다.

    python tools/analyze_timeout_habfm.py --logdir <경로> --window-sec 30 \
        --habfm-ratio-threshold 0.7 --habfm-continuous-sec 10

    python tools/analyze_timeout_habfm.py compare --before <전> --after <후> \
        --output analysis/habfm_timeout_comparison

임계값은 전부 CLI 옵션이다. 코드에 고정하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import BFM_MODES, load_episodes, normalize_bfm, warn  # noqa: E402

TIMEOUT_OUTCOMES = ("draw", "timeout")


def load_segments(logdir: Path) -> dict[str, list[dict[str, object]]]:
    """에피소드별 BFM 구간. 없으면 빈 dict."""
    paths = sorted(logdir.rglob("bfm_timeline.csv"))
    if not paths:
        return {}
    by_ep: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                dur = r.get("duration_sec")
                by_ep[f"{path.parent.name}#{r.get('episode', '')}"].append({
                    "start": float(r["start_sec"]) if r.get("start_sec") else None,
                    "end": float(r["end_sec"]) if r.get("end_sec") not in (None, "", "None") else None,
                    "duration": float(dur) if dur not in (None, "", "None") else None,
                    "mode": normalize_bfm(r.get("mode")),
                    "circle": r.get("circle") or None,
                })
    return by_ep


def analyze_window(segments: list[dict[str, object]], window_sec: float,
                   ratio_th: float, cont_th: float) -> dict[str, object]:
    """마지막 window_sec 구간의 BFM 분포와 고착 판정."""
    ends = [s["end"] for s in segments if s["end"] is not None]
    starts = [s["start"] for s in segments if s["start"] is not None]
    if not starts:
        return {"error": "구간 정보 없음"}
    finish = max(ends) if ends else max(starts)
    cutoff = finish - window_sec

    dur: dict[str, float] = {m: 0.0 for m in BFM_MODES}
    order: list[str] = []
    longest_habfm = 0.0
    for s in segments:
        st, en = s["start"], s["end"]
        if st is None:
            continue
        if en is None:
            en = finish            # 마지막 구간은 종료 시각으로 잘라 쓴다
        lo, hi = max(st, cutoff), min(en, finish)
        if hi <= lo:
            continue
        mode = str(s["mode"])
        dur[mode] = dur.get(mode, 0.0) + (hi - lo)
        if not order or order[-1] != mode:
            order.append(mode)
        if mode == "HABFM":
            longest_habfm = max(longest_habfm, hi - lo)

    total = sum(dur.values())
    ratio = {m: (dur[m] / total if total else None) for m in BFM_MODES}
    habfm_ratio = ratio.get("HABFM")
    transitions = max(0, len(order) - 1)

    suspicious = bool(
        (habfm_ratio is not None and habfm_ratio >= ratio_th)
        or longest_habfm >= cont_th
    )
    return {
        "window_sec": window_sec,
        "window_covered_sec": round(min(window_sec, finish - min(starts)), 3),
        "mode_duration_sec": {m: round(dur[m], 4) for m in BFM_MODES},
        "mode_ratio": ratio,
        "habfm_ratio": habfm_ratio,
        "habfm_longest_continuous_sec": round(longest_habfm, 3),
        "mode_transitions": transitions,
        "last_mode": order[-1] if order else None,
        "mode_order": order,
        "habfm_stuck_suspected": suspicious,
    }


def run_analyze(args: argparse.Namespace) -> int:
    episodes = load_episodes(args.logdir)
    timeouts = [e for e in episodes if e.outcome in TIMEOUT_OUTCOMES]
    print(f"전체 {len(episodes)}경기 중 타임아웃/무승부 {len(timeouts)}건")

    segments = load_segments(args.logdir)
    if not segments:
        warn("bfm_timeline.csv 가 없다. BFM 분포를 계산할 수 없다.")
        print("\nBFM 모드는 BT stdout 에만 있다. 먼저 추출해야 한다:")
        print("  python run_local_dogfight.py ... > bt_stdout.log 2>&1")
        print("  python tools/extract_bfm_log.py --stdout bt_stdout.log --output analysis/bfm")
        payload = {
            "logdir": str(args.logdir),
            "timeout_episode_count": len(timeouts),
            "timeout_match_ids": [e.match_id for e in timeouts],
            "bfm_available": False,
            "reason": "bfm_timeline.csv 없음 (BFM 은 BT stdout 에만 기록된다)",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "habfm_timeout_analysis.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n타임아웃 경기 목록만 저장했다: {args.output}")
        return 2

    results = []
    for ep in timeouts:
        key = next((k for k in segments if k.endswith(f"#{ep.episode}")), None)
        if key is None:
            warn(f"{ep.match_id}: BFM 구간을 찾지 못했다")
            continue
        stat = analyze_window(segments[key], args.window_sec,
                              args.habfm_ratio_threshold, args.habfm_continuous_sec)
        stat.update({"match_id": ep.match_id, "outcome": ep.outcome,
                     "end_condition": ep.end_condition})
        results.append(stat)

    args.output.mkdir(parents=True, exist_ok=True)
    suspects = [r for r in results if r.get("habfm_stuck_suspected")]
    payload = {
        "logdir": str(args.logdir),
        "window_sec": args.window_sec,
        "thresholds": {
            "habfm_ratio": args.habfm_ratio_threshold,
            "habfm_continuous_sec": args.habfm_continuous_sec,
        },
        "timeout_episode_count": len(timeouts),
        "analyzed": len(results),
        "habfm_stuck_suspected_count": len(suspects),
        "habfm_stuck_suspected_ids": [r["match_id"] for r in suspects],
        "bfm_available": True,
        "episodes": results,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output / "habfm_timeout_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"분석 {len(results)}건, HABFM 고착 의심 {len(suspects)}건")
    for r in suspects[:10]:
        print(f"  {r['match_id']}  HABFM {r['habfm_ratio']:.1%}  "
              f"연속 {r['habfm_longest_continuous_sec']}s  전환 {r['mode_transitions']}회")
    print(f"\n출력: {args.output}")
    return 0


def _side_stats(logdir: Path, window_sec: float, ratio_th: float,
                cont_th: float) -> dict[str, object]:
    """비교 한쪽의 지표. BFM 없으면 계산 가능한 것만 채운다."""
    episodes = load_episodes(logdir)
    total = len(episodes)
    timeouts = [e for e in episodes if e.outcome in TIMEOUT_OUTCOMES]
    losses = [e for e in episodes if e.outcome in ("loss", "crash")]
    wins = [e for e in episodes if e.outcome == "win"]
    rewards = [e.total_reward for e in episodes]

    out: dict[str, object] = {
        "logdir": str(logdir),
        "match_count": total,
        "timeout_draw_count": len(timeouts),
        "timeout_draw_ratio": (len(timeouts) / total) if total else None,
        "win_rate": (len(wins) / total) if total else None,
        "crash_rate": (len(losses) / total) if total else None,
        "avg_total_reward": (sum(rewards) / total) if total else None,
        "end_condition_counts": dict(Counter(e.end_condition for e in episodes)),
    }

    segments = load_segments(logdir)
    if not segments:
        out.update({
            "bfm_available": False,
            "bfm_unavailable_reason": "bfm_timeline.csv 없음 (BFM 은 BT stdout 에만 있다)",
            "habfm_ratio_mean": None,
            "habfm_longest_continuous_sec_max": None,
            "mode_transitions_mean": None,
            "one_circle_mean_sec": None,
            "two_circle_mean_sec": None,
            "habfm_exit_count": None,
        })
        return out

    stats = []
    for ep in timeouts:
        key = next((k for k in segments if k.endswith(f"#{ep.episode}")), None)
        if key:
            stats.append(analyze_window(segments[key], window_sec, ratio_th, cont_th))
    ratios = [s["habfm_ratio"] for s in stats if s.get("habfm_ratio") is not None]
    longest = [s["habfm_longest_continuous_sec"] for s in stats if s.get("habfm_longest_continuous_sec") is not None]
    trans = [s["mode_transitions"] for s in stats if s.get("mode_transitions") is not None]

    circle_dur: dict[str, list[float]] = {"1C": [], "2C": []}
    exits = 0
    for segs in segments.values():
        prev = None
        for s in segs:
            if s["circle"] in circle_dur and s["duration"] is not None:
                circle_dur[str(s["circle"])].append(float(s["duration"]))
            if prev == "HABFM" and s["mode"] != "HABFM":
                exits += 1
            prev = str(s["mode"])

    out.update({
        "bfm_available": True,
        "habfm_ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
        "habfm_longest_continuous_sec_max": max(longest) if longest else None,
        "mode_transitions_mean": (sum(trans) / len(trans)) if trans else None,
        "one_circle_mean_sec": (sum(circle_dur["1C"]) / len(circle_dur["1C"])) if circle_dur["1C"] else None,
        "two_circle_mean_sec": (sum(circle_dur["2C"]) / len(circle_dur["2C"])) if circle_dur["2C"] else None,
        "habfm_exit_count": exits,
    })
    return out


def run_compare(args: argparse.Namespace) -> int:
    before = _side_stats(args.before, args.window_sec,
                         args.habfm_ratio_threshold, args.habfm_continuous_sec)
    after = _side_stats(args.after, args.window_sec,
                        args.habfm_ratio_threshold, args.habfm_continuous_sec)

    keys = ("match_count", "timeout_draw_count", "timeout_draw_ratio", "win_rate",
            "crash_rate", "avg_total_reward", "habfm_ratio_mean",
            "habfm_longest_continuous_sec_max", "mode_transitions_mean",
            "one_circle_mean_sec", "two_circle_mean_sec", "habfm_exit_count")
    diff = {}
    for k in keys:
        b, a = before.get(k), after.get(k)
        diff[k] = {
            "before": b, "after": a,
            "delta": (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None,
        }

    computable = [k for k in keys if diff[k]["delta"] is not None]
    not_computable = [k for k in keys if diff[k]["delta"] is None]

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "before": before, "after": after, "diff": diff,
        "computable_metrics": computable,
        "not_computable_metrics": not_computable,
        "bfm_available_before": before.get("bfm_available"),
        "bfm_available_after": after.get("bfm_available"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output / "habfm_timeout_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'지표':<34}{'before':>14}{'after':>14}{'delta':>12}")
    print("-" * 74)
    for k in keys:
        d = diff[k]
        def f(v):
            if v is None:
                return "N/A"
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k:<34}{f(d['before']):>14}{f(d['after']):>14}{f(d['delta']):>12}")
    if not_computable:
        print(f"\n계산 불가: {', '.join(not_computable)}")
        if not before.get("bfm_available") or not after.get("bfm_available"):
            print("  사유: BFM 모드가 로그에 없다. extract_bfm_log.py 를 먼저 실행해야 한다.")
    print(f"\n출력: {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="HABFM 타임아웃 분석")
    sub = ap.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--window-sec", type=float, default=30.0)
    common.add_argument("--habfm-ratio-threshold", type=float, default=0.7)
    common.add_argument("--habfm-continuous-sec", type=float, default=10.0)

    a = sub.add_parser("analyze", parents=[common], help="타임아웃 경기 BFM 분포")
    a.add_argument("--logdir", required=True, type=Path)
    a.add_argument("--output", type=Path, default=Path("analysis/habfm_timeout"))

    c = sub.add_parser("compare", parents=[common], help="수정 전후 비교")
    c.add_argument("--before", required=True, type=Path)
    c.add_argument("--after", required=True, type=Path)
    c.add_argument("--output", type=Path, default=Path("analysis/habfm_timeout_comparison"))

    # 서브커맨드 없이 --logdir 만 줘도 analyze 로 동작하게 한다.
    ap.add_argument("--logdir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("analysis/habfm_timeout"))
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--habfm-ratio-threshold", type=float, default=0.7)
    ap.add_argument("--habfm-continuous-sec", type=float, default=10.0)

    args = ap.parse_args()
    if args.command == "compare":
        return run_compare(args)
    if args.logdir is None:
        ap.error("--logdir 이 필요하다 (또는 compare 서브커맨드를 쓴다)")
    return run_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
