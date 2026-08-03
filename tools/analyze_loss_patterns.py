# -*- coding: utf-8 -*-
"""패배 직전 N초의 공통 패턴을 뽑는다.

축 세 가지 중 현재 로그로 계산 가능한 것만 실행한다.
  A. BFM 전환 실패  -> 계산 불가. BFM 모드가 어떤 로그에도 없다(2026-08-04 조사).
  B. 에너지 역전    -> 계산 가능. specific energy = g*h + 0.5*v^2 근사.
                       Tacview 에 속도 컬럼이 없어 위경도 차분으로 속도를 추정한다.
  C. 고도 위험      -> 계산 가능.

실행:
    python tools/analyze_loss_patterns.py --logdir <경로> --window-sec 5 \
        --min-altitude 300 --output analysis/loss_patterns
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

from log_analysis import load_episodes, load_track, mean_ignoring_nan, warn  # noqa: E402
from log_analysis.events import (  # noqa: E402
    Thresholds,
    build_sequence,
    detect_altitude_events,
    detect_bfm_events,
    detect_energy_events,
)
from log_analysis.metrics import (  # noqa: E402
    G,
    descent_rate_series,
    specific_energy_series,
    speed_series,
    window_indices,
)

LOSS_OUTCOMES = ("loss", "crash")


def analyze_episode(ep, window_sec: float, th: Thresholds) -> dict[str, object] | None:
    """경기 하나를 분석해 이벤트 시퀀스와 종료 시점 지표를 만든다."""
    if ep.ownship_log is None:
        warn(f"{ep.match_id}: ownship Tacview CSV 경로를 찾지 못했다")
        return None
    own = load_track(ep.ownship_log)
    if len(own) < 2:
        warn(f"{ep.match_id}: ownship 샘플이 부족하다 ({len(own)}행)")
        return None
    tgt = load_track(ep.target_log) if ep.target_log else None

    lo, hi = window_indices(own.time, window_sec, anchor="end")
    covered = own.time[hi - 1] - own.time[lo] if hi > lo else 0.0
    short = covered < window_sec * 0.9

    events = detect_energy_events(own.time, own, tgt or own, lo, hi, th)
    events += detect_altitude_events(own.time, own, lo, hi, th)
    bfm_events, bfm_reason = detect_bfm_events(own.time, own, lo, hi, th)
    events += bfm_events

    speed = speed_series(own.time, own.lat, own.lon, own.alt)
    se = specific_energy_series(own.alt, speed)
    descent = descent_rate_series(own.time, own.alt)

    tgt_se_final = None
    if tgt is not None and len(tgt) >= 2:
        tspeed = speed_series(tgt.time, tgt.lat, tgt.lon, tgt.alt)
        tse = specific_energy_series(tgt.alt, tspeed)
        tgt_se_final = tse[-1] if tse else None

    seq = build_sequence(events, ep.end_condition)
    return {
        "match_id": ep.match_id,
        "iteration": ep.iteration,
        "episode": ep.episode,
        "outcome": ep.outcome,
        "end_condition": ep.end_condition,
        "end_condition_raw": ep.end_condition_raw,
        "total_reward": ep.total_reward,
        "steps": ep.steps,
        "window_sec_requested": window_sec,
        "window_sec_covered": round(covered, 3),
        "window_too_short": short,
        "final_altitude_m": own.alt[-1] if own.alt else None,
        "final_speed_ms": speed[-1] if speed else None,
        "final_specific_energy": se[-1] if se else None,
        "final_energy_diff": (se[-1] - tgt_se_final)
        if (se and tgt_se_final is not None) else None,
        "max_descent_rate_ms": max((d for d in descent[lo:hi] if d == d), default=None),
        "min_altitude_in_window_m": min((a for a in own.alt[lo:hi] if a == a), default=None),
        "mean_speed_in_window_ms": mean_ignoring_nan(speed[lo:hi]),
        "events": [{"code": e.code, "time": round(e.time, 3), "detail": e.detail}
                   for e in sorted(events, key=lambda e: e.time)],
        "sequence": seq,
        "pattern": " -> ".join(seq) if seq else "(이벤트 없음)",
        "bfm_unavailable_reason": bfm_reason,
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    """패턴별로 묶어 통계를 낸다."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in results:
        groups[str(r["pattern"])].append(r)

    total = len(results)
    patterns = []
    for pattern, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        def avg(key: str) -> float | None:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else None

        patterns.append({
            "pattern": pattern,
            "count": len(rows),
            "ratio_of_losses": len(rows) / total if total else None,
            "example_match_ids": [r["match_id"] for r in rows[:3]],
            "example_window": [
                {"match_id": r["match_id"],
                 "window_sec_covered": r["window_sec_covered"]} for r in rows[:3]
            ],
            "end_conditions": dict(Counter(str(r["end_condition"]) for r in rows)),
            "avg_total_reward": avg("total_reward"),
            "avg_final_altitude_m": avg("final_altitude_m"),
            "avg_final_speed_ms": avg("final_speed_ms"),
            "avg_final_energy_diff": avg("final_energy_diff"),
            "avg_max_descent_rate_ms": avg("max_descent_rate_ms"),
        })

    return {
        "loss_episode_count": total,
        "distinct_patterns": len(patterns),
        "patterns": patterns,
        "event_frequency": dict(Counter(
            e["code"] for r in results for e in r["events"]  # type: ignore[index]
        )),
        "short_window_episodes": [r["match_id"] for r in results if r["window_too_short"]],
        "bfm_axis_available": not any(r.get("bfm_unavailable_reason") for r in results),
        "bfm_unavailable_reason": next(
            (r["bfm_unavailable_reason"] for r in results if r.get("bfm_unavailable_reason")),
            None),
        "specific_energy_note":
            f"질량 미상이라 절대 에너지 대신 specific energy = g*h + 0.5*v^2 근사 사용 (g={G}). "
            "Tacview 에 속도 컬럼이 없어 속도는 위경도/고도 차분으로 추정했다.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_outputs(results: list[dict[str, object]], summary: dict[str, object],
                  outdir: Path, window_sec: float, th: Thresholds) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "loss_event_timeline.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "iteration", "episode", "event_index",
                    "event_code", "event_time_sec", "detail"])
        for r in results:
            for i, e in enumerate(r["events"]):  # type: ignore[index]
                w.writerow([r["match_id"], r["iteration"], r["episode"], i,
                            e["code"], e["time"], e["detail"]])

    (outdir / "loss_pattern_summary.json").write_text(
        json.dumps({"thresholds": vars(th), "window_sec": window_sec,
                    "summary": summary, "episodes": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    with (outdir / "loss_pattern_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pattern", "count", "ratio_of_losses", "avg_total_reward",
                    "avg_final_altitude_m", "avg_final_speed_ms",
                    "avg_final_energy_diff", "avg_max_descent_rate_ms",
                    "example_match_ids"])
        for p in summary["patterns"]:  # type: ignore[index]
            def f(key: str) -> str:
                v = p[key]
                return "N/A" if v is None else f"{v:.4f}"
            w.writerow([p["pattern"], p["count"], f("ratio_of_losses"),
                        f("avg_total_reward"), f("avg_final_altitude_m"),
                        f("avg_final_speed_ms"), f("avg_final_energy_diff"),
                        f("avg_max_descent_rate_ms"),
                        " ".join(p["example_match_ids"])])

    lines = [
        "# 패배 직전 구간 패턴 분석",
        "",
        f"- 분석 대상: 패배 경기 {summary['loss_episode_count']}건",
        f"- 구간: 종료 직전 {window_sec}초",
        f"- 임계값: {vars(th)}",
        f"- 발견된 패턴: {summary['distinct_patterns']}종",
        "",
        "## 계산 축",
        "",
        "| 축 | 상태 |",
        "| --- | --- |",
        "| B. 에너지 역전 | 계산됨 |",
        "| C. 고도 위험 | 계산됨 |",
        f"| A. BFM 전환 실패 | {'계산됨' if summary['bfm_axis_available'] else '계산 불가 — ' + str(summary['bfm_unavailable_reason'])} |",
        "",
        f"> {summary['specific_energy_note']}",
        "",
        "## 이벤트 발생 빈도",
        "",
        "| 이벤트 | 횟수 |",
        "| --- | ---: |",
    ]
    for code, cnt in sorted(summary["event_frequency"].items(), key=lambda kv: -kv[1]):  # type: ignore[index]
        lines.append(f"| {code} | {cnt} |")

    lines += ["", "## 반복 패턴", "",
              "| 순위 | 패턴 | 횟수 | 비율 | 평균 보상 | 종료 고도 | 대표 경기 |",
              "| ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for i, p in enumerate(summary["patterns"], 1):  # type: ignore[index]
        def g(key: str, nd: int = 1) -> str:
            v = p[key]
            return "N/A" if v is None else f"{v:.{nd}f}"
        ratio = p["ratio_of_losses"]
        lines.append(
            f"| {i} | {p['pattern']} | {p['count']} | "
            f"{'N/A' if ratio is None else f'{ratio:.1%}'} | {g('avg_total_reward')} | "
            f"{g('avg_final_altitude_m')} m | {', '.join(p['example_match_ids'])} |")

    if summary["distinct_patterns"] < 3:
        lines += ["", "## 패턴이 3개 미만인 이유", "",
                  f"실제로 발견된 패턴은 {summary['distinct_patterns']}종이다. 억지로 늘리지 않았다.",
                  "표본이 부족하거나 종료 원인이 한 가지로 쏠려 있을 때 이렇게 나온다.",
                  "임계값(--descent-rate, --speed-loss 등)을 조정하거나 경기 수를 늘려야 한다."]

    if summary["short_window_episodes"]:
        lines += ["", "## 구간보다 짧은 경기", "",
                  f"{len(summary['short_window_episodes'])}건: "
                  + ", ".join(summary["short_window_episodes"][:10])]

    (outdir / "loss_pattern_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="패배 직전 패턴 분석")
    ap.add_argument("--logdir", required=True, type=Path)
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--min-altitude", type=float, default=300.0,
                    help="환경의 min_altitude 기본값과 맞춘다")
    ap.add_argument("--low-altitude-margin", type=float, default=700.0)
    ap.add_argument("--descent-rate", type=float, default=40.0)
    ap.add_argument("--speed-loss", type=float, default=30.0)
    ap.add_argument("--bfm-stuck-sec", type=float, default=10.0)
    ap.add_argument("--bfm-thrash-count", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("analysis/loss_patterns"))
    args = ap.parse_args()

    th = Thresholds(
        min_altitude_m=args.min_altitude,
        low_altitude_margin_m=args.low_altitude_margin,
        descent_rate_ms=args.descent_rate,
        speed_loss_ms=args.speed_loss,
        bfm_stuck_sec=args.bfm_stuck_sec,
        bfm_thrash_count=args.bfm_thrash_count,
    )

    episodes = load_episodes(args.logdir)
    losses = [e for e in episodes if e.outcome in LOSS_OUTCOMES]
    print(f"전체 경기 {len(episodes)}건 중 패배(loss/crash) {len(losses)}건")
    if not losses:
        warn("패배 경기가 없다. 분석할 대상이 없다.")
        return 1

    results = [r for r in (analyze_episode(e, args.window_sec, th) for e in losses) if r]
    if not results:
        warn("Tacview 궤적을 읽을 수 있는 패배 경기가 없다.")
        return 1

    summary = summarize(results)
    write_outputs(results, summary, args.output, args.window_sec, th)

    print(f"분석된 경기: {len(results)}건")
    print(f"발견된 패턴: {summary['distinct_patterns']}종\n")
    for i, p in enumerate(summary["patterns"][:8], 1):
        ratio = p["ratio_of_losses"]
        print(f"  {i}. {p['pattern']}")
        print(f"     {p['count']}건 ({'N/A' if ratio is None else f'{ratio:.1%}'})"
              f"  대표: {', '.join(p['example_match_ids'])}")
    print(f"\n이벤트 빈도: {summary['event_frequency']}")
    if not summary["bfm_axis_available"]:
        print(f"\nBFM 축 계산 불가: {summary['bfm_unavailable_reason']}")
    print(f"\n출력: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
