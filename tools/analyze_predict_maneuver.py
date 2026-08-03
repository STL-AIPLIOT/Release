# -*- coding: utf-8 -*-
"""PredictManeuver wrap-around 수정 전/후 로그 비교.

무엇을 비교하는가
-----------------
BT 의 PredictManeuverCsvLogger 가 남긴 CSV 두 묶음(before / after)을 읽어

  1. avgDelta 분포와 ±180 부근 이상값
  2. wrap 보정이 실제로 적용됐는지의 직접 증거(rawDelta vs normalizedDelta)
  3. SCISSORS 진입 빈도 (transition 기준, 상태 행 수 아님)

를 비교하고 wraparound_verdict 를 낸다.

로그 만드는 법
--------------
CSV 는 환경변수 PM_CSV_LOG 가 지정된 실행에서만 생긴다.

    # 수정 전 (해당 커밋을 체크아웃해 DLL 을 만든 뒤)
    $env:PM_CSV_LOG   = "logs/predict/before/run01.csv"
    $env:PM_CSV_RUNTYPE = "before"
    python run_local_dogfight.py --ownship-backend rl `
        --ownship-bundle-dir artifacts/models/stil/sac_mlp_obs8_iter400 `
        --observation-module student.my_observation `
        --target-backend bt --target-bt-dll AIP_STIL.dll --save-log

    # 수정 후 (현재 트리로 만든 DLL)
    $env:PM_CSV_LOG   = "logs/predict/after/run01.csv"
    $env:PM_CSV_RUNTYPE = "after"
    python run_local_dogfight.py ...   # 나머지 인자 동일

실행
----
    python tools/analyze_predict_maneuver.py compare \
        --before logs/predict/before \
        --after  logs/predict/after \
        --output analysis/predict_maneuver_comparison

    python tools/analyze_predict_maneuver.py compare \
        --before ... --after ... \
        --outlier-threshold-deg 170 --spike-threshold-deg 90 \
        --min-matches 20

종료 코드
    0  비교 완료 (verdict 는 JSON 에 있다)
    2  입력 로그가 없거나 필수 컬럼이 없어 비교 불가 (INSUFFICIENT_DATA)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import warn  # noqa: E402
from log_analysis.geometry import (  # noqa: E402
    DEFAULT_WEZ_ANGLE_DEG,
    DEFAULT_WEZ_MAX_M,
    DEFAULT_WEZ_MIN_M,
)
from log_analysis.predict_maneuver import (  # noqa: E402
    AvgDeltaStats,
    Outlier,
    PredictLog,
    ScissorsSegment,
    ScissorsStats,
    avg_delta_stats,
    detect_outliers,
    load_predict_log,
    scissors_after_outliers,
    scissors_stats,
    wraparound_evidence,
)

# 판정 기준. 보고서에도 그대로 싣는다.
VERDICT_RULES = {
    "PASS": (
        "after 로그에서 (a) |normalizedDelta| > 180 인 프레임이 0 이고, "
        "(b) |avgDelta| >= outlier_threshold 인 프레임이 0 이며, "
        "(c) normalizedDelta 가 전부 wrap_angle_deg(rawDelta) 와 일치한다."
    ),
    "PARTIAL": (
        "wrap 경계 급등(|normalizedDelta| > 180, rawDelta 미보정)은 사라졌으나 "
        "|avgDelta| 이상값이 남아 있다. 원인이 wrap 이 아닌 다른 곳에 있다는 뜻이다."
    ),
    "FAIL": (
        "after 로그에도 보정되지 않은 값이 남아 있다 "
        "(|normalizedDelta| > 180 또는 normalizedDelta != wrap(rawDelta))."
    ),
    "INSUFFICIENT_DATA": (
        "before/after 중 한쪽 이상에 프레임이 없거나, avgDelta/rawDelta/"
        "normalizedDelta 컬럼이 없어 비교할 수 없다."
    ),
}

# 같은 조건에서 비교했는지 사람이 확인해야 하는 항목. 자동으로 알 수 없다.
FAIRNESS_CHECKLIST = (
    "seed",
    "상대 정책 또는 상대 bundle (--target-backend / --target-bt-dll / --target-bundle-dir)",
    "aircraft 설정 (Release/aircraft, engine)",
    "scenario (initial_scenario: altitude_m / distance_m / heading)",
    "episode 제한시간 (--max-engage-time / --episode-step-limit)",
    "observation 설정 (--observation-mode / --observation-module)",
    "reward 설정 (reward_module / MY_REWARD_CONFIG)",
    "PredictManeuver 설정 (historySize=5, TURN_THRESHOLD_DEG=1.5)",
    "BFM 상태 전환 설정 (Rule.xml, SetBFMMode_* 임계값)",
    "경기 수 (--min-matches 이상)",
    "checkpoint / bundle 디렉터리",
    "코드 commit hash (Release / Behaviortree 각각)",
)


def _f(value: float | None, digits: int = 4) -> str:
    """None 을 0 으로 위장하지 않고 N/A 로 쓴다."""
    return "N/A" if value is None else f"{value:.{digits}f}"


def _diff(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _rel(after: float | None, before: float | None) -> float | None:
    """상대 변화율. before 가 0 이거나 없으면 None."""
    if after is None or before is None or before == 0:
        return None
    return (after - before) / abs(before)


# --------------------------------------------------------------------------- 판정
def decide_verdict(after_log: PredictLog, after_stats: AvgDeltaStats,
                   after_evidence: dict[str, object],
                   before_log: PredictLog,
                   outlier_threshold_deg: float) -> tuple[str, list[str]]:
    """wraparound_verdict 와 그 근거 문장들을 돌려준다."""
    reasons: list[str] = []

    if not after_log.frames:
        return "INSUFFICIENT_DATA", ["after 로그에 프레임이 없다."]
    if after_stats.sample_count == 0:
        return "INSUFFICIENT_DATA", ["after 로그에 avgDelta 값이 없다."]
    if after_evidence["checked_frames"] == 0:
        return "INSUFFICIENT_DATA", [
            "after 로그에 rawDelta/normalizedDelta 쌍이 없어 wrap 보정을 확인할 수 없다."]

    beyond = int(after_evidence["normalized_delta_beyond_180"])
    mismatch = int(after_evidence["normalized_vs_expected_mismatch"])
    magnitude_outliers = after_stats.outlier_counts.get(int(outlier_threshold_deg), 0)

    if beyond > 0 or mismatch > 0:
        reasons.append(
            f"보정되지 않은 프레임이 남아 있다: |normalizedDelta|>180 {beyond}건, "
            f"wrap(rawDelta) 불일치 {mismatch}건.")
        return "FAIL", reasons

    reasons.append(
        f"after 로그 {after_evidence['checked_frames']}프레임 전부 "
        f"normalizedDelta == wrap(rawDelta) 이고 |normalizedDelta| <= 180 이다.")
    reasons.append(
        f"그중 {after_evidence['wrap_corrected_frames']}프레임에서 실제로 wrap 보정이 "
        f"일어났다(rawDelta != normalizedDelta). "
        f"rawDelta 가 ±180 을 넘은 프레임은 {after_evidence['raw_delta_beyond_180']}건.")

    if after_stats.out_of_range_count > 0:
        reasons.append(
            f"avgDelta 가 정의 범위(|v|<=180)를 벗어난 프레임 "
            f"{after_stats.out_of_range_count}건이 남아 있다.")
        return "FAIL", reasons

    if magnitude_outliers > 0:
        reasons.append(
            f"wrap 오류는 사라졌으나 |avgDelta| >= {outlier_threshold_deg:g}도 인 프레임이 "
            f"{magnitude_outliers}건 남아 있다. wrap 이 아닌 다른 원인을 봐야 한다.")
        return "PARTIAL", reasons

    reasons.append(
        f"|avgDelta| >= {outlier_threshold_deg:g}도 인 프레임이 0건이다.")
    if not before_log.frames:
        reasons.append(
            "before 로그가 없어 '수정 전에는 있었다'는 대조는 하지 못했다. "
            "판정은 after 로그 단독 근거에 기반한다.")
    return "PASS", reasons


# --------------------------------------------------------------------------- 출력
def write_avg_delta_statistics(path: Path, groups: dict[str, AvgDeltaStats],
                               evidence: dict[str, dict[str, object]]) -> None:
    thresholds = sorted({int(k) for st in groups.values() for k in st.outlier_counts})
    fields = [
        "group", "sample_count", "nan_count", "min", "max", "mean", "median",
        "stdev", "p95", "p99", "abs_max", "out_of_range_count", "spike_count",
    ] + [f"outlier_ge_{t}" for t in thresholds] + [
        "checked_frames", "raw_delta_beyond_180", "normalized_delta_beyond_180",
        "wrap_corrected_frames", "max_abs_normalized_delta",
        "normalized_vs_expected_mismatch",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, st in groups.items():
            row: dict[str, object] = {"group": name}
            d = st.as_dict()
            for key in ("sample_count", "nan_count", "min", "max", "mean", "median",
                        "stdev", "p95", "p99", "abs_max", "out_of_range_count",
                        "spike_count"):
                row[key] = "" if d[key] is None else d[key]
            for t in thresholds:
                row[f"outlier_ge_{t}"] = st.outlier_counts.get(t, 0)
            ev = evidence.get(name, {})
            for key in ("checked_frames", "raw_delta_beyond_180",
                        "normalized_delta_beyond_180", "wrap_corrected_frames",
                        "max_abs_normalized_delta", "normalized_vs_expected_mismatch"):
                val = ev.get(key)
                row[key] = "" if val is None else val
            w.writerow(row)


def write_outliers(path: Path, groups: dict[str, list[Outlier]]) -> None:
    fields = ["group"] + list(asdict(Outlier(
        match_id="", run_type="", episode=0, frame=None, time_sec=None,
        avg_delta_deg=0.0, raw_delta_deg=None, normalized_delta_deg=None,
        kind="", bfm_before=None, bfm_at="", bfm_after=None,
        scissors_before=None, scissors_at=False, scissors_after=None,
        own_ata_deg=None, target_aa_deg=None, distance_m=None,
        derived_in_wez=None, wez_reason="")).keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, items in groups.items():
            for o in items:
                row = {"group": name}
                for k, v in asdict(o).items():
                    row[k] = "" if v is None else v
                w.writerow(row)


def write_scissors_statistics(path: Path, groups: dict[str, ScissorsStats],
                              after_outlier_entries: dict[str, int]) -> None:
    metrics = [
        "match_count", "matches_with_entry", "entry_match_ratio", "entry_count",
        "entries_per_match_mean", "entries_per_match_median", "total_dwell_sec",
        "dwell_per_match_mean", "dwell_per_entry_mean", "longest_dwell_sec",
        "reentry_count", "open_segment_count", "logged_entry_count",
    ]
    names = list(groups)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + names + ["difference", "relative_change"])
        for m in metrics:
            vals = [groups[n].as_dict()[m] for n in names]
            before = vals[0] if len(vals) > 0 else None
            after = vals[1] if len(vals) > 1 else None
            num = [v if isinstance(v, (int, float)) else None for v in vals]
            d = _diff(num[1] if len(num) > 1 else None, num[0] if num else None)
            r = _rel(num[1] if len(num) > 1 else None, num[0] if num else None)
            w.writerow([m] + ["" if v is None else v for v in vals]
                       + ["" if d is None else d, "" if r is None else r])
        # percentage point 는 비율 지표에만 의미가 있다.
        ratios = [groups[n].entry_match_ratio for n in names]
        pp = None
        if len(ratios) > 1 and ratios[0] is not None and ratios[1] is not None:
            pp = (ratios[1] - ratios[0]) * 100.0
        w.writerow(["entry_match_ratio_percentage_point"] + [""] * len(names)
                   + ["" if pp is None else pp, ""])
        # 이상값 직후 진입 수: 그룹별 값이 있으므로 지표 행과 같은 열 배치를 쓴다.
        w.writerow(["scissors_entry_after_outlier"]
                   + [after_outlier_entries.get(n, 0) for n in names]
                   + [_diff(after_outlier_entries.get(names[1]) if len(names) > 1 else None,
                            after_outlier_entries.get(names[0]) if names else None), ""])
        # 진입 전/후 모드 분포는 dict 이라 JSON 문자열로 같은 열에 넣는다.
        for key in ("entered_from", "exited_to"):
            w.writerow([key]
                       + [json.dumps(groups[n].as_dict()[key], ensure_ascii=False)
                          for n in names] + ["", ""])


def write_episode_comparison(path: Path, logs: dict[str, PredictLog],
                             outliers: dict[str, list[Outlier]],
                             segments: dict[str, list[ScissorsSegment]],
                             outlier_threshold_deg: float) -> None:
    """경기별 한 행. 경기 = runType + episode."""
    fields = ["group", "match_id", "run_type", "episode", "frame_count",
              "time_start_sec", "time_end_sec", "avg_delta_abs_max",
              "outlier_count", "scissors_entry_count", "scissors_dwell_sec",
              "scissors_open_segment", "episode_derived"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for group, log in logs.items():
            for match_id, frames in log.by_match().items():
                times = [f.time_sec for f in frames if f.time_sec is not None]
                avgs = [abs(f.avg_delta_deg) for f in frames
                        if f.avg_delta_deg is not None and f.avg_delta_deg == f.avg_delta_deg]
                segs = [s for s in segments[group] if s.match_id == match_id]
                entries = [s for s in segs if s.entered_from is not None]
                dwell = [s.duration_sec for s in segs if s.duration_sec is not None]
                w.writerow({
                    "group": group,
                    "match_id": match_id,
                    "run_type": frames[0].run_type,
                    "episode": frames[0].episode,
                    "frame_count": len(frames),
                    "time_start_sec": min(times) if times else "",
                    "time_end_sec": max(times) if times else "",
                    "avg_delta_abs_max": max(avgs) if avgs else "",
                    "outlier_count": sum(1 for o in outliers[group] if o.match_id == match_id),
                    "scissors_entry_count": len(entries),
                    "scissors_dwell_sec": sum(dwell) if dwell else "",
                    "scissors_open_segment": sum(1 for s in segs if s.open_ended),
                    "episode_derived": frames[0].episode_derived,
                })


def write_representative_events(path: Path, outliers: dict[str, list[Outlier]],
                                segments: dict[str, list[ScissorsSegment]],
                                limit: int) -> None:
    """보고서에 인용할 대표 이벤트. |avgDelta| 가 큰 순으로 고른다."""
    fields = ["group", "event_type", "match_id", "time_sec", "frame", "detail",
              "bfm_before", "bfm_at", "bfm_after", "own_ata_deg", "target_aa_deg",
              "distance_m", "derived_in_wez"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for group, items in outliers.items():
            top = sorted(items, key=lambda o: abs(o.avg_delta_deg), reverse=True)[:limit]
            for o in top:
                w.writerow({
                    "group": group, "event_type": f"AVG_DELTA_{o.kind.upper()}",
                    "match_id": o.match_id, "time_sec": o.time_sec, "frame": o.frame,
                    "detail": f"avgDelta={o.avg_delta_deg:.3f} raw={o.raw_delta_deg} "
                              f"norm={o.normalized_delta_deg}",
                    "bfm_before": o.bfm_before or "", "bfm_at": o.bfm_at,
                    "bfm_after": o.bfm_after or "",
                    "own_ata_deg": "" if o.own_ata_deg is None else o.own_ata_deg,
                    "target_aa_deg": "" if o.target_aa_deg is None else o.target_aa_deg,
                    "distance_m": "" if o.distance_m is None else o.distance_m,
                    "derived_in_wez": "" if o.derived_in_wez is None else o.derived_in_wez,
                })
        for group, segs in segments.items():
            longest = sorted((s for s in segs if s.duration_sec is not None),
                             key=lambda s: s.duration_sec or 0.0, reverse=True)[:limit]
            for s in longest:
                w.writerow({
                    "group": group, "event_type": "SCISSORS_SEGMENT",
                    "match_id": s.match_id, "time_sec": s.start_sec, "frame": "",
                    "detail": f"dwell={s.duration_sec:.2f}s from={s.entered_from} "
                              f"to={s.exited_to}",
                    "bfm_before": s.entered_from or "", "bfm_at": "SCISSORS",
                    "bfm_after": s.exited_to or "", "own_ata_deg": "",
                    "target_aa_deg": "", "distance_m": "", "derived_in_wez": "",
                })


def write_report(path: Path, payload: dict[str, object], reasons: list[str],
                 args: argparse.Namespace) -> None:
    b, a = payload["before"], payload["after"]
    d = payload["difference"]
    lines = [
        "# PredictManeuver wrap-around 수정 전/후 비교",
        "",
        f"- 생성 시각: {payload['generated_at']}",
        f"- 판정: **{payload['wraparound_verdict']}**",
        f"- 각도 단위: {payload['units']['angle']} / 거리 {payload['units']['distance']}"
        f" / 시간 {payload['units']['time']}",
        "",
        "## 1. 입력",
        "",
        "| 항목 | before | after |",
        "|---|---|---|",
        f"| logdir | `{b['logdir']}` | `{a['logdir']}` |",
        f"| CSV 파일 수 | {b['source_count']} | {a['source_count']} |",
        f"| 프레임 수 | {b['frame_count']} | {a['frame_count']} |",
        f"| 경기 수 | {b['episode_count']} | {a['episode_count']} |",
        f"| episode 파생 여부 | {b['episode_derived']} | {a['episode_derived']} |",
        f"| 없는 컬럼 | {', '.join(b['missing_columns']) or '-'} "
        f"| {', '.join(a['missing_columns']) or '-'} |",
        "",
    ]

    if payload["wraparound_verdict"] == "INSUFFICIENT_DATA":
        lines += [
            "## 2. 판정 근거", "",
            *[f"- {r}" for r in reasons], "",
            "## 3. 로그를 만드는 방법", "",
            "```powershell",
            "# 수정 전: wrap 보정이 없던 커밋으로 DLL 을 만든 뒤",
            '$env:PM_CSV_RUNTYPE = "before"',
            '$env:PM_CSV_LOG     = "logs/predict/before/run01.csv"',
            "python run_local_dogfight.py --ownship-backend rl `",
            "    --ownship-bundle-dir artifacts/models/stil/sac_mlp_obs8_iter400 `",
            "    --observation-module student.my_observation `",
            "    --target-backend bt --target-bt-dll AIP_STIL.dll --save-log",
            "",
            "# 수정 후: 현재 트리로 만든 DLL 로 같은 인자 반복",
            '$env:PM_CSV_RUNTYPE = "after"',
            '$env:PM_CSV_LOG     = "logs/predict/after/run01.csv"',
            "python run_local_dogfight.py ...",
            "```",
            "",
            f"권장 경기 수: 그룹당 최소 {args.min_matches}경기 "
            "(`--min-matches` 로 조정).",
            "",
            "## 4. 같은 조건인지 확인할 항목", "",
            *[f"- [ ] {item}" for item in FAIRNESS_CHECKLIST],
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "## 2. avgDelta",
        "",
        "| 지표 | before | after | 차이 |",
        "|---|---:|---:|---:|",
        f"| 표본 수 | {b['avg_delta']['sample_count']} | {a['avg_delta']['sample_count']} "
        f"| {_diff(a['avg_delta']['sample_count'], b['avg_delta']['sample_count'])} |",
        f"| 최소 | {_f(b['avg_delta']['min'])} | {_f(a['avg_delta']['min'])} "
        f"| {_f(_diff(a['avg_delta']['min'], b['avg_delta']['min']))} |",
        f"| 최대 | {_f(b['avg_delta']['max'])} | {_f(a['avg_delta']['max'])} "
        f"| {_f(_diff(a['avg_delta']['max'], b['avg_delta']['max']))} |",
        f"| 평균 | {_f(b['avg_delta']['mean'])} | {_f(a['avg_delta']['mean'])} "
        f"| {_f(_diff(a['avg_delta']['mean'], b['avg_delta']['mean']))} |",
        f"| 중앙값 | {_f(b['avg_delta']['median'])} | {_f(a['avg_delta']['median'])} "
        f"| {_f(_diff(a['avg_delta']['median'], b['avg_delta']['median']))} |",
        f"| 표준편차 | {_f(b['avg_delta']['stdev'])} | {_f(a['avg_delta']['stdev'])} "
        f"| {_f(_diff(a['avg_delta']['stdev'], b['avg_delta']['stdev']))} |",
        f"| p95 | {_f(b['avg_delta']['p95'])} | {_f(a['avg_delta']['p95'])} "
        f"| {_f(_diff(a['avg_delta']['p95'], b['avg_delta']['p95']))} |",
        f"| p99 | {_f(b['avg_delta']['p99'])} | {_f(a['avg_delta']['p99'])} "
        f"| {_f(_diff(a['avg_delta']['p99'], b['avg_delta']['p99']))} |",
        f"| 절댓값 최대 | {_f(b['avg_delta']['abs_max'])} | {_f(a['avg_delta']['abs_max'])} "
        f"| {_f(_diff(a['avg_delta']['abs_max'], b['avg_delta']['abs_max']))} |",
        f"| 범위 이탈(\\|v\\|>180) | {b['avg_delta']['out_of_range_count']} "
        f"| {a['avg_delta']['out_of_range_count']} "
        f"| {_diff(a['avg_delta']['out_of_range_count'], b['avg_delta']['out_of_range_count'])} |",
        f"| 급변(>= {args.spike_threshold_deg:g}도) | {b['avg_delta']['spike_count']} "
        f"| {a['avg_delta']['spike_count']} "
        f"| {_diff(a['avg_delta']['spike_count'], b['avg_delta']['spike_count'])} |",
        "",
        "이상값 개수(절댓값 기준):",
        "",
        "| 임계값 | before | after | 차이 |",
        "|---:|---:|---:|---:|",
    ]
    for th in sorted(a["avg_delta"]["outlier_counts"], key=int):
        bv = b["avg_delta"]["outlier_counts"].get(th, 0)
        av = a["avg_delta"]["outlier_counts"][th]
        lines.append(f"| >= {th}도 | {bv} | {av} | {av - bv} |")

    lines += [
        "",
        "## 3. wrap 보정 직접 증거",
        "",
        "| 지표 | before | after |",
        "|---|---:|---:|",
        f"| 확인 프레임 | {b['wrap_evidence']['checked_frames']} "
        f"| {a['wrap_evidence']['checked_frames']} |",
        f"| rawDelta 가 ±180 초과 | {b['wrap_evidence']['raw_delta_beyond_180']} "
        f"| {a['wrap_evidence']['raw_delta_beyond_180']} |",
        f"| normalizedDelta 가 ±180 초과 | {b['wrap_evidence']['normalized_delta_beyond_180']} "
        f"| {a['wrap_evidence']['normalized_delta_beyond_180']} |",
        f"| 실제로 보정된 프레임 | {b['wrap_evidence']['wrap_corrected_frames']} "
        f"| {a['wrap_evidence']['wrap_corrected_frames']} |",
        f"| wrap(raw) 불일치 | {b['wrap_evidence']['normalized_vs_expected_mismatch']} "
        f"| {a['wrap_evidence']['normalized_vs_expected_mismatch']} |",
        "",
        "## 4. SCISSORS 진입 빈도",
        "",
        "진입은 **상태 행 수가 아니라 비-SCISSORS -> SCISSORS transition** 으로 센다.",
        "",
        "| 지표 | before | after | 차이 | 상대변화 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("match_count", "경기 수"),
        ("matches_with_entry", "진입 경기 수"),
        ("entry_count", "총 진입 수"),
        ("entries_per_match_mean", "경기당 평균 진입"),
        ("entries_per_match_median", "경기당 중앙값"),
        ("total_dwell_sec", "총 체류(초)"),
        ("dwell_per_match_mean", "경기당 평균 체류(초)"),
        ("dwell_per_entry_mean", "1회당 평균 체류(초)"),
        ("longest_dwell_sec", "최장 연속 체류(초)"),
        ("reentry_count", "재진입 수"),
    ):
        bv, av = b["scissors"][key], a["scissors"][key]
        lines.append(
            f"| {label} | {_f(bv, 3) if isinstance(bv, float) else (bv if bv is not None else 'N/A')} "
            f"| {_f(av, 3) if isinstance(av, float) else (av if av is not None else 'N/A')} "
            f"| {_f(_diff(av, bv), 3)} | {_f(_rel(av, bv), 3)} |")
    br, ar = b["scissors"]["entry_match_ratio"], a["scissors"]["entry_match_ratio"]
    pp = None if (br is None or ar is None) else (ar - br) * 100.0
    lines += [
        f"| 진입 경기 비율 | {_f(br, 3)} | {_f(ar, 3)} | {_f(_diff(ar, br), 3)} "
        f"| {_f(_rel(ar, br), 3)} |",
        f"| 진입 경기 비율 차(pp) | | | {_f(pp, 2)} | |",
        f"| 이상값 직후 진입 | {b['scissors_entry_after_outlier']} "
        f"| {a['scissors_entry_after_outlier']} "
        f"| {_diff(a['scissors_entry_after_outlier'], b['scissors_entry_after_outlier'])} | |",
        "",
        f"- 진입 직전 모드 분포 — before: {b['scissors']['entered_from']} / "
        f"after: {a['scissors']['entered_from']}",
        f"- 종료 후 전환 모드 분포 — before: {b['scissors']['exited_to']} / "
        f"after: {a['scissors']['exited_to']}",
        "",
        "> **해석 주의.** SCISSORS 진입 빈도의 변화만으로 성능이 좋아졌다고 볼 수 없다. ",
        "> 승률·패배율·타임아웃 비율·WEZ 상태와 함께 읽어야 한다. 이 표는 그 지표들을 ",
        "> 담지 않는다(PredictManeuver CSV 에 경기 결과가 없다). ",
        "> `python tools/count_end_conditions.py --logdir <경기 로그>` 로 따로 뽑아 함께 볼 것.",
        "",
        "## 5. 판정",
        "",
        f"**{payload['wraparound_verdict']}**",
        "",
        *[f"- {r}" for r in reasons],
        "",
        "판정 기준:",
        "",
    ]
    for name, rule in VERDICT_RULES.items():
        lines.append(f"- `{name}` — {rule}")
    lines += [
        "",
        "## 6. 같은 조건인지 확인할 항목",
        "",
        "이 도구는 아래를 자동으로 확인할 수 없다. 사람이 대조해야 한다.",
        "",
        *[f"- [ ] {item}" for item in FAIRNESS_CHECKLIST],
        "",
    ]
    if payload["insufficient_matches"]:
        lines += [
            f"> 경고: 그룹 {payload['insufficient_matches']} 의 경기 수가 "
            f"--min-matches({args.min_matches}) 에 못 미친다. 표본이 부족한 비교다.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- CLI
def run_compare(args: argparse.Namespace) -> int:
    outdir: Path = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = tuple(args.outlier_thresholds_deg)
    logs = {
        "before": load_predict_log(args.before, "before"),
        "after": load_predict_log(args.after, "after"),
    }

    stats: dict[str, AvgDeltaStats] = {}
    evidence: dict[str, dict[str, object]] = {}
    outliers: dict[str, list[Outlier]] = {}
    sc_stats: dict[str, ScissorsStats] = {}
    sc_segments: dict[str, list[ScissorsSegment]] = {}
    sc_after_outlier: dict[str, int] = {}

    for name, log in logs.items():
        stats[name] = avg_delta_stats(log.frames, thresholds, args.spike_threshold_deg)
        evidence[name] = wraparound_evidence(log.frames)
        outliers[name] = detect_outliers(
            log.frames, args.outlier_threshold_deg, args.spike_threshold_deg,
            args.wez_angle_deg, args.wez_min_range_m, args.wez_max_range_m)
        sc_stats[name], sc_segments[name] = scissors_stats(log.frames)
        sc_after_outlier[name] = scissors_after_outliers(
            log.frames, outliers[name], args.outlier_followup_frames)

    verdict, reasons = decide_verdict(
        logs["after"], stats["after"], evidence["after"], logs["before"],
        args.outlier_threshold_deg)

    insufficient = [n for n, s in sc_stats.items()
                    if s.match_count and s.match_count < args.min_matches]

    payload: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "units": {"angle": "degree", "distance": "meter", "time": "second"},
        "source_units_note": (
            "PredictManeuver CSV 의 각도는 원본 그대로 degree 다. 변환하지 않았다."),
        "thresholds": {
            "outlier_threshold_deg": args.outlier_threshold_deg,
            "outlier_thresholds_deg": list(thresholds),
            "spike_threshold_deg": args.spike_threshold_deg,
            "outlier_followup_frames": args.outlier_followup_frames,
            "min_matches": args.min_matches,
        },
        "wez_config": {
            "angle_deg": args.wez_angle_deg,
            "min_range_m": args.wez_min_range_m,
            "max_range_m": args.wez_max_range_m,
            "gate": "update_damage 와 동일: min<=d<=max and angle_deg/2 >= |ATA|",
        },
        "insufficient_matches": insufficient,
        "verdict_rules": VERDICT_RULES,
        "verdict_reasons": reasons,
        "fairness_checklist": list(FAIRNESS_CHECKLIST),
        "wraparound_verdict": verdict,
    }

    for name in ("before", "after"):
        log = logs[name]
        payload[name] = {
            "logdir": str(getattr(args, name)),
            "source_count": len(log.sources),
            "sources": [str(p) for p in log.sources],
            "frame_count": len(log.frames),
            "episode_count": sc_stats[name].match_count,
            "episode_derived": log.episode_derived,
            "missing_columns": list(log.missing_columns),
            "avg_delta": stats[name].as_dict(),
            "avg_delta_outlier_count": stats[name].outlier_counts.get(
                int(args.outlier_threshold_deg), 0),
            "wrap_evidence": evidence[name],
            "scissors": sc_stats[name].as_dict(),
            "scissors_entry_count": sc_stats[name].entry_count,
            "scissors_entry_episode_ratio": sc_stats[name].entry_match_ratio,
            "scissors_entry_after_outlier": sc_after_outlier[name],
        }

    before_ratio = sc_stats["before"].entry_match_ratio
    after_ratio = sc_stats["after"].entry_match_ratio
    payload["difference"] = {
        "avg_delta_outlier_count": _diff(
            payload["after"]["avg_delta_outlier_count"],
            payload["before"]["avg_delta_outlier_count"]),
        "avg_delta_abs_max": _diff(stats["after"].abs_max, stats["before"].abs_max),
        "scissors_entry_count": _diff(sc_stats["after"].entry_count,
                                      sc_stats["before"].entry_count),
        "scissors_entry_episode_ratio_percentage_point": (
            None if (before_ratio is None or after_ratio is None)
            else (after_ratio - before_ratio) * 100.0),
    }

    (outdir / "predict_maneuver_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_avg_delta_statistics(outdir / "avg_delta_statistics.csv", stats, evidence)
    write_outliers(outdir / "avg_delta_outliers.csv", outliers)
    write_scissors_statistics(outdir / "scissors_entry_statistics.csv",
                              sc_stats, sc_after_outlier)
    write_episode_comparison(outdir / "episode_comparison.csv", logs, outliers,
                             sc_segments, args.outlier_threshold_deg)
    write_representative_events(outdir / "representative_events.csv", outliers,
                                sc_segments, args.representative_limit)
    write_report(outdir / "predict_maneuver_report.md", payload, reasons, args)

    print(f"판정: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print(f"\nbefore: 프레임 {len(logs['before'].frames)} / 경기 {sc_stats['before'].match_count}")
    print(f"after : 프레임 {len(logs['after'].frames)} / 경기 {sc_stats['after'].match_count}")
    print(f"출력: {outdir}")

    if verdict == "INSUFFICIENT_DATA":
        warn("비교할 로그가 없다. 보고서의 '로그를 만드는 방법' 절을 따라 실행하라.")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PredictManeuver wrap-around 전/후 비교")
    sub = ap.add_subparsers(dest="command", required=True)

    cmp_ = sub.add_parser("compare", help="before/after 로그 비교")
    cmp_.add_argument("--before", required=True, type=Path,
                      help="수정 전 PredictManeuver CSV 파일 또는 디렉터리")
    cmp_.add_argument("--after", required=True, type=Path,
                      help="수정 후 PredictManeuver CSV 파일 또는 디렉터리")
    cmp_.add_argument("--output", type=Path,
                      default=Path("analysis/predict_maneuver_comparison"))
    cmp_.add_argument("--outlier-threshold-deg", type=float, default=170.0,
                      help="이 값 이상의 |avgDelta| 를 이상값으로 본다 (기본 170)")
    cmp_.add_argument("--outlier-thresholds-deg", type=float, nargs="*",
                      default=[170.0, 175.0, 179.0],
                      help="통계표에 넣을 임계값 목록")
    cmp_.add_argument("--spike-threshold-deg", type=float, default=90.0,
                      help="연속 프레임 avgDelta 변화량이 이 값 이상이면 급변 (기본 90)")
    cmp_.add_argument("--outlier-followup-frames", type=int, default=20,
                      help="이상값 직후 SCISSORS 진입을 찾을 프레임 창 (기본 20)")
    cmp_.add_argument("--min-matches", type=int, default=20,
                      help="그룹당 권장 최소 경기 수 (기본 20)")
    cmp_.add_argument("--representative-limit", type=int, default=20,
                      help="representative_events.csv 에 담을 이벤트 수 (그룹·유형별)")
    cmp_.add_argument("--wez-angle-deg", type=float, default=DEFAULT_WEZ_ANGLE_DEG)
    cmp_.add_argument("--wez-min-range-m", type=float, default=DEFAULT_WEZ_MIN_M)
    cmp_.add_argument("--wez-max-range-m", type=float, default=DEFAULT_WEZ_MAX_M)
    cmp_.set_defaults(func=run_compare)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
