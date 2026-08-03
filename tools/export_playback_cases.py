# -*- coding: utf-8 -*-
"""대표 패배 경기를 골라 DogFight Log Playback 뷰어용 데이터로 내보낸다.

무엇을 만드는가
---------------
    analysis/playback_cases/
      manifest.json          케이스 목록 (뷰어가 읽는 진입점)
      case_001/
        playback.json        뷰어가 재생하는 시계열 + 이벤트
        trajectory.csv       같은 내용의 CSV (엑셀/판다스용)
        source_summary.json  원본 <ts>_summary.json + replay_index 행
        source_training_log.csv  해당 iteration 의 training_log.csv 행
        case_report.md       선정 이유와 핵심 timestamp

원본 로그는 통째로 복사하지 않는다. 경로만 기록하고 필요한 구간만 담는다.

원본 필드와 파생 필드
---------------------
Tacview CSV 에 있는 값만 원본이다.

    Time, Longitude, Latitude, Altitude, Roll (deg), Pitch (deg), Yaw (deg), Health

ATA / AA / WEZ / 속도 / 에너지는 어떤 로그에도 없다. 두 기체의 위치·자세에서
다시 계산한 값이며 전부 `derived_` 접두사를 붙인다. playback.json 의
`field_origin` 이 필드별 출처를 명시한다. 원본에 없는 값을 지어내지 않는다.

BFM 모드 / SCISSORS / avgDelta 는 Python 로그에 없다. PredictManeuver CSV
(--predict-log) 를 함께 주면 시간축으로 붙이고, 없으면 해당 필드를 넣지 않고
`unavailable` 에 사유를 남긴다.

실행
----
    python tools/export_playback_cases.py \
        --logdir C:/AIP_LIB/DogFightEnv/Release/artifacts/logs \
        --output analysis/playback_cases \
        --handoff analysis/rl_trajectory_handoff

    # PredictManeuver / BFM 로그가 있으면 유형 A·B 도 선정된다
    python tools/export_playback_cases.py --logdir ... \
        --predict-log logs/predict/after --output analysis/playback_cases
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import (  # noqa: E402
    Episode,
    Track,
    load_episodes,
    load_summary,
    load_track,
    specific_energy_series,
    speed_series,
    warn,
)
from log_analysis.geometry import (  # noqa: E402
    DEFAULT_WEZ_ANGLE_DEG,
    DEFAULT_WEZ_MAX_M,
    DEFAULT_WEZ_MIN_M,
    GeoSample,
    derive_series,
    wez_intervals,
)
from log_analysis.predict_maneuver import (  # noqa: E402
    detect_outliers,
    load_predict_log,
    scissors_segments,
)

SCHEMA_VERSION = "1.0"

# 케이스 유형. 요구된 A/B/C 와 추가 유형.
TYPE_A = "A_AVG_DELTA_OUTLIER"
TYPE_B = "B_SCISSORS"
TYPE_C = "C_WEZ_OR_DEFENSE_FAILURE"
TYPE_LOW_ALT = "D_LOW_ALTITUDE_CRASH"
TYPE_COLLISION = "E_COLLISION"
TYPE_TIMEOUT = "F_TIMEOUT_DRAW"
TYPE_HABFM = "G_HABFM_STUCK"
TYPE_ENERGY = "H_ENERGY_REVERSAL"

FIELD_ORIGIN = {
    "time_sec": ("source (Tacview Time) — **실제 경과 시간이 아니다.** "
                 "호스트가 env step 1회마다 1행을 쓰면서 Time 은 내부 스텝 1개분"
                 "(1/60초)만 올린다. 실제로는 step_ratio 배 더 흘렀다."),
    "derived_real_time_sec": "derived — time_sec * step_ratio. 실제 경과 시간.",
    "own_lat": "source (Tacview Latitude)",
    "own_lon": "source (Tacview Longitude)",
    "own_alt_m": "source (Tacview Altitude, meter)",
    "own_roll_deg": "source (Tacview Roll)",
    "own_pitch_deg": "source (Tacview Pitch)",
    "own_yaw_deg": "source (Tacview Yaw) — playback 의 own heading",
    "own_health": "source (Tacview Health)",
    "target_lat": "source", "target_lon": "source", "target_alt_m": "source",
    "target_roll_deg": "source", "target_pitch_deg": "source",
    "target_yaw_deg": "source — playback 의 target heading",
    "target_health": "source",
    "derived_own_speed_ms": ("derived — 위치 차분 (Tacview 에 속도 컬럼 없음). "
                             "dt 에 step_ratio 보정을 적용했다."),
    "derived_target_speed_ms": "derived — 위치 차분, step_ratio 보정 적용",
    "derived_distance_m": "derived — NED 유클리드 거리",
    "derived_own_ata_deg": "derived — GeoMathUtil._get_antenna_train_angle 재계산",
    "derived_target_ata_deg": "derived — 표적 기준 ATA",
    "derived_target_aa_deg": "derived — GeoMathUtil._get_aspect_angle 재계산 (0=표적의 6시)",
    "derived_own_in_wez": "derived — update_damage 와 동일 게이트 (angle_deg/2)",
    "derived_target_in_wez": "derived — 표적이 나를 조준 중인지",
    "derived_own_specific_energy": "derived — g*h + 0.5*v^2 (질량 미상)",
    "derived_target_specific_energy": "derived",
    "derived_ata_sign_degenerate": "derived — GeoMathUtil 부호 붕괴 구간 표시",
    "bfm_mode": "source (PredictManeuver CSV bfmMode) — 로그를 줬을 때만",
    "avg_delta_deg": "source (PredictManeuver CSV avgDelta) — 로그를 줬을 때만",
    "scissors_active": "derived — bfm_mode == SCISSORS",
}

ANGLE_CONVENTIONS = {
    "own_ata_deg": (
        "내 기수와 표적 LOS 사이의 각. 부호 있음. 0 = 정조준, "
        "양수/음수는 좌우. |ATA| 가 작을수록 유리하다."),
    "target_aa_deg": (
        "표적 기준 aspect angle. **0 = 내가 표적의 6시**, 180 = 표적의 정면. "
        "GeoMathUtil 규약이다. BT 의 MyAspectAngle_Degree 는 반대 규약이니 "
        "섞지 말 것."),
    "in_wez": (
        "update_damage 와 같은 게이트: min_range <= 거리 <= max_range 이고 "
        "angle_deg/2 >= |ATA|. 기본 2.0도이므로 실제 원뿔은 **1도**다."),
}


@dataclass
class CaseCandidate:
    """대표 경기 후보 하나."""

    episode: Episode
    case_type: str
    reason: str
    score: float
    key_timestamp_sec: float | None
    evidence: dict[str, object] = field(default_factory=dict)


def classify_result(ep: Episode) -> str:
    """경기 결과를 LOSS / WIN / DRAW / UNKNOWN 으로 판정한다.

    **환경이 기록한 outcome 이 최우선이다.** 이 값이 그 경기의 공식 판정이다.
    실제 로그(2026-08-04)에서 관측되는 값은 두 가지뿐이다.

        outcome="crash", end_condition="ownship altitude below min"
        outcome="draw",  end_condition="target altitude below min"

    두 번째를 표적 격추(=승리)로 읽으면 안 된다. 환경은 그것을 draw 로 판정한다.
    outcome 이 승패를 직접 말하지 않는 'crash' 일 때만 end_condition 으로
    누가 떨어졌는지를 본다.
    """
    outcome = (ep.outcome or "").lower()
    if outcome in ("win", "loss", "draw"):
        return outcome.upper()
    if outcome == "timeout":
        # 시간 초과. 체력 차가 있으면 그걸 따른다.
        if ep.ownship_health is not None and ep.target_health is not None:
            if ep.ownship_health < ep.target_health:
                return "LOSS"
            if ep.ownship_health > ep.target_health:
                return "WIN"
        return "DRAW"

    raw = (ep.end_condition_raw or "").lower()
    if outcome == "crash":
        if "ownship" in raw:
            return "LOSS"
        if "target" in raw:
            return "WIN"
    if "ownship destroyed" in raw:
        return "LOSS"
    if "target destroyed" in raw:
        return "WIN"
    return "UNKNOWN"


def _nan_to_none(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def build_frames(own: Track, tgt: Track, geo: list[GeoSample],
                 predict_by_time: dict[float, dict[str, object]] | None,
                 step_ratio: float = 1.0) -> list[dict[str, object]]:
    """playback.json 의 frames 배열을 만든다.

    step_ratio 는 Tacview Time 컬럼을 실제 경과 시간으로 바꾸는 배율이다
    (호스트가 env step 1회마다 1행을 쓰면서 Time 은 내부 스텝 1개분만 올린다).
    속도·에너지는 이 보정을 적용해 계산하고, `derived_real_time_sec` 로 보정된
    시각도 함께 담는다. 원본 `time_sec` 은 그대로 둔다.
    """
    own_speed = speed_series(own.time, own.lat, own.lon, own.alt, step_ratio)
    tgt_speed = speed_series(tgt.time, tgt.lat, tgt.lon, tgt.alt, step_ratio)
    own_se = specific_energy_series(own.alt, own_speed)
    tgt_se = specific_energy_series(tgt.alt, tgt_speed)

    n = min(len(geo), len(own.time))
    frames: list[dict[str, object]] = []
    pm_times = sorted(predict_by_time) if predict_by_time else []

    for i in range(n):
        g = geo[i]
        row: dict[str, object] = {
            "time_sec": own.time[i],
            "derived_real_time_sec": own.time[i] * step_ratio,
            "own_lat": _nan_to_none(own.lat[i] if i < len(own.lat) else None),
            "own_lon": _nan_to_none(own.lon[i] if i < len(own.lon) else None),
            "own_alt_m": _nan_to_none(own.alt[i] if i < len(own.alt) else None),
            "own_roll_deg": _nan_to_none(own.roll[i] if i < len(own.roll) else None),
            "own_pitch_deg": _nan_to_none(own.pitch[i] if i < len(own.pitch) else None),
            "own_yaw_deg": _nan_to_none(own.yaw[i] if i < len(own.yaw) else None),
            "own_health": _nan_to_none(own.health[i] if i < len(own.health) else None),
            "target_lat": _nan_to_none(tgt.lat[i] if i < len(tgt.lat) else None),
            "target_lon": _nan_to_none(tgt.lon[i] if i < len(tgt.lon) else None),
            "target_alt_m": _nan_to_none(tgt.alt[i] if i < len(tgt.alt) else None),
            "target_roll_deg": _nan_to_none(tgt.roll[i] if i < len(tgt.roll) else None),
            "target_pitch_deg": _nan_to_none(tgt.pitch[i] if i < len(tgt.pitch) else None),
            "target_yaw_deg": _nan_to_none(tgt.yaw[i] if i < len(tgt.yaw) else None),
            "target_health": _nan_to_none(tgt.health[i] if i < len(tgt.health) else None),
            "derived_own_speed_ms": _nan_to_none(own_speed[i] if i < len(own_speed) else None),
            "derived_target_speed_ms": _nan_to_none(tgt_speed[i] if i < len(tgt_speed) else None),
            "derived_distance_m": g.distance_m,
            "derived_own_ata_deg": g.own_ata_deg,
            "derived_target_ata_deg": g.target_ata_deg,
            "derived_target_aa_deg": g.target_aa_deg,
            "derived_own_in_wez": g.own_in_wez,
            "derived_target_in_wez": g.target_in_wez,
            "derived_own_specific_energy": _nan_to_none(own_se[i] if i < len(own_se) else None),
            "derived_target_specific_energy": _nan_to_none(tgt_se[i] if i < len(tgt_se) else None),
            "derived_ata_sign_degenerate": g.ata_sign_degenerate,
        }
        if pm_times:
            # 가장 가까운 PredictManeuver 프레임을 붙인다(시각 기준).
            t = own.time[i]
            nearest = min(pm_times, key=lambda x: abs(x - t))
            if abs(nearest - t) <= 0.5:
                row.update(predict_by_time[nearest])
        frames.append(row)
    return frames


def build_events(ep: Episode, frames: list[dict[str, object]],
                 geo: list[GeoSample]) -> list[dict[str, object]]:
    """타임라인 marker 로 쓸 이벤트. 계산 가능한 것만 만든다."""
    events: list[dict[str, object]] = []

    for side in ("own", "target"):
        for start, end in wez_intervals(geo, side):
            events.append({
                "time_sec": start,
                "end_sec": end,
                "type": f"WEZ_ENTER_{side.upper()}",
                "detail": (f"{'내가 표적을' if side == 'own' else '표적이 나를'} WEZ 안에 "
                           f"{'유지' if end is None else f'{end - start:.2f}s 유지'}"),
            })

    # BFM 전환 / SCISSORS (PredictManeuver CSV 를 붙였을 때만 존재한다)
    prev_mode: str | None = None
    for row in frames:
        mode = row.get("bfm_mode")
        if mode is None:
            continue
        if prev_mode is not None and mode != prev_mode:
            events.append({
                "time_sec": row["time_sec"], "end_sec": None,
                "type": "BFM_TRANSITION",
                "detail": f"{prev_mode} -> {mode}",
            })
            if mode == "SCISSORS":
                events.append({
                    "time_sec": row["time_sec"], "end_sec": None,
                    "type": "SCISSORS_ENTER",
                    "detail": f"{prev_mode} 에서 진입",
                })
        prev_mode = mode

    # 체력 감소 = 피격
    prev_hp: float | None = None
    for row in frames:
        hp = row.get("own_health")
        if hp is None:
            continue
        if prev_hp is not None and hp < prev_hp - 1e-9:
            events.append({
                "time_sec": row["time_sec"], "end_sec": None,
                "type": "OWN_DAMAGE",
                "detail": f"체력 {prev_hp:.4f} -> {hp:.4f}",
            })
        prev_hp = hp

    if frames:
        events.append({
            "time_sec": frames[-1]["time_sec"], "end_sec": None,
            "type": "EPISODE_END",
            "detail": f"{ep.end_condition_raw} (outcome={ep.outcome_raw})",
        })
    events.sort(key=lambda e: (e["time_sec"] if e["time_sec"] is not None else 0.0))
    return events


def score_candidates(episodes: list[Episode], tracks: dict[str, tuple[Track, Track]],
                     geos: dict[str, list[GeoSample]],
                     predict_index: dict[str, dict[str, object]],
                     args: argparse.Namespace) -> list[CaseCandidate]:
    """유형별 후보를 점수와 함께 만든다. 각 유형 안에서 점수가 높은 순으로 고른다."""
    out: list[CaseCandidate] = []

    for ep in episodes:
        result = classify_result(ep)
        pair = tracks.get(ep.match_id)
        geo = geos.get(ep.match_id) or []
        if pair is None:
            continue
        own, _tgt = pair

        # --- 유형 C: WEZ / 방어 실패
        target_wez = wez_intervals(geo, "target")
        target_wez_time = sum((e - s) for s, e in target_wez if e is not None)
        hp_drop = None
        if ep.ownship_health is not None:
            hp_drop = 1.0 - ep.ownship_health
        if result == "LOSS" and (target_wez_time > 0 or (hp_drop or 0) > 1e-6):
            key_t = target_wez[0][0] if target_wez else None
            out.append(CaseCandidate(
                episode=ep, case_type=TYPE_C,
                reason=(f"표적이 나를 WEZ 안에 둔 시간 {target_wez_time:.2f}s, "
                        f"내 체력 손실 {hp_drop:.4f}. WEZ/방어 실패로 끝난 패배."),
                score=target_wez_time * 100.0 + (hp_drop or 0.0),
                key_timestamp_sec=key_t,
                evidence={"target_wez_seconds": target_wez_time,
                          "ownship_health_loss": hp_drop},
            ))

        # --- 유형 D: 저고도 추락
        if result == "LOSS" and "altitude below min" in (ep.end_condition_raw or "").lower():
            alts = [a for a in own.alt if not math.isnan(a)]
            min_alt = min(alts) if alts else None
            key_t = None
            if alts:
                idx = min(range(len(own.alt)),
                          key=lambda i: own.alt[i] if not math.isnan(own.alt[i]) else 1e18)
                key_t = own.time[idx] if idx < len(own.time) else None
            out.append(CaseCandidate(
                episode=ep, case_type=TYPE_LOW_ALT,
                reason=(f"end_condition='{ep.end_condition_raw}'. 최저 고도 "
                        f"{min_alt:.1f} m 로 최소 고도 아래에서 종료."),
                score=-(min_alt if min_alt is not None else 0.0),
                key_timestamp_sec=key_t,
                evidence={"min_altitude_m": min_alt,
                          "ep_min_distance_m": ep.ep_min_distance},
            ))

        # --- 유형 E: 충돌 (최근접 거리가 매우 작음)
        if ep.ep_min_distance is not None and ep.ep_min_distance <= args.collision_distance_m:
            key_t = None
            dists = [(g.time_sec, g.distance_m) for g in geo if g.distance_m is not None]
            if dists:
                key_t = min(dists, key=lambda x: x[1])[0]
            out.append(CaseCandidate(
                episode=ep, case_type=TYPE_COLLISION,
                reason=(f"최근접 거리 {ep.ep_min_distance:.1f} m "
                        f"(<= {args.collision_distance_m:g} m). 충돌에 준하는 근접."),
                score=-ep.ep_min_distance,
                key_timestamp_sec=key_t,
                evidence={"ep_min_distance_m": ep.ep_min_distance},
            ))

        # --- 유형 F: 타임아웃 무승부
        if result == "DRAW" or ep.outcome in ("draw", "timeout"):
            out.append(CaseCandidate(
                episode=ep, case_type=TYPE_TIMEOUT,
                reason=(f"outcome='{ep.outcome_raw}', end_condition="
                        f"'{ep.end_condition_raw}'. 결착 없이 종료."),
                score=float(ep.steps),
                key_timestamp_sec=own.time[-1] if own.time else None,
                evidence={"steps": ep.steps},
            ))

        # --- 유형 H: 에너지 역전
        own_speed = speed_series(own.time, own.lat, own.lon, own.alt, args.step_ratio)
        tgt_track = tracks[ep.match_id][1]
        tgt_speed = speed_series(tgt_track.time, tgt_track.lat, tgt_track.lon,
                                 tgt_track.alt, args.step_ratio)
        own_se = specific_energy_series(own.alt, own_speed)
        tgt_se = specific_energy_series(tgt_track.alt, tgt_speed)
        m = min(len(own_se), len(tgt_se), len(own.time))
        reversal_t = None
        worst = 0.0
        prev_sign = None
        # 속도는 위치 차분으로 추정하므로 앞쪽 몇 샘플은 값이 튄다
        # (speed_series 는 첫 샘플을 두 번째 값으로 복사한다). 그 구간의
        # '에너지 역전'은 기동이 아니라 추정 잡음이므로 제외한다.
        t_start = own.time[0] if own.time else 0.0
        for i in range(m):
            if own.time[i] - t_start < args.energy_warmup_sec:
                continue
            if math.isnan(own_se[i]) or math.isnan(tgt_se[i]):
                continue
            diff = own_se[i] - tgt_se[i]
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if prev_sign is not None and prev_sign > 0 and sign < 0 and reversal_t is None:
                reversal_t = own.time[i]
            prev_sign = sign
            worst = min(worst, diff)
        if result == "LOSS" and reversal_t is not None:
            out.append(CaseCandidate(
                episode=ep, case_type=TYPE_ENERGY,
                reason=(f"t={reversal_t:.2f}s 에 비에너지 우위가 역전됐고 "
                        f"최저 차이 {worst:.0f} J/kg 까지 벌어진 뒤 패배."),
                score=-worst,
                key_timestamp_sec=reversal_t,
                evidence={"reversal_time_sec": reversal_t,
                          "worst_specific_energy_diff": worst},
            ))

        # --- 유형 A / B: PredictManeuver CSV 가 있을 때만
        pm = predict_index.get(ep.match_id)
        if pm:
            if pm.get("outlier_count", 0) > 0:
                out.append(CaseCandidate(
                    episode=ep, case_type=TYPE_A,
                    reason=(f"avgDelta 이상값 {pm['outlier_count']}건 "
                            f"(최대 |avgDelta| {pm['abs_max']:.2f}도)."),
                    score=float(pm["outlier_count"]),
                    key_timestamp_sec=pm.get("first_outlier_time"),
                    evidence=dict(pm),
                ))
            if pm.get("scissors_entries", 0) > 0:
                out.append(CaseCandidate(
                    episode=ep, case_type=TYPE_B,
                    reason=(f"SCISSORS 진입 {pm['scissors_entries']}회, "
                            f"총 체류 {pm.get('scissors_dwell_sec')}초."),
                    score=float(pm["scissors_entries"]),
                    key_timestamp_sec=pm.get("first_scissors_time"),
                    evidence=dict(pm),
                ))
            if pm.get("habfm_dwell_sec", 0) >= args.habfm_stuck_sec:
                out.append(CaseCandidate(
                    episode=ep, case_type=TYPE_HABFM,
                    reason=(f"HABFM 연속 체류 {pm['habfm_dwell_sec']:.1f}초 "
                            f"(>= {args.habfm_stuck_sec:g}초)."),
                    score=float(pm["habfm_dwell_sec"]),
                    key_timestamp_sec=pm.get("first_habfm_time"),
                    evidence=dict(pm),
                ))

    return out


def build_predict_index(path: Path, outlier_threshold_deg: float,
                        spike_threshold_deg: float) -> tuple[dict[str, dict[str, object]], str]:
    """PredictManeuver CSV 를 경기별로 요약한다.

    반환: (match_id -> 요약, 사유). PM CSV 의 경기 ID(runType/epNNN)는
    replay_index 의 경기 ID(run/iterNNNNNN_epNN)와 체계가 다르다. 자동으로
    이어붙이지 않고, 사용자가 --predict-match-map 으로 지정하지 않는 한
    유형 A/B 후보를 만들지 않는다. 없는 대응을 지어내지 않기 위해서다.
    """
    log = load_predict_log(path, "after")
    if not log.frames:
        return {}, f"PredictManeuver CSV 를 읽지 못했다: {path}"
    outliers = detect_outliers(log.frames, outlier_threshold_deg, spike_threshold_deg)
    segments = scissors_segments(log.frames)

    index: dict[str, dict[str, object]] = {}
    for match_id, frames in log.by_match().items():
        o = [x for x in outliers if x.match_id == match_id]
        segs = [s for s in segments if s.match_id == match_id]
        entries = [s for s in segs if s.entered_from is not None]
        dwell = [s.duration_sec for s in segs if s.duration_sec is not None]
        avgs = [abs(f.avg_delta_deg) for f in frames
                if f.avg_delta_deg is not None and f.avg_delta_deg == f.avg_delta_deg]

        habfm_dwell, habfm_first = 0.0, None
        run_start = None
        for i, f in enumerate(frames):
            if f.bfm_mode == "HABFM" and run_start is None:
                run_start = i
            elif f.bfm_mode != "HABFM" and run_start is not None:
                t0, t1 = frames[run_start].time_sec, f.time_sec
                if t0 is not None and t1 is not None and (t1 - t0) > habfm_dwell:
                    habfm_dwell, habfm_first = t1 - t0, t0
                run_start = None

        index[match_id] = {
            "outlier_count": len(o),
            "abs_max": max(avgs) if avgs else 0.0,
            "first_outlier_time": o[0].time_sec if o else None,
            "scissors_entries": len(entries),
            "scissors_dwell_sec": sum(dwell) if dwell else 0.0,
            "first_scissors_time": entries[0].start_sec if entries else None,
            "habfm_dwell_sec": habfm_dwell,
            "first_habfm_time": habfm_first,
        }
    return index, ""


def load_match_map(path: Path | None) -> dict[str, str]:
    """PM CSV 경기 ID -> replay_index 경기 ID 매핑 JSON.

    형식: {"after/ep000": "sac_mlp_obs8_iter400/iter000000_ep00", ...}
    """
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"{path} 읽기 실패: {exc}")
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def write_case(outdir: Path, case_id: str, cand: CaseCandidate,
               own: Track, tgt: Track, geo: list[GeoSample],
               predict_by_time: dict[float, dict[str, object]] | None,
               unavailable: list[str], args: argparse.Namespace) -> dict[str, object]:
    """케이스 디렉터리 하나를 만든다. manifest 에 넣을 항목을 돌려준다."""
    case_dir = outdir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    ep = cand.episode

    frames = build_frames(own, tgt, geo, predict_by_time, args.step_ratio)
    stride = 1
    if args.max_frames and len(frames) > args.max_frames:
        stride = math.ceil(len(frames) / args.max_frames)
        frames = frames[::stride]

    events = build_events(ep, frames, geo)
    result = classify_result(ep)

    playback = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "case_type": cand.case_type,
        "episode_id": ep.match_id,
        "run": ep.run,
        "iteration": ep.iteration,
        "episode": ep.episode,
        "result": result,
        "outcome_raw": ep.outcome_raw,
        "end_condition": ep.end_condition,
        "end_condition_raw": ep.end_condition_raw,
        "reason_selected": cand.reason,
        "key_timestamp_sec": cand.key_timestamp_sec,
        "ownship_health": ep.ownship_health,
        "target_health": ep.target_health,
        "steps": ep.steps,
        "total_reward": ep.total_reward,
        "sample_count": len(frames),
        "sample_stride": stride,
        "duration_sec": (frames[-1]["time_sec"] - frames[0]["time_sec"]) if frames else None,
        "real_duration_sec": ((frames[-1]["time_sec"] - frames[0]["time_sec"]) * args.step_ratio
                              if frames else None),
        "step_ratio": args.step_ratio,
        "time_base_note": (
            "Tacview Time 컬럼은 실제 경과 시간이 아니다. 호스트가 env step 1회마다 "
            "1행을 쓰면서 Time 은 내부 스텝 1개분(1/60초)만 올린다"
            "(single_agent_env.py:405, 289, 996). 실제 경과는 step_ratio 배다. "
            "속도/에너지는 이 보정을 적용해 계산했고, derived_real_time_sec 에 "
            "보정된 시각이 들어 있다. 각도·거리는 시간과 무관해 영향이 없다."),
        "units": {"angle": "degree", "distance": "meter", "altitude": "meter",
                  "speed": "meter/second", "time": "second"},
        "angle_conventions": ANGLE_CONVENTIONS,
        "wez_config": {"angle_deg": args.wez_angle_deg,
                       "min_range_m": args.wez_min_range_m,
                       "max_range_m": args.wez_max_range_m,
                       "note": "실제 원뿔은 angle_deg/2 다 (update_damage 와 동일)"},
        "field_origin": FIELD_ORIGIN,
        "unavailable": unavailable,
        "source_files": {
            "ownship_log": str(ep.ownship_log) if ep.ownship_log else None,
            "target_log": str(ep.target_log) if ep.target_log else None,
            "summary_json": str(ep.summary_json) if ep.summary_json else None,
            "replay_index": str(ep.source_index) if ep.source_index else None,
        },
        "evidence": cand.evidence,
        "events": events,
        "frames": frames,
    }
    (case_dir / "playback.json").write_text(
        json.dumps(playback, ensure_ascii=False, indent=1), encoding="utf-8")

    # trajectory.csv
    if frames:
        with (case_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
            fields = list(frames[0])
            for f in frames:
                for k in f:
                    if k not in fields:
                        fields.append(k)
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for f in frames:
                w.writerow({k: ("" if f.get(k) is None else f.get(k)) for k in fields})

    # source_summary.json — 원본 요약 + replay_index 행
    summary = load_summary(ep.summary_json) if ep.summary_json else {}
    (case_dir / "source_summary.json").write_text(json.dumps({
        "summary_json_path": str(ep.summary_json) if ep.summary_json else None,
        "summary": summary,
        "replay_index_row": {
            "run": ep.run, "iteration": ep.iteration, "episode": ep.episode,
            "steps": ep.steps, "total_reward": ep.total_reward,
            "outcome": ep.outcome_raw, "end_condition": ep.end_condition_raw,
            "ownship_health": ep.ownship_health, "target_health": ep.target_health,
            "ep_min_distance": ep.ep_min_distance,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # source_training_log.csv — 해당 iteration 행만
    tl_written = _write_training_log_row(case_dir, ep, args.logdir)

    # case_report.md
    (case_dir / "case_report.md").write_text(
        _render_case_report(case_id, cand, playback, tl_written), encoding="utf-8")

    return {
        "case_id": case_id,
        "case_type": cand.case_type,
        "episode_id": ep.match_id,
        "result": result,
        "end_condition": ep.end_condition_raw,
        "reason_selected": cand.reason,
        "playback_file": f"{case_id}/playback.json",
        "source_files": [v for v in playback["source_files"].values() if v],
        "important_timestamp_sec": cand.key_timestamp_sec,
        "sample_count": len(frames),
        "sample_stride": stride,
    }


def _write_training_log_row(case_dir: Path, ep: Episode, logdir: Path) -> bool:
    """해당 iteration 의 training_log.csv 행만 옮긴다. 없으면 False."""
    if ep.source_index is None:
        return False
    base = ep.source_index.parent
    candidates = [base / "training_log.csv", base.parent / "training_log.csv"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return False
    try:
        with src.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if str(r.get("iter", "")).strip() == str(ep.iteration)]
            fields = reader.fieldnames or []
    except OSError as exc:
        warn(f"{src} 읽기 실패: {exc}")
        return False
    if not rows:
        return False
    with (case_dir / "source_training_log.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return True


def _render_case_report(case_id: str, cand: CaseCandidate,
                        playback: dict[str, object], tl_written: bool) -> str:
    frames: list[dict[str, object]] = playback["frames"]  # type: ignore[assignment]
    ata = [f["derived_own_ata_deg"] for f in frames if f.get("derived_own_ata_deg") is not None]
    aa = [f["derived_target_aa_deg"] for f in frames if f.get("derived_target_aa_deg") is not None]
    own_wez = sum(1 for f in frames if f.get("derived_own_in_wez"))
    tgt_wez = sum(1 for f in frames if f.get("derived_target_in_wez"))
    degen = sum(1 for f in frames if f.get("derived_ata_sign_degenerate"))

    lines = [
        f"# {case_id} — {cand.case_type}",
        "",
        f"- episode_id: `{playback['episode_id']}`",
        f"- 결과: **{playback['result']}** / end_condition: `{playback['end_condition_raw']}`",
        f"- outcome(원문): `{playback['outcome_raw']}`",
        f"- 핵심 timestamp: {playback['key_timestamp_sec']}",
        f"- 길이: Time 기준 {playback['duration_sec']} 초 / "f"실제 {playback['real_duration_sec']} 초 (step_ratio={playback['step_ratio']}) / "f"샘플 {playback['sample_count']}개"
        f" (stride {playback['sample_stride']})",
        f"- 체력: 아군 {playback['ownship_health']} / 표적 {playback['target_health']}",
        "",
        "## 선정 이유",
        "",
        cand.reason,
        "",
        "## 시계열 요약 (파생값)",
        "",
        f"- Own ATA: 최소 {min(map(abs, ata)):.2f}도 / 최대 {max(map(abs, ata)):.2f}도"
        if ata else "- Own ATA: 계산 불가",
        f"- Target AA: 최소 {min(map(abs, aa)):.2f}도 / 최대 {max(map(abs, aa)):.2f}도"
        if aa else "- Target AA: 계산 불가",
        f"- 내가 WEZ 안이던 프레임: {own_wez} / {len(frames)}",
        f"- 표적이 나를 WEZ 안에 둔 프레임: {tgt_wez} / {len(frames)}",
        f"- GeoMathUtil 부호 붕괴 구간 프레임: {degen} / {len(frames)}"
        " (플랫폼 결함 1 — 이 구간의 ATA 부호는 신뢰할 수 없다)",
        "",
        "## 이벤트",
        "",
        "| 시각(초) | 유형 | 내용 |",
        "|---:|---|---|",
    ]
    for e in playback["events"]:  # type: ignore[union-attr]
        t = e["time_sec"]
        lines.append(f"| {'' if t is None else f'{t:.3f}'} | `{e['type']}` | {e['detail']} |")

    lines += [
        "",
        "## 원본 파일",
        "",
    ]
    for k, v in playback["source_files"].items():  # type: ignore[union-attr]
        lines.append(f"- {k}: `{v}`")
    if not tl_written:
        lines.append("- source_training_log.csv: 해당 iteration 행을 찾지 못해 만들지 않았다")
    if playback["unavailable"]:
        lines += ["", "## 담지 못한 값", ""]
        for u in playback["unavailable"]:  # type: ignore[union-attr]
            lines.append(f"- {u}")
    return "\n".join(lines) + "\n"


def write_handoff(handoff: Path, cases: list[dict[str, object]], playback_dir: Path,
                  args: argparse.Namespace, unavailable: list[str],
                  episodes: list[Episode]) -> None:
    """RL 담당자 공유용 자료. 대용량 원본을 복사하지 않는다."""
    handoff.mkdir(parents=True, exist_ok=True)
    cases_dst = handoff / "playback_cases"
    if cases_dst.exists():
        shutil.rmtree(cases_dst)
    shutil.copytree(playback_dir, cases_dst)

    with (handoff / "representative_episodes.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case_id", "case_type", "episode_id", "result", "end_condition",
            "important_timestamp_sec", "reason_selected", "playback_file"])
        w.writeheader()
        for c in cases:
            w.writerow({k: c.get(k) for k in w.fieldnames})

    with (handoff / "event_timeline.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case_id", "episode_id", "time_sec", "end_sec", "type", "detail"])
        w.writeheader()
        for c in cases:
            pb = json.loads((playback_dir / c["case_id"] / "playback.json")
                            .read_text(encoding="utf-8"))
            for e in pb["events"]:
                w.writerow({"case_id": c["case_id"], "episode_id": pb["episode_id"],
                            "time_sec": e["time_sec"], "end_sec": e.get("end_sec"),
                            "type": e["type"], "detail": e["detail"]})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": cases,
        "source_logdir": str(args.logdir),
        "note": "원본 로그는 복사하지 않았다. source_files 경로를 참조하라.",
    }
    (handoff / "trajectory_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    obs = args.observation_note
    lines = [
        "# RL 담당자용 대표 궤적 전달 자료",
        "",
        f"생성: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. 분석 목적",
        "",
        "PredictManeuver 의 각도 wrap-around 수정 이후 남은 패배 패턴을 확인하고,",
        "보상·관측 설계에 반영할 이벤트를 고르기 위한 자료다. 코드를 실행하지 않아도",
        "대표 경기의 궤적과 핵심 이벤트를 볼 수 있게 정리했다.",
        "",
        "## 2. 사용한 실험과 checkpoint",
        "",
        f"- 로그 위치: `{args.logdir}`",
        f"- 실험(run): {sorted({e.run for e in episodes})}",
        f"- observation 설정: {obs}",
        "",
        "## 3. 대표 경기 목록과 선정 이유",
        "",
        "| case | 유형 | episode_id | 결과 | 핵심 시각(초) | 선정 이유 |",
        "|---|---|---|---|---:|---|",
    ]
    for c in cases:
        t = c["important_timestamp_sec"]
        lines.append(
            f"| {c['case_id']} | `{c['case_type']}` | `{c['episode_id']}` | "
            f"{c['result']} | {'' if t is None else f'{t:.2f}'} | {c['reason_selected']} |")

    lines += [
        "",
        "## 4. 웹 뷰어 실행 방법",
        "",
        "```powershell",
        "# Release 에서 실행. 추가 의존성 없음(표준 라이브러리만 쓴다).",
        f"python tools/dashboard.py --playback-dir {playback_dir} --port 7860",
        "# 브라우저: http://localhost:7860/  -> 상단 'Replay' 탭",
        "```",
        "",
        "학습 지표까지 함께 보려면:",
        "",
        "```powershell",
        "python tools/dashboard.py --logdir artifacts/logs/stil "
        f"--playback-dir {playback_dir} --port 7860",
        "```",
        "",
        "## 5. 각 파일의 역할",
        "",
        "| 파일 | 내용 |",
        "|---|---|",
        "| `trajectory_manifest.json` | 케이스 목록. 뷰어와 스크립트의 진입점 |",
        "| `representative_episodes.csv` | 대표 경기 한 줄 요약 |",
        "| `event_timeline.csv` | 모든 케이스의 이벤트를 한 파일에 모은 것 |",
        "| `playback_cases/<case>/playback.json` | 시계열 + 이벤트 (뷰어가 읽는다) |",
        "| `playback_cases/<case>/trajectory.csv` | 같은 내용의 CSV |",
        "| `playback_cases/<case>/source_summary.json` | 원본 summary + replay_index 행 |",
        "| `playback_cases/<case>/case_report.md` | 선정 이유와 핵심 시각 |",
        "",
        "원본 Tacview CSV 는 복사하지 않았다. 각 케이스의 `source_files` 에 경로가 있다.",
        "",
        "## 6. Own ATA / Target AA / WEZ 해석",
        "",
    ]
    for k, v in ANGLE_CONVENTIONS.items():
        lines.append(f"- **{k}** — {v}")
    lines += [
        "",
        "이 세 값은 **로그에 없어서 다시 계산한 파생값**이다. 계산식은 호스트의",
        "`GeoMathUtil.GeometryInfo` / `single_agent_env.update_damage` 와 같다.",
        "`derived_ata_sign_degenerate` 가 True 인 프레임은 GeoMathUtil 의 부호 규칙이",
        "붕괴하는 구간이라(플랫폼 결함 1) ATA 부호를 믿으면 안 된다.",
        "",
        "## 7. PredictManeuver 이상값 / SCISSORS 관찰",
        "",
    ]
    if unavailable:
        for u in unavailable:
            lines.append(f"- {u}")
    else:
        lines.append("- PredictManeuver CSV 를 붙였다. 각 케이스의 이벤트를 참조하라.")

    lines += [
        "",
        "## 8. 보상 / 관측 설계에 참고할 이벤트",
        "",
        "- `WEZ_ENTER_TARGET` — 표적이 나를 조준한 구간. 이 구간의 진입 조건이",
        "  방어 보상의 1차 후보다.",
        "- `OWN_DAMAGE` — 실제 체력이 깎인 시점. WEZ 보상 가중치를 이 시점 기준으로",
        "  검증할 수 있다.",
        "- `EPISODE_END` — 종료 원인. 현재 로그에서는 고도 하한 위반이 지배적이다.",
        "- `BFM_TRANSITION` / `SCISSORS_ENTER` — PredictManeuver CSV 를 붙였을 때만 존재.",
        "",
        "## 9. 재현 명령",
        "",
        "```powershell",
        "# 대표 경기 데이터 재생성",
        f"python tools/export_playback_cases.py --logdir {args.logdir} \\",
        f"    --output {playback_dir} --handoff {handoff}",
        "```",
        "",
        "> seed: 현재 로그(replay_index.jsonl / summary.json)에는 seed 필드가 없다.",
        "> 정확한 재현이 필요하면 실험 YAML 과 commit hash 를 함께 고정해야 한다.",
        "",
        "## 10. observation 설정",
        "",
        f"{obs}",
        "",
        "```powershell",
        "python tools/check_observation_consistency.py \\",
        "    --config experiments/stil_sac_mlp_obs8_iter400.yaml \\",
        "    --metadata <bundle>/metadata.json \\",
        "    --bundle-weights <bundle>/policy_weights.pkl.gz",
        "```",
        "",
    ]
    (handoff / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="대표 패배 경기 playback 데이터 생성")
    ap.add_argument("--logdir", required=True, type=Path,
                    help="replay_index.jsonl 이 있는 로그 루트")
    ap.add_argument("--output", type=Path, default=Path("analysis/playback_cases"))
    ap.add_argument("--handoff", type=Path,
                    help="RL 담당자 공유용 디렉터리 (지정하면 함께 만든다)")
    ap.add_argument("--predict-log", type=Path,
                    help="PredictManeuver CSV 파일 또는 디렉터리 (유형 A/B/G 용)")
    ap.add_argument("--predict-match-map", type=Path,
                    help="PM 경기 ID -> replay_index 경기 ID 매핑 JSON")
    ap.add_argument("--per-type", type=int, default=1,
                    help="유형별로 고를 경기 수 (기본 1)")
    ap.add_argument("--max-frames", type=int, default=1200,
                    help="playback.json 에 담을 최대 프레임 수 (초과하면 균등 샘플링)")
    ap.add_argument("--collision-distance-m", type=float, default=30.0,
                    help="이 거리 이하 최근접을 충돌로 본다")
    ap.add_argument("--habfm-stuck-sec", type=float, default=10.0)
    ap.add_argument("--step-ratio", type=float, default=6.0,
                    help="Tacview Time 을 실제 경과 시간으로 바꾸는 배율. "
                         "학습 env_config.step_ratio 와 같게 준다(배포 YAML 전부 6). "
                         "호스트가 env step 1회마다 1행을 쓰면서 Time 은 내부 스텝 "
                         "1개분만 올리기 때문이다. 1 을 주면 보정하지 않는다.")
    ap.add_argument("--energy-warmup-sec", type=float, default=1.0,
                    help="에너지 역전 판정에서 앞쪽 이 시간만큼은 무시한다 "
                         "(속도 차분 추정이 안정되기 전 구간)")
    ap.add_argument("--wez-angle-deg", type=float, default=DEFAULT_WEZ_ANGLE_DEG)
    ap.add_argument("--wez-min-range-m", type=float, default=DEFAULT_WEZ_MIN_M)
    ap.add_argument("--wez-max-range-m", type=float, default=DEFAULT_WEZ_MAX_M)
    ap.add_argument("--outlier-threshold-deg", type=float, default=170.0)
    ap.add_argument("--spike-threshold-deg", type=float, default=90.0)
    ap.add_argument("--run", action="append",
                    help="특정 실험만 대상으로 한다 (여러 번 지정 가능)")
    ap.add_argument("--observation-note", default=(
        "observation_module=student.my_observation, mode=stil8, size=8 "
        "(tools/check_observation_consistency.py 로 확인)"))
    args = ap.parse_args()

    episodes = load_episodes(args.logdir)
    if args.run:
        episodes = [e for e in episodes if e.run in set(args.run)]
    if not episodes:
        warn(f"경기를 찾지 못했다: {args.logdir}")
        return 2

    unavailable: list[str] = []
    predict_index: dict[str, dict[str, object]] = {}
    predict_frames_by_match: dict[str, dict[float, dict[str, object]]] = {}

    if args.predict_log is not None:
        raw_index, reason = build_predict_index(
            args.predict_log, args.outlier_threshold_deg, args.spike_threshold_deg)
        if reason:
            unavailable.append(reason)
        match_map = load_match_map(args.predict_match_map)
        if raw_index and not match_map:
            unavailable.append(
                "PredictManeuver CSV 는 읽었지만 경기 ID 매핑(--predict-match-map)이 없어 "
                "RL 경기와 이어붙이지 않았다. PM 경기 ID 는 runType/epNNN, RL 경기 ID 는 "
                "run/iterNNNNNN_epNN 로 체계가 다르다.")
        for pm_id, summary in raw_index.items():
            mapped = match_map.get(pm_id)
            if mapped:
                predict_index[mapped] = summary
        if match_map:
            log = load_predict_log(args.predict_log, "after")
            for pm_id, frames in log.by_match().items():
                mapped = match_map.get(pm_id)
                if not mapped:
                    continue
                predict_frames_by_match[mapped] = {
                    f.time_sec: {"bfm_mode": f.bfm_mode,
                                 "avg_delta_deg": f.avg_delta_deg,
                                 "scissors_active": f.bfm_mode == "SCISSORS"}
                    for f in frames if f.time_sec is not None}
    else:
        unavailable.append(
            "PredictManeuver CSV(--predict-log)를 주지 않아 BFM 모드 / SCISSORS / "
            "avgDelta 를 담지 못했다. 유형 A(avgDelta 이상값)와 유형 B(SCISSORS)는 "
            "선정하지 않았다. 로그를 만들려면 PM_CSV_LOG 를 설정하고 교전을 실행하라.")

    # 궤적 로드
    tracks: dict[str, tuple[Track, Track]] = {}
    geos: dict[str, list[GeoSample]] = {}
    for ep in episodes:
        if not ep.ownship_log or not ep.target_log:
            warn(f"{ep.match_id}: 궤적 로그 경로가 없다")
            continue
        own, tgt = load_track(ep.ownship_log), load_track(ep.target_log)
        if len(own) == 0 or len(tgt) == 0:
            warn(f"{ep.match_id}: 궤적이 비어 있다")
            continue
        tracks[ep.match_id] = (own, tgt)
        geos[ep.match_id] = derive_series(own, tgt, args.wez_angle_deg,
                                          args.wez_min_range_m, args.wez_max_range_m)

    candidates = score_candidates(episodes, tracks, geos, predict_index, args)
    by_type: dict[str, list[CaseCandidate]] = {}
    for c in candidates:
        by_type.setdefault(c.case_type, []).append(c)

    outdir: Path = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    used: set[str] = set()
    index = 0
    for case_type in sorted(by_type):
        chosen = sorted(by_type[case_type], key=lambda c: c.score, reverse=True)
        picked = 0
        for cand in chosen:
            if picked >= args.per_type:
                break
            key = f"{cand.episode.match_id}|{case_type}"
            if key in used:
                continue
            used.add(key)
            index += 1
            case_id = f"case_{index:03d}"
            own, tgt = tracks[cand.episode.match_id]
            cases.append(write_case(
                outdir, case_id, cand, own, tgt, geos[cand.episode.match_id],
                predict_frames_by_match.get(cand.episode.match_id),
                unavailable, args))
            picked += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_logdir": str(args.logdir),
        "episode_count": len(episodes),
        "case_types_selected": sorted(by_type),
        "case_types_unavailable": [t for t in (TYPE_A, TYPE_B, TYPE_HABFM)
                                   if t not in by_type],
        "unavailable_reasons": unavailable,
        "units": {"angle": "degree", "distance": "meter", "time": "second"},
        "angle_conventions": ANGLE_CONVENTIONS,
        "cases": cases,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"경기 {len(episodes)}개 중 케이스 {len(cases)}개 생성")
    for c in cases:
        print(f"  {c['case_id']}  {c['case_type']:26s} {c['episode_id']}  {c['result']}")
    for u in unavailable:
        print(f"  [담지 못함] {u}")
    print(f"출력: {outdir}")

    if args.handoff is not None:
        write_handoff(args.handoff, cases, outdir, args, unavailable, episodes)
        print(f"공유 자료: {args.handoff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
