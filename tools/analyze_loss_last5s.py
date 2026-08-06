# -*- coding: utf-8 -*-
"""패배 경기의 마지막 5초 패턴 추출 (Gate 3 입력).

무엇을 하는가
-------------
1. replay_index.jsonl + <ts>_summary.json 의 end_condition/outcome 으로 패배 판을 고른다.
2. 각 판의 Tacview CSV(LLA + 자세)를 마지막 5초만 잘라낸다.
3. LLA -> NED 변환 후 **호스트 GeoMathUtil 로 ATA/AA 를 재계산**한다.
   삼각함수를 재구현하지 않는다. student/tests/check_against_release.py 의
   `_load_by_path()` 와 같은 방식으로 실제 모듈을 경로 로드한다.
4. `student/my_observation._repair_degenerate_angle()` 을 반드시 적용한다.
   GeoMathUtil 은 3D 부호를 `sign = np.sign(p_unit_t[2])` 로 잡는데, 동고도·정후방이면
   `np.sign(0.0) == 0.0` 이라 실제 180도인 ATA 가 0도(nose-on)로 붕괴한다.
   보정을 빼면 "패배 직전 ATA 가 0이었다"는 가짜 패턴이 나온다.
5. 판마다 ①에너지 역전 ②고도 위험 ③BFM 전환 실패의 시점·순서를 기록한다.

시간축 주의
-----------
Tacview `Time` 컬럼은 실제 경과 시간이 아니다. 호스트는 env step 1회마다 1행을
쓰면서(`single_agent_env.py:405`) Time 은 내부 스텝 1개분(1/60초)만 올린다(`:996`).
1 env step = step_ratio 내부 스텝(`:289`)이므로 **Time 은 실제보다 step_ratio 배 느리다.**
따라서 "마지막 5초"는 `--step-ratio`(기본 6)로 보정한 실제 시간 기준으로 자른다.

단위 (호스트 확정값)
    거리 meter / ALT meter / KCAS m/s / 각도 signed degree
    ATA 0 = nose-on(적 정면), AA 0 = 적의 six(후방)

BFM 모드는 Python 로그 어디에도 없다(2026-08-04 조사). BT stdout 에만 있으므로
`--bfm-events` 로 tools/extract_bfm_log.py 산출물을 주면 ③을 실제 값으로 판정하고,
없으면 기하 기반 대리지표로만 판정하고 그 사실을 명시한다.

실행
----
    python tools/analyze_loss_last5s.py --logdir artifacts/logs \
        --release-root . --output analysis/loss_last5s
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import Episode, Track, load_episodes, load_summary, load_track, warn  # noqa: E402
from log_analysis.metrics import specific_energy_series, speed_series  # noqa: E402

# 패배로 볼 outcome / end_condition. 환경이 기록한 값이 우선이다.
LOSS_OUTCOMES = {"loss", "crash"}
LOSS_END_SUBSTRINGS = ("ownship altitude below min", "ownship destroyed",
                       "fdm update fail", "two circle headon guard fail")

EV_ENERGY_REVERSAL = "ENERGY_REVERSAL"
EV_ENERGY_DEFICIT = "ENERGY_DEFICIT"
EV_ALTITUDE_RISK = "ALTITUDE_RISK"
EV_HIGH_DESCENT = "HIGH_DESCENT"
EV_BFM_SWITCH_FAIL = "BFM_SWITCH_FAIL"
EV_ATA_DEGENERATE = "ATA_SIGN_DEGENERATE"


# --------------------------------------------------------------------------- 호스트 로드
def _load_by_path(module_name: str, path: Path):
    """경로로 모듈을 로드한다. check_against_release.py 와 같은 방식이다."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_host_modules(release_root: Path):
    """실제 GeoMathUtil / state_schema / my_observation 을 로드한다.

    재구현 금지. 호스트 트리가 없으면 즉시 멈춘다.
    """
    schema_path = release_root / "src" / "dogfight" / "sim" / "state_schema.py"
    geo_path = release_root / "GeoMathUtil.py"
    if not schema_path.exists() or not geo_path.exists():
        raise SystemExit(
            f"Release 본체를 찾지 못했다 (state_schema.py / GeoMathUtil.py): {release_root}\n"
            "  --release-root 로 DogFightEnv/Release 경로를 지정하라.")

    if str(release_root) not in sys.path:
        sys.path.insert(0, str(release_root))

    real_schema = _load_by_path("_real_state_schema", schema_path)
    pkg = types.ModuleType("dogfight")
    pkg.__path__ = []
    sim = types.ModuleType("dogfight.sim")
    sim.__path__ = []
    sys.modules.update({"dogfight": pkg, "dogfight.sim": sim,
                        "dogfight.sim.state_schema": real_schema})

    geomath = _load_by_path("_real_geomath", geo_path)
    from student import my_observation as obs_mod  # noqa: PLC0415

    return real_schema.StateIndex, geomath.GeometryInfo(), obs_mod


# --------------------------------------------------------------------------- 상태 구성
# FighterSim.py:54-56 의 datum. LLA -> NED 변환 기준점이다.
ORIGIN_LAT = 37.91455691666666
ORIGIN_LON = 128.18188127777776
ORIGIN_ALT = 0.0
STATE_LEN = 51


def build_state(state_index, lat, lon, alt_m, roll, pitch, yaw):
    """GeoMathUtil 이 기대하는 상태 배열을 만든다.

    NED 변환은 FighterSim 과 동일하게 pymap3d.geodetic2ned + 같은 datum 을 쓴다.
    각도 계산에 필요한 인덱스(N/E/D/ROLL/PITCH/YAW)만 채운다.
    """
    import numpy as np
    import pymap3d as pm

    n, e, d = pm.geodetic2ned(lat, lon, alt_m, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
    state = np.zeros(STATE_LEN, dtype=float)
    state[state_index.N] = n
    state[state_index.E] = e
    state[state_index.D] = d
    state[state_index.ROLL] = roll
    state[state_index.PITCH] = pitch
    state[state_index.YAW] = yaw
    state[state_index.ALT] = alt_m
    return state


@dataclass
class Sample:
    """한 시점의 재계산 결과."""

    time_sec: float          # Tacview Time 컬럼 원본
    real_time_sec: float     # step_ratio 보정
    ata_deg: float           # 부호 있는 ATA, 보정 적용
    aa_deg: float            # 부호 있는 AA, 보정 적용
    ata_raw_deg: float       # 보정 전(붕괴 여부 확인용)
    distance_m: float
    own_alt_m: float
    tgt_alt_m: float
    own_speed_ms: float
    tgt_speed_ms: float
    own_se: float
    tgt_se: float
    repaired: bool


@dataclass
class Event:
    code: str
    real_time_sec: float
    detail: str


@dataclass
class CaseResult:
    episode: Episode
    samples: list[Sample]
    events: list[Event] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def recompute(own: Track, tgt: Track, state_index, geo, obs_mod,
              step_ratio: float, window_sec: float) -> list[Sample]:
    """마지막 window_sec(실제 시간) 구간의 ATA/AA 를 호스트 GeoMathUtil 로 재계산한다."""
    n = min(len(own), len(tgt))
    if n == 0:
        return []

    own_speed = speed_series(own.time, own.lat, own.lon, own.alt, step_ratio)
    tgt_speed = speed_series(tgt.time, tgt.lat, tgt.lon, tgt.alt, step_ratio)
    own_se = specific_energy_series(own.alt, own_speed)
    tgt_se = specific_energy_series(tgt.alt, tgt_speed)

    # 실제 시간 기준으로 마지막 window_sec 만 남긴다.
    real_end = own.time[n - 1] * step_ratio
    lo = 0
    for i in range(n):
        if own.time[i] * step_ratio >= real_end - window_sec:
            lo = i
            break

    out: list[Sample] = []
    for i in range(lo, n):
        vals = (own.lat[i], own.lon[i], own.alt[i], own.roll[i], own.pitch[i], own.yaw[i],
                tgt.lat[i], tgt.lon[i], tgt.alt[i], tgt.roll[i], tgt.pitch[i], tgt.yaw[i])
        if any(v is None or math.isnan(v) for v in vals):
            continue

        own_state = build_state(state_index, own.lat[i], own.lon[i], own.alt[i],
                                own.roll[i], own.pitch[i], own.yaw[i])
        tgt_state = build_state(state_index, tgt.lat[i], tgt.lon[i], tgt.alt[i],
                                tgt.roll[i], tgt.pitch[i], tgt.yaw[i])

        # 실제 GeoMathUtil. proj=False (3D) 가 update_damage 와 같은 경로다.
        ata_raw = float(geo._get_antenna_train_angle(own_state, tgt_state, False))
        aa_raw = float(geo._get_aspect_angle(own_state, tgt_state, False))
        dist = float(geo._get_distance(own_state, tgt_state))

        # 필수 보정: sign=0 붕괴를 실제 크기로 되돌린다.
        ata = obs_mod._repair_degenerate_angle(
            ata_raw, obs_mod._ata_magnitude_deg, own_state, tgt_state)
        aa = obs_mod._repair_degenerate_angle(
            aa_raw, obs_mod._aa_magnitude_deg, own_state, tgt_state)

        out.append(Sample(
            time_sec=own.time[i],
            real_time_sec=own.time[i] * step_ratio,
            ata_deg=ata, aa_deg=aa, ata_raw_deg=ata_raw,
            distance_m=dist,
            own_alt_m=own.alt[i], tgt_alt_m=tgt.alt[i],
            own_speed_ms=own_speed[i] if i < len(own_speed) else math.nan,
            tgt_speed_ms=tgt_speed[i] if i < len(tgt_speed) else math.nan,
            own_se=own_se[i] if i < len(own_se) else math.nan,
            tgt_se=tgt_se[i] if i < len(tgt_se) else math.nan,
            repaired=abs(ata - ata_raw) > 1e-6,
        ))
    return out


def detect_events(samples: list[Sample], args: argparse.Namespace,
                  bfm_by_time: dict[float, str] | None) -> tuple[list[Event], list[str]]:
    """① 에너지 역전 ② 고도 위험 ③ BFM 전환 실패를 시점과 함께 잡는다."""
    events: list[Event] = []
    notes: list[str] = []
    if not samples:
        return events, ["구간에 유효 샘플이 없다"]

    # --- ① 에너지 역전: 비에너지 우위 부호가 양 -> 음
    prev_sign = None
    worst = 0.0
    for s in samples:
        if math.isnan(s.own_se) or math.isnan(s.tgt_se):
            continue
        diff = s.own_se - s.tgt_se
        worst = min(worst, diff)
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if prev_sign is not None and prev_sign > 0 and sign < 0:
            events.append(Event(EV_ENERGY_REVERSAL, s.real_time_sec,
                                f"비에너지 우위 상실 (차이 {diff:+.0f} J/kg)"))
        prev_sign = sign
    valid = [s.own_se - s.tgt_se for s in samples
             if not (math.isnan(s.own_se) or math.isnan(s.tgt_se))]
    if valid and all(v < 0 for v in valid):
        events.append(Event(EV_ENERGY_DEFICIT, samples[-1].real_time_sec,
                            f"구간 내내 에너지 열세 (평균 {sum(valid)/len(valid):+.0f} J/kg)"))

    # --- ② 고도 위험
    low_line = args.min_altitude_m + args.low_altitude_margin_m
    flagged_low = flagged_desc = False
    for i, s in enumerate(samples):
        if not flagged_low and s.own_alt_m <= low_line:
            flagged_low = True
            events.append(Event(EV_ALTITUDE_RISK, s.real_time_sec,
                                f"고도 {s.own_alt_m:.0f} m (<= {low_line:.0f})"))
        if i > 0:
            dt = s.real_time_sec - samples[i - 1].real_time_sec
            if dt > 0:
                rate = -(s.own_alt_m - samples[i - 1].own_alt_m) / dt
                if not flagged_desc and rate >= args.descent_rate_ms:
                    flagged_desc = True
                    events.append(Event(EV_HIGH_DESCENT, s.real_time_sec,
                                        f"하강률 {rate:.0f} m/s"))

    # --- 부호 붕괴가 실제로 있었는지 (보정 안 했다면 가짜 패턴이 나왔을 구간)
    rep = [s for s in samples if s.repaired]
    if rep:
        events.append(Event(EV_ATA_DEGENERATE, rep[0].real_time_sec,
                            f"sign=0 붕괴 {len(rep)}/{len(samples)} 프레임 보정됨 "
                            f"(보정 전 {rep[0].ata_raw_deg:.1f}도 -> {rep[0].ata_deg:.1f}도)"))

    # --- ③ BFM 전환 실패
    if bfm_by_time:
        modes = []
        for s in samples:
            near = min(bfm_by_time, key=lambda t: abs(t - s.real_time_sec))
            if abs(near - s.real_time_sec) <= 1.0:
                modes.append((s.real_time_sec, bfm_by_time[near]))
        if modes:
            runs = 1
            for i in range(1, len(modes)):
                if modes[i][1] != modes[i - 1][1]:
                    runs += 1
            span = modes[-1][0] - modes[0][0]
            if runs == 1 and span >= args.bfm_stuck_sec:
                events.append(Event(EV_BFM_SWITCH_FAIL, modes[0][0],
                                    f"{modes[0][1]} 에 {span:.1f}초 고착 (전환 0회)"))
        else:
            notes.append("BFM 이벤트가 이 구간과 시간이 맞지 않는다")
    else:
        # BFM 모드가 없으므로 기하 기반 대리지표로만 본다. 대리지표임을 명시한다.
        notes.append(
            "BFM 모드 로그가 없어 ③은 실제 전환 실패가 아니라 기하 대리지표로 판정했다 "
            "(--bfm-events 로 extract_bfm_log.py 산출물을 주면 실제 값으로 바뀐다)")
        defensive = [s for s in samples if abs(s.aa_deg) >= args.defensive_aa_deg
                     and s.distance_m <= args.defensive_distance_m]
        if len(defensive) >= max(2, int(len(samples) * args.defensive_ratio)):
            first = defensive[0]
            events.append(Event(
                EV_BFM_SWITCH_FAIL, first.real_time_sec,
                f"[대리지표] |AA| >= {args.defensive_aa_deg:g}도 & 거리 <= "
                f"{args.defensive_distance_m:g} m 가 {len(defensive)}/{len(samples)} 프레임 "
                f"지속 (방어 기동으로 전환하지 못한 상태)"))

    events.sort(key=lambda e: e.real_time_sec)
    return events, notes


def load_bfm_events(path: Path | None) -> dict[float, str] | None:
    """extract_bfm_log.py 의 bfm_timeline.csv 를 시각->모드 로 읽는다."""
    if path is None:
        return None
    target = path / "bfm_timeline.csv" if path.is_dir() else path
    if not target.exists():
        warn(f"BFM 타임라인이 없다: {target}")
        return None
    out: dict[float, str] = {}
    try:
        for row in csv.DictReader(target.open(encoding="utf-8", newline="")):
            try:
                out[float(row["start_sec"])] = row["mode"]
            except (TypeError, ValueError, KeyError):
                continue
    except OSError as exc:
        warn(f"{target} 읽기 실패: {exc}")
        return None
    return out or None


def is_loss(ep: Episode, summary: dict) -> bool:
    """패배 판정. 환경이 기록한 outcome 이 우선, end_condition 으로 보강한다."""
    outcome = (str(summary.get("outcome", ep.outcome_raw)) or "").strip().lower()
    end = (str(summary.get("end_condition", ep.end_condition_raw)) or "").strip().lower()
    if outcome in ("win", "draw"):
        return False
    if outcome in LOSS_OUTCOMES:
        # crash 는 누가 떨어졌는지로 갈린다.
        if outcome == "crash" and "target" in end:
            return False
        return True
    return any(s in end for s in LOSS_END_SUBSTRINGS)


def main() -> int:
    ap = argparse.ArgumentParser(description="패배 경기 마지막 5초 패턴 추출")
    ap.add_argument("--logdir", required=True, type=Path)
    ap.add_argument("--release-root", type=Path, default=Path("."),
                    help="DogFightEnv/Release 경로 (GeoMathUtil.py 가 있는 곳)")
    ap.add_argument("--output", type=Path, default=Path("analysis/loss_last5s"))
    ap.add_argument("--window-sec", type=float, default=5.0,
                    help="종료 직전 몇 초를 볼지 (실제 시간 기준)")
    ap.add_argument("--step-ratio", type=float, default=6.0,
                    help="Tacview Time 을 실제 시간으로 바꾸는 배율")
    ap.add_argument("--min-altitude-m", type=float, default=300.0)
    ap.add_argument("--low-altitude-margin-m", type=float, default=700.0)
    ap.add_argument("--descent-rate-ms", type=float, default=40.0)
    ap.add_argument("--bfm-stuck-sec", type=float, default=3.0)
    ap.add_argument("--bfm-events", type=Path,
                    help="extract_bfm_log.py 산출 디렉터리 또는 bfm_timeline.csv")
    ap.add_argument("--defensive-aa-deg", type=float, default=120.0,
                    help="대리지표: 이 |AA| 이상이면 내가 적의 전방(불리)")
    ap.add_argument("--defensive-distance-m", type=float, default=1500.0)
    ap.add_argument("--defensive-ratio", type=float, default=0.6)
    ap.add_argument("--run", action="append")
    args = ap.parse_args()

    state_index, geo, obs_mod = load_host_modules(args.release_root.resolve())
    bfm = load_bfm_events(args.bfm_events)

    episodes = load_episodes(args.logdir)
    if args.run:
        episodes = [e for e in episodes if e.run in set(args.run)]

    losses = []
    for ep in episodes:
        summary = load_summary(ep.summary_json) if ep.summary_json else {}
        if is_loss(ep, summary):
            losses.append(ep)

    print(f"전체 {len(episodes)}판 / 패배 {len(losses)}판")
    if len(losses) < 3:
        warn(f"표본 부족: 패배 {len(losses)}판 (3판 미만). 나온 만큼만 잠정 보고한다.")

    results: list[CaseResult] = []
    for ep in losses:
        if not ep.ownship_log or not ep.target_log:
            continue
        own, tgt = load_track(ep.ownship_log), load_track(ep.target_log)
        samples = recompute(own, tgt, state_index, geo, obs_mod,
                            args.step_ratio, args.window_sec)
        events, notes = detect_events(samples, args, bfm)
        results.append(CaseResult(episode=ep, samples=samples, events=events, notes=notes))

    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "loss_last5s_events.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "end_condition", "outcome", "event_order",
                    "code", "real_time_sec", "t_minus_sec", "detail"])
        for r in results:
            end_t = r.samples[-1].real_time_sec if r.samples else 0.0
            for i, e in enumerate(r.events, 1):
                w.writerow([r.episode.match_id, r.episode.end_condition_raw,
                            r.episode.outcome_raw, i, e.code, f"{e.real_time_sec:.3f}",
                            f"{e.real_time_sec - end_t:+.3f}", e.detail])

    with (args.output / "loss_last5s_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "time_sec", "real_time_sec", "ata_deg", "aa_deg",
                    "ata_raw_deg", "repaired", "distance_m", "own_alt_m",
                    "own_speed_ms", "own_se", "tgt_se"])
        for r in results:
            for s in r.samples:
                w.writerow([r.episode.match_id, f"{s.time_sec:.4f}", f"{s.real_time_sec:.4f}",
                            f"{s.ata_deg:.3f}", f"{s.aa_deg:.3f}", f"{s.ata_raw_deg:.3f}",
                            int(s.repaired), f"{s.distance_m:.1f}", f"{s.own_alt_m:.1f}",
                            f"{s.own_speed_ms:.1f}", f"{s.own_se:.0f}", f"{s.tgt_se:.0f}"])

    # 패턴 빈도 집계
    freq: dict[str, int] = {}
    for r in results:
        for code in {e.code for e in r.events}:
            freq[code] = freq.get(code, 0) + 1

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "logdir": str(args.logdir),
        "episode_count": len(episodes),
        "loss_count": len(losses),
        "analyzed": len(results),
        "sample_shortage": len(losses) < 3,
        "window_sec_real": args.window_sec,
        "step_ratio": args.step_ratio,
        "units": {"distance": "meter", "altitude": "meter", "speed": "meter/second",
                  "angle": "signed degree", "specific_energy": "J/kg"},
        "angle_source": "호스트 GeoMathUtil (재구현 아님) + _repair_degenerate_angle 적용",
        "angle_convention": "ATA 0 = nose-on(적 정면), AA 0 = 적의 six(후방)",
        "bfm_source": "extract_bfm_log 산출물" if bfm else "없음 (③은 기하 대리지표)",
        "pattern_frequency": {k: f"{v}/{len(results)}판" for k, v in
                              sorted(freq.items(), key=lambda kv: -kv[1])},
        "cases": [
            {
                "match_id": r.episode.match_id,
                "end_condition": r.episode.end_condition_raw,
                "outcome": r.episode.outcome_raw,
                "sample_count": len(r.samples),
                "repaired_frames": sum(1 for s in r.samples if s.repaired),
                "sequence": [e.code for e in r.events],
                "events": [{"code": e.code, "real_time_sec": round(e.real_time_sec, 3),
                            "detail": e.detail} for e in r.events],
                "notes": r.notes,
            } for r in results
        ],
    }
    (args.output / "loss_last5s_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n분석 {len(results)}판 / 패턴 빈도:")
    for k, v in sorted(freq.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v}/{len(results)}판")
    rep_total = sum(1 for r in results for s in r.samples if s.repaired)
    print(f"\nsign=0 붕괴 보정 프레임: {rep_total}건 "
          f"(보정을 빼면 이 프레임들이 ATA≈0 인 가짜 nose-on 으로 보인다)")
    print(f"출력: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
