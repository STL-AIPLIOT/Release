# -*- coding: utf-8 -*-
"""WEZ 밴드 통과 실태 분석 — "사거리엔 들어가는데 왜 못 쏘는가".

배경
----
BT 상대 30판에서 `ep_min_distance` 는 30/30 판 모두 125.5 m 였다. WEZ 최대
사거리(914.4 m) 안쪽이다. 그런데 30/30 판 모두 양측 체력이 1.0 이다.
사거리에는 들어가는데 피해가 0 이라는 뜻이고, 원인은 셋 중 하나다.

    (a) 밴드 안에 있는 시간이 너무 짧다        -> 통과 프레임 수로 본다
    (b) 밴드 안에서 기수가 안 맞는다           -> 그 구간 |ATA| 로 본다
    (c) 최근접이 최소 사거리보다 더 가깝다     -> min_range 미만 프레임으로 본다

이 도구는 셋을 분리해서 센다.

판정 기준은 호스트와 **비트 단위로 같아야 한다.** `single_agent_env.update_damage()`
(`single_agent_env.py:572,577-579`) 의 조건은

    min_range_m <= distance <= max_range_m  and  |ATA| <= angle_deg / 2.0

이고, 각도 게이트가 `angle_deg` 가 아니라 **`angle_deg / 2`** 라는 점이 핵심이다.
기본 설정 `angle_deg: 2.0` 이면 실제 피해 원뿔은 2도가 아니라 **1도**다. 여기에
각도 epsilon 을 더하면 실제로는 맞지 않는 샷을 "명중"으로 세게 되므로 넣지 않는다.

ATA/AA 는 재구현하지 않는다. 호스트 `GeoMathUtil` 을 경로 로드해서 그대로 쓰고,
`my_observation._repair_degenerate_angle()` 을 반드시 적용한다(sign=0 붕괴 보정).
`analyze_loss_last5s.py` 의 로더를 재사용한다.

시간축
------
`run_local_dogfight` 로그는 step_ratio=1(60 Hz)이라 Time 이 실제 시간이다.
학습 로그는 step_ratio=6 이므로 `--step-ratio 6` 을 줘야 한다.

실행
----
    python tools/analyze_wez_window.py --logdir artifacts/logs \
        --release-root . --step-ratio 1 --output analysis/bt30/wez
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import Episode, load_episodes, load_track, warn  # noqa: E402

from analyze_loss_last5s import build_state, load_host_modules  # noqa: E402

# config.py:36-39 의 실제 기본값. 각각 500 ft / 3000 ft.
DEFAULT_MIN_RANGE_M = 152.4
DEFAULT_MAX_RANGE_M = 914.4
DEFAULT_ANGLE_DEG = 2.0


@dataclass
class BandStats:
    """한 교전에서 WEZ 사거리 밴드를 통과한 기록."""

    episode: str
    frames_total: int = 0
    frames_in_range: int = 0        # min_range <= D <= max_range
    frames_too_close: int = 0       # D < min_range (최소 사거리 미만)
    frames_hit: int = 0             # 사거리 + 각도 둘 다 충족 = 실제 피해 조건
    min_distance_m: float = math.inf
    ata_in_range: list[float] = field(default_factory=list)
    best_ata_in_range: float = math.inf   # 밴드 안에서 가장 잘 맞춘 |ATA|
    passes: int = 0                 # 밴드에 들어갔다 나온 횟수
    dwell_frames: list[int] = field(default_factory=list)  # 통과별 체류 프레임
    # replay_index 가 기록한 교전 결과. 재측정 비교에서 같은 파일만 읽으면 되도록 함께 담는다.
    outcome: str = ""
    ownship_health: float | None = None
    target_health: float | None = None


def analyse_episode(ep: Episode, state_index, geo, obs_mod, wez: dict) -> BandStats | None:
    if ep.ownship_log is None or ep.target_log is None:
        return None
    own = load_track(ep.ownship_log)
    tgt = load_track(ep.target_log)
    n = min(len(own), len(tgt))
    if n == 0:
        return None

    lo_r = float(wez["min_range_m"])
    hi_r = float(wez["max_range_m"])
    # update_damage() 와 같은 반각 게이트. epsilon 을 더하지 않는다.
    half_angle = float(wez["angle_deg"]) / 2.0

    st = BandStats(episode=ep.match_id,
                   outcome=ep.outcome or "",
                   ownship_health=ep.ownship_health,
                   target_health=ep.target_health)
    in_band = False
    dwell = 0

    for i in range(n):
        vals = (own.lat[i], own.lon[i], own.alt[i], own.roll[i], own.pitch[i], own.yaw[i],
                tgt.lat[i], tgt.lon[i], tgt.alt[i], tgt.roll[i], tgt.pitch[i], tgt.yaw[i])
        if any(v is None or math.isnan(v) for v in vals):
            continue

        own_state = build_state(state_index, own.lat[i], own.lon[i], own.alt[i],
                                own.roll[i], own.pitch[i], own.yaw[i])
        tgt_state = build_state(state_index, tgt.lat[i], tgt.lon[i], tgt.alt[i],
                                tgt.roll[i], tgt.pitch[i], tgt.yaw[i])

        dist = float(geo._get_distance(own_state, tgt_state))
        if not math.isfinite(dist):
            continue

        st.frames_total += 1
        st.min_distance_m = min(st.min_distance_m, dist)

        if dist < lo_r:
            st.frames_too_close += 1

        hit_range = lo_r <= dist <= hi_r
        if hit_range:
            # proj=False (3D) 가 update_damage 와 같은 경로다.
            ata_raw = float(geo._get_antenna_train_angle(own_state, tgt_state, False))
            ata = obs_mod._repair_degenerate_angle(
                ata_raw, obs_mod._ata_magnitude_deg, own_state, tgt_state)
            if math.isfinite(ata):
                st.frames_in_range += 1
                st.ata_in_range.append(abs(ata))
                st.best_ata_in_range = min(st.best_ata_in_range, abs(ata))
                if abs(ata) <= half_angle:
                    st.frames_hit += 1

        if hit_range and not in_band:
            in_band, dwell = True, 0
        if in_band:
            dwell += 1
            if not hit_range:
                in_band = False
                st.passes += 1
                st.dwell_frames.append(dwell)
    if in_band:
        st.passes += 1
        st.dwell_frames.append(dwell)
    return st


def summarise(stats: list[BandStats], step_ratio: float, wez: dict) -> dict:
    """판별 통계를 합쳐 (a)/(b)/(c) 세 원인 중 무엇인지 가린다."""
    hz = 60.0 / max(step_ratio, 1e-9)   # 로그 1행이 차지하는 실제 시간의 역수

    def med(vals):
        return statistics.median(vals) if vals else None

    outcomes: dict[str, int] = {}
    for s in stats:
        outcomes[s.outcome or "unknown"] = outcomes.get(s.outcome or "unknown", 0) + 1
    # 피해를 준/받은 판. 체력이 1.0 미만이면 맞은 것이다.
    damaged_target = sum(1 for s in stats
                         if s.target_health is not None and s.target_health < 1.0)
    damaged_own = sum(1 for s in stats
                      if s.ownship_health is not None and s.ownship_health < 1.0)

    in_range_eps = [s for s in stats if s.frames_in_range > 0]
    all_ata = [a for s in stats for a in s.ata_in_range]
    dwell = [d for s in stats for d in s.dwell_frames]

    return {
        "wez": wez,
        "half_angle_deg": float(wez["angle_deg"]) / 2.0,
        "step_ratio": step_ratio,
        "episodes": len(stats),
        "episodes_entering_range": len(in_range_eps),
        "episodes_with_hit_frames": sum(1 for s in stats if s.frames_hit > 0),
        "min_distance_m": {
            "median": med([s.min_distance_m for s in stats]),
            "min": min((s.min_distance_m for s in stats), default=None),
            "max": max((s.min_distance_m for s in stats), default=None),
        },
        "frames_in_range_per_episode": {
            "median": med([s.frames_in_range for s in stats]),
            "max": max((s.frames_in_range for s in stats), default=None),
        },
        "dwell_seconds": {
            "median": (med(dwell) / hz) if dwell else None,
            "max": (max(dwell) / hz) if dwell else None,
            "passes_total": sum(s.passes for s in stats),
        },
        "ata_in_range_deg": {
            "median": med(all_ata),
            "best": min(all_ata) if all_ata else None,
            "samples": len(all_ata),
        },
        "frames_too_close_total": sum(s.frames_too_close for s in stats),
        "frames_hit_total": sum(s.frames_hit for s in stats),
        "outcomes": outcomes,
        "episodes_damaging_target": damaged_target,
        "episodes_taking_damage": damaged_own,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="WEZ 밴드 통과 실태 분석")
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--step-ratio", type=float, default=1.0,
                    help="로그 1행 = 몇 sim step. run_local_dogfight 은 1, 학습은 6")
    # analyze_loss_last5s.py 와 같은 의미로 맞춘다: Episode.run 완전 일치, 여러 번 지정 가능.
    ap.add_argument("--run", action="append", help="Episode.run(실험 태그) 필터. 여러 번 지정 가능")
    ap.add_argument("--min-range-m", type=float, default=DEFAULT_MIN_RANGE_M)
    ap.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M)
    ap.add_argument("--angle-deg", type=float, default=DEFAULT_ANGLE_DEG,
                    help="wez.angle_deg. 실제 게이트는 이 값의 절반이다")
    args = ap.parse_args()

    wez = {"min_range_m": args.min_range_m,
           "max_range_m": args.max_range_m,
           "angle_deg": args.angle_deg}

    state_index, geo, obs_mod = load_host_modules(args.release_root.resolve())

    episodes = load_episodes(args.logdir)
    if args.run:
        episodes = [e for e in episodes if e.run in set(args.run)]
    if not episodes:
        warn(f"교전 로그를 찾지 못했다: {args.logdir}")
        return 2

    stats: list[BandStats] = []
    for ep in episodes:
        s = analyse_episode(ep, state_index, geo, obs_mod, wez)
        if s is not None:
            stats.append(s)
    if not stats:
        warn("트랙을 읽을 수 있는 교전이 없다")
        return 2

    summary = summarise(stats, args.step_ratio, wez)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "wez_window.json").write_text(
        json.dumps({"summary": summary,
                    "episodes": [{"episode": s.episode,
                                  "frames_total": s.frames_total,
                                  "frames_in_range": s.frames_in_range,
                                  "frames_too_close": s.frames_too_close,
                                  "frames_hit": s.frames_hit,
                                  "min_distance_m": s.min_distance_m,
                                  "best_ata_in_range_deg": (
                                      None if math.isinf(s.best_ata_in_range)
                                      else s.best_ata_in_range),
                                  "passes": s.passes,
                                  "outcome": s.outcome,
                                  "ownship_health": s.ownship_health,
                                  "target_health": s.target_health}
                                 for s in stats]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    d = summary
    print(f"\n교전 {d['episodes']}판  (step_ratio={args.step_ratio})")
    print(f"  최근접 거리 중앙 {d['min_distance_m']['median']:.1f} m "
          f"(최소 {d['min_distance_m']['min']:.1f} / 최대 {d['min_distance_m']['max']:.1f})")
    print(f"\n(c) 최소 사거리({wez['min_range_m']} m) 미만 프레임: {d['frames_too_close_total']}")
    print(f"(a) 사거리 밴드 진입: {d['episodes_entering_range']}/{d['episodes']}판, "
          f"통과 {d['dwell_seconds']['passes_total']}회")
    if d['dwell_seconds']['median'] is not None:
        print(f"    체류 시간 중앙 {d['dwell_seconds']['median']:.2f}초 "
              f"(최대 {d['dwell_seconds']['max']:.2f}초)")
    print(f"    판당 밴드 내 프레임 중앙 {d['frames_in_range_per_episode']['median']}")
    if d['ata_in_range_deg']['median'] is not None:
        print(f"(b) 밴드 안 |ATA| 중앙 {d['ata_in_range_deg']['median']:.1f}° "
              f"/ 최선 {d['ata_in_range_deg']['best']:.2f}° "
              f"(게이트 {d['half_angle_deg']}°)")
    print(f"\n피해 조건(사거리+각도) 충족 프레임: {d['frames_hit_total']} "
          f"({d['episodes_with_hit_frames']}/{d['episodes']}판)")
    print(f"outcome: {d['outcomes']}")
    print(f"  적기에 피해를 준 판 {d['episodes_damaging_target']}/{d['episodes']}, "
          f"피해를 입은 판 {d['episodes_taking_damage']}/{d['episodes']}")
    print(f"출력: {args.output / 'wez_window.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
