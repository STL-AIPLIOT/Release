# -*- coding: utf-8 -*-
"""두 BT 교전 배치를 비교해 판정 표를 만든다.

무엇을 하는가
-------------
`analyze_wez_window.py` 가 낸 `wez_window.json` 두 개를 읽어, 보상/정책 변경 전후를
같은 지표로 나란히 놓는다. 판정 기준은 코드에 박아 둔다 — 결과를 보고 나서 기준을
고르면 아무 결론이나 나오기 때문이다.

지표별 방향(개선이 증가인지 감소인지)은 `METRICS` 에 명시한다.

**계산할 수 없는 값은 만들어내지 않는다.** 한쪽에 값이 없으면 `N/A` 로 두고
판정도 `판정불가` 로 남긴다.

표본 주의
---------
`TimeoutNode` 가 wall-clock 타이머라 같은 초기 조건에서도 궤적이 갈린다
(analysis/bt_remeasure §6). 판수가 적으면 차이가 분산에 묻힌다. 양쪽 판수가
`--min-episodes` 미만이면 경고를 남기고 "표본 부족" 을 표에 적는다.

실행
----
    python tools/compare_bt_batches.py \
        --before analysis/bt30/wez --after analysis/bt30_after/wez \
        --output analysis/bt30_after/comparison.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import warn  # noqa: E402

# (표시 이름, JSON 경로, 개선 방향, 소수점)
#   direction: "down" = 줄어야 개선, "up" = 늘어야 개선, "info" = 판정하지 않음
METRICS = [
    ("교전 판수",                 ("episodes",),                          "info", 0),
    ("거리 최근접 중앙 [m]",      ("min_distance_m", "median"),           "down", 1),
    ("사거리 밴드 진입 판",       ("episodes_entering_range",),           "up",   0),
    ("밴드 통과 횟수",            ("dwell_seconds", "passes_total"),      "up",   0),
    ("밴드 체류 중앙 [초]",       ("dwell_seconds", "median"),            "up",   2),
    (r"밴드 내 \|ATA\| 중앙 [도]", ("ata_in_range_deg", "median"),        "down", 1),
    (r"밴드 내 \|ATA\| 최선 [도]", ("ata_in_range_deg", "best"),          "down", 2),
    ("피해 조건 충족 프레임",     ("frames_hit_total",),                  "up",   0),
    ("피해를 준 판",              ("episodes_damaging_target",),          "up",   0),
    ("피해를 입은 판",            ("episodes_taking_damage",),            "down", 0),
    ("최소 사거리 미만 프레임",   ("frames_too_close_total",),            "down", 0),
]


def dig(data: dict, path: tuple):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


def load_summary(target: Path) -> dict:
    """디렉터리든 파일이든 받아 wez_window.json 의 summary 를 돌려준다."""
    path = target / "wez_window.json" if target.is_dir() else target
    if not path.exists():
        raise SystemExit(f"wez_window.json 을 찾지 못했다: {path}\n"
                         "  tools/analyze_wez_window.py 를 먼저 돌려라.")
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def verdict(before, after, direction: str) -> str:
    if direction == "info":
        return "—"
    if before is None or after is None:
        return "판정불가"
    if after == before:
        return "변화 없음"
    improved = (after < before) if direction == "down" else (after > before)
    return "개선" if improved else "악화"


def fmt(value, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}" if digits else f"{value:.0f}"


def delta(before, after, digits: int) -> str:
    if before is None or after is None:
        return "N/A"
    d = after - before
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.{digits}f}" if digits else f"{sign}{d:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="두 BT 교전 배치 비교")
    ap.add_argument("--before", type=Path, required=True,
                    help="변경 전 analyze_wez_window 출력 디렉터리(또는 json)")
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None, help="markdown 저장 경로")
    ap.add_argument("--before-label", default="변경 전")
    ap.add_argument("--after-label", default="변경 후")
    ap.add_argument("--min-episodes", type=int, default=20)
    args = ap.parse_args()

    before = load_summary(args.before)
    after = load_summary(args.after)

    n_before = before.get("episodes", 0)
    n_after = after.get("episodes", 0)
    scarce = n_before < args.min_episodes or n_after < args.min_episodes
    if scarce:
        warn(f"표본 부족: 전 {n_before}판 / 후 {n_after}판 "
             f"(권장 {args.min_episodes}판 이상). 판정을 확정으로 읽지 말 것.")

    # 게이트 값이 다르면 |ATA| 비교가 성립하지 않는다.
    if before.get("half_angle_deg") != after.get("half_angle_deg"):
        warn(f"WEZ 반각이 다르다: {before.get('half_angle_deg')} vs "
             f"{after.get('half_angle_deg')}. 각도 지표는 비교 불가다.")

    rows = []
    for name, path, direction, digits in METRICS:
        b, a = dig(before, path), dig(after, path)
        rows.append((name, fmt(b, digits), fmt(a, digits),
                     delta(b, a, digits), verdict(b, a, direction)))

    lines = [
        "# BT 교전 배치 비교",
        "",
        f"- {args.before_label}: `{args.before}` ({n_before}판)",
        f"- {args.after_label}: `{args.after}` ({n_after}판)",
        f"- WEZ 피해 게이트: 사거리 {after['wez']['min_range_m']}~{after['wez']['max_range_m']} m, "
        f"|ATA| <= {after['half_angle_deg']}도 (`angle_deg`의 절반)",
        "",
    ]
    if scarce:
        lines += [f"> **표본 부족** — 전 {n_before}판 / 후 {n_after}판. "
                  f"`TimeoutNode` 가 wall-clock 이라 분산이 크므로 "
                  f"{args.min_episodes}판 이상에서 다시 볼 것.", ""]

    lines += [f"| 지표 | {args.before_label} | {args.after_label} | 변화 | 판정 |",
              "|---|---:|---:|---:|---|"]
    lines += [f"| {n} | {b} | {a} | {d} | {v} |" for n, b, a, d, v in rows]

    lines += ["", "## outcome 분포", "",
              f"- {args.before_label}: `{before.get('outcomes')}`",
              f"- {args.after_label}: `{after.get('outcomes')}`", ""]

    # 가장 중요한 한 줄. 피해가 0에서 벗어났는지가 이 실험의 목적이다.
    hit_b, hit_a = before.get("frames_hit_total"), after.get("frames_hit_total")
    if hit_b == 0 and (hit_a or 0) > 0:
        lines += ["## 핵심", "",
                  f"**피해 조건 충족 프레임이 0 -> {hit_a} 로 바뀌었다. 최초 유효타다.**", ""]
    elif (hit_a or 0) == 0:
        lines += ["## 핵심", "",
                  "**피해 조건 충족 프레임이 여전히 0 이다.** 거리가 줄었더라도 "
                  "조준이 게이트에 못 닿았다는 뜻이므로, 다음 병목은 각도다.", ""]

    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\n저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
