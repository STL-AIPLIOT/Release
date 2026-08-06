# -*- coding: utf-8 -*-
"""BT stdout 에서 BFM 모드 타임라인을 뽑아 CSV 로 만든다.

배경
----
BFM 모드는 Python 쪽 어떤 로그에도 없다(2026-08-04 조사: training_log.csv /
Tacview CSV / summary.json / replay_index.jsonl 전부 BFM 필드 없음).
유일한 출처는 BT(C++) 노드가 std::cout 으로 찍는 진단 로그다.

    [SetBFMMode_OBFM] t=12.34s | Enter OBFM (AA=..., D=..., EnergySup=...)
    [SetBFMMode_HABFM] t=12.36s | Blocked | sight=1, AA=..., D=..., E=...

`t=` 는 BB->RunningTime(초)이다. 이 형식은 SetBFMMode_{SCISSORS,OBFM,DBFM,HABFM}
네 노드 모두 동일하다.

이 로그는 파일로 저장되지 않으므로 실행할 때 리다이렉트해서 받아야 한다.

    python run_local_dogfight.py --ownship-backend rl ... > bt_stdout.log 2>&1
    python tools/extract_bfm_log.py --stdout bt_stdout.log --output analysis/bfm

`t=` 가 감소하면 새 에피소드가 시작된 것으로 본다(RunningTime 이 reset 된다).

실행:
    python tools/extract_bfm_log.py --stdout <로그파일> [--output <디렉터리>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import normalize_bfm, warn  # noqa: E402

# [SetBFMMode_OBFM] t=12.34s | Enter OBFM (...)
LINE_RE = re.compile(
    r"\[SetBFMMode_(?P<mode>SCISSORS|OBFM|DBFM|HABFM)\]\s*"
    r"t=(?P<time>-?\d+(?:\.\d+)?)s\s*\|\s*(?P<verdict>Enter|Blocked)",
    re.IGNORECASE,
)
# 1C/2C 는 HABFM Enter 로그 끝에 Circle=1C / 2C / NONE 으로 붙는다.
CIRCLE_RE = re.compile(r"Circle=(?P<circle>1C|2C|NONE)")


# 파싱 실패 통계. write_outputs 가 메타에 싣는다.
# stdout 은 라인 원자성이 없어 JSBSim 출력과 섞이면 그 줄을 잃는다.
_PARSE_STATS: dict[str, int] = {"seen": 0, "unmatched": 0}


def parse(stdout_path: Path) -> list[dict[str, object]]:
    """stdout 로그를 (episode, time, mode, verdict, circle) 행으로 바꾼다."""
    rows: list[dict[str, object]] = []
    episode = 0
    prev_t: float | None = None
    unmatched = 0

    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warn(f"{stdout_path} 읽기 실패: {exc}")
        return rows

    for lineno, line in enumerate(text.splitlines(), 1):
        if "SetBFMMode_" not in line:
            continue
        _PARSE_STATS["seen"] += 1
        m = LINE_RE.search(line)
        if not m:
            unmatched += 1
            _PARSE_STATS["unmatched"] += 1
            if unmatched <= 5:
                warn(f"{stdout_path.name}:{lineno} 형식이 맞지 않는다: {line.strip()[:90]}")
            continue
        t = float(m.group("time"))
        if prev_t is not None and t < prev_t - 1e-6:
            episode += 1          # RunningTime 이 되감겼다 = 새 에피소드
        prev_t = t
        circle = None
        if m.group("verdict").lower() == "enter":
            cm = CIRCLE_RE.search(line)
            circle = cm.group("circle") if cm else None
        rows.append({
            "episode": episode,
            "time_sec": t,
            "mode": normalize_bfm(m.group("mode")),
            "verdict": m.group("verdict").capitalize(),
            "circle": circle,
            "raw_line": line.strip(),
        })

    if unmatched > 5:
        warn(f"{stdout_path.name}: 형식 불일치 {unmatched}줄 (앞 5줄만 출력)")
    rows.sort(key=lambda r: (r["episode"], r["time_sec"]))
    return rows


def build_timeline(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Enter 로그만 남겨 모드 체류 구간을 만든다.

    체류시간은 다음 Enter 까지의 간격이다. 각 에피소드의 마지막 구간은 종료 시각을
    알 수 없으므로 duration 을 None 으로 두고 last_segment=True 로 표시한다
    (0 으로 채우거나 임의 값으로 메우지 않는다).
    """
    enters = [r for r in rows if r["verdict"] == "Enter"]
    timeline: list[dict[str, object]] = []
    for i, r in enumerate(enters):
        nxt = enters[i + 1] if i + 1 < len(enters) else None
        same_ep = nxt is not None and nxt["episode"] == r["episode"]
        duration = (float(nxt["time_sec"]) - float(r["time_sec"])) if same_ep else None
        timeline.append({
            "episode": r["episode"],
            "start_sec": r["time_sec"],
            "end_sec": nxt["time_sec"] if same_ep else None,
            "duration_sec": duration,
            "mode": r["mode"],
            "circle": r["circle"],
            "last_segment": not same_ep,
        })
    return timeline


def write_outputs(rows: list[dict[str, object]], timeline: list[dict[str, object]],
                  outdir: Path, source: Path) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "bfm_events.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode", "time_sec", "mode", "verdict",
                                           "circle", "raw_line"])
        w.writeheader()
        w.writerows(rows)

    with (outdir / "bfm_timeline.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode", "start_sec", "end_sec",
                                           "duration_sec", "mode", "circle",
                                           "last_segment"])
        w.writeheader()
        w.writerows(timeline)

    episodes = sorted({r["episode"] for r in rows})
    meta = {
        "source_stdout": str(source),
        "episode_count": len(episodes),
        "episode_ids": episodes,
        "event_count": len(rows),
        "enter_count": sum(1 for r in rows if r["verdict"] == "Enter"),
        "blocked_count": sum(1 for r in rows if r["verdict"] == "Blocked"),
        "mode_enter_counts": dict(Counter(r["mode"] for r in rows
                                          if r["verdict"] == "Enter")),
        "circle_counts": dict(Counter(str(r["circle"]) for r in rows
                                      if r["circle"] is not None)),
        "last_segment_count": sum(1 for t in timeline if t["last_segment"]),
        "note": ("각 에피소드의 마지막 구간은 종료 시각을 알 수 없어 duration_sec 이 "
                 "비어 있다. 체류시간 집계 시 제외하거나 별도 처리해야 한다."),
        # stdout 은 라인 원자성이 없다. JSBSim(다른 DLL)이 같은 stdout 에 쓰면서
        # BT 한 줄 중간에 끼어들면 그 줄은 파싱되지 않는다. 조용히 넘기지 않고 수치로 남긴다.
        "parsed_lines": _PARSE_STATS["seen"] - _PARSE_STATS["unmatched"],
        "unmatched_lines": _PARSE_STATS["unmatched"],
        "unmatched_ratio": (_PARSE_STATS["unmatched"] / _PARSE_STATS["seen"]
                            if _PARSE_STATS["seen"] else None),
        "unmatched_cause": ("stdout 라인 원자성 없음 — JSBSim 출력과 섞인 줄. "
                            "손실률이 1% 를 넘으면 집계를 신뢰하지 말 것."),
        # BB->BFM 은 SetBFMMode_* 가 성공할 때만 쓰이고, 전부 실패해도 초기화되지 않는다.
        # 따라서 PredictManeuver CSV 의 bfmMode 는 '지금 모드'가 아니라
        # '마지막으로 진입에 성공한 모드'다. 이 타임라인(Enter 기준)이 정확한 쪽이다.
        "sticky_bfm_warning": ("BB->BFM 은 초기화되지 않는다(sticky). 'BFM=X 가 N tick 지속' 은 "
                               "'N tick 동안 X 모드였다'가 아니라 'X 가 마지막 성공 모드다' 라는 뜻이다. "
                               "체류 시간은 이 파일의 Enter 이벤트 간격으로 계산하라."),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (outdir / "bfm_extract_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="BT stdout 에서 BFM 타임라인 추출")
    ap.add_argument("--stdout", required=True, type=Path, help="BT stdout 을 받은 텍스트 파일")
    ap.add_argument("--output", type=Path, default=Path("analysis/bfm"))
    args = ap.parse_args()

    if not args.stdout.exists():
        warn(f"파일이 없다: {args.stdout}")
        print("\nBT stdout 을 받으려면 실행할 때 리다이렉트해야 한다:")
        print("  python run_local_dogfight.py --ownship-backend rl ... > bt_stdout.log 2>&1")
        return 1

    rows = parse(args.stdout)
    if not rows:
        warn("SetBFMMode 로그를 한 줄도 찾지 못했다.")
        print("\n확인할 것:")
        print("  - BT DLL 이 실제로 로드됐는가 (--target-backend bt 등)")
        print("  - 빌드된 DLL 에 B-1 시간 로그가 들어 있는가 ([SetBFMMode_*] t=...s | ...)")
        return 1

    timeline = build_timeline(rows)
    meta = write_outputs(rows, timeline, args.output, args.stdout)

    print(f"이벤트 {meta['event_count']}건 (Enter {meta['enter_count']} / "
          f"Blocked {meta['blocked_count']})")
    print(f"에피소드 {meta['episode_count']}개")
    print(f"모드별 진입 횟수: {meta['mode_enter_counts']}")
    if meta["circle_counts"]:
        print(f"1C/2C 기록: {meta['circle_counts']}")
    print(f"\n출력: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
