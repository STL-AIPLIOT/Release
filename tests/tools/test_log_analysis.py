# -*- coding: utf-8 -*-
"""로그 분석 도구 테스트.

외부 프레임워크 없이 실패 개수를 세는 방식이다.
저장소의 기존 검증 스크립트(student/tests/check_*.py)와 같은 형태를 따른다.

    python tests/tools/test_log_analysis.py

실제 로그를 fixture 로 복사하지 않는다. 필요한 최소 데이터를 임시 디렉터리에
그때그때 만들어 쓴다.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from log_analysis import (  # noqa: E402
    load_episodes,
    load_track,
    normalize_bfm,
    normalize_end_condition,
    normalize_outcome,
)
from log_analysis.events import Thresholds, build_sequence, detect_altitude_events  # noqa: E402
from log_analysis.metrics import (  # noqa: E402
    mean_ignoring_nan,
    specific_energy_series,
    speed_series,
    window_indices,
)

_checks = 0
_failures = 0


def check(cond: bool, label: str) -> None:
    global _checks, _failures
    _checks += 1
    if cond:
        print(f"  [OK]   {label}")
    else:
        _failures += 1
        print(f"  [FAIL] {label}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- 정규화
def test_normalization() -> None:
    section("1. 정규화")
    check(normalize_end_condition("ownship altitude below min") == "altitude_check_failed",
          "실제 로그 값 -> altitude_check_failed")
    check(normalize_end_condition("target altitude below min") == "altitude_check_failed",
          "target 쪽도 같은 유형")
    check(normalize_end_condition("max time out") == "timeout", "max time out -> timeout")
    check(normalize_end_condition("episode step limit") == "timeout", "step limit -> timeout")
    check(normalize_end_condition("ownship destroyed") == "wez_hit", "destroyed -> wez_hit")
    check(normalize_end_condition("FDM Update Fail") == "fdm_fail", "대소문자 무시")
    check(normalize_end_condition("듣도보도못한값") == "unknown",
          "미지의 end condition -> unknown (other 로 묻지 않는다)")
    check(normalize_end_condition(None) == "unknown", "None -> unknown")
    check(normalize_end_condition("") == "unknown", "빈 문자열 -> unknown")

    check(normalize_outcome("crash") == "crash", "outcome crash")
    check(normalize_outcome("DRAW") == "draw", "outcome 대문자")
    check(normalize_outcome(None) == "unknown", "outcome None")

    check(normalize_bfm("OBFM") == "OBFM", "BFM 그대로")
    check(normalize_bfm("habfm") == "HABFM", "BFM 소문자")
    check(normalize_bfm("모르는모드") == "UNKNOWN", "미지의 BFM -> UNKNOWN")
    check(normalize_bfm(None) == "UNKNOWN", "BFM None")


# --------------------------------------------------------------------------- 로더
def test_loaders(tmp: Path) -> None:
    section("2. 로더")

    empty = write_csv(tmp / "empty.csv", "Time,Altitude", [])
    track = load_track(empty)
    check(len(track) == 0, "빈 CSV -> 길이 0 (예외 없음)")

    missing = write_csv(tmp / "missing.csv", "Time,Foo", ["0.0,1", "1.0,2"])
    track = load_track(missing)
    check("alt" in track.missing, "없는 컬럼이 missing 에 기록된다")
    check(track.alt == [], "없는 컬럼을 0 으로 채우지 않는다")

    bad = write_csv(tmp / "bad_time.csv", "Time,Altitude",
                    ["abc,7000", "1.0,6900", "2.0,6800"])
    track = load_track(bad)
    check(len(track) == 2, "잘못된 timestamp 행은 건너뛴다")

    shuffled = write_csv(tmp / "shuffled.csv", "Time,Altitude",
                         ["2.0,6800", "0.0,7000", "1.0,6900"])
    track = load_track(shuffled)
    check(track.time == [0.0, 1.0, 2.0], "시간이 뒤섞인 로그는 정렬된다")
    check(track.alt == [7000.0, 6900.0, 6800.0], "정렬 시 다른 컬럼도 함께 움직인다")

    same = write_csv(tmp / "same_time.csv", "Time,Altitude",
                     ["1.0,7000", "1.0,6900"])
    track = load_track(same)
    check(len(track) == 2, "동일 timestamp 도 버리지 않는다")

    nan_csv = write_csv(tmp / "nan.csv", "Time,Altitude", ["0.0,", "1.0,6900"])
    track = load_track(nan_csv)
    check(math.isnan(track.alt[0]), "빈 값은 NaN 으로 남긴다 (0 아님)")

    # replay_index: 여러 episode 가 한 파일에
    idx = tmp / "runA" / "engagement_replays" / "replay_index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text("\n".join(json.dumps(r) for r in [
        {"iteration": 0, "episode": 0, "steps": 10, "total_reward": -1.0,
         "outcome": "crash", "end_condition": "ownship altitude below min"},
        {"iteration": 1, "episode": 0, "steps": 20, "total_reward": -2.0,
         "outcome": "draw", "end_condition": "target altitude below min"},
    ]) + "\n", encoding="utf-8")
    eps = load_episodes(tmp)
    check(len(eps) == 2, "replay_index 여러 episode 로딩")
    check(eps[0].run == "runA", "실험 태그가 run 에 들어간다")
    check(eps[0].match_id.startswith("runA/"), "match_id 에 실험 태그가 붙어 충돌하지 않는다")
    check(eps[0].end_condition == "altitude_check_failed", "로딩 시 정규화된다")

    bad_json = tmp / "runB" / "engagement_replays" / "replay_index.jsonl"
    bad_json.parent.mkdir(parents=True, exist_ok=True)
    bad_json.write_text("{깨진 json\n", encoding="utf-8")
    before = len(load_episodes(tmp))
    check(before == 2, "깨진 JSON 줄은 건너뛰고 나머지는 살린다")


# --------------------------------------------------------------------------- 지표
def test_metrics() -> None:
    section("3. 지표")
    check(window_indices([], 5.0) == (0, 0), "빈 시계열 -> (0,0)")
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = window_indices(t, 2.0)
    check(t[lo] >= 3.0 and hi == len(t), "마지막 2초 창")
    lo, hi = window_indices([0.0, 1.0], 30.0)
    check((lo, hi) == (0, 2), "창보다 짧은 경기는 전체를 돌려준다")

    speed = speed_series([0.0, 1.0], [37.0, 37.0], [128.0, 128.0], [7000.0, 6900.0])
    check(abs(speed[1] - 100.0) < 1.0, "수직 하강 100 m 를 1초에 -> 약 100 m/s")

    se = specific_energy_series([1000.0], [100.0])
    check(abs(se[0] - (9.80665 * 1000.0 + 0.5 * 100.0 ** 2)) < 1e-6,
          "specific energy = g*h + 0.5*v^2")

    check(mean_ignoring_nan([float("nan")]) is None,
          "전부 NaN 이면 None (0 으로 대체하지 않는다)")
    check(mean_ignoring_nan([1.0, float("nan"), 3.0]) == 2.0, "NaN 제외 평균")


# --------------------------------------------------------------------------- 이벤트
def test_events(tmp: Path) -> None:
    section("4. 이벤트")
    from log_analysis.schemas import Track

    th = Thresholds(min_altitude_m=300.0, low_altitude_margin_m=700.0,
                    descent_rate_ms=40.0)
    tr = Track(path=tmp / "x.csv")
    tr.time = [0.0, 1.0, 2.0]
    tr.alt = [7000.0, 5000.0, 900.0]
    events = detect_altitude_events(tr.time, tr, 0, 3, th)
    codes = {e.code for e in events}
    check("LOW_ALTITUDE" in codes, "저고도 진입 감지")
    check("HIGH_DESCENT" in codes, "급하강 감지")

    tr2 = Track(path=tmp / "y.csv")
    tr2.time = [0.0, 1.0]
    tr2.alt = [7000.0, 6999.0]
    check(detect_altitude_events(tr2.time, tr2, 0, 2, th) == [],
          "정상 순항에서는 이벤트가 없다")

    seq = build_sequence([], "altitude_check_failed")
    check(seq == ["ALTITUDE_CHECK_FAILED"], "종료 원인이 시퀀스 끝에 붙는다")


# --------------------------------------------------------------------------- BFM 추출
def test_bfm_extract(tmp: Path) -> None:
    section("5. BFM stdout 추출")
    sys.path.insert(0, str(ROOT / "tools"))
    import extract_bfm_log as ex

    tmp.mkdir(parents=True, exist_ok=True)
    log = tmp / "bt.log"
    log.write_text("\n".join([
        "무관한 줄",
        "[SetBFMMode_OBFM] t=1.00s | Enter OBFM (AA=10, D=1200)",
        "[SetBFMMode_HABFM] t=3.00s | Blocked | sight=1",
        "[SetBFMMode_HABFM] t=4.00s | Enter HABFM | AA=170 | Circle=2C",
        "[SetBFMMode_OBFM] t=0.50s | Enter OBFM (AA=5, D=900)",
    ]) + "\n", encoding="utf-8")

    rows = ex.parse(log)
    check(len(rows) == 4, "SetBFMMode 줄만 파싱한다")
    check(rows[0]["episode"] == 0 and rows[-1]["episode"] == 1,
          "t 가 되감기면 새 에피소드로 나눈다")
    circle = [r["circle"] for r in rows if r["circle"]]
    check(circle == ["2C"], "Circle=2C 를 뽑아낸다")

    timeline = ex.build_timeline(rows)
    check(all(t["mode"] != "UNKNOWN" for t in timeline), "모드가 정규화된다")
    last = [t for t in timeline if t["last_segment"]]
    check(all(t["duration_sec"] is None for t in last),
          "마지막 구간 duration 은 None (0 으로 채우지 않는다)")

    empty = tmp / "empty.log"
    empty.write_text("아무 BFM 로그 없음\n", encoding="utf-8")
    check(ex.parse(empty) == [], "BFM 로그가 없으면 빈 목록")


def main() -> int:
    print("로그 분석 도구 테스트")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_normalization()
        test_loaders(tmp / "loaders")
        test_metrics()
        test_events(tmp / "events")
        test_bfm_extract(tmp / "bfm")
    print("\n" + "=" * 60)
    if _failures:
        print(f"{_failures} / {_checks} 실패")
    else:
        print(f"전부 통과 ({_checks}건)")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
