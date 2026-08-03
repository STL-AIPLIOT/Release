# -*- coding: utf-8 -*-
"""각도 wrap-around / PredictManeuver / SCISSORS 집계 테스트.

외부 프레임워크 없이 실패 개수를 세는 방식이다.
저장소의 기존 검증 스크립트(tests/tools/test_log_analysis.py,
student/tests/check_*.py)와 같은 형태를 따른다.

    python tests/tools/test_predict_maneuver.py

C++ 쪽 같은 표는 Behaviortree/tests/PredictManeuverAngleTest.cpp 가 검증한다.
두 구현이 갈라지지 않도록 **같은 케이스 표**를 쓴다.

    powershell -File Behaviortree/tests/build_and_run_predict_angle.ps1
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from log_analysis.angles import (  # noqa: E402
    circular_mean_deg,
    mean_signed_delta_deg,
    signed_angle_delta_deg,
    signed_angle_delta_rad,
    signed_delta_series_deg,
    wrap_angle_deg,
    wrap_angle_rad,
)
from log_analysis.geometry import in_wez  # noqa: E402
from log_analysis.predict_maneuver import (  # noqa: E402
    avg_delta_stats,
    detect_outliers,
    load_predict_log,
    scissors_segments,
    scissors_stats,
    wraparound_evidence,
)

_checks = 0
_failures = 0


def check(condition: bool, what: str) -> None:
    global _checks, _failures
    _checks += 1
    if condition:
        print(f"  [OK]   {what}")
    else:
        print(f"  [FAIL] {what}")
        _failures += 1


def close(a: float | None, b: float, tol: float = 1e-6) -> bool:
    return a is not None and math.isfinite(a) and abs(a - b) <= tol


# --------------------------------------------------------------------------- 1
# C++ 테스트와 동일한 케이스 표. 경계 정책: 범위가 [-180, 180) 이라 ±180 은 -180.
DELTA_CASES: tuple[tuple[float, float, float, str], ...] = (
    (10.0, 5.0, 5.0, "10 - 5 = +5"),
    (5.0, 10.0, -5.0, "5 - 10 = -5"),
    (-179.0, 179.0, 2.0, "-179 - 179 = +2 (경계 통과)"),
    (179.0, -179.0, -2.0, "179 - (-179) = -2 (경계 통과)"),
    (1.0, 359.0, 2.0, "1 - 359 = +2"),
    (359.0, 1.0, -2.0, "359 - 1 = -2"),
    (180.0, 0.0, -180.0, "180 - 0 = -180 (경계 정책)"),
    (0.0, 180.0, -180.0, "0 - 180 = -180 (경계 정책)"),
    (360.0, 0.0, 0.0, "360 - 0 = 0"),
    (0.0, 360.0, 0.0, "0 - 360 = 0"),
)


def test_signed_delta() -> None:
    print("\n[1] 최소 부호 각도 차이 (degree)")
    for current, previous, expected, label in DELTA_CASES:
        got = signed_angle_delta_deg(current, previous)
        check(close(got, expected, 1e-9), f"{label} -> {got}")


def test_range_and_no_spike() -> None:
    print("\n[2] 반환 범위와 급등 부재")
    in_range = True
    no_spike = True
    for i in range(-2160, 2161):
        v = wrap_angle_deg(i * 0.5)
        if not (-180.0 <= v < 180.0):
            in_range = False
        if abs(v) >= 300.0:
            no_spike = False
    check(in_range, "-1080~+1080 전 구간에서 반환값이 [-180, 180)")
    check(no_spike, "wrap 경계에서 |값| >= 300 이 한 번도 나오지 않는다")
    check(close(wrap_angle_deg(179.9), 179.9, 1e-9), "179.9 는 그대로")
    check(close(wrap_angle_deg(180.1), -179.9, 1e-9), "180.1 은 -179.9 로 접힌다")
    check(close(wrap_angle_deg(-180.0), -180.0, 1e-9), "-180 은 -180")
    check(close(wrap_angle_deg(180.0), -180.0, 1e-9), "+180 도 -180 (반열린 구간)")


def test_radian_equivalent() -> None:
    print("\n[3] radian 대응 함수")
    for current, previous, expected, label in DELTA_CASES:
        got = signed_angle_delta_rad(math.radians(current), math.radians(previous))
        check(close(math.degrees(got), expected, 1e-6), f"[rad] {label}")
    ok = all(-math.pi <= wrap_angle_rad(i * 0.1) < math.pi for i in range(-200, 201))
    check(ok, "wrap_angle_rad 반환 범위가 [-pi, pi)")


def test_non_finite() -> None:
    print("\n[4] NaN / inf 처리")
    check(math.isnan(wrap_angle_deg(math.nan)), "NaN 은 NaN 그대로 (0 으로 위장하지 않는다)")
    check(math.isinf(wrap_angle_deg(math.inf)), "inf 는 inf 그대로")
    check(math.isnan(signed_angle_delta_deg(math.nan, 10.0)), "한쪽이 NaN 이면 결과도 NaN")
    check(mean_signed_delta_deg([math.nan, math.nan]) is None,
          "전부 NaN 이면 평균은 None (0 이 아니다)")
    check(close(mean_signed_delta_deg([2.0, math.nan, 4.0]), 3.0),
          "NaN 을 뺀 평균을 쓴다")
    check(mean_signed_delta_deg([]) is None, "표본이 없으면 None")


def test_series_and_means() -> None:
    print("\n[5] 차이 계열과 평균")
    yaws = [176.0, 178.0, -180.0, -178.0, -176.0]
    deltas = signed_delta_series_deg(yaws)
    check(len(deltas) == len(yaws) - 1, "차이 개수는 표본보다 1 적다 (초기 프레임 처리)")
    check(all(close(d, 2.0, 1e-6) for d in deltas),
          f"경계를 넘는 +2도/프레임이 전부 +2 로 나온다: {deltas}")
    avg = mean_signed_delta_deg(deltas)
    check(close(avg, 2.0, 1e-6), f"avgDelta = {avg} (보정 없으면 -88 근처가 된다)")
    check(abs(avg) < 300.0, "avgDelta 에 300도 이상 급등이 들어가지 않는다")

    naive = [yaws[i] - yaws[i - 1] for i in range(1, len(yaws))]
    check(any(abs(d) > 300.0 for d in naive),
          f"단순 뺄셈이었다면 실제로 300도 넘는 값이 생긴다: {naive}")

    # circular mean 은 '각도 자체' 평균에만 쓴다.
    check(close(circular_mean_deg([179.0, -179.0]), 180.0, 1e-6)
          or close(circular_mean_deg([179.0, -179.0]), -180.0, 1e-6),
          "circular_mean_deg([179, -179]) 은 ±180 근처")
    check(circular_mean_deg([]) is None, "표본이 없으면 circular mean 도 None")


# --------------------------------------------------------------------------- CSV
V2_HEADER = ("runType,episode,frame,time,prevAngle,currAngle,rawDelta,normalizedDelta,"
             "avgDelta,predictedTurn,bfmMode,scissorsEntered,distance_m,ownAta_deg,"
             "targetAa_deg,angleOff_deg,enemyInSight")
V1_HEADER = ("runType,frame,time,prevAngle,currAngle,rawDelta,normalizedDelta,"
             "avgDelta,bfmMode,scissorsEntered")


def _v2_row(ep: int, frame: int, t: float, raw: float, norm: float, avg: float,
            mode: str, entered: int, dist: float = 500.0, ata: float = 30.0,
            aa: float = 120.0) -> str:
    return (f"after,{ep},{frame},{t},0,0,{raw},{norm},{avg},STRAIGHT,{mode},"
            f"{entered},{dist},{ata},{aa},10,1")


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_predict_loader(tmp: Path) -> None:
    print("\n[6] PredictManeuver CSV 로더")
    v2 = tmp / "v2" / "after.csv"
    write_csv(v2, V2_HEADER, [
        _v2_row(0, 0, 0.0, 2.0, 2.0, 2.0, "OBFM", 0),
        _v2_row(0, 1, 0.1, -358.0, 2.0, 2.0, "OBFM", 0),
        _v2_row(1, 2, 0.0, 2.0, 2.0, 2.0, "DBFM", 0),
    ])
    log = load_predict_log(tmp / "v2", "after")
    check(len(log.frames) == 3, f"3행을 읽는다 (읽은 수 {len(log.frames)})")
    check(len(log.match_ids) == 2, f"episode 컬럼으로 경기 2개 (얻은 수 {len(log.match_ids)})")
    check(log.episode_derived is False, "v2 는 episode 를 파생시키지 않는다")
    check(log.missing_columns == (), f"v2 는 없는 컬럼이 없다: {log.missing_columns}")
    check(log.frames[0].own_ata_deg == 30.0, "ownAta_deg 를 읽는다")

    v1 = tmp / "v1" / "before.csv"
    write_csv(v1, V1_HEADER, [
        "before,0,0.0,0,0,2,2,2,OBFM,0",
        "before,1,0.1,0,0,2,2,2,OBFM,0",
        "before,2,0.0,0,0,2,2,2,DBFM,0",   # time 되감김 -> 새 경기
    ])
    log1 = load_predict_log(v1, "before")
    check(log1.episode_derived is True, "v1 은 time 되감김으로 episode 를 파생시킨다")
    check(len(log1.match_ids) == 2, f"파생 경기 2개 (얻은 수 {len(log1.match_ids)})")
    check("distance_m" in log1.missing_columns,
          f"없는 컬럼을 보고한다: {log1.missing_columns}")
    check(log1.frames[0].own_ata_deg is None, "없는 컬럼은 None (0 으로 채우지 않는다)")

    other = tmp / "v1" / "not_predict.csv"
    other.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    log2 = load_predict_log(tmp / "v1", "before")
    check(len(log2.frames) == 3, "PredictManeuver 형식이 아닌 CSV 는 건너뛴다")


def test_avg_delta_stats(tmp: Path) -> None:
    print("\n[7] avgDelta 통계와 이상값")
    path = tmp / "stats.csv"
    write_csv(path, V2_HEADER, [
        _v2_row(0, 0, 0.0, 2.0, 2.0, 2.0, "OBFM", 0),
        _v2_row(0, 1, 0.1, 2.0, 2.0, 175.0, "OBFM", 0),     # 이상값
        _v2_row(0, 2, 0.2, 2.0, 2.0, 3.0, "OBFM", 0),       # 급변
        _v2_row(1, 3, 0.0, 2.0, 2.0, 179.5, "DBFM", 0),     # 다른 경기의 이상값
    ])
    frames = load_predict_log(path, "after").frames
    st = avg_delta_stats(frames, (170.0, 175.0, 179.0), 90.0)
    check(st.sample_count == 4, f"표본 4 (얻은 값 {st.sample_count})")
    check(st.outlier_counts[170] == 2, f"|avgDelta|>=170 은 2건 (얻은 값 {st.outlier_counts[170]})")
    check(st.outlier_counts[179] == 1, f"|avgDelta|>=179 는 1건 (얻은 값 {st.outlier_counts[179]})")
    check(close(st.abs_max, 179.5), f"절댓값 최대 179.5 (얻은 값 {st.abs_max})")
    check(st.out_of_range_count == 0, "범위 이탈 없음")
    # 경기 0 안에서 2->175, 175->3 두 번 급변. 경기 경계(3->179.5)는 세지 않는다.
    check(st.spike_count == 2, f"급변 2회, 경기 경계는 제외 (얻은 값 {st.spike_count})")
    check(st.median is not None and st.p95 is not None, "중앙값/분위수가 계산된다")

    outliers = detect_outliers(frames, 170.0, 90.0)
    kinds = {o.kind for o in outliers}
    check(any("magnitude" in k for k in kinds), "magnitude 이상값을 잡는다")
    check(any("spike" in k for k in kinds), "spike 이상값을 잡는다")
    o = next(o for o in outliers if close(o.avg_delta_deg, 175.0))
    check(o.bfm_at == "OBFM" and o.bfm_before == "OBFM" and o.bfm_after == "OBFM",
          "이상값 전후 BFM 모드를 함께 담는다")
    check(o.derived_in_wez is False,
          f"distance 500m / ATA 30도 -> WEZ 밖 (얻은 값 {o.derived_in_wez})")


def test_wraparound_evidence(tmp: Path) -> None:
    print("\n[8] wrap 보정 직접 증거")
    good = tmp / "good.csv"
    write_csv(good, V2_HEADER, [
        _v2_row(0, 0, 0.0, -358.0, 2.0, 2.0, "OBFM", 0),   # 보정됨
        _v2_row(0, 1, 0.1, 2.0, 2.0, 2.0, "OBFM", 0),      # 보정 불필요
    ])
    ev = wraparound_evidence(load_predict_log(good, "after").frames)
    check(ev["checked_frames"] == 2, "2프레임을 확인한다")
    check(ev["raw_delta_beyond_180"] == 1, "rawDelta 가 ±180 을 넘은 프레임 1건")
    check(ev["normalized_delta_beyond_180"] == 0, "보정 후에는 범위를 넘지 않는다")
    check(ev["wrap_corrected_frames"] == 1, "실제로 보정된 프레임 1건")
    check(ev["normalized_vs_expected_mismatch"] == 0, "wrap(raw) 와 전부 일치")

    bad = tmp / "bad.csv"
    write_csv(bad, V2_HEADER, [
        _v2_row(0, 0, 0.0, -358.0, -358.0, -358.0, "OBFM", 0),   # 보정 안 됨
    ])
    ev2 = wraparound_evidence(load_predict_log(bad, "before").frames)
    check(ev2["normalized_delta_beyond_180"] == 1, "미보정 프레임을 잡아낸다")
    check(ev2["normalized_vs_expected_mismatch"] == 1, "wrap(raw) 불일치를 잡아낸다")


def test_scissors(tmp: Path) -> None:
    print("\n[9] SCISSORS 진입 집계 (transition 기준)")
    path = tmp / "scissors.csv"
    # 경기 0: OBFM OBFM SCISSORS SCISSORS SCISSORS DBFM SCISSORS DBFM
    #         -> 진입 2회, 재진입 1회, 체류 0.3s + 0.1s
    rows = []
    seq0 = ["OBFM", "OBFM", "SCISSORS", "SCISSORS", "SCISSORS", "DBFM", "SCISSORS", "DBFM"]
    for i, mode in enumerate(seq0):
        rows.append(_v2_row(0, i, round(i * 0.1, 6), 2.0, 2.0, 2.0, mode, 0))
    # 경기 1: SCISSORS 로 시작해서 끝까지 유지 -> 진입 0회, open segment 1개
    for i, mode in enumerate(["SCISSORS", "SCISSORS"]):
        rows.append(_v2_row(1, 100 + i, round(i * 0.1, 6), 2.0, 2.0, 2.0, mode, 0))
    # 경기 2: SCISSORS 없음
    rows.append(_v2_row(2, 200, 0.0, 2.0, 2.0, 2.0, "OBFM", 0))
    write_csv(path, V2_HEADER, rows)

    frames = load_predict_log(path, "after").frames
    st, segs = scissors_stats(frames)

    check(st.match_count == 3, f"경기 3개 (얻은 값 {st.match_count})")
    check(st.entry_count == 2,
          f"진입은 transition 2회다. 상태 행 수(4)가 아니다 (얻은 값 {st.entry_count})")
    check(st.matches_with_entry == 1, f"진입한 경기 1개 (얻은 값 {st.matches_with_entry})")
    check(close(st.entry_match_ratio, 1 / 3), f"진입 경기 비율 1/3 (얻은 값 {st.entry_match_ratio})")
    check(st.reentry_count == 1, f"재진입 1회 (얻은 값 {st.reentry_count})")
    check(st.open_segment_count == 1,
          f"끝까지 SCISSORS 인 구간 1개 (얻은 값 {st.open_segment_count})")
    check(close(st.total_dwell_sec, 0.4, 1e-6),
          f"닫힌 구간 체류 합 0.4s (얻은 값 {st.total_dwell_sec})")
    check(close(st.longest_dwell_sec, 0.3, 1e-6),
          f"최장 연속 체류 0.3s (얻은 값 {st.longest_dwell_sec})")
    check(close(st.dwell_per_match_mean, 0.4 / 3, 1e-6),
          "경기당 평균 체류는 진입 없는 경기도 분모에 넣는다")
    check(st.entered_from.get("OBFM") == 1 and st.entered_from.get("DBFM") == 1,
          f"진입 직전 모드 분포 {st.entered_from}")
    check(st.exited_to.get("DBFM") == 2, f"종료 후 전환 모드 분포 {st.exited_to}")
    check(st.logged_entry_count == 0, "CSV scissorsEntered 합계도 함께 보고한다")

    # 첫 프레임부터 SCISSORS 인 구간은 진입으로 세지 않는다.
    first = [s for s in segs if s.match_id.endswith("ep001")]
    check(len(first) == 1 and first[0].entered_from is None,
          "직전 상태를 모르는 구간은 entered_from=None 이며 진입에서 제외된다")

    # timestamp 간격이 일정하지 않아도 체류시간은 end-start 로 계산된다.
    uneven = tmp / "uneven.csv"
    write_csv(uneven, V2_HEADER, [
        _v2_row(0, 0, 0.0, 2.0, 2.0, 2.0, "OBFM", 0),
        _v2_row(0, 1, 1.0, 2.0, 2.0, 2.0, "SCISSORS", 1),
        _v2_row(0, 2, 7.5, 2.0, 2.0, 2.0, "SCISSORS", 0),
        _v2_row(0, 3, 9.0, 2.0, 2.0, 2.0, "OBFM", 0),
    ])
    st2, _ = scissors_stats(load_predict_log(uneven, "after").frames)
    check(close(st2.total_dwell_sec, 8.0, 1e-6),
          f"불규칙 간격에서도 체류 = 9.0-1.0 = 8.0 (얻은 값 {st2.total_dwell_sec})")

    st3, segs3 = scissors_stats([])
    check(st3.match_count == 0 and st3.entry_count == 0 and segs3 == [],
          "빈 경기 데이터에서도 예외 없이 0 을 돌려준다")
    check(st3.entry_match_ratio is None, "경기가 없으면 비율은 None (0 이 아니다)")


def test_wez_gate() -> None:
    print("\n[10] WEZ 게이트가 update_damage 와 같은가")
    # angle_deg=2.0 이면 실제 원뿔은 1.0도다 (플랫폼 결함 2).
    check(in_wez(500.0, 0.9) is True, "거리 500m, |ATA| 0.9도 -> WEZ 안")
    check(in_wez(500.0, 1.0) is True, "|ATA| 정확히 1.0도 -> 안 (>= 비교)")
    check(in_wez(500.0, 1.1) is False, "|ATA| 1.1도 -> 밖 (full angle_deg 였다면 안이 된다)")
    check(in_wez(500.0, 1.9) is False, "|ATA| 1.9도 -> 밖. angle_deg 그대로 쓰면 안 된다")
    check(in_wez(152.4, 0.5) is True, "최소 사거리 경계는 포함")
    check(in_wez(914.4, 0.5) is True, "최대 사거리 경계는 포함")
    check(in_wez(152.3, 0.5) is False, "최소 사거리 미만은 밖")
    check(in_wez(914.5, 0.5) is False, "최대 사거리 초과는 밖")
    check(in_wez(500.0, -0.9) is True, "부호는 abs 로 처리")


def main() -> int:
    print("PredictManeuver / SCISSORS 분석 테스트")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_signed_delta()
        test_range_and_no_spike()
        test_radian_equivalent()
        test_non_finite()
        test_series_and_means()
        test_predict_loader(tmp / "loader")
        test_avg_delta_stats(tmp / "stats")
        test_wraparound_evidence(tmp / "evidence")
        test_scissors(tmp / "scissors")
        test_wez_gate()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{_failures} / {_checks} 실패")
    else:
        print(f"전부 통과 ({_checks}건)")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
