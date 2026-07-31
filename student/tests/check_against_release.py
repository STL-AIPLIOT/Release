# -*- coding: utf-8 -*-
"""student hook의 기하·단위 가정을 Release 본체 실물과 대조 검증.

check_student_contracts.py 는 geo_info를 스텁으로 대신하므로 "계약을 지키는가"만
본다. 이 스크립트는 Release 본체의 **실제** `GeoMathUtil.GeometryInfo` 와
`dogfight/sim/state_schema.py` 를 파일에서 직접 읽어와, 부호·단위 해석이 실제
구현과 맞는지 확인한다.

Ray / gymnasium / pymap3d 없이 numpy만으로 돈다 (`dogfight/__init__.py`가
pymap3d를 끌고 오므로 state_schema는 패키지가 아니라 파일 경로로 로드한다).

실행 (Release 루트에서):
    python student/tests/check_against_release.py
종료 코드 0 = 전부 통과, 2 = Release 본체를 찾지 못해 건너뜀.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

STATE_SCHEMA_PATH = RELEASE_ROOT / "src" / "dogfight" / "sim" / "state_schema.py"
GEO_MATH_PATH = RELEASE_ROOT / "GeoMathUtil.py"


def _load_by_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if not STATE_SCHEMA_PATH.exists() or not GEO_MATH_PATH.exists():
    print("Release 본체를 찾지 못했습니다 (state_schema.py / GeoMathUtil.py).")
    print("이 스크립트는 DogFightEnv/Release 루트에서 실행해야 합니다. 건너뜁니다.")
    raise SystemExit(2)


# 실제 state_schema를 dogfight.sim.state_schema 자리에 끼워 넣는다.
# dogfight/__init__.py 를 실행하면 pymap3d가 필요하므로 껍데기 패키지만 만든다.
_real_schema = _load_by_path("_real_state_schema", STATE_SCHEMA_PATH)
_pkg = types.ModuleType("dogfight")
_pkg.__path__ = []
_sim = types.ModuleType("dogfight.sim")
_sim.__path__ = []
sys.modules.update(
    {"dogfight": _pkg, "dogfight.sim": _sim, "dogfight.sim.state_schema": _real_schema}
)

StateIndex = _real_schema.StateIndex
GeoMathUtil = _load_by_path("_real_geomath", GEO_MATH_PATH)

from student import my_observation as obs_mod  # noqa: E402
from student import my_reward as rew_mod  # noqa: E402


GEO = GeoMathUtil.GeometryInfo()
WEZ = {"angle_deg": 2.0, "min_range_m": 152.4, "max_range_m": 914.4}
STATE_LEN = 51
IDX_U, IDX_V, IDX_W = 6, 7, 8


def make_state(n=0.0, e=0.0, d=-7000.0, roll=0.0, pitch=0.0, yaw=0.0,
               u=200.0, v=0.0, w=0.0, kcas=200.0, alt=7000.0, health=1.0):
    """실제 FighterSim 상태 배열 배치를 그대로 흉내낸다."""
    state = np.zeros(STATE_LEN, dtype=np.float64)
    state[StateIndex.N], state[StateIndex.E], state[StateIndex.D] = n, e, d
    state[StateIndex.ROLL], state[StateIndex.PITCH], state[StateIndex.YAW] = roll, pitch, yaw
    state[IDX_U], state[IDX_V], state[IDX_W] = u, v, w
    state[StateIndex.KCAS] = kcas
    state[StateIndex.ALT] = alt
    state[StateIndex.HEALTH] = health
    return state


def observe(own, tgt):
    return obs_mod.build_observation(own, tgt, GEO, WEZ)


def reward_components(own, tgt, terminated=False, end_condition=""):
    return rew_mod.compute_reward(
        own, tgt, 0.0, 0.0, GEO, WEZ, rew_mod.MY_REWARD_CONFIG,
        terminated, False, end_condition,
    )[1]


_failures: list[str] = []
_passed = 0


def check(condition: bool, what: str) -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  [OK]   {what}")
    else:
        _failures.append(what)
        print(f"  [FAIL] {what}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


F = obs_mod.describe_observation()["features"]
I_ALT, I_KCAS, I_ATA, I_AA = (F.index(k) for k in
                              ("own_alt_norm", "own_kcas_norm", "ata_norm", "aa_norm"))
I_DIST, I_ENERGY, I_CLOSURE, I_WEZ = (F.index(k) for k in
                                      ("distance_norm", "energy_advantage_norm",
                                       "closure_rate_norm", "in_wez_flag"))


# ---------------------------------------------------------------------------
def test_units():
    section("1. 단위 해석 (실제 state_schema 인덱스 사용)")
    check(int(StateIndex.ALT) == 44 and int(StateIndex.KCAS) == 12,
          f"StateIndex ALT={int(StateIndex.ALT)}, KCAS={int(StateIndex.KCAS)}")

    own = make_state(n=0.0)
    tgt = make_state(n=1234.5)
    real_distance = float(GEO._get_distance(own, tgt))
    check(abs(real_distance - 1234.5) < 1e-6,
          "_get_distance가 NED 좌표 차 그대로 (meter)")

    obs = observe(own, tgt)
    expected = real_distance / obs_mod.DISTANCE_MAX_M * 2.0 - 1.0
    check(abs(float(obs[I_DIST]) - expected) < 1e-5,
          "distance feature가 실제 거리와 일치")

    # 고도·속도가 meter / meter-per-sec로 해석되는지
    es = obs_mod._specific_energy_m(make_state(alt=7000.0, kcas=250.0))
    check(abs(es - (7000.0 + 250.0 ** 2 / (2 * 9.80665))) < 1e-6,
          "Es = ALT[m] + KCAS[m/s]^2/2g")


def test_angle_conventions():
    section("2. ATA / AA 부호 규약 (실제 GeoMathUtil)")

    # (A) 공세: 적의 6시 뒤에서 같은 방향
    own = make_state(n=0.0, yaw=0.0)
    tgt = make_state(n=1000.0, yaw=0.0)
    ata = float(GEO._get_antenna_train_angle(own, tgt, False))
    aa = float(GEO._get_aspect_angle(own, tgt, False))
    check(abs(ata) < 1e-6, f"적의 6시: ATA ~ 0 (got {ata:.3f})")
    check(abs(aa) < 1e-6, f"적의 6시: AA ~ 0 (got {aa:.3f})")

    comp = reward_components(own, tgt)
    check(comp["pursuit"] > 0 and comp["position"] > 0,
          "적의 6시 -> pursuit > 0, position > 0 (가장 좋은 자세)")

    # (B) 헤드온 — 고도차를 두면 GeoMathUtil이 정상 동작한다
    own = make_state(n=0.0, yaw=0.0, d=-7000.0)
    tgt = make_state(n=1000.0, yaw=180.0, d=-7001.0)
    ata = float(GEO._get_antenna_train_angle(own, tgt, False))
    aa = float(GEO._get_aspect_angle(own, tgt, False))
    check(abs(ata) < 0.1, f"헤드온: ATA ~ 0 (got {ata:.3f})")
    check(abs(abs(aa) - 180.0) < 0.1, f"헤드온: |AA| ~ 180 (got {aa:.3f})")

    comp = reward_components(own, tgt)
    check(comp["pursuit"] > 0 > comp["position"],
          "헤드온 -> pursuit > 0 > position (서로 반대 방향)")
    check(comp["pursuit"] + comp["position"] > 0,
          "헤드온 순합이 양수 (상쇄 가드가 실제 기하에서도 성립)")

    # (C) 열세: 적이 내 뒤
    own = make_state(n=1000.0, yaw=0.0, d=-7000.0)
    tgt = make_state(n=0.0, yaw=0.0, d=-7001.0)
    ata = float(GEO._get_antenna_train_angle(own, tgt, False))
    check(abs(abs(ata) - 180.0) < 0.1, f"적이 내 뒤: |ATA| ~ 180 (got {ata:.3f})")
    comp = reward_components(own, tgt)
    check(comp["pursuit"] < 0, "적이 내 뒤 -> pursuit < 0")

    # (D) ATA 좌우 부호
    own = make_state(n=0.0, e=0.0, yaw=0.0)
    right = float(GEO._get_antenna_train_angle(own, make_state(n=0.0, e=1000.0), False))
    left = float(GEO._get_antenna_train_angle(own, make_state(n=0.0, e=-1000.0), False))
    check(left < 0 < right, f"우측 적 ATA>0, 좌측 적 ATA<0 (L={left:.1f}, R={right:.1f})")

    obs_r = observe(own, make_state(n=0.0, e=1000.0))
    obs_l = observe(own, make_state(n=0.0, e=-1000.0))
    check(float(obs_l[I_ATA]) < 0 < float(obs_r[I_ATA]),
          "관측 ata_norm이 실제 부호를 그대로 전달")


def test_degenerate_sign_repair():
    section("3. GeoMathUtil sign=0 축퇴와 보정")
    # 재현: 같은 고도에서 적이 정확히 내 6시.
    # GeoMathUtil은 sign = np.sign(p_unit_t[2]) = np.sign(0.0) = 0.0 이라
    # ATA 180을 0으로 보고한다 (GeoMathUtil.py L131-137).
    own = make_state(n=1000.0, d=-7000.0, yaw=0.0)
    tgt = make_state(n=0.0, d=-7000.0, yaw=0.0)
    raw_ata = float(GEO._get_antenna_train_angle(own, tgt, False))
    check(abs(raw_ata) < 1e-9,
          f"플랫폼 축퇴 재현: 동일 고도 정후방에서 ATA={raw_ata} (실제는 180)")

    tgt_offset = make_state(n=0.0, d=-7000.000000001, yaw=0.0)
    raw_offset = float(GEO._get_antenna_train_angle(own, tgt_offset, False))
    check(abs(abs(raw_offset) - 180.0) < 1e-6,
          f"고도차 1e-9 m만 있어도 ATA={raw_offset:.1f}로 복귀 (불연속)")

    # 보정 후: 내 hook은 실제 크기를 쓴다.
    check(abs(obs_mod._ata_magnitude_deg(own, tgt) - 180.0) < 1e-6,
          "보정 함수가 실제 ATA 크기 180을 계산")
    comp = reward_components(own, tgt)
    check(comp["pursuit"] < 0,
          "보정 후: 적이 내 6시 -> pursuit < 0 (보정 없으면 +0.6이 된다)")
    obs = observe(own, tgt)
    check(abs(float(obs[I_ATA])) > 0.99,
          "보정 후: 관측 ata_norm이 ~ +-1")

    # AA 축퇴 (동일 고도 헤드온)
    own_h = make_state(n=0.0, d=-7000.0, yaw=0.0)
    tgt_h = make_state(n=1000.0, d=-7000.0, yaw=180.0)
    raw_aa = float(GEO._get_aspect_angle(own_h, tgt_h, False))
    check(abs(raw_aa) < 1e-9, f"플랫폼 축퇴 재현: 동일 고도 헤드온에서 AA={raw_aa}")
    check(abs(obs_mod._aa_magnitude_deg(own_h, tgt_h) - 180.0) < 1e-6,
          "보정 함수가 실제 AA 크기 180을 계산")
    comp = reward_components(own_h, tgt_h)
    check(comp["position"] < 0, "보정 후: 헤드온 -> position < 0")

    # 축퇴가 아닌 경우에는 아무것도 바꾸지 않아야 한다.
    normal_own = make_state(n=0.0, e=0.0, d=-7000.0, yaw=0.0)
    normal_tgt = make_state(n=1000.0, e=400.0, d=-6800.0, yaw=45.0)
    reported = float(GEO._get_antenna_train_angle(normal_own, normal_tgt, False))
    repaired = obs_mod._repair_degenerate_angle(
        reported, obs_mod._ata_magnitude_deg, normal_own, normal_tgt
    )
    check(repaired == reported, "정상 구간에서는 보정이 값을 건드리지 않는다")


def test_wez_half_angle_matches_damage_model():
    section("4. WEZ 판정 기준이 update_damage와 일치")
    # single_agent_env.update_damage(): half_wez_angle_deg = wez["angle_deg"] / 2.0
    half, min_m, max_m = obs_mod._wez_thresholds(WEZ)
    check(abs(half - WEZ["angle_deg"] / 2.0) < 1e-12,
          f"관측 half angle = angle_deg/2 = {half}")
    r_half, r_min_ft, r_max_ft = rew_mod._wez_thresholds_ft(WEZ)
    check(abs(r_half - WEZ["angle_deg"] / 2.0) < 1e-12,
          f"보상 half angle = angle_deg/2 = {r_half}")
    check(abs(min_m - WEZ["min_range_m"]) < 1e-9 and abs(max_m - WEZ["max_range_m"]) < 1e-9,
          "사거리 기준은 wez_config 그대로")

    # 실제 대미지 판정과 같은 결론을 내는지 각도를 훑어 확인한다.
    own = make_state(n=0.0, e=0.0, d=-7000.0, yaw=0.0)
    mismatch = 0
    for offset_deg in np.linspace(0.0, 4.0, 41):
        # 500 m 앞, offset_deg 만큼 옆으로 벌어진 표적
        rad = math.radians(offset_deg)
        tgt = make_state(n=500.0 * math.cos(rad), e=500.0 * math.sin(rad), d=-7000.0)
        ata = float(GEO._get_antenna_train_angle(own, tgt, False))
        dis = float(GEO._get_distance(own, tgt))
        env_says_damage = (
            WEZ["min_range_m"] <= dis <= WEZ["max_range_m"]
            and WEZ["angle_deg"] / 2.0 >= abs(ata)
        )
        hook_says_in_wez = float(observe(own, tgt)[I_WEZ]) > 0.0
        if env_says_damage != hook_says_in_wez:
            mismatch += 1
    check(mismatch == 0, f"ATA 0~4도 구간 41개 지점에서 판정 일치 (불일치 {mismatch}건)")


def test_closure_against_finite_difference():
    section("5. closure rate vs 실제 거리의 유한차분")
    dt = 0.01

    scenarios = {
        "정면 접근 (250 + 250 m/s)": (
            make_state(n=0.0, yaw=0.0, u=250.0),
            make_state(n=2000.0, yaw=180.0, u=250.0),
        ),
        "추격 (300 vs 200 m/s)": (
            make_state(n=0.0, yaw=0.0, u=300.0),
            make_state(n=1000.0, yaw=0.0, u=200.0),
        ),
        "이탈 (반대 방향)": (
            make_state(n=0.0, yaw=180.0, u=200.0),
            make_state(n=1000.0, yaw=0.0, u=200.0),
        ),
        "수직 성분 포함": (
            make_state(n=0.0, d=-7000.0, yaw=30.0, pitch=10.0, roll=20.0, u=220.0),
            make_state(n=1500.0, e=800.0, d=-6500.0, yaw=200.0, u=180.0),
        ),
    }

    for label, (own, tgt) in scenarios.items():
        d0 = float(GEO._get_distance(own, tgt))

        own_next = own.copy()
        tgt_next = tgt.copy()
        for state, nxt in ((own, own_next), (tgt, tgt_next)):
            vn, ve, vd = obs_mod._velocity_ned(state)
            nxt[StateIndex.N] = state[StateIndex.N] + vn * dt
            nxt[StateIndex.E] = state[StateIndex.E] + ve * dt
            nxt[StateIndex.D] = state[StateIndex.D] + vd * dt
        d1 = float(GEO._get_distance(own_next, tgt_next))

        finite_diff = (d0 - d1) / dt
        analytic = obs_mod._closure_rate_ms(own, tgt)
        check(abs(finite_diff - analytic) < 0.5,
              f"{label}: 해석해 {analytic:.2f} ~ 유한차분 {finite_diff:.2f} m/s")

        obs = observe(own, tgt)
        expected_norm = max(-1.0, min(1.0, analytic / obs_mod.CLOSURE_MAX_MS))
        check(abs(float(obs[I_CLOSURE]) - expected_norm) < 1e-5,
              f"{label}: 관측 closure_rate_norm이 해석해와 일치")


def test_wez_against_real_geometry():
    section("6. WEZ 판정 (실제 기하)")
    # 정확히 적의 6시, 밴드 한가운데 (500~3000 ft = 152.4~914.4 m)
    own = make_state(n=0.0, yaw=0.0)
    tgt = make_state(n=500.0, yaw=0.0)
    obs = observe(own, tgt)
    check(float(obs[I_WEZ]) == 1.0, "밴드 안 + 정조준 -> in_wez = +1")

    comp = reward_components(own, tgt)
    check(comp["wez_entry"] > 0, "같은 상황에서 보상도 WEZ 진입으로 인식")

    # 500 ft 미만
    rew_mod.reset_reward_state()
    tgt_close = make_state(n=100.0, yaw=0.0)
    obs = observe(own, tgt_close)
    check(float(obs[I_WEZ]) == -1.0, "500 ft 미만 -> in_wez = -1")
    comp = reward_components(own, tgt_close)
    check(comp["overclose"] < 0, "500 ft 미만 -> overclose 패널티")

    # 각도만 벗어난 경우 (동일 거리, 옆으로)
    rew_mod.reset_reward_state()
    tgt_side = make_state(n=0.0, e=500.0, yaw=0.0)
    obs = observe(own, tgt_side)
    check(float(obs[I_WEZ]) == -1.0, "거리는 맞지만 정조준 아님 -> in_wez = -1")


def test_random_states_are_finite():
    section("7. 무작위 상태에서 finite 보장 (실제 기하)")
    rng = np.random.default_rng(20260729)
    bad_obs = 0
    bad_reward = 0
    mismatched_sum = 0

    for _ in range(2000):
        own = make_state(
            n=rng.uniform(-20000, 20000), e=rng.uniform(-20000, 20000),
            d=rng.uniform(-15000, -100),
            roll=rng.uniform(-180, 180), pitch=rng.uniform(-90, 90), yaw=rng.uniform(0, 360),
            u=rng.uniform(0, 400), v=rng.uniform(-50, 50), w=rng.uniform(-50, 50),
            kcas=rng.uniform(0, 400), alt=rng.uniform(0, 15000),
            health=float(rng.integers(0, 2)),
        )
        tgt = make_state(
            n=rng.uniform(-20000, 20000), e=rng.uniform(-20000, 20000),
            d=rng.uniform(-15000, -100),
            roll=rng.uniform(-180, 180), pitch=rng.uniform(-90, 90), yaw=rng.uniform(0, 360),
            u=rng.uniform(0, 400), v=rng.uniform(-50, 50), w=rng.uniform(-50, 50),
            kcas=rng.uniform(0, 400), alt=rng.uniform(0, 15000),
            health=float(rng.integers(0, 2)),
        )

        obs = observe(own, tgt)
        if not (np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= 1.0 + 1e-6)
                and obs.shape == (obs_mod.OBSERVATION_SIZE,)):
            bad_obs += 1

        terminated = bool(rng.integers(0, 2))
        total, comps = rew_mod.compute_reward(
            own, tgt, 0.0, 0.0, GEO, WEZ, rew_mod.MY_REWARD_CONFIG,
            terminated, False, "ownship_alt" if terminated else "",
        )
        if not (math.isfinite(total) and all(math.isfinite(v) for v in comps.values())):
            bad_reward += 1
        if abs(sum(comps.values()) - total) > 1e-9:
            mismatched_sum += 1

    check(bad_obs == 0, f"관측 2000건 전부 finite + 범위 안 (위반 {bad_obs}건)")
    check(bad_reward == 0, f"보상 2000건 전부 finite (위반 {bad_reward}건)")
    check(mismatched_sum == 0, f"보상 2000건 전부 sum == total (위반 {mismatched_sum}건)")


def test_degenerate_real_geometry():
    section("8. 축퇴 상황 (실제 기하)")
    same = make_state(n=100.0, e=200.0, d=-7000.0)
    obs = observe(same, same.copy())
    check(bool(np.all(np.isfinite(obs))), "완전히 같은 위치/자세여도 finite")

    rew_mod.reset_reward_state()
    total, comps = rew_mod.compute_reward(
        same, same.copy(), 0.0, 0.0, GEO, WEZ, rew_mod.MY_REWARD_CONFIG, False, False, "",
    )
    check(math.isfinite(total) and all(math.isfinite(v) for v in comps.values()),
          "같은 위치에서 보상도 finite (0 division 없음)")

    zero_speed = make_state(u=0.0, v=0.0, w=0.0, kcas=0.0)
    obs = observe(zero_speed, make_state(n=1000.0, u=0.0, kcas=0.0))
    check(bool(np.all(np.isfinite(obs))) and abs(float(obs[I_CLOSURE])) < 1e-9,
          "양쪽 정지 -> closure 0, finite")


def main() -> int:
    print("student hook <-> Release 본체 대조 검증")
    print(f"  state_schema: {STATE_SCHEMA_PATH}")
    print(f"  GeoMathUtil : {GEO_MATH_PATH}")

    test_units()
    test_angle_conventions()
    test_degenerate_sign_repair()
    test_wez_half_angle_matches_damage_model()
    test_closure_against_finite_difference()
    test_wez_against_real_geometry()
    test_random_states_are_finite()
    test_degenerate_real_geometry()

    print("\n" + "=" * 60)
    if _failures:
        print(f"실패 {len(_failures)}건 / 통과 {_passed}건")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print(f"전부 통과 ({_passed}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
