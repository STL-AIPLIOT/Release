# -*- coding: utf-8 -*-
"""student/my_observation.py + my_reward.py 계약 오프라인 검증.

DogFightEnv/Release 본체(src/dogfight, JSBSim DLL, Ray) 없이 돌아간다.
dogfight.sim.state_schema 와 geo_info 를 스텁으로 갈아끼우고 두 hook 모듈을
직접 호출해, 본실행 전에 shape / NaN / 컴포넌트 합 / 무상태성을 확인한다.

스텁의 StateIndex 값과 상태 배열 배치는 본체 실물과 동일하다:
    src/dogfight/sim/state_schema.py, FighterSim.py L170-244

이건 `--iterations 1` smoke의 **대체가 아니라 선행 검사**다. 여기를 통과해도
Release 본체에서 dry-run -> smoke -> 본실행 순서는 그대로 밟아야 한다.

실행:
    python student/tests/check_student_contracts.py
종료 코드 0 = 전부 통과.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))


# ---------------------------------------------------------------------------
# dogfight.sim.state_schema 스텁 (본체와 같은 인덱스 값)
# ---------------------------------------------------------------------------
class StateIndex:
    N = 0
    E = 1
    D = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    KCAS = 12
    FUEL = 23
    SIM_TIME = 41
    LAT = 42
    LON = 43
    ALT = 44
    HEALTH = 45


IDX_U, IDX_V, IDX_W = 6, 7, 8   # body 속도 [m/s], FighterSim.py L178-180
STATE_LEN = 51


def _install_stub_dogfight() -> None:
    pkg = types.ModuleType("dogfight")
    pkg.__path__ = []  # 패키지로 인식시킨다
    sim = types.ModuleType("dogfight.sim")
    sim.__path__ = []
    schema = types.ModuleType("dogfight.sim.state_schema")
    schema.StateIndex = StateIndex

    envs = types.ModuleType("dogfight.envs")
    envs.__path__ = []
    observation = types.ModuleType("dogfight.envs.observation")

    def normalize(value, low, high):
        span = high - low
        if span == 0:
            return 0.0
        return (float(value) - low) / span * 2.0 - 1.0

    observation.normalize = normalize

    sys.modules.update(
        {
            "dogfight": pkg,
            "dogfight.sim": sim,
            "dogfight.sim.state_schema": schema,
            "dogfight.envs": envs,
            "dogfight.envs.observation": observation,
        }
    )


_install_stub_dogfight()

from student import my_observation as obs_mod  # noqa: E402
from student import my_reward as rew_mod  # noqa: E402


# ---------------------------------------------------------------------------
# 스텁 helper
# ---------------------------------------------------------------------------
class GeoStub:
    """geo_info 스텁. 반환값을 직접 지정하거나 예외를 던지게 만들 수 있다."""

    def __init__(self, distance=1000.0, ata=0.0, aa=0.0, raise_on_call=False):
        self.distance = distance
        self.ata = ata
        self.aa = aa
        self.raise_on_call = raise_on_call

    def _maybe_raise(self):
        if self.raise_on_call:
            raise RuntimeError("geo_info 내부 예외")

    def _get_distance(self, own, tgt):
        self._maybe_raise()
        return self.distance

    def _get_antenna_train_angle(self, own, tgt, flag):
        self._maybe_raise()
        return self.ata

    def _get_aspect_angle(self, own, tgt, flag):
        self._maybe_raise()
        return self.aa


def make_state(
    alt=7000.0,      # meter (initial_scenario.altitude_m 기본값)
    kcas=200.0,      # meter/sec
    health=1.0,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    n=0.0,
    e=0.0,
    d=-7000.0,
    u=0.0,
    v=0.0,
    w=0.0,
):
    state = np.zeros(STATE_LEN, dtype=np.float64)
    state[StateIndex.N] = n
    state[StateIndex.E] = e
    state[StateIndex.D] = d
    state[StateIndex.ROLL] = roll
    state[StateIndex.PITCH] = pitch
    state[StateIndex.YAW] = yaw
    state[IDX_U] = u
    state[IDX_V] = v
    state[IDX_W] = w
    state[StateIndex.KCAS] = kcas
    state[StateIndex.ALT] = alt
    state[StateIndex.HEALTH] = health
    return state


F = obs_mod.describe_observation()["features"]
WEZ = {"angle_deg": 2.0, "min_range_m": 152.4, "max_range_m": 914.4}
FT_TO_M = 0.3048


def reward(
    own,
    tgt,
    geo,
    terminated=False,
    truncated=False,
    end_condition="",
    cfg=None,
    wez=None,
):
    return rew_mod.compute_reward(
        own,
        tgt,
        0.0,
        0.0,
        geo,
        WEZ if wez is None else wez,
        rew_mod.MY_REWARD_CONFIG if cfg is None else cfg,
        terminated,
        truncated,
        end_condition,
    )


# ---------------------------------------------------------------------------
# 검사 프레임
# ---------------------------------------------------------------------------
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


def assert_valid_obs(vector, what: str) -> None:
    check(isinstance(vector, np.ndarray), f"{what}: ndarray 반환")
    check(vector.dtype == np.float32, f"{what}: dtype float32 (got {vector.dtype})")
    check(vector.shape == (obs_mod.OBSERVATION_SIZE,), f"{what}: shape {vector.shape}")
    check(bool(np.all(np.isfinite(vector))), f"{what}: 모든 값 finite")
    check(
        bool(np.all(vector >= obs_mod.OBSERVATION_LOW))
        and bool(np.all(vector <= obs_mod.OBSERVATION_HIGH)),
        f"{what}: 값이 [{obs_mod.OBSERVATION_LOW}, {obs_mod.OBSERVATION_HIGH}] 안",
    )


# ---------------------------------------------------------------------------
# 1. 관측: 기본 계약
# ---------------------------------------------------------------------------
def test_observation_contract():
    section("1. 관측 기본 계약")
    described = obs_mod.describe_observation()
    check(
        described["size"] == obs_mod.OBSERVATION_SIZE,
        f"describe_observation().size == OBSERVATION_SIZE ({obs_mod.OBSERVATION_SIZE})",
    )
    check(
        len(described["features"]) == obs_mod.OBSERVATION_SIZE,
        f"feature 이름 개수 == OBSERVATION_SIZE (got {len(described['features'])})",
    )
    vector = obs_mod.build_observation(make_state(), make_state(), GeoStub(), WEZ)
    assert_valid_obs(vector, "정상 입력")


# ---------------------------------------------------------------------------
# 2. 관측: NaN 안전 가드
# ---------------------------------------------------------------------------
def test_observation_nan_guards():
    section("2. 관측 NaN 안전 가드")

    cases = {
        "상태에 NaN": (make_state(alt=math.nan, kcas=math.nan, u=math.nan), make_state(), GeoStub()),
        "상태에 +inf": (make_state(alt=math.inf, kcas=math.inf), make_state(), GeoStub()),
        "상태에 -inf": (make_state(alt=-math.inf), make_state(), GeoStub()),
        "자세에 NaN": (make_state(roll=math.nan, yaw=math.nan), make_state(), GeoStub()),
        "위치에 NaN": (make_state(n=math.nan), make_state(), GeoStub()),
        "geo가 NaN 반환": (
            make_state(),
            make_state(),
            GeoStub(distance=math.nan, ata=math.nan, aa=math.nan),
        ),
        "geo가 예외": (make_state(), make_state(), GeoStub(raise_on_call=True)),
        "거리 0": (make_state(), make_state(), GeoStub(distance=0.0)),
        "거리 음수": (make_state(), make_state(), GeoStub(distance=-500.0)),
        "거리 초대형": (make_state(), make_state(), GeoStub(distance=1e12)),
        "각도 범위 초과": (make_state(), make_state(), GeoStub(ata=1e6, aa=-1e6)),
        "속도 음수": (make_state(kcas=-200.0), make_state(), GeoStub()),
        "속도 초대형": (make_state(kcas=1e9, u=1e9), make_state(), GeoStub()),
        "두 기체 같은 위치": (
            make_state(n=0.0, e=0.0, d=0.0),
            make_state(n=0.0, e=0.0, d=0.0),
            GeoStub(distance=0.0),
        ),
    }
    for label, (own, tgt, geo) in cases.items():
        assert_valid_obs(obs_mod.build_observation(own, tgt, geo, WEZ), label)

    for label, wez in {
        "wez_config=None": None,
        "wez_config={}": {},
        "wez_config에 NaN": {"angle_deg": math.nan, "min_range_m": math.nan, "max_range_m": math.nan},
        "wez_config가 dict 아님": "garbage",
    }.items():
        assert_valid_obs(
            obs_mod.build_observation(make_state(), make_state(), GeoStub(), wez), label
        )

    short = np.zeros(2, dtype=np.float64)
    assert_valid_obs(
        obs_mod.build_observation(short, short, GeoStub(), WEZ), "상태 배열 길이 부족"
    )


# ---------------------------------------------------------------------------
# 3. 관측: 무상태성 (env가 step당 여러 번, 인자를 바꿔서 부른다)
# ---------------------------------------------------------------------------
def test_observation_stateless():
    section("3. 관측 무상태성")
    own = make_state(n=0.0, yaw=0.0, u=200.0)
    tgt = make_state(n=1000.0, yaw=0.0, u=100.0)
    geo = GeoStub(distance=1000.0, ata=10.0, aa=20.0)

    first = obs_mod.build_observation(own, tgt, geo, WEZ)
    again = obs_mod.build_observation(own, tgt, geo, WEZ)
    check(np.array_equal(first, again), "같은 입력을 반복 호출하면 같은 값")

    # target action provider 경로: 인자를 바꿔 호출해도 원래 호출이 오염되지 않아야 한다.
    obs_mod.build_observation(tgt, own, geo, WEZ)
    after_swap = obs_mod.build_observation(own, tgt, geo, WEZ)
    check(
        np.array_equal(first, after_swap),
        "ownship/target을 바꿔 부른 뒤에도 원래 호출 결과가 그대로",
    )

    # step_ratio 만큼 반복 호출되는 상황
    for _ in range(6):
        obs_mod.build_observation(tgt, own, geo, WEZ)
    after_burst = obs_mod.build_observation(own, tgt, geo, WEZ)
    check(np.array_equal(first, after_burst), "step_ratio 횟수만큼 추가 호출해도 동일")

    check(
        not hasattr(obs_mod, "reset_observation_state"),
        "이전 step 캐시가 남아 있지 않다 (reset 훅 불필요)",
    )


# ---------------------------------------------------------------------------
# 4. 관측: closure rate (속도 벡터 기반)
# ---------------------------------------------------------------------------
def test_observation_closure():
    section("4. closure rate")
    idx = obs_mod.describe_observation()["features"].index("closure_rate_norm")
    geo = GeoStub(distance=1000.0)

    # 적은 북쪽 1000 m. 내가 기수 북(yaw=0)으로 200 m/s -> 접근.
    closing = obs_mod.build_observation(
        make_state(n=0.0, yaw=0.0, u=200.0), make_state(n=1000.0, u=0.0), geo, WEZ
    )
    check(closing[idx] > 0.0, "적을 향해 비행 -> closure > 0")
    check(
        abs(float(closing[idx]) - 200.0 / obs_mod.CLOSURE_MAX_MS) < 1e-5,
        "closure 크기가 상대속도 투영과 일치 (200 m/s)",
    )

    opening = obs_mod.build_observation(
        make_state(n=0.0, yaw=180.0, u=200.0), make_state(n=1000.0, u=0.0), geo, WEZ
    )
    check(opening[idx] < 0.0, "적 반대로 비행 -> closure < 0")

    parallel = obs_mod.build_observation(
        make_state(n=0.0, yaw=0.0, u=200.0), make_state(n=1000.0, yaw=0.0, u=200.0), geo, WEZ
    )
    check(abs(float(parallel[idx])) < 1e-6, "같은 방향 같은 속도 -> closure 0")

    crossing = obs_mod.build_observation(
        make_state(n=0.0, yaw=90.0, u=200.0), make_state(n=1000.0, u=0.0), geo, WEZ
    )
    check(abs(float(crossing[idx])) < 1e-6, "LOS와 직각으로 비행 -> closure 0")

    head_on = obs_mod.build_observation(
        make_state(n=0.0, yaw=0.0, u=250.0), make_state(n=1000.0, yaw=180.0, u=250.0), geo, WEZ
    )
    check(head_on[idx] > float(closing[idx]), "정면 접근이 단독 접근보다 closure가 크다")


# ---------------------------------------------------------------------------
# 5. 관측: 의미 있는 값이 실제로 실리는가
# ---------------------------------------------------------------------------
def test_observation_semantics():
    section("5. 관측 의미 검증")
    features = obs_mod.describe_observation()["features"]
    i_energy = features.index("energy_advantage_norm")
    i_wez = features.index("in_wez_flag")
    i_ata = features.index("ata_norm")
    i_dist = features.index("distance_norm")
    i_alt = features.index("own_alt_norm")
    i_kcas = features.index("own_kcas_norm")

    inside = obs_mod.build_observation(
        make_state(), make_state(), GeoStub(distance=500.0, ata=0.0), WEZ
    )
    check(inside[i_wez] == 1.0, "WEZ 안이면 in_wez = +1")

    outside_angle = obs_mod.build_observation(
        make_state(), make_state(), GeoStub(distance=500.0, ata=30.0), WEZ
    )
    check(outside_angle[i_wez] == -1.0, "각도 벗어나면 in_wez = -1")

    outside_range = obs_mod.build_observation(
        make_state(), make_state(), GeoStub(distance=5000.0, ata=0.0), WEZ
    )
    check(outside_range[i_wez] == -1.0, "사거리 벗어나면 in_wez = -1")

    # 고도 우위 (meter). 3000 m 차이면 정규화 폭(3000 m)에서 포화 직전.
    superior = obs_mod.build_observation(
        make_state(alt=9000.0, kcas=300.0), make_state(alt=6000.0, kcas=150.0), GeoStub(), WEZ
    )
    check(superior[i_energy] > 0.0, "고도·속도 우위면 energy_advantage > 0")

    inferior = obs_mod.build_observation(
        make_state(alt=6000.0, kcas=150.0), make_state(alt=9000.0, kcas=300.0), GeoStub(), WEZ
    )
    check(inferior[i_energy] < 0.0, "고도·속도 열세면 energy_advantage < 0")

    # 속도만 다른 경우도 에너지 차로 잡혀야 한다 (KCAS가 m/s로 해석되는지 확인)
    fast = obs_mod.build_observation(
        make_state(alt=7000.0, kcas=250.0), make_state(alt=7000.0, kcas=200.0), GeoStub(), WEZ
    )
    expected = (250.0 ** 2 - 200.0 ** 2) / (2.0 * 9.80665) / obs_mod.ENERGY_SPAN_M
    check(
        abs(float(fast[i_energy]) - expected) < 1e-5,
        "속도차만 있을 때 energy_advantage = (V1^2-V2^2)/2g / span",
    )

    left = obs_mod.build_observation(make_state(), make_state(), GeoStub(ata=-90.0), WEZ)
    right = obs_mod.build_observation(make_state(), make_state(), GeoStub(ata=90.0), WEZ)
    check(left[i_ata] < 0.0 < right[i_ata], "ATA 부호가 보존된다")

    near = obs_mod.build_observation(make_state(), make_state(), GeoStub(distance=500.0), WEZ)
    far = obs_mod.build_observation(make_state(), make_state(), GeoStub(distance=15000.0), WEZ)
    check(near[i_dist] < far[i_dist], "거리 정규화가 단조 증가")

    # 교전 고도·속도가 정규화 범위 중앙 근처에 오는지 (해상도 확인)
    typical = obs_mod.build_observation(
        make_state(alt=7000.0, kcas=200.0), make_state(), GeoStub(), WEZ
    )
    check(
        -0.6 < float(typical[i_alt]) < 0.6 and -0.6 < float(typical[i_kcas]) < 0.6,
        "교전 고도 7000 m / 속도 200 m/s가 정규화 중앙 근처",
    )


# ---------------------------------------------------------------------------
# 6. 보상: 합 == total, NaN 없음
# ---------------------------------------------------------------------------
def test_reward_contract():
    section("6. 보상 기본 계약")
    rew_mod.reset_reward_state()
    total, components = reward(make_state(), make_state(), GeoStub())

    check(isinstance(total, float), "total이 float")
    check(isinstance(components, dict), "components가 dict")
    check(math.isfinite(total), "total이 finite")
    check(
        all(isinstance(v, float) and math.isfinite(v) for v in components.values()),
        "모든 컴포넌트가 finite float",
    )
    check(abs(sum(components.values()) - total) < 1e-9, "sum(components) == total")

    expected = {
        "step",
        "pursuit",
        "position",
        "wez_entry",
        "wez_hold",
        "overclose",
        "energy",
        "crash",
        "terminal",
    }
    check(set(components) == expected, f"컴포넌트 키 집합 고정 {sorted(expected)}")


def test_reward_nan_guards():
    section("7. 보상 NaN 안전 가드")
    cases = {
        "상태에 NaN": (make_state(alt=math.nan, kcas=math.nan, health=math.nan), make_state(), GeoStub()),
        "상태에 inf": (make_state(alt=math.inf), make_state(alt=-math.inf), GeoStub()),
        "geo가 NaN": (make_state(), make_state(), GeoStub(distance=math.nan, ata=math.nan, aa=math.nan)),
        "geo가 예외": (make_state(), make_state(), GeoStub(raise_on_call=True)),
        "거리 0 (0 division 후보)": (make_state(), make_state(), GeoStub(distance=0.0)),
        "거리 음수": (make_state(), make_state(), GeoStub(distance=-100.0)),
        "속도 0": (make_state(kcas=0.0), make_state(kcas=0.0), GeoStub()),
        "상태 배열 길이 부족": (np.zeros(2), np.zeros(2), GeoStub()),
    }
    for label, (own, tgt, geo) in cases.items():
        rew_mod.reset_reward_state()
        total, components = reward(own, tgt, geo)
        ok = (
            math.isfinite(total)
            and all(math.isfinite(v) for v in components.values())
            and abs(sum(components.values()) - total) < 1e-9
        )
        check(ok, f"{label}: finite + 합 일치")

    for label, cfg in {
        "reward_config=None": None,
        "reward_config={}": {},
        "reward_config에 NaN": {k: math.nan for k in rew_mod.MY_REWARD_CONFIG},
        "reward_config가 dict 아님": "garbage",
    }.items():
        rew_mod.reset_reward_state()
        total, components = reward(make_state(), make_state(), GeoStub(), cfg=cfg)
        check(
            math.isfinite(total) and all(math.isfinite(v) for v in components.values()),
            f"{label}: finite",
        )

    for label, wez in {
        "wez_config=None": None,
        "wez_config에 NaN": {"angle_deg": math.nan, "min_range_m": math.nan, "max_range_m": math.nan},
        "wez_config 역전(min>max)": {"angle_deg": 2.0, "min_range_m": 5000.0, "max_range_m": 100.0},
    }.items():
        rew_mod.reset_reward_state()
        total, components = reward(make_state(), make_state(), GeoStub(), wez=wez)
        check(
            math.isfinite(total) and all(math.isfinite(v) for v in components.values()),
            f"{label}: finite",
        )


# ---------------------------------------------------------------------------
# 8. 보상: 각도 shaping 과 상쇄 위험
# ---------------------------------------------------------------------------
def test_reward_angle_shaping():
    section("8. 각도 shaping / 상쇄 점검")
    rew_mod.reset_reward_state()
    _, nose_on = reward(make_state(), make_state(), GeoStub(distance=3000.0, ata=0.0, aa=0.0))
    check(nose_on["pursuit"] > 0.0, "ATA=0 (기수가 적을 향함) -> pursuit > 0")
    check(nose_on["position"] > 0.0, "AA=0 (적의 6시) -> position > 0")

    rew_mod.reset_reward_state()
    _, nose_off = reward(make_state(), make_state(), GeoStub(distance=3000.0, ata=180.0, aa=0.0))
    check(nose_off["pursuit"] < 0.0, "ATA=180 (적이 내 뒤) -> pursuit < 0")

    rew_mod.reset_reward_state()
    _, head_on = reward(make_state(), make_state(), GeoStub(distance=3000.0, ata=0.0, aa=180.0))
    check(head_on["position"] < 0.0, "헤드온(AA=180) -> position < 0")
    check(
        head_on["pursuit"] + head_on["position"] > 0.0,
        "헤드온에서 pursuit+position 순합 > 0 (상쇄로 뒤집히지 않음)",
    )

    bad_cfg = dict(rew_mod.MY_REWARD_CONFIG)
    bad_cfg["pursuit_weight"] = 0.3
    bad_cfg["position_weight"] = 0.9
    rew_mod.reset_reward_state()
    _, guarded = reward(
        make_state(), make_state(), GeoStub(distance=3000.0, ata=0.0, aa=180.0), cfg=bad_cfg
    )
    check(
        guarded["pursuit"] + guarded["position"] > 0.0,
        "position_weight >= pursuit_weight 로 설정해도 상쇄 가드가 동작",
    )


# ---------------------------------------------------------------------------
# 9. 보상: WEZ 안전 밴드
# ---------------------------------------------------------------------------
def test_reward_wez_band():
    section("9. WEZ 안전 밴드 (500~3000 ft)")
    in_band_m = 500.0 * FT_TO_M + 50.0
    too_close_m = 300.0 * FT_TO_M
    way_too_close_m = 50.0 * FT_TO_M

    rew_mod.reset_reward_state()
    geo = GeoStub(distance=in_band_m, ata=0.0, aa=0.0)
    _, first = reward(make_state(), make_state(), geo)
    check(first["wez_entry"] > 0.0, "밴드 최초 진입 -> wez_entry > 0")
    check(first["overclose"] == 0.0, "밴드 안에서는 overclose 없음")

    _, second = reward(make_state(), make_state(), geo)
    check(second["wez_hold"] > 0.0, "밴드 유지 -> wez_hold > 0")
    check(second["wez_entry"] == 0.0, "유지 중에는 진입 보너스 재지급 안 함")

    geo.distance = 5000.0
    reward(make_state(), make_state(), geo)
    geo.distance = in_band_m
    _, re_entry = reward(make_state(), make_state(), geo)
    check(
        0.0 < re_entry["wez_entry"] < first["wez_entry"],
        "재진입 보너스는 최초보다 작다 (decay)",
    )

    rew_mod.reset_reward_state()
    _, close = reward(make_state(), make_state(), GeoStub(distance=too_close_m, ata=0.0))
    check(close["overclose"] < 0.0, "500 ft 미만 -> overclose 패널티")
    check(close["wez_entry"] == 0.0, "과근접에서는 진입 보너스 없음")

    rew_mod.reset_reward_state()
    _, closer = reward(make_state(), make_state(), GeoStub(distance=way_too_close_m, ata=0.0))
    check(closer["overclose"] < close["overclose"], "더 파고들수록 패널티가 커진다")

    rew_mod.reset_reward_state()
    _, outside = reward(make_state(), make_state(), GeoStub(distance=5000.0, ata=0.0))
    check(
        outside["wez_entry"] == 0.0 and outside["wez_hold"] == 0.0 and outside["overclose"] == 0.0,
        "밴드 밖에서는 WEZ 항이 전부 0",
    )


# ---------------------------------------------------------------------------
# 10. 보상: 에너지 항
# ---------------------------------------------------------------------------
def test_reward_energy():
    section("10. 에너지 항")
    geo = GeoStub(distance=3000.0)

    rew_mod.reset_reward_state()
    _, first = reward(make_state(alt=7000.0), make_state(alt=7000.0), geo)
    check(first["energy"] == 0.0, "첫 step의 energy는 0 (이전 값 없음)")

    _, gained = reward(make_state(alt=7010.0), make_state(alt=7000.0), geo)
    check(gained["energy"] > 0.0, "에너지 우위가 늘면 energy > 0")

    _, lost = reward(make_state(alt=6990.0), make_state(alt=7000.0), geo)
    check(lost["energy"] < 0.0, "에너지 우위가 줄면 energy < 0")

    rew_mod.reset_reward_state()
    reward(make_state(alt=7000.0), make_state(alt=7000.0), geo)
    _, huge = reward(make_state(alt=7300.0, kcas=350.0), make_state(alt=7000.0), geo)
    check(
        abs(huge["energy"]) <= abs(rew_mod.MY_REWARD_CONFIG["energy_weight"]) + 1e-9,
        "energy 항은 energy_weight로 상한이 걸린다 (소가중치)",
    )


# ---------------------------------------------------------------------------
# 11. 보상: 종료 처리
# ---------------------------------------------------------------------------
def test_reward_terminal():
    section("11. 종료 보상 / 추락 패널티")
    geo = GeoStub(distance=3000.0)

    rew_mod.reset_reward_state()
    _, win = reward(
        make_state(health=1.0), make_state(health=0.0), geo, terminated=True, end_condition="target_dead"
    )
    check(win["terminal"] > 0.0, "적 격추 -> terminal > 0")
    check(win["crash"] == 0.0, "격추승에는 crash 패널티 없음")

    rew_mod.reset_reward_state()
    _, lose = reward(
        make_state(health=0.0), make_state(health=1.0), geo, terminated=True, end_condition="ownship_dead"
    )
    check(lose["terminal"] < 0.0, "피격추 -> terminal < 0")

    rew_mod.reset_reward_state()
    _, draw = reward(
        make_state(health=1.0), make_state(health=1.0), geo, truncated=True, end_condition="max_time"
    )
    check(draw["terminal"] < 0.0, "시간 초과 무승부 -> terminal < 0 (작게)")
    check(draw["crash"] == 0.0, "max_time은 추락이 아니다")

    # 본체가 실제로 쓰는 문자열 (envs/termination.py L22-53, single_agent_env.py L292/L299)
    for cond in ("ownship altitude below min", "FDM Update Fail", "Ownship FDM output Fall"):
        rew_mod.reset_reward_state()
        _, crashed = reward(
            make_state(health=1.0), make_state(health=1.0), geo, terminated=True, end_condition=cond
        )
        check(crashed["crash"] < 0.0, f"아군 추락 '{cond}' -> crash 패널티")

    # 적기 쪽 종료를 우리 추락으로 세면 승리 직전에 -150을 맞는다.
    for cond in ("target altitude below min", "Target FDM output Fall", "target destroyed"):
        rew_mod.reset_reward_state()
        _, not_crash = reward(
            make_state(health=1.0), make_state(health=0.0), geo, terminated=True, end_condition=cond
        )
        check(not_crash["crash"] == 0.0, f"적기 쪽 '{cond}' -> crash 패널티 없음")

    for cond in ("max time out", "episode step limit", "fuel fail", "two circle headon guard fail"):
        rew_mod.reset_reward_state()
        _, benign = reward(
            make_state(health=1.0), make_state(health=1.0), geo, truncated=True, end_condition=cond
        )
        check(benign["crash"] == 0.0, f"'{cond}'은 추락으로 오인하지 않는다")

    rew_mod.reset_reward_state()
    _, running = reward(make_state(), make_state(), geo)
    check(
        running["terminal"] == 0.0 and running["crash"] == 0.0,
        "진행 중 step에는 terminal/crash 항이 0",
    )


# ---------------------------------------------------------------------------
# 12. 보상: 에피소드 경계 / env 격리
# ---------------------------------------------------------------------------
def test_reward_state_isolation():
    section("12. 에피소드 경계 / env 격리")
    in_band_m = 500.0 * FT_TO_M + 50.0

    rew_mod.reset_reward_state()
    geo = GeoStub(distance=in_band_m, ata=0.0)
    _, first = reward(make_state(), make_state(), geo)
    check(first["wez_entry"] > 0.0, "에피소드 1: 최초 진입 보너스")

    reward(make_state(health=1.0), make_state(health=0.0), geo, terminated=True, end_condition="target_dead")

    _, next_ep = reward(make_state(), make_state(), geo)
    check(
        abs(next_ep["wez_entry"] - first["wez_entry"]) < 1e-9,
        "에피소드가 끝나면 진입 보너스 decay가 초기화된다",
    )

    rew_mod.reset_reward_state()
    geo_a = GeoStub(distance=in_band_m, ata=0.0)
    geo_b = GeoStub(distance=in_band_m, ata=0.0)
    _, a_first = reward(make_state(), make_state(), geo_a)
    _, b_first = reward(make_state(), make_state(), geo_b)
    check(
        a_first["wez_entry"] > 0.0 and abs(b_first["wez_entry"] - a_first["wez_entry"]) < 1e-9,
        "env별 WEZ 상태 격리 (geo_info 키 분리)",
    )


# ---------------------------------------------------------------------------
# 13. 두 모듈의 단위 해석 일치
# ---------------------------------------------------------------------------
def test_cross_module_consistency():
    section("13. 모듈 간 단위 해석 일치")
    state = make_state(alt=7000.0, kcas=250.0)
    check(
        abs(obs_mod._specific_energy_m(state) - rew_mod._specific_energy_m(state)) < 1e-9,
        "비에너지 계산이 my_observation / my_reward에서 동일",
    )
    # ALT=meter, KCAS=m/s 해석이 맞는지 직접 확인
    expected = 7000.0 + 250.0 ** 2 / (2.0 * 9.80665)
    check(
        abs(obs_mod._specific_energy_m(state) - expected) < 1e-6,
        "Es = ALT[m] + KCAS[m/s]^2 / 2g (단위 변환 없음)",
    )
    check(
        obs_mod.DEFAULT_WEZ_MIN_RANGE_M == rew_mod.DEFAULT_WEZ_MIN_RANGE_M
        and obs_mod.DEFAULT_WEZ_MAX_RANGE_M == rew_mod.DEFAULT_WEZ_MAX_RANGE_M
        and obs_mod.DEFAULT_WEZ_ANGLE_DEG == rew_mod.DEFAULT_WEZ_ANGLE_DEG,
        "WEZ fallback 상수가 두 모듈에서 동일",
    )

    # single_agent_env.update_damage(): half_wez_angle_deg = wez["angle_deg"] / 2.0
    obs_half, _, _ = obs_mod._wez_thresholds(WEZ)
    rew_half, _, _ = rew_mod._wez_thresholds_ft(WEZ)
    check(
        abs(obs_half - WEZ["angle_deg"] / 2.0) < 1e-12
        and abs(rew_half - WEZ["angle_deg"] / 2.0) < 1e-12,
        "WEZ half angle = angle_deg/2 (update_damage와 같은 기준)",
    )

    # 대미지 판정 밖인 1.5도에서 WEZ로 오인하지 않는지
    obs = obs_mod.build_observation(
        make_state(), make_state(), GeoStub(distance=500.0, ata=1.5), WEZ
    )
    check(float(obs[F.index("in_wez_flag")]) == -1.0, "ATA 1.5도는 WEZ 밖 (half angle 1.0도)")
    rew_mod.reset_reward_state()
    _, comps = reward(make_state(), make_state(), GeoStub(distance=500.0, ata=1.5))
    check(comps["wez_entry"] == 0.0, "ATA 1.5도에서는 진입 보너스 없음")


def test_degenerate_angle_repair():
    section("14. GeoMathUtil sign=0 축퇴 보정")
    # 실제 GeoMathUtil은 같은 고도 정후방에서 ATA 180 대신 0을 반환한다.
    # 스텁으로 그 상황을 재현하고, 보정이 실제 크기를 복구하는지 본다.
    own = make_state(n=1000.0, d=-7000.0, yaw=0.0)
    tgt = make_state(n=0.0, d=-7000.0, yaw=0.0)
    check(
        abs(obs_mod._ata_magnitude_deg(own, tgt) - 180.0) < 1e-9,
        "보정 함수가 실제 ATA 크기 180을 계산",
    )
    check(
        abs(obs_mod._aa_magnitude_deg(own, tgt) - 180.0) < 1e-9,
        "같은 상황의 실제 AA 크기는 180 (내가 적의 앞 = 적이 나를 겨눔)",
    )

    # geo가 0을 보고하면 보정이 개입한다
    obs = obs_mod.build_observation(own, tgt, GeoStub(distance=1000.0, ata=0.0), WEZ)
    check(
        abs(float(obs[F.index("ata_norm")])) > 0.99,
        "geo가 0을 보고해도 관측 ata_norm은 ~ +-1로 복구",
    )
    rew_mod.reset_reward_state()
    _, comps = reward(own, tgt, GeoStub(distance=1000.0, ata=0.0))
    check(comps["pursuit"] < 0.0, "보정 후 pursuit < 0 (보정 없으면 +0.6)")

    # 진짜로 정조준인 경우 (실제 크기도 0) 에는 그대로 0
    nose_on_own = make_state(n=0.0, d=-7000.0, yaw=0.0)
    nose_on_tgt = make_state(n=1000.0, d=-7000.0, yaw=0.0)
    obs = obs_mod.build_observation(
        nose_on_own, nose_on_tgt, GeoStub(distance=500.0, ata=0.0), WEZ
    )
    check(
        abs(float(obs[F.index("ata_norm")])) < 1e-9,
        "실제로 정조준이면 0을 유지 (오보정 없음)",
    )
    check(float(obs[F.index("in_wez_flag")]) == 1.0, "실제 정조준 + 밴드 안 -> WEZ 인정")

    # 축퇴가 아닌 값은 건드리지 않는다
    reported = 37.5
    check(
        obs_mod._repair_degenerate_angle(
            reported, obs_mod._ata_magnitude_deg, own, tgt
        ) == reported,
        "0이 아닌 보고값은 그대로 통과",
    )


# ---------------------------------------------------------------------------
def main() -> int:
    print("student hook 계약 검증 (DogFightEnv 본체 없이 실행)")
    print(f"  observation: {obs_mod.OBSERVATION_MODE} / size={obs_mod.OBSERVATION_SIZE}")

    test_observation_contract()
    test_observation_nan_guards()
    test_observation_stateless()
    test_observation_closure()
    test_observation_semantics()
    test_reward_contract()
    test_reward_nan_guards()
    test_reward_angle_shaping()
    test_reward_wez_band()
    test_reward_energy()
    test_reward_terminal()
    test_reward_state_isolation()
    test_cross_module_consistency()
    test_degenerate_angle_repair()

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
