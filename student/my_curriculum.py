# -*- coding: utf-8 -*-
"""STIL 고도 커리큘럼 — 7,000 m 에서 대회 초기 배치(762 m)까지 계단으로 내려간다.

왜 필요한가 (2026-08-05)
------------------------
대회 초기 배치는 약 2000~3000 ft(609.6~914.4 m)이고 추락 판정은 300 m 다
(`Docs/competiton_rules.md` §5). 시작하자마자 지면까지 여유가 462 m 뿐이다.
그 조건에서 처음부터 학습시키면 정책이 교전을 배우기 전에 지면에 닿는다 —
`sac_mlp_obs8_match_v1` 실측에서 에피소드 길이가 17초에서 12.6초로 **줄었고**
추락률은 1.00 이었다. 200초 라운드의 6% 만 살아남는다.

그래서 고도를 계단으로 내린다. 높은 곳에서 비행과 교전을 먼저 배우고,
추락률이 기준 아래로 내려갈 때만 다음 고도로 내려간다.

실행
----
    python train_curriculum.py --algorithm sac --stages-module student.my_curriculum \\
        --reward-module student.my_reward --observation-module student.my_observation \\
        --output-name stil --output-tag curriculum_alt_v1

무작위화를 켜지 않는다 — 플랫폼 결함이다
----------------------------------------
`ownship_randomization.enabled: True` 는 `default` 시나리오 모드에서 **누적된다.**
`add_random_init_position()` 이 `fighter._init_pos_* += ...` 로 더하는데
(`single_agent_env.py:844-849`), `default` 모드에는 매 reset 마다 기준값을 다시
세우는 곳이 없다. `change_init_position()` 은 `two_circle_headon` /
`ref_old_random` 경로에서만 불린다(`:657,667,694,705`).

결과는 에피소드마다 이어지는 random walk 다. radius=500 기준 추정:

    100 에피소드 후  |D축 누적| 중앙 2,056 m (최대 7,683 m)
    1000 에피소드 후                6,396 m (최대 25,535 m)

762 m 에서 시작하면 수십 판 만에 지면 아래가 된다. 게다가 `default` 모드에서는
**ownship 만** 흔들려서(`:252`) 교전 기하 자체가 무너진다.

> 벤더 커리큘럼(`src/dogfight/ai/curriculum.py`)의 스테이지 0~4 도 이 경로를 쓴다.
> 참고할 때 그대로 가져오지 말 것.

초기 위치를 다양화하려면 `env_overrides` 로 스테이지마다 `ownship`/`target` 을 직접
바꾸는 편이 안전하다. 그건 매번 절대값을 덮어쓰므로 누적되지 않는다.

고도와 `altitude_safe_band_m` 은 함께 내려간다
-----------------------------------------------
보상의 고도 항은 `margin = clip((alt - 300) / band, 0, 1)` 을 쓴다. band 를 고정한 채
시작 고도만 내리면 첫 step 부터 상수 패널티가 깔려 교전 신호를 덮는다
(762 m + band 2000 이면 -0.237/step, 200초 누적 -474 로 crash_penalty 보다 크다).
그래서 `band = clip(시작고도 - 300, 500, 2000)` 으로 함께 줄인다. 스테이지 시작
고도에서는 신호가 거의 0 이고 내려갈수록 가팔라진다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage

# 200초 라운드 x sim_hz 60. episode_step_limit 은 내부 tick 단위다
# (호스트 기본값 18000 = 300초 x 60 에서 확인).
MATCH_STEP_LIMIT = 12000

# 대회 초기 배치 고도. 2000~3000 ft 의 중앙인 2500 ft.
MATCH_ALTITUDE_M = 762.0

# env min_altitude. 추락 판정선.
FLOOR_M = 300.0


def _placement(altitude_m: float) -> dict:
    """양 기체를 같은 고도에 헤드온으로 놓는다.

    [N, E, D, roll, pitch, heading, speed]. D 는 아래 방향이라 고도는 -D 다.
    N 간격 5,000 m / 헤드온 / 300 m/s 는 호스트 기본값 그대로 두고 고도만 바꾼다.
    """
    d = -abs(float(altitude_m))
    return {
        "ownship": [1000.0, 0.0, d, 0.0, 0.0, 0.0, 300.0],
        "target": [6000.0, 0.0, d, 0.0, 0.0, 180.0, 300.0],
    }


def _altitude_band(altitude_m: float) -> float:
    """그 고도에서 쓸 `altitude_safe_band_m`.

    시작 고도에서 패널티가 거의 0 이 되도록 `시작고도 - 추락선` 으로 잡되
    2000 m 를 넘기지 않는다. 넘기면 정상 교전 고도에서도 패널티가 깔린다.
    """
    return float(min(max(altitude_m - FLOOR_M, 500.0), 2000.0))


def _stage(index: int, name: str, altitude_m: float, target_mode: str,
           max_iterations: int, crash_max: float | None,
           description: str) -> CurriculumStage:
    # 추락률이 이 아래로 내려갈 때만 다음 고도로 간다.
    # 마지막 스테이지는 조건이 없다 — 더 내려갈 곳이 없다.
    conditions: dict = {} if crash_max is None else {"crash_rate_max": crash_max}

    return CurriculumStage(
        index=index,
        name=name,
        description=description,
        target_mode=target_mode,
        episode_step_limit=MATCH_STEP_LIMIT,
        max_iterations=max_iterations,
        checkpoint_interval=25,
        reward_overrides={"altitude_safe_band_m": _altitude_band(altitude_m)},
        randomization={"enabled": False},   # 위 '플랫폼 결함' 참조
        advance_conditions=conditions,
        advance_window=10,
        env_overrides={
            **_placement(altitude_m),
            "max_engage_time": 200.0,
            "target_behavior_dll": "AIP_STIL.dll",
        },
    )


def get_stages() -> list[CurriculumStage]:
    """고도 계단: 7000 -> 3000 -> 1500 -> 762 m.

    스테이지 0 만 고정 표적이다. "떨어지지 않고 나는 법" 을 먼저 배우게 하고
    그 다음부터 기동하는 상대를 붙인다.

    `crash_rate_max` 0.25 는 실측 근거가 있는 값이 아니라 출발점이다.
    스테이지 0 에서 이 조건을 못 넘으면 문제는 고도 배치가 아니라 다른 데 있다.
    """
    return [
        _stage(0, "alt7000_fixed", 7000.0, "fixed", 100, 0.25,
               "고고도 고정 표적. 비행 자체를 먼저 배운다."),
        _stage(1, "alt3000_bt", 3000.0, "behavior_tree", 100, 0.25,
               "중고도 BT 상대. 교전을 배우되 지면 여유는 아직 넉넉하다."),
        _stage(2, "alt1500_bt", 1500.0, "behavior_tree", 100, 0.25,
               "저고도 진입. 지면 여유 1,200 m."),
        _stage(3, "alt762_match", MATCH_ALTITUDE_M, "behavior_tree", 100, None,
               "대회 초기 배치(2500 ft). 지면 여유 462 m."),
    ]


__all__ = ["get_stages"]
