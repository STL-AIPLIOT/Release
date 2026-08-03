# -*- coding: utf-8 -*-
"""RL bundle vs BT 반복 스파링 루프.

run_local_dogfight.py 는 1회 교전만 수행하고 --episodes / --seed / 결과 저장
옵션이 없다(parse_args 확인). 이 스크립트는 같은 구성 요소를 그대로 쓰되
반복 실행과 경기별 기록을 추가한다.

실행 예:
    python student\\tools\\sparring_loop.py ^
        --rl-bundle artifacts\\models\\stil\\sac_mlp_obs8_iter400 ^
        --bt-dll AIP_STIL.dll ^
        --observation-mode custom ^
        --observation-module student.my_observation ^
        --episodes 10 ^
        --out artifacts\\sparring\\obs8_vs_btv3.json

관측 설정은 train / evaluation / bundle / sparring 에서 동일해야 한다.
이 스크립트는 실제 사용된 값을 결과 파일에 함께 기록해 사후 대조가 가능하게 한다.

알려진 플랫폼 결함 우회
----------------------
dogfight.ai.inference_env.RLLibInferenceEnv 는 관측 크기를

    size = config.get("observation_size", observation_size(mode))

로 정하는데, observation_size() 는 tactical16/relative14 외의 모드에 대해
전부 12를 돌려준다(envs/observation.py:17-22). bundle 의 env_config 에는
observation_size 키가 없고 정답은 observation_summary.size 에만 있다.
그 결과 8차원 bundle 을 로드하면 12차원 모델이 만들어져

    RuntimeError: size mismatch for pi_encoder.net.mlp.0.weight
    copying a param with shape torch.Size([256, 8]) ... shape ... torch.Size([256, 12])

로 실패한다. src/dogfight/ 는 수정 금지 대상이므로, 여기서는 메모리상의
metadata 사본에 observation_size 를 주입해 우회한다. 파일은 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from DogFightEnvWrapper import DogFightWrapper                      # noqa: E402
from dogfight.ai.bt_action_provider import BTActionProvider         # noqa: E402
from dogfight.ai.bt_rule_manager import activate_rule_xml           # noqa: E402
from dogfight.ai.rl_action_provider import RLActionProvider         # noqa: E402
from dogfight.ai.rllib_utils import build_algorithm_from_bundle     # noqa: E402
from dogfight.ai.student_hooks import load_observation_hook         # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RL bundle vs BT 반복 스파링")
    p.add_argument("--rl-bundle", required=True, help="RL bundle 디렉터리")
    p.add_argument("--bt-dll", default="AIP_BASE_target.dll", help="상대 BT DLL")
    p.add_argument("--bt-rule-xml", default="", help="교전 중 활성화할 Rule.xml")
    p.add_argument("--bt-version", default="v3", help="결과에 기록할 BT 버전 문자열")
    p.add_argument("--observation-mode", default="custom")
    p.add_argument("--observation-module", default="student.my_observation")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=0, help="episode i 의 seed = seed_base + i")
    p.add_argument("--max-engage-time", type=float, default=120.0)
    p.add_argument("--episode-step-limit", type=int, default=7200)
    p.add_argument("--min-altitude", type=float, default=300.0)
    p.add_argument("--out", default="artifacts/sparring/sparring_result.json")
    # seed 만으로는 초기 조건이 바뀌지 않는다. config.py:45-51 의 기본값이
    # ownship_randomization.enabled = False 이고 initial_scenario.mode = "default" 라
    # single_agent_env.reset:246-247 이 아무 무작위화도 적용하지 않기 때문이다.
    # 경기마다 다른 상황을 보려면 아래를 켜야 한다.
    p.add_argument("--randomize-init", action="store_true",
                   help="경기마다 초기 위치/자세를 흩뿌린다 (seed 가 실제로 작동하게 됨)")
    p.add_argument("--rand-radius", type=float, default=500.0)
    p.add_argument("--rand-roll", type=float, default=10.0)
    p.add_argument("--rand-pitch", type=float, default=5.0)
    p.add_argument("--rand-heading", type=float, default=30.0)
    return p.parse_args()


def make_rl_provider(bundle_dir: str, observation_size: int):
    """관측 크기를 주입한 metadata 로 RLActionProvider 를 만든다."""

    def factory(metadata: dict):
        md = copy.deepcopy(metadata)
        cfg = md.setdefault("algorithm_config", {})
        env_cfg = cfg.setdefault("env_config", {})
        if "observation_size" not in env_cfg:
            env_cfg["observation_size"] = int(observation_size)
        return build_algorithm_from_bundle(md)

    return RLActionProvider(bundle_dir=bundle_dir, algorithm_factory=factory)


def classify(info: dict, total_reward: float) -> tuple[str, bool, bool]:
    """(winner, rl_crash, bt_crash) 를 종료 사유와 체력으로 판정한다."""
    end = str(info.get("end_condition", "")).lower()
    own_hp = info.get("ownship_health")
    tgt_hp = info.get("target_health")

    rl_crash = "ownship altitude below min" in end or ("fdm" in end and "ownship" in end)
    bt_crash = "target altitude below min" in end

    if "ownship destroyed" in end or rl_crash:
        winner = "bt"
    elif "target destroyed" in end or bt_crash:
        winner = "rl"
    elif isinstance(own_hp, (int, float)) and isinstance(tgt_hp, (int, float)):
        if own_hp > tgt_hp:
            winner = "rl"
        elif tgt_hp > own_hp:
            winner = "bt"
        else:
            winner = "draw"
    else:
        winner = "draw"
    return winner, bool(rl_crash), bool(bt_crash)


def main() -> int:
    args = parse_args()

    hook = load_observation_hook(args.observation_module) if args.observation_module else None
    obs_size = hook["size"] if hook else None
    obs_mode = hook["mode"] if hook else args.observation_mode

    print(f"[sparring] observation_mode={obs_mode} module={args.observation_module} size={obs_size}")
    print(f"[sparring] rl_bundle={args.rl_bundle}  bt_dll={args.bt_dll}  episodes={args.episodes}")

    records: list[dict] = []
    failures = 0

    for i in range(args.episodes):
        seed = args.seed_base + i
        rec = {
            "episode": i,
            "seed": seed,
            "winner": None,
            "termination_reason": None,
            "rl_reward": None,
            "rl_crash": None,
            "bt_crash": None,
            "duration": None,
            "rl_bundle": args.rl_bundle,
            "bt_version": args.bt_version,
            "observation_mode": obs_mode,
            "observation_module": args.observation_module,
            "observation_size": obs_size,
            "error": None,
        }
        t0 = time.time()
        env = None
        try:
            rl_provider = make_rl_provider(args.rl_bundle, obs_size or 12)
            bt_provider = BTActionProvider(dll_name=args.bt_dll)
            env_config = {
                "observation_mode": obs_mode,
                "observation_module": args.observation_module,
                "ownship_control_mode": "rl",
                "target_mode": "rl",
                "max_engage_time": args.max_engage_time,
                "episode_step_limit": args.episode_step_limit,
                "min_altitude": args.min_altitude,
            }
            if args.randomize_init:
                env_config["ownship_randomization"] = {
                    "enabled": True,
                    "radius": args.rand_radius,
                    "r_roll": args.rand_roll,
                    "r_pitch": args.rand_pitch,
                    "r_heading": args.rand_heading,
                }
            with activate_rule_xml(args.bt_rule_xml or None, ROOT):
                env = DogFightWrapper(
                    env_config=env_config,
                    observation_fn=hook["build_observation"] if hook else None,
                    observation_size=obs_size,
                    observation_low=hook["low"] if hook else None,
                    observation_high=hook["high"] if hook else None,
                    ownship_action_provider=rl_provider,
                    target_action_provider=bt_provider,
                )
                # env.reset(seed=...) 로 넘겨야 한다. np.random.seed() 만으로는
                # 초기 조건이 바뀌지 않는다(single_agent_env.reset:232-233).
                np.random.seed(seed)
                _obs, info = env.reset(seed=seed)
                terminated = truncated = False
                total = 0.0
                while not (terminated or truncated):
                    _obs, r, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                    total += float(r)

            winner, rl_crash, bt_crash = classify(info, total)
            rec.update(
                winner=winner,
                termination_reason=info.get("end_condition", ""),
                rl_reward=round(total, 4),
                rl_crash=rl_crash,
                bt_crash=bt_crash,
                duration=round(time.time() - t0, 2),
            )
            print(f"  ep {i:>3} seed={seed} winner={winner:<5} "
                  f"reason={rec['termination_reason']!r} reward={rec['rl_reward']} "
                  f"({rec['duration']}s)")
        except Exception as exc:  # 개별 경기 실패는 루프를 멈추지 않는다
            failures += 1
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["duration"] = round(time.time() - t0, 2)
            print(f"  ep {i:>3} seed={seed} 실패: {rec['error']}")
            traceback.print_exc()
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        records.append(rec)

    ok = [r for r in records if r["error"] is None]
    def rate(pred):
        return round(sum(1 for r in ok if pred(r)) / len(ok), 4) if ok else None

    summary = {
        "total_episodes": len(records),
        "completed_episodes": len(ok),
        "failed_episodes": failures,
        "rl_wins": sum(1 for r in ok if r["winner"] == "rl"),
        "bt_wins": sum(1 for r in ok if r["winner"] == "bt"),
        "draws": sum(1 for r in ok if r["winner"] == "draw"),
        "rl_win_rate": rate(lambda r: r["winner"] == "rl"),
        "rl_crash_rate": rate(lambda r: r["rl_crash"]),
        "bt_crash_rate": rate(lambda r: r["bt_crash"]),
        "avg_rl_reward": round(sum(r["rl_reward"] for r in ok) / len(ok), 4) if ok else None,
        "avg_duration": round(sum(r["duration"] for r in ok) / len(ok), 2) if ok else None,
        "rl_bundle": args.rl_bundle,
        "bt_version": args.bt_version,
        "observation_mode": obs_mode,
        "observation_module": args.observation_module,
        "observation_size": obs_size,
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    print("\n=== 최종 통계 ===")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")
    print(f"\n결과 저장: {out}")
    print(f"           {csv_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
