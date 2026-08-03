# -*- coding: utf-8 -*-
"""bundle metadata.json 에 observation_size 를 주입해 추론 경로 로드 실패를 고친다.

문제
----
src/dogfight/ai/inference_env.py:21-22

    mode = config.get("observation_mode", "classic12")
    size = int(config.get("observation_size", observation_size(mode)))

src/dogfight/envs/observation.py:17-22

    def observation_size(mode):
        if mode == "tactical16": return 16
        if mode == "relative14": return 14
        return 12                      # 그 외 전부 12

bundle 의 env_config.observation_mode 는 커스텀 훅 이름(예: "stil8")이라
observation_size() 가 12를 돌려준다. 그런데 bundle env_config 에는
observation_size 키가 없고, 정답은 observation_summary.size 에만 들어 있다.
RLLibInferenceEnv 는 그 키를 읽지 않으므로 12차원 모델이 만들어지고

    RuntimeError: size mismatch for pi_encoder.net.mlp.0.weight:
      copying a param with shape torch.Size([256, 8]) ...
      the shape in current model is torch.Size([256, 12])

로 실패한다. run_local_dogfight.py 를 포함한 모든 추론 경로가 영향을 받는다.

해결
----
src/dogfight/ 는 수정 금지 대상이다. 대신 bundle 의 metadata.json 은 우리 산출물이므로
algorithm_config.env_config 에 observation_size 를 채워 넣는다. 값은 같은 metadata 안의
observation_summary.size 에서 가져오므로 새로 만들어내는 정보가 아니다.

이렇게 하면 로드 측 코드를 전혀 바꾸지 않고도 공식 경로가 그대로 동작한다.

사용
----
    python student\\tools\\fix_bundle_obs_size.py <bundle_dir> [...]      # dry-run
    python student\\tools\\fix_bundle_obs_size.py <bundle_dir> --apply    # 반영

원본은 metadata.json.bak 으로 1회 백업한다. 여러 번 실행해도 안전하다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def resolve_size(payload: dict) -> tuple[int | None, str]:
    """(크기, 출처) 를 돌려준다. 찾지 못하면 (None, 사유)."""
    cfg = payload.get("algorithm_config", {}) or {}
    env_cfg = cfg.get("env_config", {}) or {}

    summary = env_cfg.get("observation_summary") or {}
    if isinstance(summary, dict) and isinstance(summary.get("size"), int):
        return int(summary["size"]), "algorithm_config.env_config.observation_summary.size"

    meta = payload.get("metadata", {}) or {}
    if isinstance(meta.get("observation_size"), int):
        return int(meta["observation_size"]), "metadata.observation_size"

    return None, "observation_summary.size 도 metadata.observation_size 도 없다"


def process(bundle_dir: Path, apply: bool) -> bool:
    path = bundle_dir / "metadata.json"
    if not path.exists():
        print(f"  [건너뜀] metadata.json 없음: {bundle_dir}")
        return False

    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.setdefault("algorithm_config", {})
    env_cfg = cfg.setdefault("env_config", {})

    existing = env_cfg.get("observation_size")
    size, origin = resolve_size(payload)

    if existing is not None:
        print(f"  [이미 반영] {bundle_dir.name}: observation_size={existing}")
        return False
    if size is None:
        print(f"  [실패] {bundle_dir.name}: 크기를 찾을 수 없다 ({origin})")
        return False

    mode = env_cfg.get("observation_mode")
    print(f"  [대상] {bundle_dir.name}: observation_mode={mode!r} -> observation_size={size}")
    print(f"         출처: {origin}")

    if not apply:
        return True

    backup = path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"         백업: {backup.name}")

    env_cfg["observation_size"] = size
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("         반영 완료")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="bundle metadata 에 observation_size 주입")
    ap.add_argument("bundles", nargs="+", help="bundle 디렉터리")
    ap.add_argument("--apply", action="store_true", help="실제로 파일을 수정한다")
    args = ap.parse_args()

    changed = 0
    for b in args.bundles:
        changed += bool(process(Path(b), args.apply))

    print()
    if args.apply:
        print(f"{changed}개 bundle 수정 완료.")
    else:
        print(f"DRY-RUN — 수정 대상 {changed}개. --apply 로 반영.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
