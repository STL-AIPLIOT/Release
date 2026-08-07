"""
[학생 작성 파일] 경진대회 제출 — Unreal 서버 연결
=====================================================
학습한 모델을 경진대회 서버에 연결합니다.
BUNDLE_DIR 경로와 팀 이름을 설정한 뒤 이 파일을 실행하세요.

커맨드라인으로 직접 실행하는 방법 (권장)
-------------------------------------------
  # RL 모델 사용
  python run_unreal_inference.py --mode rl \\
      --bundle-dir artifacts/models/team01/v1 \\
      --team-name team01 \\
      --server-ip <서버IP> --server-port 9999

  # BT만 사용 (모델 없이)
  python run_unreal_inference.py --mode bt \\
      --bt-dll AIP_BASE.dll \\
      --bt-rule-xml Rule_forTraining.xml \\
      --team-name team01 \\
      --server-ip <서버IP>

  # RL + BT 하이브리드
  python run_unreal_inference.py --mode hybrid \\
      --bundle-dir artifacts/models/team01/v1 \\
      --bt-dll AIP_BASE.dll \\
      --bt-rule-xml Rule_팀이름.xml \\
      --hybrid-mode residual --residual-scale 0.35 \\
      --team-name team01 \\
      --server-ip <서버IP>

이 파일에서 직접 실행하려면
----------------------------
  python student/my_submission.py

아래 설정을 수정한 뒤 실행하면 됩니다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.hybrid_action_provider import HybridActionProvider
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient


# =============================================================================
# TODO: 아래 설정을 팀에 맞게 수정하세요.
# =============================================================================

TEAM_NAME = "stil"                                 # 운영 공지 등록명과 일치 확인 필요
SERVER_IP = "221.151.77.208"                       # TODO: 경진대회 서버 IP
SERVER_PORT = 9999

# 사용할 백엔드 모드 선택: "rl" | "bt" | "hybrid"
MODE = "rl"

# RL 모드 설정
# 관측 일관성 체인 (Docs/09 §2): 아래 두 값은 BUNDLE_DIR/metadata.json 의
# obs_mode / observation_module 과 반드시 같아야 한다. 불일치는 에러 없이
# 이상 행동만 남기는 silent failure가 된다.
BUNDLE_DIR = "artifacts/models/stil/sac_mlp_obs11_crash90_v1"  # 30판 결정적 평가 추락 0/30
OBSERVATION_MODE = "custom"                        # observation_module 이 있으면 훅의 OBSERVATION_MODE(stil11)가 이긴다
OBSERVATION_MODULE = "student.my_observation"      # 기본 관측이면 빈 문자열

# BT 모드 설정
# - 기본 배포 Rule은 Rule_forTraining.xml입니다.
# - 팀별 BT DLL/XML을 제출하는 경우 파일을 Release 루트에 두고 아래 이름을 바꾸세요.
BT_DLL = "AIP_BASE.dll"
BT_RULE_XML = "Rule_forTraining.xml"  # 예: "Rule_team01.xml"

# Hybrid 모드 설정 (MODE="hybrid" 일 때만 사용)
HYBRID_MODE = "residual"   # "residual" | "blend" | "switch"
RESIDUAL_SCALE = 0.35      # residual 모드 강도 (0~1, 클수록 RL 비중 증가)
ALPHA = 0.5                # blend 모드 비율 (alpha × RL + (1-alpha) × BT)

# 연결 설정
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6          # 학습 step_ratio=6과 맞춰 6개 PlaneInfo pair마다 새 policy 호출
DEBUG_ACTION_REPEAT = False


# =============================================================================
# 예시: 학습 결과 확인 (로컬 테스트용 백엔드)
# =============================================================================
# 경진대회 제출 전 로컬에서 결과 확인:
#   python run_local_dogfight.py \\
#       --ownship-backend rl \\
#       --ownship-bundle-dir artifacts/models/team01/v1 \\
#       --target-backend bt \\
#       --save-log


# =============================================================================
# 실행 로직 (수정 불필요)
# =============================================================================

class ObservationContractError(RuntimeError):
    """observation 설정이 bundle 과 어긋난다. 조용히 진행하지 않고 즉시 멈춘다."""


def _bundle_observation_info(bundle_path: Path) -> dict[str, object]:
    """bundle metadata.json 에서 observation 관련 값만 뽑는다.

    구 스키마에는 없는 필드가 있으므로, 없으면 None 으로 두고 "MISSING" 으로
    보고한다. 즉시 틀린 값으로 단정하지 않는다.
    """
    meta_path = bundle_path / "metadata.json"
    if not meta_path.exists():
        return {"available": False, "reason": f"metadata.json 없음: {meta_path}"}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"metadata.json 파싱 실패: {exc}"}

    meta = payload.get("metadata") or {}
    env_cfg = (payload.get("algorithm_config") or {}).get("env_config") or {}
    summary = env_cfg.get("observation_summary") or {}

    size = env_cfg.get("observation_size")
    size_source = "algorithm_config.env_config.observation_size"
    if size is None and isinstance(summary, dict) and isinstance(summary.get("size"), int):
        size = summary["size"]
        size_source = "algorithm_config.env_config.observation_summary.size"
    if size is None and isinstance(meta.get("observation_size"), int):
        size = meta["observation_size"]
        size_source = "metadata.observation_size"

    return {
        "available": True,
        "path": str(meta_path),
        "mode": meta.get("obs_mode", env_cfg.get("observation_mode")),
        "module": meta.get("observation_module", env_cfg.get("observation_module")),
        "size": size,
        "size_source": size_source if size is not None else None,
        "env_config_size": env_cfg.get("observation_size"),
    }


def verify_observation_contract(bundle_path: Path, observation_hook: dict | None) -> None:
    """실행 직전에 observation 계약을 검증하고 최종 설정을 출력한다.

    확인 항목
      - bundle metadata 의 observation_module 과 이 파일의 OBSERVATION_MODULE
      - bundle metadata 의 observation_size 와 로드된 훅의 size
      - RLLibInferenceEnv 가 읽는 env_config.observation_size 의 존재

    어긋나면 ObservationContractError 로 즉시 멈춘다. padding / truncation /
    조용한 fallback 을 하지 않는다 — 그러면 정책이 돌긴 하면서 성능만 무너진다.
    """
    info = _bundle_observation_info(bundle_path)

    declared_module = OBSERVATION_MODULE.strip()
    hook_mode = observation_hook["mode"] if observation_hook else None
    hook_size = observation_hook["size"] if observation_hook else None
    effective_mode = hook_mode or OBSERVATION_MODE

    source = "metadata" if info.get("available") else "my_submission 상수"
    print("[ObservationConfig]")
    print(f"mode={effective_mode}")
    print(f"module={declared_module or '(없음 — 기본 관측)'}")
    print(f"size={hook_size if hook_size is not None else 'UNKNOWN'}")
    print(f"source={source}")
    print(f"checkpoint={bundle_path}")

    if not info.get("available"):
        # metadata 를 못 읽는 것 자체는 계약 위반이 아니다(구 bundle 일 수 있다).
        # 다만 검증을 못 했다는 사실은 반드시 보인다.
        print(f"[ObservationConfig] 경고: bundle 검증을 하지 못했다 — {info.get('reason')}")
        return

    meta_module = str(info.get("module") or "").strip()
    meta_size = info.get("size")

    problems: list[str] = []
    if meta_module != declared_module:
        problems.append(
            f"- bundle metadata observation_module: {meta_module or '(없음)'}\n"
            f"- my_submission OBSERVATION_MODULE : {declared_module or '(없음)'}")
    if meta_size is not None and hook_size is not None and int(meta_size) != int(hook_size):
        problems.append(
            f"- bundle metadata observation_size : {meta_size}\n"
            f"- loaded module observation_size   : {hook_size}")

    if problems:
        raise ObservationContractError(
            "Observation configuration mismatch:\n"
            + "\n".join(problems)
            + f"\n\nbundle: {info.get('path')}\n"
            "This checkpoint cannot be reused with the selected observation module.\n"
            "Use the matching module or train a new checkpoint with a new output tag."
        )

    if info.get("env_config_size") is None:
        # RLLibInferenceEnv 는 env_config.observation_size 만 읽는다. 이 키가 없으면
        # observation_size(mode) 가 12 를 돌려주어 모델 로드 자체가 실패한다.
        raise ObservationContractError(
            "Observation configuration mismatch:\n"
            f"- bundle metadata observation_size: {meta_size} "
            f"({info.get('size_source')})\n"
            "- algorithm_config.env_config.observation_size: MISSING\n\n"
            f"bundle: {info.get('path')}\n"
            "RLLibInferenceEnv reads env_config.observation_size only; without it the\n"
            "model is built with observation_size(mode)=12 and weight loading fails.\n"
            "Fix the bundle first (it only fills in a value already present in the file):\n"
            f"  python student\\tools\\fix_bundle_obs_size.py {bundle_path} --apply"
        )

    print(f"[ObservationConfig] bundle 검증 통과 (size={meta_size}, "
          f"출처 {info.get('size_source')})")


def build_action_provider():
    if MODE == "bt":
        print(f"[{TEAM_NAME}] BT 백엔드 사용: {BT_DLL}")
        return BTActionProvider(dll_name=BT_DLL)

    bundle_path = ROOT / BUNDLE_DIR
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"모델 번들을 찾을 수 없습니다: {bundle_path}\n"
            f"먼저 학습을 완료하고 BUNDLE_DIR 경로를 확인하세요."
        )

    print(f"[{TEAM_NAME}] RL 모델 로드: {bundle_path}")
    rl_provider = RLActionProvider(
        bundle_dir=str(bundle_path),
        algorithm_factory=build_algorithm_from_bundle,
    )

    if MODE == "rl":
        print(f"[{TEAM_NAME}] RL 전용 모드")
        return rl_provider

    # hybrid
    bt_provider = BTActionProvider(dll_name=BT_DLL)
    print(f"[{TEAM_NAME}] Hybrid 모드: {HYBRID_MODE} (scale={RESIDUAL_SCALE}, alpha={ALPHA})")
    return HybridActionProvider(
        primary_provider=rl_provider,
        secondary_provider=bt_provider,
        mode=HYBRID_MODE,
        alpha=ALPHA,
        residual_scale=RESIDUAL_SCALE,
    )


def main():
    print(f"=== {TEAM_NAME} 경진대회 클라이언트 시작 ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT}")
    print(f"모드: {MODE}")
    if MODE in {"bt", "hybrid"}:
        print(f"BT DLL/XML: {BT_DLL} / {BT_RULE_XML}")

    with activate_rule_xml(BT_RULE_XML, ROOT):
        observation_hook = (
            load_observation_hook(OBSERVATION_MODULE)
            if OBSERVATION_MODULE
            else None
        )
        # 모델을 만들기 전에 계약을 먼저 확인한다. 어긋나면 여기서 멈춘다.
        if MODE in {"rl", "hybrid"}:
            verify_observation_contract(ROOT / BUNDLE_DIR, observation_hook)

        action_provider = build_action_provider()
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=observation_hook["mode"] if observation_hook else OBSERVATION_MODE,
            observation_fn=observation_hook["build_observation"]
            if observation_hook
            else None,
            ownship_force_side=1,
            target_force_side=2,
            action_repeat=ACTION_REPEAT,
            debug_action_repeat=DEBUG_ACTION_REPEAT,
        )

        client = UnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=SERVER_IP,
            server_port=SERVER_PORT,
            team_name=TEAM_NAME,
            ai_type=AI_TYPE,
            heartbeat_interval_sec=HEARTBEAT_SEC,
            command_delay_sec=COMMAND_DELAY_SEC,
            recv_timeout_sec=RECV_TIMEOUT_SEC,
            enable_terminal_monitor=True,   # 패킷 모니터 표시
        )

        try:
            client.run()
        finally:
            action_provider.close()
            print(f"[{TEAM_NAME}] 클라이언트 종료")


if __name__ == "__main__":
    main()
