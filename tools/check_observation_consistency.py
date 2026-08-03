# -*- coding: utf-8 -*-
"""observation 설정 5개 위치 일관성 검사.

배경
----
같은 observation 설정이 아래 다섯 곳에 흩어져 있고, 어긋나도 예외가 나지 않는다.
정책은 그냥 돌면서 이상하게 행동할 뿐이라 사람이 알아채기 어렵다.

    A. experiments/*.yaml          env.observation_mode / env.observation_module
    B. train_rllib.py              YAML -> env_config -> bundle metadata 로 흐르는 규칙
    C. <bundle>/metadata.json      학습 당시 확정값 (metadata.obs_mode 등)
    D. run_local_dogfight.py       CLI 인자 기본값
    E. student/my_submission.py    모듈 상수 OBSERVATION_MODE / OBSERVATION_MODULE

여기에 두 개를 더 본다.

    F. observation module 자체      OBSERVATION_MODE / OBSERVATION_SIZE
    G. bundle 가중치                1층 weight 의 입력 차원 (= 모델이 실제로 받는 크기)

무엇이 정답인가 (source of truth)
---------------------------------
    새 학습을 시작할 때      -> 실험 YAML
    기존 checkpoint 를 돌릴 때 -> 그 bundle 의 metadata.json + 모델 입력 shape

--source-of-truth 로 고를 수 있고, 기본값은 --metadata 를 줬으면 metadata,
안 줬으면 yaml 이다.

정적 분석의 한계
----------------
B/D/E 는 파일을 import 하지 않고 AST 로만 읽는다. 실행 중 CLI 인자나 환경변수로
바뀌는 값은 알 수 없으므로, 그런 항목은 상태를 UNVERIFIED 로 두고 어떤 규칙으로
결정되는지를 함께 보고한다. 값을 추측해서 채우지 않는다.

실행
----
    python tools/check_observation_consistency.py \
        --config experiments/stil_sac_mlp_obs8_iter400.yaml \
        --metadata <bundle>/metadata.json \
        --train-script train_rllib.py \
        --local-runner run_local_dogfight.py \
        --submission student/my_submission.py \
        --observation-module-file student/my_observation.py \
        --bundle-weights <bundle>/policy_weights.pkl.gz \
        --output analysis/observation_consistency \
        --json

종료 코드
    0  필수 설정이 모두 일치
    1  불일치
    2  필수 파일 또는 필드 누락
    3  파일 파싱 실패
"""
from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import pickle
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import warn  # noqa: E402

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_MISSING = 2
EXIT_PARSE_ERROR = 3

MATCH = "MATCH"
MISMATCH = "MISMATCH"
MISSING = "MISSING"
HARDCODED = "HARDCODED"
OVERRIDDEN = "OVERRIDDEN"
UNVERIFIED = "UNVERIFIED"


class ParseError(RuntimeError):
    """설정 파일을 읽지 못했다. 종료 코드 3 으로 이어진다."""


# --------------------------------------------------------------------------- YAML
def _yaml_scalar(text: str) -> object:
    """YAML 스칼라 하나를 파이썬 값으로. 최소 구현이다."""
    s = text.strip()
    if s.startswith("#") or s == "":
        return None
    # 인라인 주석 제거 (따옴표 밖의 ' #' 만)
    if not (s.startswith('"') or s.startswith("'")):
        for i in range(len(s) - 1):
            if s[i] in " \t" and s[i + 1] == "#":
                s = s[:i].strip()
                break
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_yaml_fallback(text: str) -> dict[str, object]:
    """PyYAML 이 없을 때 쓰는 최소 파서.

    지원: 들여쓰기 중첩 매핑, 스칼라, `- ` 리스트, `#` 주석.
    미지원: 앵커/별칭, 여러 문서, 블록 스칼라(| >), 복합 키.
    실험 YAML 은 이 범위 안에 있다(2026-08-04 기준 experiments/*.yaml 전부).
    """
    root: dict[str, object] = {}
    # (indent, container) 스택
    stack: list[tuple[int, object]] = [(-1, root)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if line.startswith("- "):
            item = _yaml_scalar(line[2:])
            if isinstance(container, list):
                container.append(item)
            continue

        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        value_text = rest.strip()

        if not isinstance(container, dict):
            continue

        if value_text == "" or value_text.startswith("#"):
            # 다음 줄들이 중첩 매핑 또는 리스트다. 첫 자식을 볼 때까지 dict 로 둔다.
            child: dict[str, object] = {}
            container[key] = child
            stack.append((indent, child))
            continue
        if value_text.startswith("[") and value_text.endswith("]"):
            inner = value_text[1:-1].strip()
            container[key] = [_yaml_scalar(p) for p in inner.split(",")] if inner else []
            continue
        container[key] = _yaml_scalar(value_text)

    return root


def load_yaml(path: Path) -> dict[str, object]:
    """실험 YAML 을 읽는다. PyYAML 이 있으면 그것을 쓴다."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParseError(f"{path} 읽기 실패: {exc}") from exc
    try:
        import yaml  # type: ignore
    except ImportError:
        warn(f"PyYAML 이 없어 최소 파서로 {path.name} 을 읽는다 "
             "(앵커/블록 스칼라 미지원).")
        return _parse_yaml_fallback(text)
    try:
        data = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001 - yaml 예외 종류가 다양하다
        raise ParseError(f"{path} YAML 파싱 실패: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError(f"{path} 최상위가 매핑이 아니다")
    return data


# --------------------------------------------------------------------------- AST
def _parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError as exc:
        raise ParseError(f"{path} 읽기 실패: {exc}") from exc
    except SyntaxError as exc:
        raise ParseError(f"{path} 파싱 실패: {exc}") from exc


def module_constants(tree: ast.Module) -> dict[str, object]:
    """모듈 최상위의 `NAME = <literal>` 만 뽑는다. 함수 안은 보지 않는다."""
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out[t.id] = value
    return out


def dict_literal_entries(tree: ast.Module, name: str) -> dict[str, object]:
    """최상위 `NAME = {...}` 딕셔너리의 리터럴 항목만 뽑는다."""
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        out: dict[str, object] = {}
        for k, v in zip(node.value.keys, node.value.values):
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                continue
            try:
                out[k.value] = ast.literal_eval(v)
            except (ValueError, SyntaxError, TypeError):
                out[k.value] = UNVERIFIED
        return out
    return {}


def argparse_defaults(tree: ast.Module) -> dict[str, object]:
    """`add_argument("--x", default=...)` 의 옵션 -> 기본값.

    default 가 리터럴이 아니면(다른 변수 참조 등) UNVERIFIED 를 넣는다.
    """
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        option: str | None = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.startswith("--"):
                option = arg.value
                break
        if option is None:
            continue
        found = False
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            found = True
            try:
                out[option] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError, TypeError):
                out[option] = UNVERIFIED
        if not found:
            out[option] = None      # default 미지정 -> None
    return out


def has_subscript_assignment(tree: ast.Module, target_key: str) -> bool:
    """`something["target_key"] = ...` 형태의 대입이 있는지."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                    and t.slice.value == target_key):
                return True
    return False


def reads_metadata_json(tree: ast.Module) -> bool:
    """소스가 metadata.json 문자열을 언급하는지 (AST 상수 기준)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "metadata.json" in node.value:
            return True
    return False


# --------------------------------------------------------------------------- bundle
class _PickleStub:
    """복원할 필요가 없는 객체(numpy.dtype 등)를 대신하는 자리표시자."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        pass


class _ShapeOnlyUnpickler(pickle.Unpickler):
    """numpy 없이 shape 만 뽑기 위한 제한 Unpickler.

    policy_weights.pkl.gz 는 numpy 배열의 OrderedDict 다. 배열은
    `numpy._core.numeric._frombuffer(buffer, dtype, shape, order)` 로 복원되는데,
    이 함수를 shape 만 돌려주는 대체 함수로 바꿔치기하면 numpy 버전과 무관하게
    (그리고 torch 없이) 입력 차원을 읽을 수 있다. 실제 가중치 값은 읽지 않는다.

    허용하지 않은 전역은 전부 _PickleStub 으로 바꾼다. 임의 코드가 실행되지 않는다.
    """

    @staticmethod
    def _frombuffer(buffer: object, dtype: object, shape: object,
                    order: object = None) -> dict[str, tuple[int, ...]]:
        return {"__shape__": tuple(shape)}  # type: ignore[arg-type]

    def find_class(self, module: str, name: str):  # noqa: ANN201
        if name == "_frombuffer":
            return self._frombuffer
        if module == "collections" and name == "OrderedDict":
            return dict
        return _PickleStub


def read_bundle_input_size(path: Path) -> tuple[int | None, str]:
    """bundle 가중치에서 모델의 실제 입력 차원을 읽는다.

    반환: (입력 차원, 근거 문자열). 읽지 못하면 (None, 사유).
    첫 MLP 층의 weight shape 는 (hidden, input) 이므로 shape[1] 이 입력 차원이다.
    """
    if not path.exists():
        return None, f"가중치 파일이 없다: {path}"
    try:
        raw = gzip.open(path, "rb").read()
    except OSError as exc:
        return None, f"가중치 파일 읽기 실패: {exc}"
    try:
        state = _ShapeOnlyUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:  # noqa: BLE001 - 어떤 pickle 오류든 UNVERIFIED 로 떨어뜨린다
        return None, f"가중치 언피클 실패({type(exc).__name__}: {exc})"
    if not isinstance(state, dict):
        return None, "가중치 최상위가 매핑이 아니다"

    # 입력층 후보. pi_encoder(정책 인코더)를 우선한다.
    candidates = [k for k in state
                  if k.endswith(".0.weight") and isinstance(state[k], dict)
                  and "__shape__" in state[k]]
    if not candidates:
        return None, "입력층 weight(*.0.weight)를 찾지 못했다"
    preferred = [k for k in candidates if "pi_encoder" in k] or candidates
    key = sorted(preferred)[0]
    shape = state[key]["__shape__"]
    if len(shape) != 2:
        return None, f"{key} 의 shape 가 2차원이 아니다: {shape}"
    return int(shape[1]), f"{key} shape={tuple(shape)} -> 입력 {shape[1]}"


# --------------------------------------------------------------------------- 수집
# 위치가 비교에서 하는 역할.
#   authoritative  실제로 쓰이는 값이다. 어긋나면 곧 불일치다.
#   default_only   CLI 기본값일 뿐이라 YAML/CLI 가 덮어쓴다. 값 자체를 불일치로
#                  세지 않고, 기본값이 실효값과 다르면 '위험'으로만 보고한다.
ROLE_AUTHORITATIVE = "authoritative"
ROLE_DEFAULT_ONLY = "default_only"


@dataclass
class Location:
    """설정 위치 한 곳에서 읽어낸 값."""

    name: str
    path: str
    observation_mode: object = None
    observation_module: object = None
    observation_size: object = None
    value_source: str = ""
    status: str = UNVERIFIED
    role: str = ROLE_AUTHORITATIVE
    notes: list[str] = field(default_factory=list)
    # 값 비교와 무관하게 이 위치 자체에서 발견한 문제.
    # (severity, field_name, detail) 튜플. run() 이 Finding 으로 바꿔 합친다.
    extra_issues: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Finding:
    """불일치 한 건.

    severity
        mismatch  실제로 쓰이는 값이 어긋난다. 종료 코드 1.
        missing   필수 필드가 없다. 종료 코드 2.
        risk      지금 당장 틀리지는 않지만, 인자를 빠뜨리면 조용히 틀려진다.
                  기본적으로 종료 코드에 영향을 주지 않는다(--strict 로 승격).
    """

    severity: str          # "mismatch" | "missing" | "risk"
    field_name: str
    detail: str


def collect_yaml(path: Path) -> Location:
    data = load_yaml(path)
    env = data.get("env") or {}
    env_config = data.get("env_config") or {}
    output = data.get("output") or {}
    loc = Location(name="YAML", path=str(path), value_source="직접 설정")
    if not isinstance(env, dict):
        loc.notes.append("env 블록이 매핑이 아니다")
        return loc
    loc.observation_mode = env.get("observation_mode")
    loc.observation_module = env.get("observation_module")
    size = None
    if isinstance(env_config, dict):
        size = env_config.get("observation_size")
    loc.observation_size = size
    if size is None:
        loc.notes.append(
            "YAML 에는 observation_size 가 없다. 크기는 observation_module 의 "
            "OBSERVATION_SIZE 가 정한다(정상).")
    if isinstance(output, dict):
        loc.notes.append(
            f"output: name={output.get('name')!r} tag={output.get('tag')!r}")
    return loc


def collect_train_script(path: Path) -> Location:
    tree = _parse_python(path)
    defaults = argparse_defaults(tree)
    loc = Location(name="train_rllib.py", path=str(path),
                   value_source="YAML/CLI 값을 받아 env_config 로 전달",
                   role=ROLE_DEFAULT_ONLY)
    loc.observation_mode = defaults.get("--observation-mode", MISSING)
    loc.observation_module = defaults.get("--observation-module", MISSING)
    loc.observation_size = None

    overrides = has_subscript_assignment(tree, "observation_mode")
    if overrides:
        loc.status = OVERRIDDEN
        loc.notes.append(
            "observation_module 이 있으면 cfg['observation_mode'] 를 훅의 "
            "OBSERVATION_MODE 로 덮어쓴다. 그래서 YAML 이 'custom' 이어도 "
            "metadata 에는 훅 이름(예: stil8)이 기록된다 — 정상 동작이다.")
    else:
        loc.notes.append(
            "observation_mode 재대입을 찾지 못했다. 덮어쓰기 규칙이 바뀌었을 수 있다.")
    loc.notes.append(
        f"CLI 기본값: --observation-mode={defaults.get('--observation-mode')!r}, "
        f"--observation-module={defaults.get('--observation-module')!r}")
    loc.notes.append(
        "CLI 인자는 YAML 값을 덮어쓴다(run_experiment.py 가 YAML -> CLI 로 변환).")
    loc.notes.append("정적 분석 한계: 실행 시 실제로 넘어온 인자는 알 수 없다.")
    return loc


def collect_metadata(path: Path) -> Location:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ParseError(f"{path} 읽기 실패: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path} JSON 파싱 실패: {exc}") from exc

    meta = data.get("metadata") or {}
    env_cfg = (data.get("algorithm_config") or {}).get("env_config") or {}
    summary = env_cfg.get("observation_summary") or {}

    loc = Location(name="metadata.json", path=str(path),
                   value_source="학습 시 저장")
    loc.observation_mode = meta.get("obs_mode", env_cfg.get("observation_mode", MISSING))
    loc.observation_module = meta.get(
        "observation_module", env_cfg.get("observation_module", MISSING))

    size = env_cfg.get("observation_size")
    size_origin = "algorithm_config.env_config.observation_size"
    if size is None and isinstance(summary, dict) and isinstance(summary.get("size"), int):
        size = summary["size"]
        size_origin = "algorithm_config.env_config.observation_summary.size"
    if size is None and isinstance(meta.get("observation_size"), int):
        size = meta["observation_size"]
        size_origin = "metadata.observation_size"
    loc.observation_size = size if size is not None else MISSING
    loc.notes.append(f"observation_size 출처: {size_origin}")

    if env_cfg.get("observation_size") is None:
        detail = (
            "algorithm_config.env_config.observation_size 가 없다. "
            "RLLibInferenceEnv(src/dogfight/ai/inference_env.py:21-22)는 이 키만 읽으므로 "
            "추론 경로가 observation_size(mode)=12 로 모델을 만들어 가중치 로드에 실패한다. "
            f"고치는 법: python student\\tools\\fix_bundle_obs_size.py {path.parent} --apply")
        loc.notes.append(detail)
        loc.extra_issues.append(("missing", "env_config.observation_size", detail))

    # 하위 호환: 옛 metadata 에 없는 필드는 MISSING 으로만 표시하고 값을 지어내지 않는다.
    for key in ("observation_schema_version", "experiment_config",
                "checkpoint_path", "git_commit"):
        if key not in meta:
            loc.notes.append(f"metadata.{key} 없음 (구 스키마) -> MISSING")
    if isinstance(summary, dict) and summary.get("features"):
        loc.notes.append(f"feature 순서: {list(summary['features'])}")
    return loc


def collect_local_runner(path: Path) -> Location:
    tree = _parse_python(path)
    defaults = argparse_defaults(tree)
    loc = Location(name="run_local_dogfight.py", path=str(path),
                   role=ROLE_DEFAULT_ONLY)
    loc.observation_mode = defaults.get("--observation-mode", MISSING)
    loc.observation_module = defaults.get("--observation-module", MISSING)
    loc.observation_size = None
    loc.status = HARDCODED
    loc.value_source = "CLI 인자 기본값 (metadata 를 읽지 않는다)"
    if reads_metadata_json(tree):
        loc.notes.append("소스에 metadata.json 참조가 있다. 규칙을 다시 확인하라.")
        loc.value_source = "metadata 참조 가능"
    else:
        loc.notes.append(
            "이 스크립트는 bundle 의 metadata.json 을 읽지 않는다. "
            "--observation-module 을 주지 않으면 --observation-mode 기본값이 그대로 쓰인다.")
    loc.notes.append(
        "observation_module 을 주면 훅의 OBSERVATION_MODE 가 --observation-mode 를 "
        "이긴다(run_local_dogfight.py 의 observation_hook['mode'] 우선).")
    loc.notes.append("정적 분석 한계: 실행 시 준 CLI 인자는 알 수 없다.")
    return loc


def collect_submission(path: Path) -> Location:
    tree = _parse_python(path)
    consts = module_constants(tree)
    loc = Location(name="my_submission.py", path=str(path))
    loc.observation_mode = consts.get("OBSERVATION_MODE", MISSING)
    loc.observation_module = consts.get("OBSERVATION_MODULE", MISSING)
    loc.observation_size = None
    loc.status = HARDCODED
    loc.value_source = "모듈 상수 (하드코딩)"
    if reads_metadata_json(tree):
        loc.value_source = "metadata 참조"
        loc.notes.append("소스가 metadata.json 을 참조한다.")
    else:
        loc.notes.append(
            "bundle 의 metadata.json 을 읽지 않는다. 상수와 bundle 이 어긋나면 "
            "예외 없이 이상 행동만 남는다.")
    if "BUNDLE_DIR" in consts:
        loc.notes.append(f"BUNDLE_DIR={consts['BUNDLE_DIR']!r}")
    if "ACTION_REPEAT" in consts:
        loc.notes.append(f"ACTION_REPEAT={consts['ACTION_REPEAT']!r} "
                         "(학습 env_config.step_ratio 와 같아야 한다)")
    loc.notes.append(
        "OBSERVATION_MODULE 이 비어 있지 않으면 훅의 OBSERVATION_MODE 가 "
        "OBSERVATION_MODE 상수를 이긴다.")
    return loc


def collect_observation_module_file(path: Path) -> Location:
    tree = _parse_python(path)
    consts = module_constants(tree)
    loc = Location(name="observation module", path=str(path),
                   value_source="모듈 상수 (실제 반환 크기의 근거)")
    loc.observation_mode = consts.get("OBSERVATION_MODE", MISSING)
    loc.observation_module = None
    loc.observation_size = consts.get("OBSERVATION_SIZE", MISSING)
    if loc.observation_size is MISSING:
        loc.notes.append(
            "OBSERVATION_SIZE 상수가 없다. observation_size() 함수를 쓰는 모듈이면 "
            "정적 분석으로는 값을 알 수 없다 -> UNVERIFIED.")
    loc.notes.append(
        "정적 분석이라 build_observation() 이 실제로 이 길이를 돌려주는지는 "
        "확인하지 않는다. --import-check 를 주면 import 해서 확인한다.")
    return loc


def import_check_observation(module_name: str, release_root: Path) -> tuple[int | None, str]:
    """observation 모듈을 실제로 import 해 OBSERVATION_SIZE 를 읽는다.

    호스트 트리(src/dogfight)가 있어야 import 가 되므로 실패할 수 있다.
    실패하면 (None, 사유) 를 돌려주고 UNVERIFIED 로 남긴다.
    """
    import importlib

    for p in (release_root, release_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return None, f"import 실패({type(exc).__name__}: {exc})"
    size = getattr(mod, "OBSERVATION_SIZE", None)
    if size is None:
        fn = getattr(mod, "observation_size", None)
        if callable(fn):
            try:
                size = int(fn())
            except Exception as exc:  # noqa: BLE001
                return None, f"observation_size() 호출 실패: {exc}"
    if size is None:
        return None, "OBSERVATION_SIZE 도 observation_size() 도 없다"
    return int(size), f"import 성공: OBSERVATION_SIZE={size}"


# --------------------------------------------------------------------------- 판정
def _norm(value: object) -> object:
    """비교용 정규화. 빈 문자열과 None 을 같게 본다."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def evaluate(locations: list[Location], truth: Location,
             effective_mode: object) -> list[Finding]:
    """각 위치의 status 를 정하고 불일치 목록을 만든다.

    observation_mode 비교는 **훅 이름 기준**이다. observation_module 이 설정된
    위치에서는 선언된 mode('custom' 등)가 무시되고 훅의 OBSERVATION_MODE 가 쓰이므로,
    'custom' 은 effective_mode 와 같은 것으로 본다(플랫폼 규칙, 결함 아님).
    """
    findings: list[Finding] = []
    truth_module = _norm(truth.observation_module)
    truth_size = truth.observation_size

    for loc in locations:
        problems: list[str] = []

        if loc.role == ROLE_DEFAULT_ONLY:
            # CLI 기본값은 '설정'이 아니다. YAML/CLI 가 덮어쓰므로 불일치로 세지 않는다.
            # 대신 기본값이 실효값과 다르면, 인자를 빠뜨렸을 때 조용히 틀려진다는
            # 사실을 위험으로 보고한다.
            mod = _norm(loc.observation_module)
            mode = _norm(loc.observation_mode)
            if mod != _norm(truth.observation_module):
                findings.append(Finding(
                    "risk", "observation_module",
                    f"{loc.name} 의 --observation-module 기본값이 {mod!r} 이다. "
                    f"인자를 주지 않으면 {truth_module!r} 이 아닌 기본 관측으로 실행된다."))
            if mode not in (_norm(effective_mode), "custom"):
                findings.append(Finding(
                    "risk", "observation_mode",
                    f"{loc.name} 의 --observation-mode 기본값이 {mode!r} 이라 "
                    f"실효 모드 {effective_mode!r} 와 다르다."))
            if loc.status == UNVERIFIED:
                loc.status = HARDCODED
            continue

        # --- module: 가장 중요한 값. 이것만은 다섯 곳이 같아야 한다.
        mod = _norm(loc.observation_module)
        if loc.observation_module is None:
            pass  # 이 위치에는 module 개념이 없다
        elif mod is MISSING or mod == MISSING:
            problems.append("observation_module 필드 없음")
            findings.append(Finding("missing", "observation_module",
                                    f"{loc.name}: 값이 없다"))
        elif mod != truth_module:
            problems.append(f"observation_module {mod!r} != {truth_module!r}")
            findings.append(Finding(
                "mismatch", "observation_module",
                f"{loc.name}={mod!r} vs source_of_truth({truth.name})={truth_module!r}"))

        # --- mode
        mode = _norm(loc.observation_mode)
        if loc.observation_mode is None:
            pass
        elif mode is MISSING or mode == MISSING:
            problems.append("observation_mode 필드 없음")
            findings.append(Finding("missing", "observation_mode",
                                    f"{loc.name}: 값이 없다"))
        elif mode not in (_norm(effective_mode), "custom"):
            problems.append(f"observation_mode {mode!r} != {effective_mode!r}")
            findings.append(Finding(
                "mismatch", "observation_mode",
                f"{loc.name}={mode!r} vs 실효 모드={effective_mode!r}"))

        # --- size
        if loc.observation_size is None:
            pass
        elif loc.observation_size is MISSING or loc.observation_size == MISSING:
            problems.append("observation_size 필드 없음")
            findings.append(Finding("missing", "observation_size",
                                    f"{loc.name}: 값이 없다"))
        elif truth_size not in (None, MISSING) and loc.observation_size != truth_size:
            problems.append(f"observation_size {loc.observation_size} != {truth_size}")
            findings.append(Finding(
                "mismatch", "observation_size",
                f"{loc.name}={loc.observation_size} vs "
                f"source_of_truth({truth.name})={truth_size}"))

        if problems:
            loc.status = MISMATCH if any("!=" in p for p in problems) else MISSING
            loc.notes.extend(problems)
        elif loc.status in (UNVERIFIED,):
            loc.status = MATCH
        elif loc.status in (HARDCODED, OVERRIDDEN):
            pass       # 값은 맞지만 결정 방식을 표시로 남긴다
        else:
            loc.status = MATCH

    return findings


def render_markdown(payload: dict[str, object]) -> str:
    locs: list[dict[str, object]] = payload["locations"]  # type: ignore[assignment]

    def cell(v: object) -> str:
        if v is None:
            return "—"
        if v == "":
            return "(빈 문자열)"
        return f"`{v}`"

    lines = [
        "# observation 설정 일관성 점검",
        "",
        f"- 생성 시각: {payload['generated_at']}",
        f"- source of truth: **{payload['source_of_truth']}** ({payload['source_of_truth_reason']})",
        f"- 실효 observation 모드: `{payload['effective_mode']}`",
        f"- 실효 observation 크기: {payload['effective_size']}",
        f"- 판정: **{payload['verdict']}**",
        "",
        "## 1. 위치별 값",
        "",
        "| 위치 | observation_mode | observation_module | observation_size | 값의 출처 | 상태 |",
        "|---|---|---|---:|---|---|",
    ]
    for loc in locs:
        size = loc["observation_size"]
        lines.append(
            f"| {loc['name']} | {cell(loc['observation_mode'])} "
            f"| {cell(loc['observation_module'])} "
            f"| {'—' if size is None else cell(size)} "
            f"| {loc['value_source'] or '—'} | `{loc['status']}` |")

    lines += [
        "",
        "> `observation_mode` 가 YAML/제출 스크립트에서 `custom` 인데 metadata 에서 ",
        "> `stil8` 인 것은 **정상**이다. `observation_module` 이 설정되면 훅의 ",
        "> `OBSERVATION_MODE` 가 선언값을 덮어쓴다. 다섯 곳이 반드시 같아야 하는 값은 ",
        "> `observation_module` 이다.",
        "",
        "## 2. 최종적으로 쓰이는 값",
        "",
        "| 경로 | 값 | 결정 규칙 |",
        "|---|---|---|",
    ]
    for row in payload["effective_values"]:  # type: ignore[union-attr]
        lines.append(f"| {row['path']} | `{row['value']}` | {row['rule']} |")

    lines += [
        "",
        "## 3. 우선순위",
        "",
    ]
    for i, rule in enumerate(payload["precedence"], 1):  # type: ignore[union-attr]
        lines.append(f"{i}. {rule}")

    lines += [
        "",
        "## 4. checkpoint 재사용 가능 여부",
        "",
        f"- 판정: **{payload['checkpoint_reuse']['verdict']}**",
        f"- 근거: {payload['checkpoint_reuse']['reason']}",
        f"- 모델 입력 차원: {payload['checkpoint_reuse']['model_input_size']} "
        f"({payload['checkpoint_reuse']['model_input_source']})",
        f"- 새 output tag 필요: {payload['checkpoint_reuse']['needs_new_tag']}",
        "",
        "## 5. 발견한 문제",
        "",
    ]
    if payload["findings"]:
        lines.append("| 심각도 | 항목 | 내용 |")
        lines.append("|---|---|---|")
        for f in payload["findings"]:  # type: ignore[union-attr]
            lines.append(f"| {f['severity']} | `{f['field_name']}` | {f['detail']} |")
        lines += [
            "",
            "- `mismatch` 실제로 쓰이는 값이 어긋난다 (종료 코드 1)",
            "- `missing` 필수 필드가 없다 (종료 코드 2)",
            "- `risk` 지금은 맞지만 인자를 빠뜨리면 조용히 틀려진다 "
            "(기본적으로 종료 코드에 영향 없음, `--strict` 로 승격)",
        ]
    else:
        lines.append("없음. 필수 설정이 모두 일치한다.")

    lines += ["", "## 6. 위치별 비고", ""]
    for loc in locs:
        lines.append(f"### {loc['name']}")
        lines.append("")
        lines.append(f"`{loc['path']}`")
        lines.append("")
        for note in loc["notes"]:  # type: ignore[union-attr]
            lines.append(f"- {note}")
        if not loc["notes"]:
            lines.append("- (없음)")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def run(args: argparse.Namespace) -> int:
    locations: list[Location] = []
    missing_files: list[str] = []

    def add(path: Path | None, collector, required: bool) -> Location | None:
        if path is None:
            return None
        if not path.exists():
            if required:
                missing_files.append(str(path))
            else:
                warn(f"건너뜀(파일 없음): {path}")
            return None
        loc = collector(path)
        locations.append(loc)
        return loc

    yaml_loc = add(args.config, collect_yaml, True)
    train_loc = add(args.train_script, collect_train_script, False)
    meta_loc = add(args.metadata, collect_metadata, args.metadata is not None)
    runner_loc = add(args.local_runner, collect_local_runner, False)
    sub_loc = add(args.submission, collect_submission, False)
    module_loc = add(args.observation_module_file, collect_observation_module_file, False)

    if missing_files:
        for p in missing_files:
            warn(f"필수 파일이 없다: {p}")
        return EXIT_MISSING

    # --- source of truth
    if args.source_of_truth == "auto":
        truth_name = "metadata" if meta_loc is not None else "yaml"
    else:
        truth_name = args.source_of_truth
    truth = meta_loc if truth_name == "metadata" else yaml_loc
    if truth is None:
        warn(f"source of truth 로 지정한 {truth_name} 을 읽지 못했다.")
        return EXIT_MISSING
    truth_reason = (
        "기존 checkpoint/bundle 을 실행·제출하는 경로이므로 학습 시 확정된 값이 기준이다."
        if truth_name == "metadata" else
        "새 학습을 시작하는 경로이므로 실험 YAML 이 기준이다.")

    # --- 실효 모드/크기
    effective_mode = truth.observation_mode
    if effective_mode in (None, "", "custom", MISSING) and module_loc is not None:
        effective_mode = module_loc.observation_mode
    effective_size = truth.observation_size
    if effective_size in (None, MISSING) and module_loc is not None:
        effective_size = module_loc.observation_size

    # --- observation 모듈 실제 import 확인 (선택)
    import_note = "실행하지 않음 (--import-check 로 활성화)"
    imported_size: int | None = None
    if args.import_check:
        module_name = str(_norm(truth.observation_module) or "")
        if module_name:
            imported_size, import_note = import_check_observation(
                module_name, args.release_root)
            if imported_size is not None and module_loc is not None:
                module_loc.notes.append(f"import 확인: OBSERVATION_SIZE={imported_size}")
                if (module_loc.observation_size not in (None, MISSING)
                        and module_loc.observation_size != imported_size):
                    module_loc.notes.append(
                        f"정적 값({module_loc.observation_size})과 import 값"
                        f"({imported_size})이 다르다.")
                    module_loc.observation_size = imported_size

    findings = evaluate(locations, truth, effective_mode)
    for loc in locations:
        for severity, field_name, detail in loc.extra_issues:
            findings.append(Finding(severity, field_name, f"{loc.name}: {detail}"))

    # --- 모델 입력 차원
    model_input: int | None = None
    model_input_source = "확인하지 않음 (--bundle-weights 미지정)"
    if args.bundle_weights is not None:
        model_input, model_input_source = read_bundle_input_size(args.bundle_weights)

    # 재사용 판정은 **실행 시 실제로 들어갈 크기**와 비교해야 한다.
    # 그 값은 observation 모듈이 돌려주는 크기이고, 모듈을 못 읽었을 때만
    # metadata 의 실효 크기로 대신한다.
    runtime_size = effective_size
    runtime_size_source = f"source of truth({truth.name})"
    if module_loc is not None and isinstance(module_loc.observation_size, int):
        runtime_size = module_loc.observation_size
        runtime_size_source = f"observation 모듈({module_loc.path})"

    reuse_verdict = "UNVERIFIED"
    reuse_reason = "모델 입력 차원을 확인하지 못했다."
    needs_new_tag = "UNVERIFIED"
    if model_input is not None and isinstance(runtime_size, int):
        if model_input == runtime_size:
            reuse_verdict = "COMPATIBLE"
            reuse_reason = (
                f"모델 입력 차원({model_input})과 실행 시 observation 크기"
                f"({runtime_size}, 출처: {runtime_size_source})가 같다.")
            needs_new_tag = "불필요"
        else:
            reuse_verdict = "INCOMPATIBLE"
            reuse_reason = (
                f"모델 입력 차원({model_input}) != 실행 시 observation 크기"
                f"({runtime_size}, 출처: {runtime_size_source}). "
                "이 checkpoint 는 선택한 observation 모듈로 재사용할 수 없다.")
            needs_new_tag = "필요 — 새 output tag 로 새로 학습하라"
            findings.append(Finding(
                "mismatch", "model_input_size",
                f"모델 입력 {model_input} vs 실행 시 observation 크기 {runtime_size}"))
    elif model_input is not None:
        reuse_reason = (
            f"모델 입력 차원은 {model_input} 이지만 실행 시 observation 크기를 "
            "확정하지 못했다.")

    blocking = [f for f in findings
                if f.severity != "risk" or args.strict]
    verdict = "PASS" if not blocking else "FAIL"

    payload: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_of_truth": truth.name,
        "source_of_truth_reason": truth_reason,
        "effective_mode": effective_mode,
        "effective_size": effective_size,
        "observation_module_import": import_note,
        "verdict": verdict,
        "locations": [asdict(loc) for loc in locations],
        "findings": [asdict(f) for f in findings],
        "effective_values": [
            {"path": "학습",
             "value": f"{_norm(yaml_loc.observation_module) if yaml_loc else '?'} "
                      f"/ {effective_mode} / {effective_size}",
             "rule": "YAML env.observation_module -> train_rllib.py 가 훅을 로드하고 "
                     "observation_mode 를 훅의 OBSERVATION_MODE 로 덮어쓴다."},
            {"path": "로컬 실행",
             "value": f"{_norm(runner_loc.observation_module) if runner_loc else '?'} "
                      f"(CLI 미지정 시) / {_norm(runner_loc.observation_mode) if runner_loc else '?'}",
             "rule": "run_local_dogfight.py 는 metadata 를 읽지 않는다. "
                     "--observation-module 을 반드시 명시해야 한다."},
            {"path": "제출 실행",
             "value": f"{_norm(sub_loc.observation_module) if sub_loc else '?'} "
                      f"/ {_norm(sub_loc.observation_mode) if sub_loc else '?'}",
             "rule": "my_submission.py 의 모듈 상수. metadata 를 읽지 않는다."},
        ],
        "precedence": [
            "observation_module 이 비어 있지 않으면 훅의 OBSERVATION_MODE 가 "
            "선언된 observation_mode 를 이긴다 (train_rllib.py / run_local_dogfight.py / "
            "my_submission.py 모두 같은 규칙).",
            "CLI 인자는 YAML 값을 이긴다 (run_experiment.py 가 YAML 을 CLI 로 변환).",
            "bundle 을 로드할 때는 metadata.json 의 env_config.observation_size 가 "
            "모델 입력 차원을 정한다 (RLLibInferenceEnv).",
            "그 키가 없으면 observation_size(mode) 가 12 를 돌려주어 조용히 틀린 "
            "크기가 된다. fix_bundle_obs_size.py 로 채워 둘 것.",
        ],
        "checkpoint_reuse": {
            "verdict": reuse_verdict,
            "reason": reuse_reason,
            "model_input_size": model_input,
            "model_input_source": model_input_source,
            "runtime_observation_size": runtime_size,
            "runtime_observation_size_source": runtime_size_source,
            "needs_new_tag": needs_new_tag,
        },
        "static_analysis_limits": [
            "train_rllib.py / run_local_dogfight.py 는 AST 로만 읽는다. "
            "실행 시 준 CLI 인자는 알 수 없다.",
            "my_submission.py 의 상수가 실행 중 바뀌는지는 확인하지 않는다.",
            "--import-check 를 주지 않으면 build_observation() 의 실제 반환 shape 은 "
            "확인하지 않는다.",
        ],
    }

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "observation_consistency.md").write_text(
            render_markdown(payload), encoding="utf-8")
        (args.output / "observation_consistency.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"출력: {args.output}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(payload))

    if not blocking:
        return EXIT_OK
    if any(f.severity == "mismatch" for f in blocking):
        return EXIT_MISMATCH
    if any(f.severity == "missing" for f in blocking):
        return EXIT_MISSING
    return EXIT_MISMATCH


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="observation 설정 일관성 검사")
    ap.add_argument("--config", type=Path, required=True, help="실험 YAML")
    ap.add_argument("--metadata", type=Path, help="bundle 또는 checkpoint 의 metadata.json")
    ap.add_argument("--train-script", type=Path, help="train_rllib.py")
    ap.add_argument("--local-runner", type=Path, help="run_local_dogfight.py")
    ap.add_argument("--submission", type=Path, help="student/my_submission.py")
    ap.add_argument("--observation-module-file", type=Path,
                    help="student/my_observation.py")
    ap.add_argument("--bundle-weights", type=Path,
                    help="policy_weights.pkl.gz (모델 입력 차원 확인)")
    ap.add_argument("--source-of-truth", choices=["auto", "yaml", "metadata"],
                    default="auto",
                    help="auto: metadata 를 주면 metadata, 아니면 yaml")
    ap.add_argument("--import-check", action="store_true",
                    help="observation 모듈을 실제로 import 해 크기를 확인한다")
    ap.add_argument("--release-root", type=Path, default=Path("."),
                    help="--import-check 시 sys.path 에 넣을 Release 루트")
    ap.add_argument("--strict", action="store_true",
                    help="risk 항목도 실패로 취급한다 (CI 에서 엄격 검사용)")
    ap.add_argument("--output", type=Path, help="보고서를 저장할 디렉터리")
    ap.add_argument("--json", action="store_true", help="표준출력을 JSON 으로")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except ParseError as exc:
        warn(str(exc))
        return EXIT_PARSE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
