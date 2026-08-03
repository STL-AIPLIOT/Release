# -*- coding: utf-8 -*-
"""observation 설정 일관성 검사 도구 테스트.

외부 프레임워크 없이 실패 개수를 세는 방식이다.

    python tests/tools/test_observation_consistency.py

실제 로그/번들을 fixture 로 복사하지 않는다. 필요한 최소 파일을 임시 디렉터리에
그때그때 만들어 쓴다.
"""
from __future__ import annotations

import gzip
import io
import json
import pickle
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import check_observation_consistency as occ  # noqa: E402

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


# --------------------------------------------------------------------------- fixture
YAML_TEXT = """\
name: test_experiment
output:
  name: stil
  tag: obs8_test

env:
  observation_mode: custom
  observation_module: student.my_observation
  reward_module: student.my_reward
  target_mode: fixed
  max_engage_time: 120.0

env_config:
  step_ratio: 6
  wez:
    angle_deg: 2.0
    min_range_m: 152.4
algo:
  mlp:
    fcnet_hiddens: [256, 256]
"""

TRAIN_TEXT = '''\
import argparse
def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-mode", default="tactical16")
    parser.add_argument("--observation-module", default="")
    return parser
def apply(cfg, observation_hook):
    cfg["observation_mode"] = observation_hook["mode"]
    return cfg
'''

RUNNER_TEXT = '''\
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-mode", default="tactical16")
    parser.add_argument("--observation-module", default="")
    return parser.parse_args()
'''

SUBMISSION_TEXT = '''\
OBSERVATION_MODE = "custom"
OBSERVATION_MODULE = "student.my_observation"
BUNDLE_DIR = "artifacts/models/stil/obs8_test"
ACTION_REPEAT = 6
'''

MODULE_TEXT = '''\
OBSERVATION_MODE = "stil8"
OBSERVATION_SIZE = 8
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0
def build_observation(*args, **kwargs):
    return None
'''


def make_metadata(size: int | None, module: str, mode: str,
                  summary_size: int | None = 8) -> dict:
    env_cfg: dict[str, object] = {
        "observation_mode": mode,
        "observation_module": module,
    }
    if summary_size is not None:
        env_cfg["observation_summary"] = {
            "mode": mode, "size": summary_size,
            "features": ["a"] * summary_size, "description": "test"}
    if size is not None:
        env_cfg["observation_size"] = size
    return {
        "algorithm_class": "SAC",
        "policy_id": "default_policy",
        "algorithm_config": {"env_config": env_cfg},
        "metadata": {"obs_mode": mode, "observation_module": module,
                     "algorithm": "sac"},
    }


def make_weights(path: Path, input_size: int) -> None:
    """policy_weights.pkl.gz 와 같은 구조(numpy 배열의 OrderedDict)를 흉내낸다.

    numpy 없이도 만들 수 있도록 _frombuffer 호출 형태를 직접 피클한다.
    """
    class _Arr:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __reduce__(self):
            return (_frombuffer_marker, (b"", "float32", self.shape, "C"))

    state = OrderedDict()
    state["pi_encoder.net.mlp.0.weight"] = _Arr((256, input_size))
    state["pi_encoder.net.mlp.0.bias"] = _Arr((256,))
    state["pi.net.mlp.0.weight"] = _Arr((4, 256))

    buf = io.BytesIO()
    pickle.Pickler(buf, protocol=5).dump(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(buf.getvalue()))


def _frombuffer_marker(buffer, dtype, shape, order):  # noqa: ANN001, ANN201
    """피클 안에서 numpy._core.numeric._frombuffer 자리를 대신한다."""
    return {"__shape__": tuple(shape)}


# 피클된 전역 참조가 numpy 경로처럼 보이도록 이름을 맞춰 둔다.
_frombuffer_marker.__module__ = "numpy._core.numeric"
_frombuffer_marker.__qualname__ = "_frombuffer"
_frombuffer_marker.__name__ = "_frombuffer"
sys.modules.setdefault("numpy._core", SimpleNamespace())
sys.modules["numpy._core.numeric"] = SimpleNamespace(_frombuffer=_frombuffer_marker)


def build_fixture(tmp: Path, *, metadata: dict, module_text: str = MODULE_TEXT,
                  submission_text: str = SUBMISSION_TEXT,
                  weights_input: int | None = 8) -> SimpleNamespace:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "exp.yaml").write_text(YAML_TEXT, encoding="utf-8")
    (tmp / "train_rllib.py").write_text(TRAIN_TEXT, encoding="utf-8")
    (tmp / "run_local_dogfight.py").write_text(RUNNER_TEXT, encoding="utf-8")
    (tmp / "my_submission.py").write_text(submission_text, encoding="utf-8")
    (tmp / "my_observation.py").write_text(module_text, encoding="utf-8")
    (tmp / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    weights = None
    if weights_input is not None:
        weights = tmp / "policy_weights.pkl.gz"
        make_weights(weights, weights_input)
    return SimpleNamespace(
        config=tmp / "exp.yaml",
        metadata=tmp / "metadata.json",
        train_script=tmp / "train_rllib.py",
        local_runner=tmp / "run_local_dogfight.py",
        submission=tmp / "my_submission.py",
        observation_module_file=tmp / "my_observation.py",
        bundle_weights=weights,
        source_of_truth="auto",
        import_check=False,
        release_root=ROOT,
        output=None,
        json=True,
        strict=False,
    )


def run(args: SimpleNamespace) -> tuple[int, dict]:
    """checker 를 돌리고 (종료 코드, payload) 를 돌려준다."""
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = occ.run(args)
    text = buf.getvalue()
    start = text.index("{")
    return code, json.loads(text[start:])


# --------------------------------------------------------------------------- 테스트
def test_yaml_fallback_parser() -> None:
    print("\n[1] YAML 최소 파서")
    data = occ._parse_yaml_fallback(YAML_TEXT)
    check(data["env"]["observation_mode"] == "custom", "env.observation_mode 를 읽는다")
    check(data["env"]["observation_module"] == "student.my_observation",
          "env.observation_module 를 읽는다")
    check(data["env_config"]["step_ratio"] == 6, "정수 스칼라를 읽는다")
    check(data["env_config"]["wez"]["angle_deg"] == 2.0, "2단 중첩 매핑을 읽는다")
    check(data["output"]["tag"] == "obs8_test", "output 블록을 읽는다")
    check(data["algo"]["mlp"]["fcnet_hiddens"] == [256, 256], "인라인 리스트를 읽는다")
    commented = occ._parse_yaml_fallback("a: 1  # 주석\n# 전체 주석\nb: hello\n")
    check(commented == {"a": 1, "b": "hello"}, f"주석을 제거한다: {commented}")


def test_ast_helpers() -> None:
    print("\n[2] AST 추출")
    import ast

    tree = ast.parse(TRAIN_TEXT)
    defaults = occ.argparse_defaults(tree)
    check(defaults["--observation-mode"] == "tactical16",
          "argparse 기본값을 뽑는다 (문자열 검색이 아니라 AST)")
    check(defaults["--observation-module"] == "", "빈 문자열 기본값도 구분한다")
    check(occ.has_subscript_assignment(tree, "observation_mode") is True,
          "observation_mode 재대입을 AST 로 찾는다")
    check(occ.has_subscript_assignment(tree, "reward_mode") is False,
          "없는 키는 False")

    consts = occ.module_constants(ast.parse(SUBMISSION_TEXT))
    check(consts["OBSERVATION_MODULE"] == "student.my_observation",
          "모듈 상수를 뽑는다")
    check(consts["ACTION_REPEAT"] == 6, "숫자 상수도 뽑는다")
    check(occ.reads_metadata_json(ast.parse(SUBMISSION_TEXT)) is False,
          "metadata.json 참조가 없으면 False")
    check(occ.reads_metadata_json(ast.parse('p = "bundle/metadata.json"')) is True,
          "metadata.json 참조를 찾는다")


def test_all_match(tmp: Path) -> None:
    print("\n[3] 다섯 위치가 모두 일치하는 경우")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"))
    code, payload = run(args)
    check(code == occ.EXIT_OK, f"종료 코드 0 (얻은 값 {code})")
    check(payload["verdict"] == "PASS", "판정 PASS")
    check(payload["effective_mode"] == "stil8", "실효 모드 stil8")
    check(payload["effective_size"] == 8, "실효 크기 8")
    check(payload["source_of_truth"] == "metadata.json",
          "metadata 를 주면 source of truth 는 metadata")
    mismatches = [f for f in payload["findings"] if f["severity"] == "mismatch"]
    check(not mismatches, f"mismatch 없음 (얻은 값 {mismatches})")
    risks = [f for f in payload["findings"] if f["severity"] == "risk"]
    check(len(risks) == 4,
          f"CLI 기본값 위험은 risk 로만 보고한다 (얻은 수 {len(risks)})")
    check(payload["checkpoint_reuse"]["verdict"] == "COMPATIBLE",
          "checkpoint 재사용 COMPATIBLE")
    check(payload["checkpoint_reuse"]["model_input_size"] == 8,
          "모델 입력 차원 8 을 가중치에서 직접 읽는다")


def test_module_name_mismatch(tmp: Path) -> None:
    print("\n[4] module 이름 불일치")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.other_observation", "stil8"))
    code, payload = run(args)
    check(code == occ.EXIT_MISMATCH, f"종료 코드 1 (얻은 값 {code})")
    check(payload["verdict"] == "FAIL", "판정 FAIL")
    fields = {f["field_name"] for f in payload["findings"] if f["severity"] == "mismatch"}
    check("observation_module" in fields, f"module 불일치를 잡는다 ({fields})")
    yaml_loc = next(x for x in payload["locations"] if x["name"] == "YAML")
    check(yaml_loc["status"] == occ.MISMATCH, "YAML 위치가 MISMATCH 로 표시된다")


def test_size_mismatch(tmp: Path) -> None:
    print("\n[5] observation size 불일치와 checkpoint 재사용 불가")
    meta = make_metadata(10, "student.my_observation", "stil8", summary_size=10)
    args = build_fixture(tmp, metadata=meta, weights_input=10)
    code, payload = run(args)
    check(code == occ.EXIT_MISMATCH, f"종료 코드 1 (얻은 값 {code})")
    fields = {f["field_name"] for f in payload["findings"] if f["severity"] == "mismatch"}
    check("observation_size" in fields, f"size 불일치를 잡는다 ({fields})")
    check(payload["checkpoint_reuse"]["verdict"] == "INCOMPATIBLE",
          "모델 입력 10 vs 모듈 8 -> INCOMPATIBLE")
    check("새 output tag" in payload["checkpoint_reuse"]["needs_new_tag"],
          "새 output tag 가 필요하다고 명시한다")


def test_model_input_mismatch(tmp: Path) -> None:
    print("\n[6] metadata 는 맞는데 모델 입력만 다른 경우")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"),
                         weights_input=12)
    code, payload = run(args)
    check(payload["checkpoint_reuse"]["verdict"] == "INCOMPATIBLE",
          "모델 입력 12 vs observation 8 -> INCOMPATIBLE")
    check(code == occ.EXIT_MISMATCH, f"실행을 막도록 종료 코드 1 (얻은 값 {code})")
    check(any(f["field_name"] == "model_input_size" for f in payload["findings"]),
          "model_input_size 불일치를 findings 에 남긴다")


def test_metadata_missing_size(tmp: Path) -> None:
    print("\n[7] metadata 에 env_config.observation_size 가 없는 경우")
    meta = make_metadata(None, "student.my_observation", "stil8", summary_size=8)
    args = build_fixture(tmp, metadata=meta)
    code, payload = run(args)
    loc = next(x for x in payload["locations"] if x["name"] == "metadata.json")
    check(loc["observation_size"] == 8,
          "observation_summary.size 로 대체해 읽는다 (MISSING 으로 단정하지 않는다)")
    check(any("observation_summary.size" in n for n in loc["notes"]),
          "어느 필드에서 왔는지 출처를 남긴다")
    check(any("fix_bundle_obs_size" in n for n in loc["notes"]),
          "추론 경로가 깨진다는 사실과 해결 방법을 안내한다")
    # 값은 일치하지만 추론이 로드에 실패하는 상태다. 조용히 통과시키면 안 된다.
    check(code == occ.EXIT_MISSING, f"필수 필드 누락이므로 종료 코드 2 (얻은 값 {code})")
    check(any(f["field_name"] == "env_config.observation_size"
              and f["severity"] == "missing" for f in payload["findings"]),
          "env_config.observation_size 누락을 finding 으로 남긴다")
    check(payload["checkpoint_reuse"]["verdict"] == "COMPATIBLE",
          "크기 자체는 맞으므로 재사용 판정은 COMPATIBLE (bundle 만 고치면 된다)")


def test_hardcoded_and_override(tmp: Path) -> None:
    print("\n[8] HARDCODED / OVERRIDDEN 표시")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"))
    _code, payload = run(args)
    by_name = {x["name"]: x for x in payload["locations"]}
    check(by_name["my_submission.py"]["status"] == occ.HARDCODED,
          "제출 스크립트는 상수 하드코딩으로 표시된다")
    check(by_name["train_rllib.py"]["status"] == occ.OVERRIDDEN,
          "train_rllib 는 mode 를 덮어쓰는 규칙이라 OVERRIDDEN")
    check(by_name["run_local_dogfight.py"]["status"] == occ.HARDCODED,
          "로컬 러너는 CLI 기본값이라 HARDCODED")
    check(any("metadata.json 을 읽지 않는다" in n
              for n in by_name["my_submission.py"]["notes"]),
          "metadata 를 읽지 않는다는 사실을 남긴다")


def test_strict_promotes_risk(tmp: Path) -> None:
    print("\n[9] --strict 는 risk 를 실패로 승격한다")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"))
    args.strict = True
    code, payload = run(args)
    check(code == occ.EXIT_MISMATCH, f"--strict 면 종료 코드 1 (얻은 값 {code})")
    check(payload["verdict"] == "FAIL", "--strict 면 판정 FAIL")


def test_missing_field(tmp: Path) -> None:
    print("\n[10] 필수 필드 누락")
    module_no_size = 'OBSERVATION_MODE = "stil8"\ndef build_observation(*a, **k):\n    return None\n'
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"),
                         module_text=module_no_size)
    code, payload = run(args)
    loc = next(x for x in payload["locations"] if x["name"] == "observation module")
    check(loc["observation_size"] == occ.MISSING, "OBSERVATION_SIZE 가 없으면 MISSING")
    check(code == occ.EXIT_MISSING, f"종료 코드 2 (얻은 값 {code})")
    check(any("observation_size" in f["field_name"] and f["severity"] == "missing"
              for f in payload["findings"]), "missing finding 을 남긴다")


def test_missing_file(tmp: Path) -> None:
    print("\n[11] 필수 파일 누락")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"))
    args.config = tmp / "does_not_exist.yaml"
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        code = occ.run(args)
    check(code == occ.EXIT_MISSING, f"종료 코드 2 (얻은 값 {code})")


def test_parse_error(tmp: Path) -> None:
    print("\n[12] 파싱 실패")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"))
    (tmp / "my_submission.py").write_text("def broken(:\n", encoding="utf-8")
    try:
        occ.run(args)
        check(False, "ParseError 가 발생해야 한다")
    except occ.ParseError as exc:
        check("파싱 실패" in str(exc), f"ParseError 메시지: {exc}")
    # main() 을 통하면 종료 코드 3 이다.
    code = occ.main.__wrapped__(args) if hasattr(occ.main, "__wrapped__") else None
    check(code is None, "main() 은 CLI 진입점이라 여기서는 직접 호출하지 않는다")

    (tmp / "metadata.json").write_text("{not json", encoding="utf-8")
    (tmp / "my_submission.py").write_text(SUBMISSION_TEXT, encoding="utf-8")
    try:
        occ.run(args)
        check(False, "metadata JSON 파싱 실패도 ParseError 여야 한다")
    except occ.ParseError as exc:
        check("JSON 파싱 실패" in str(exc), f"ParseError 메시지: {exc}")


def test_weights_reader_failure(tmp: Path) -> None:
    print("\n[13] 가중치를 읽지 못할 때")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"),
                         weights_input=None)
    args.bundle_weights = tmp / "no_such_weights.pkl.gz"
    code, payload = run(args)
    check(payload["checkpoint_reuse"]["verdict"] == "UNVERIFIED",
          "읽지 못하면 UNVERIFIED. 재사용 가능하다고 단정하지 않는다")
    check(payload["checkpoint_reuse"]["model_input_size"] is None,
          "모델 입력 차원은 None (0 으로 위장하지 않는다)")
    check(code == occ.EXIT_OK, "다른 값이 맞으면 종료 코드는 0")

    bad = tmp / "bad.pkl.gz"
    bad.write_bytes(b"not gzip")
    size, reason = occ.read_bundle_input_size(bad)
    check(size is None and reason, f"깨진 파일은 사유와 함께 None: {reason}")


def test_source_of_truth_yaml(tmp: Path) -> None:
    print("\n[14] metadata 없이 YAML 을 기준으로")
    args = build_fixture(tmp, metadata=make_metadata(8, "student.my_observation", "stil8"))
    args.metadata = None
    args.bundle_weights = None
    code, payload = run(args)
    check(payload["source_of_truth"] == "YAML", "metadata 가 없으면 YAML 이 기준")
    check(payload["effective_mode"] == "stil8",
          "YAML 이 'custom' 이어도 모듈의 OBSERVATION_MODE 로 실효 모드를 정한다")
    check(code == occ.EXIT_OK, f"종료 코드 0 (얻은 값 {code})")


def main() -> int:
    print("observation 설정 일관성 검사 테스트")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_yaml_fallback_parser()
        test_ast_helpers()
        test_all_match(tmp / "t3")
        test_module_name_mismatch(tmp / "t4")
        test_size_mismatch(tmp / "t5")
        test_model_input_mismatch(tmp / "t6")
        test_metadata_missing_size(tmp / "t7")
        test_hardcoded_and_override(tmp / "t8")
        test_strict_promotes_risk(tmp / "t9")
        test_missing_field(tmp / "t10")
        test_missing_file(tmp / "t11")
        test_parse_error(tmp / "t12")
        test_weights_reader_failure(tmp / "t13")
        test_source_of_truth_yaml(tmp / "t14")
    print("\n" + "=" * 60)
    if _failures:
        print(f"{_failures} / {_checks} 실패")
    else:
        print(f"전부 통과 ({_checks}건)")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
