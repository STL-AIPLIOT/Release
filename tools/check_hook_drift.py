# -*- coding: utf-8 -*-
"""번들을 학습시킨 관측 훅과 지금 `student/` 에 있는 훅이 같은지 검사한다.

왜 필요한가
-----------
번들은 **가중치만** 저장한다. `run_local_dogfight` / `run_unreal_inference` 는
`--observation-module student.my_observation` 로 **지금 디스크에 있는 모듈을 그때그때
import** 해서 관측을 만든다. 그래서 학습 뒤에 훅을 고치면, 옛 가중치에 새 관측이
들어간다. **예외도 경고도 없이 성능만 무너진다.**

특히 정규화 상수 하나만 바꿔도 그렇다. `OBSERVATION_SIZE` 가 그대로라 크기 검사
(`check_observation_consistency.py`)는 통과하고, feature 이름도 그대로라 metadata 도
일치한다. 즉 **기존 검사로는 잡히지 않는다.**

`train_rllib.py` 는 학습할 때 훅 사본을 `artifacts/records/<name>/<tag>/my_observation.py`
로 남긴다. 그 사본과 현재 파일을 비교하면 드리프트를 정확히 잡을 수 있다.

한계
----
- 보상 훅(`my_reward.py`)은 records 에 저장되지 않는다. 보상은 행동을 만들지 않으므로
  평가 정확도에는 영향이 없지만, **재현에는 영향이 있다.**
- 주석·공백만 바뀌어도 다르다고 본다. `--ignore-comments` 로 완화할 수 있으나,
  기본값은 엄격이다(정규화 상수가 주석처럼 보이는 줄에 있을 수 있다).

실행
----
    python tools/check_hook_drift.py --bundle-dir artifacts/models/stil/<tag>
종료 코드 0=동일, 1=드리프트, 2=비교 불가(사본 없음)
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import warn  # noqa: E402

# records 에 남는 학생 훅 사본. reward 는 저장되지 않는다(위 '한계' 참조).
SNAPSHOT_NAMES = ("my_observation.py",)


def records_dir_for(bundle_dir: Path) -> Path | None:
    """artifacts/models/<name>/<tag> -> artifacts/models 를 records 로 바꾼 경로."""
    parts = list(bundle_dir.resolve().parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "models":
            parts[i] = "records"
            return Path(*parts)
    return None


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        body = line.split("#", 1)[0].rstrip()
        if body:
            out.append(body)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="번들 학습 시점 훅과 현재 훅 비교")
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--student-dir", type=Path, default=Path("student"))
    ap.add_argument("--records-dir", type=Path, default=None,
                    help="자동 추론(models->records)이 안 될 때 직접 지정")
    ap.add_argument("--ignore-comments", action="store_true",
                    help="주석/빈 줄 차이는 무시한다. 기본은 엄격 비교")
    ap.add_argument("--show-diff", action="store_true")
    args = ap.parse_args()

    records = args.records_dir or records_dir_for(args.bundle_dir)
    if records is None or not records.exists():
        warn(f"학습 시점 훅 사본을 찾지 못했다: {records}\n"
             "  드리프트를 검사할 수 없다. 이 번들이 지금 student/ 로 학습된 것이 "
             "확실한지 직접 확인하라.")
        return 2

    drift = []
    checked = 0
    for name in SNAPSHOT_NAMES:
        snap, live = records / name, args.student_dir / name
        if not snap.exists():
            warn(f"사본 없음: {snap}")
            continue
        if not live.exists():
            warn(f"현재 훅 없음: {live}")
            drift.append(name)
            continue

        checked += 1
        a = snap.read_text(encoding="utf-8")
        b = live.read_text(encoding="utf-8")
        if args.ignore_comments:
            a, b = strip_comments(a), strip_comments(b)

        if hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest():
            print(f"  [OK]    {name} — 학습 시점과 동일")
            continue

        drift.append(name)
        print(f"  [DRIFT] {name} — 학습 시점과 다르다")
        if args.show_diff:
            for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                             fromfile=f"학습시점/{name}",
                                             tofile=f"현재/{name}", lineterm="", n=1):
                print(f"          {line}")

    if checked == 0:
        warn("비교한 파일이 없다")
        return 2
    if drift:
        print()
        warn("관측 훅이 학습 이후 바뀌었다. 이 번들을 지금 평가하면 "
             "**옛 가중치에 새 관측**이 들어가고, 예외 없이 성능만 무너진다.\n"
             "  선택지: (1) 평가 전에 훅을 학습 시점으로 되돌린다  "
             "(2) 바뀐 훅으로 처음부터 재학습한다")
        return 1

    print(f"\n훅 {checked}개 일치 — 이 번들을 평가해도 된다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
