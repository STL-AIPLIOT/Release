# observation 설정 일관성 점검

- 생성 시각: 2026-08-04T07:26:42
- source of truth: **metadata.json** (기존 checkpoint/bundle 을 실행·제출하는 경로이므로 학습 시 확정된 값이 기준이다.)
- 실효 observation 모드: `stil8`
- 실효 observation 크기: 8
- 판정: **PASS**

## 1. 위치별 값

| 위치 | observation_mode | observation_module | observation_size | 값의 출처 | 상태 |
|---|---|---|---:|---|---|
| YAML | `custom` | `student.my_observation` | — | 직접 설정 | `MATCH` |
| train_rllib.py | `tactical16` | (빈 문자열) | — | YAML/CLI 값을 받아 env_config 로 전달 | `OVERRIDDEN` |
| metadata.json | `stil8` | `student.my_observation` | `8` | 학습 시 저장 | `MATCH` |
| run_local_dogfight.py | `tactical16` | (빈 문자열) | — | CLI 인자 기본값 (metadata 를 읽지 않는다) | `HARDCODED` |
| my_submission.py | `custom` | `student.my_observation` | — | metadata 참조 | `HARDCODED` |
| observation module | `stil8` | — | `8` | 모듈 상수 (실제 반환 크기의 근거) | `MATCH` |

> `observation_mode` 가 YAML/제출 스크립트에서 `custom` 인데 metadata 에서 
> `stil8` 인 것은 **정상**이다. `observation_module` 이 설정되면 훅의 
> `OBSERVATION_MODE` 가 선언값을 덮어쓴다. 다섯 곳이 반드시 같아야 하는 값은 
> `observation_module` 이다.

## 2. 최종적으로 쓰이는 값

| 경로 | 값 | 결정 규칙 |
|---|---|---|
| 학습 | `student.my_observation / stil8 / 8` | YAML env.observation_module -> train_rllib.py 가 훅을 로드하고 observation_mode 를 훅의 OBSERVATION_MODE 로 덮어쓴다. |
| 로컬 실행 | ` (CLI 미지정 시) / tactical16` | run_local_dogfight.py 는 metadata 를 읽지 않는다. --observation-module 을 반드시 명시해야 한다. |
| 제출 실행 | `student.my_observation / custom` | my_submission.py 의 모듈 상수. metadata 를 읽지 않는다. |

## 3. 우선순위

1. observation_module 이 비어 있지 않으면 훅의 OBSERVATION_MODE 가 선언된 observation_mode 를 이긴다 (train_rllib.py / run_local_dogfight.py / my_submission.py 모두 같은 규칙).
2. CLI 인자는 YAML 값을 이긴다 (run_experiment.py 가 YAML 을 CLI 로 변환).
3. bundle 을 로드할 때는 metadata.json 의 env_config.observation_size 가 모델 입력 차원을 정한다 (RLLibInferenceEnv).
4. 그 키가 없으면 observation_size(mode) 가 12 를 돌려주어 조용히 틀린 크기가 된다. fix_bundle_obs_size.py 로 채워 둘 것.

## 4. checkpoint 재사용 가능 여부

- 판정: **COMPATIBLE**
- 근거: 모델 입력 차원(8)과 실행 시 observation 크기(8, 출처: observation 모듈(student\my_observation.py))가 같다.
- 모델 입력 차원: 8 (pi_encoder.net.mlp.0.weight shape=(256, 8) -> 입력 8)
- 새 output tag 필요: 불필요

## 5. 발견한 문제

| 심각도 | 항목 | 내용 |
|---|---|---|
| risk | `observation_module` | train_rllib.py 의 --observation-module 기본값이 '' 이다. 인자를 주지 않으면 'student.my_observation' 이 아닌 기본 관측으로 실행된다. |
| risk | `observation_mode` | train_rllib.py 의 --observation-mode 기본값이 'tactical16' 이라 실효 모드 'stil8' 와 다르다. |
| risk | `observation_module` | run_local_dogfight.py 의 --observation-module 기본값이 '' 이다. 인자를 주지 않으면 'student.my_observation' 이 아닌 기본 관측으로 실행된다. |
| risk | `observation_mode` | run_local_dogfight.py 의 --observation-mode 기본값이 'tactical16' 이라 실효 모드 'stil8' 와 다르다. |

- `mismatch` 실제로 쓰이는 값이 어긋난다 (종료 코드 1)
- `missing` 필수 필드가 없다 (종료 코드 2)
- `risk` 지금은 맞지만 인자를 빠뜨리면 조용히 틀려진다 (기본적으로 종료 코드에 영향 없음, `--strict` 로 승격)

## 6. 위치별 비고

### YAML

`experiments\stil_sac_mlp_obs8_iter400.yaml`

- YAML 에는 observation_size 가 없다. 크기는 observation_module 의 OBSERVATION_SIZE 가 정한다(정상).
- output: name='stil' tag='sac_mlp_obs8_iter400'

### train_rllib.py

`C:\AIP_LIB\DogFightEnv\Release\train_rllib.py`

- observation_module 이 있으면 cfg['observation_mode'] 를 훅의 OBSERVATION_MODE 로 덮어쓴다. 그래서 YAML 이 'custom' 이어도 metadata 에는 훅 이름(예: stil8)이 기록된다 — 정상 동작이다.
- CLI 기본값: --observation-mode='tactical16', --observation-module=''
- CLI 인자는 YAML 값을 덮어쓴다(run_experiment.py 가 YAML -> CLI 로 변환).
- 정적 분석 한계: 실행 시 실제로 넘어온 인자는 알 수 없다.

### metadata.json

`C:\AIP_LIB\DogFightEnv\Release\artifacts\models\stil\sac_mlp_obs8_iter400\metadata.json`

- observation_size 출처: algorithm_config.env_config.observation_size
- metadata.observation_schema_version 없음 (구 스키마) -> MISSING
- metadata.experiment_config 없음 (구 스키마) -> MISSING
- metadata.checkpoint_path 없음 (구 스키마) -> MISSING
- metadata.git_commit 없음 (구 스키마) -> MISSING
- feature 순서: ['own_alt_norm', 'own_kcas_norm', 'ata_norm', 'aa_norm', 'distance_norm', 'energy_advantage_norm', 'closure_rate_norm', 'in_wez_flag']

### run_local_dogfight.py

`C:\AIP_LIB\DogFightEnv\Release\run_local_dogfight.py`

- 이 스크립트는 bundle 의 metadata.json 을 읽지 않는다. --observation-module 을 주지 않으면 --observation-mode 기본값이 그대로 쓰인다.
- observation_module 을 주면 훅의 OBSERVATION_MODE 가 --observation-mode 를 이긴다(run_local_dogfight.py 의 observation_hook['mode'] 우선).
- 정적 분석 한계: 실행 시 준 CLI 인자는 알 수 없다.

### my_submission.py

`student\my_submission.py`

- 소스가 metadata.json 을 참조한다.
- BUNDLE_DIR='artifacts/models/stil/sac_mlp_v1'
- ACTION_REPEAT=6 (학습 env_config.step_ratio 와 같아야 한다)
- OBSERVATION_MODULE 이 비어 있지 않으면 훅의 OBSERVATION_MODE 가 OBSERVATION_MODE 상수를 이긴다.

### observation module

`student\my_observation.py`

- 정적 분석이라 build_observation() 이 실제로 이 길이를 돌려주는지는 확인하지 않는다. --import-check 를 주면 import 해서 확인한다.
