# BFM / 학습 로그 분석 도구

`tools/` 아래 분석 도구의 목적, 입력, 실행법, 출력을 정리한다.
모든 결과는 `analysis/` 에만 쓴다. 원본 로그와 Tacview CSV 는 수정하지 않는다.

## 로그 구조 (2026-08-04 실측)

| 파일 | 위치 | 내용 |
|---|---|---|
| `training_log.csv` | `artifacts/logs/<name>/<tag>/` | iteration 단위 49컬럼 |
| `replay_index.jsonl` | `artifacts/logs/<name>/<tag>/engagement_replays/` | 경기 인덱스 |
| `<ts>_summary.json` | 각 `episode_NN/` | 키 4개: `end_condition`, `outcome`, `ownship_health`, `target_health` |
| Tacview CSV | 각 `episode_NN/` | `Time`, `Longitude`, `Latitude`, `Altitude`, `Roll (deg)`, `Pitch (deg)`, `Yaw (deg)`, `Health` |
| `metrics.jsonl` | `artifacts/dashboard/<name>_<tag>/` | 키 10개, 보상 컴포넌트 없음 |

주의할 점 두 가지.

**`summary.json` 이라는 이름의 파일은 없다.** 실제 이름은 `<timestamp>_summary.json` 이다.
`find -name summary.json` 으로는 하나도 찾지 못한다.

**Tacview CSV 에 속도 컬럼이 없다.** 속도는 위경도/고도 차분으로 추정한다. 질량 정보가
없어 절대 에너지 대신 `specific_energy = g*h + 0.5*v^2` 근사를 쓴다. 산출물에도 명시된다.

## BFM 모드를 쓰려면 먼저 stdout 을 받아야 한다

BFM 모드는 위 어떤 파일에도 없다. BT(C++)가 `std::cout` 으로만 찍는다.

```
[SetBFMMode_OBFM] t=12.34s | Enter OBFM (AA=..., D=..., EnergySup=...)
[SetBFMMode_HABFM] t=12.36s | Blocked | sight=1, AA=..., D=..., E=...
```

`t=` 는 `BB->RunningTime`(초)이다. 파일로 저장되지 않으므로 실행 시 리다이렉트해야 한다.

```powershell
python run_local_dogfight.py --ownship-backend rl `
  --ownship-bundle-dir artifacts/models/stil/<tag> `
  --target-backend bt --target-bt-dll AIP_STIL.dll `
  --observation-mode custom --observation-module student.my_observation `
  *> bt_stdout.log

python tools/extract_bfm_log.py --stdout bt_stdout.log --output analysis/bfm
```

> **현재 상태 (2026-08-04): BFM 로그가 생성되지 않는다.**
> `AIP_STIL.dll` 을 30 tick 직접 구동해도 `[SetBFMMode_*]` 가 한 줄도 나오지 않고
> VP 가 `(0,0,0)` 에서 갱신되지 않는다. 같은 DLL 에서 HP=0 분기의
> `HP : 0 !!!!!!!!!` 는 정상 출력되므로 stdout 캡처 문제가 아니다.
> BT 가 BFM 판정 노드에 도달하기 전에 트리가 끊기는 것으로 보인다.
> 이 문제가 해결되기 전까지 BFM 기반 분석(baseline, HABFM 고착, 1C/2C 비교)은
> 입력이 없어 실행할 수 없다. 도구는 완성돼 있으므로 로그만 생기면 바로 동작한다.

## 도구별 사용법

### 1. 종료 원인 집계 — `count_end_conditions.py`

```powershell
python tools/count_end_conditions.py --logdir artifacts/logs/stil --output analysis/end_conditions
```

출력
- `end_condition_counts.json` — 원본 값별/정규화 유형별 카운트, 비율, 승패 교차
- `end_condition_counts.csv`
- `unmapped_end_conditions.txt` — 매핑 안 된 원본 값 (임의로 `other` 에 묻지 않는다)
- `end_condition_matches.csv` — 경기별 원본 행

정규화 유형: `altitude_check_failed`, `collision`, `timeout`, `wez_hit`, `fdm_fail`,
`fuel`, `other`, `unknown`. 매핑은 `tools/log_analysis/normalization.py` 의
`END_CONDITION_ALIASES` 에 있다.

### 2. 지표 대시보드 — `dashboard.py`

```powershell
python tools/dashboard.py --logdir artifacts/logs/stil --port 7860
python tools/dashboard.py --logdir artifacts/logs/stil --host 0.0.0.0 --port 7860 --window 20 --refresh-sec 5
```

`reward_mean`, `crash_rate`, `ep_min_distance`, `ep_wez_steps`, `win_rate` 를 한 화면에
띄운다. 지표마다 현재값 / 이동평균 / 표본 수 / 추세 / min·max·mean 과 스파크라인을 보여준다.
raw 는 회색, 이동평균은 노랑이다.

- 요청마다 로그를 다시 읽으므로 학습 중에도 재시작 없이 갱신된다.
- 없는 지표는 전체 종료가 아니라 해당 카드만 `N/A` 로 표시하고 경고를 남긴다.
- `--logdir` 아래에 `training_log.csv` 가 여러 개면 실험별로 나누어 함께 보여준다.
- 표준 라이브러리만 쓴다. 새 의존성이 없다.
- 벤더 대시보드(`DogFightEnv/tools/dogfight_dashboard`)가 있으면 그쪽을 그대로 호출한다.
  현재 드롭에는 그 디렉터리가 없어 내장 서버로 동작한다. 같은 이유로 벤더의
  `tools/web_log_viewer.py` 와 `tools/training_dashboard/server.py` 는 `ImportError` 로
  실행되지 않는다 — 복기(Replay)는 아래 `--playback-dir` 로 한다.

**Replay 탭 (DogFight Log Playback)**

```powershell
python tools/dashboard.py --playback-dir analysis/playback_cases --port 7860
python tools/dashboard.py --logdir artifacts/logs/stil --playback-dir analysis/playback_cases --port 7860
```

`export_playback_cases.py` 가 만든 케이스를 재생한다. 궤적(위에서 본 평면 + 고도),
Own ATA / Target AA / 거리 / 속도 HUD, WEZ badge, BFM·SCISSORS badge, ATA/AA 시간 그래프,
이벤트 타임라인 marker, 재생·일시정지·배속(0.25~8×)·시간 이동, hover 상세,
`playback.json` / `trajectory.csv` 내려받기를 제공한다.
`case_id` 는 manifest 에 있는 것만 열린다(경로 탈출 요청은 404).

### 3. 패배 직전 패턴 — `analyze_loss_patterns.py`

```powershell
python tools/analyze_loss_patterns.py --logdir artifacts/logs/stil --window-sec 5 `
  --min-altitude 300 --output analysis/loss_patterns
```

조정 가능한 임계값: `--low-altitude-margin`, `--descent-rate`, `--speed-loss`,
`--bfm-stuck-sec`, `--bfm-thrash-count`.

출력
- `loss_event_timeline.csv` — 경기별 이벤트 시간순
- `loss_pattern_summary.json` / `.csv`
- `loss_pattern_report.md`

이벤트: `HIGH_DESCENT`, `LOW_ALTITUDE`, `ENERGY_REVERSAL`, `ENERGY_DEFICIT`,
`SPEED_LOSS`, `ALTITUDE_CHECK_FAILED`, (`BFM_STUCK`, `BFM_THRASH` — BFM 로그 필요).

BFM 축은 데이터가 없으면 계산하지 않고 그 사유를 보고서에 남긴다.
패턴이 3종 미만이면 억지로 만들지 않고 표본 부족 사유를 적는다.

### 4. BFM stdout 추출 — `extract_bfm_log.py`

```powershell
python tools/extract_bfm_log.py --stdout bt_stdout.log --output analysis/bfm
```

출력
- `bfm_events.csv` — Enter/Blocked 원본 이벤트
- `bfm_timeline.csv` — Enter 기준 체류 구간
- `bfm_extract_meta.json` — 에피소드 수, 모드별 진입 횟수, 1C/2C 카운트

`t=` 가 되감기면 새 에피소드로 나눈다(`RunningTime` reset). 각 에피소드의 마지막
구간은 종료 시각을 알 수 없어 `duration_sec` 이 비어 있다. 0 으로 채우지 않는다.

### 5. BFM baseline — `analyze_bfm_baseline.py`

```powershell
python tools/analyze_bfm_baseline.py --logdir analysis/bfm --episodes 20 --output analysis/baseline
```

마지막 구간 처리 방식은 `--last-segment drop`(기본) 또는 `median` 이다.
`median` 은 같은 에피소드 다른 구간의 중앙값으로 대체하고 그 건수를 산출물에 기록한다.

출력
- `bfm_baseline_20.json` — 요구 스키마(`baseline_name`, `episode_ids`,
  `mode_duration_sec`, `mode_ratio` 등) + 경기별 상세
- `bfm_baseline_20.csv`
- `bfm_baseline_report.md`

시간 기준 비율과 진입 횟수 기준 비율을 모두 낸다.

baseline 과 비교:

```powershell
python tools/analyze_bfm_baseline.py --logdir analysis/bfm_new `
  --compare-baseline analysis/baseline/bfm_baseline_20.json
```

비교 결과에 모드별 비율 차이, percentage point 차이, 상대 변화율, 가장 증가/감소한 모드,
HABFM 편향 여부가 들어간다.

### 6. HABFM 타임아웃 분석 — `analyze_timeout_habfm.py`

```powershell
python tools/analyze_timeout_habfm.py --logdir artifacts/logs/stil --window-sec 30 `
  --habfm-ratio-threshold 0.7 --habfm-continuous-sec 10 --output analysis/habfm_timeout
```

타임아웃/무승부 경기의 종료 직전 구간에서 모드별 체류시간·비율, 전환 횟수,
HABFM 연속 체류 최대시간, 마지막 모드를 낸다. 고착 판정은 두 임계값 중 하나라도
넘으면 의심으로 표시한다.

수정 전후 비교:

```powershell
python tools/analyze_timeout_habfm.py compare --before <전_경로> --after <후_경로> `
  --output analysis/habfm_timeout_comparison
```

BFM 로그가 없으면 계산 가능한 지표(경기 수, 타임아웃 비율, 승률, 추락률, 평균 보상,
종료 원인 분포)만 내고, 계산 불가 지표를 이름과 사유와 함께 따로 보고한다.

### 7. PredictManeuver 전/후 비교 — `analyze_predict_maneuver.py`

```powershell
python tools/analyze_predict_maneuver.py compare `
    --before logs/predict/before --after logs/predict/after `
    --outlier-threshold-deg 170 --spike-threshold-deg 90 --min-matches 20 `
    --output analysis/predict_maneuver_comparison
```

입력은 BT 의 `PredictManeuverCsvLogger` 가 남긴 CSV 다(환경변수 `PM_CSV_LOG` 를 설정한
실행에서만 생긴다). `avgDelta` 분포와 ±180 부근 이상값, wrap 보정 직접 증거,
SCISSORS 진입 빈도(**transition 기준**)를 비교하고 `wraparound_verdict` 를 낸다.
로그가 없으면 `INSUFFICIENT_DATA` 로 판정하고 종료 코드 2 를 돌려준다 — 값을 지어내지 않는다.

자세한 절차와 판정 기준: `docs/predict_maneuver_and_observation_validation.md`.

### 8. 대표 경기 복기 데이터 — `export_playback_cases.py`

```powershell
python tools/export_playback_cases.py --logdir artifacts/logs `
    --output analysis/playback_cases --handoff analysis/rl_trajectory_handoff
```

대표 패배 경기를 유형별로 고르고, Tacview 궤적에서 ATA/AA/WEZ/속도/에너지를 다시 계산해
뷰어가 읽는 `playback.json` 을 만든다. 파생값은 전부 `derived_` 접두사가 붙고
`field_origin` 에 출처가 적힌다. 원본 로그는 복사하지 않고 경로만 기록한다.

### 9. observation 설정 검사 — `check_observation_consistency.py`

```powershell
python tools/check_observation_consistency.py `
    --config experiments/<exp>.yaml --metadata <bundle>/metadata.json `
    --train-script train_rllib.py --local-runner run_local_dogfight.py `
    --submission student/my_submission.py `
    --observation-module-file student/my_observation.py `
    --bundle-weights <bundle>/policy_weights.pkl.gz
```

YAML / `train_rllib.py` / `metadata.json` / `run_local_dogfight.py` / `my_submission.py` 와
observation 모듈·bundle 가중치를 비교한다. Python 파일은 AST 로 읽는다.
종료 코드 0/1/2/3 을 돌려주므로 CI 나 교전 preflight 에 쓸 수 있다.

## 테스트

```powershell
python tests/tools/test_log_analysis.py               # 46건
python tests/tools/test_predict_maneuver.py           # 93건
python tests/tools/test_observation_consistency.py    # 61건
```

외부 프레임워크 없이 실패 개수를 센다. 실제 로그를 fixture 로 복사하지 않고 임시
디렉터리에 최소 데이터를 만들어 쓴다. 다루는 경우: 빈 CSV, 누락 컬럼, 잘못된 timestamp,
시간 뒤섞임, 동일 timestamp, NaN, 미지의 BFM 모드, 미지의 end condition,
깨진 JSON 줄, 창보다 짧은 경기, 마지막 구간 duration 처리.

## 결과 해석 시 주의

- **속도와 에너지는 추정값이다.** Tacview 에 속도 컬럼이 없어 위치 차분으로 구한다.
  샘플 간격이 불규칙하면 오차가 커진다.
- **`min_altitude` 기본값은 300 m 다**(`run_local_dogfight.py` 기본값). 환경 설정을
  바꿨다면 `--min-altitude` 로 맞춰야 한다.
- **없는 값은 `N/A` 로 나온다.** 0 으로 채우지 않으므로 0 과 결측을 구분해서 읽어야 한다.
- **경기 ID 에 실험 태그가 붙는다**(`<tag>/iterNNNNNN_epNN`). 여러 실험을 한 번에 읽어도
  ID 가 겹치지 않는다.

## 새 실험을 baseline 과 비교하는 절차

1. 실험 실행 (stdout 리다이렉트 포함)
2. `extract_bfm_log.py` 로 BFM 타임라인 추출
3. `analyze_bfm_baseline.py --compare-baseline` 으로 비교
4. `count_end_conditions.py`, `analyze_loss_patterns.py` 로 종료 원인·패배 패턴 확인
5. `dashboard.py` 로 5대 지표 추세 확인
