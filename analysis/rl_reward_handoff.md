# RL 보상 설계 전달 문서

패배 직전 5초 구간을 분석해 얻은 반복 패턴과, 그로부터 도출한 보상 후보를 정리했다.

- 근거 데이터: `analysis/loss_patterns/` (47건, `tools/analyze_loss_patterns.py` 산출)
- 소스 로그: `artifacts/logs/stil/**/engagement_replays/`
- 분석 구간: 종료 직전 5초
- 임계값: `min_altitude=300`, `low_altitude_margin=700`, `descent_rate=40 m/s`, `speed_loss=30 m/s`

## 데이터 요약

| 항목 | 값 |
|---|---|
| 전체 경기 | 57 |
| 패배(crash) | 47 |
| 무승부(draw) | 10 |
| 승리 | 0 |
| 종료 원인 | `altitude_check_failed` 100% |

이벤트 발생 빈도 (패배 47건 기준)

| 이벤트 | 횟수 | 비율 |
|---|---:|---:|
| `HIGH_DESCENT` | 47 | 100% |
| `LOW_ALTITUDE` | 47 | 100% |
| `ENERGY_REVERSAL` | 9 | 19% |

## 패턴 1: 급하강 후 저고도 추락

### 발생 조건
- 종료 직전 5초 안에 하강률 40 m/s 이상 진입
- 이어서 고도 1,000 m 이하(= `min_altitude` 300 + margin 700) 진입
- 회복 없이 `altitude_check_failed` 로 종료

### 관측된 순서
`HIGH_DESCENT → LOW_ALTITUDE → ALTITUDE_CHECK_FAILED`

### 발생 빈도
- 전체 패배 중 **89.4%** (42 / 47)
- 대표 경기: `sac_mlp_obs8_iter400/iter000000_ep00`,
  `sac_mlp_obs8_v3_alt400/iter000020_ep00`

### 보상 설계 후보
1. **하강률 패널티** — 저고도 구간에서만 활성화. 고고도 급강하는 정상 기동이므로
   고도 조건 없이 하강률만 벌하면 전술을 망친다.
2. **고도 마진 dense 패널티** — 이미 v3 에서 시도했다(`altitude` 컴포넌트).
   400 iteration 결과 효과가 없었고 오히려 `ep_len` 이 397 → 356 으로 짧아졌다.
   **누적 패널티가 `crash_penalty` 를 넘어서면 조기 종료가 이득이 된다**(v3 실측:
   0.4 × 400 step = -160 vs `crash_penalty` -150). 재시도 시 potential-based shaping
   (고도 마진의 *변화량*에 보상)으로 바꿔야 한다.
3. **회복 성공 보상** — 저고도 진입 후 정해진 시간 안에 상승 전환에 성공하면 보상.
   패널티만 주는 것보다 학습 신호가 명확하다.

### 주의사항
- 결과(추락)만 벌하면 이미 `crash_penalty` 가 하는 일과 중복된다. 과정을 벌해야 한다.
- 고도 유지만 보상하면 정책이 고고도로 도망가 교전을 회피한다.
  실제로 v3 에서 `ep_min_distance` 가 1,968 → 5,142 m 로 멀어졌다.
- 패널티 누적 상한을 `crash_penalty` 절대값보다 확실히 작게 잡아야 한다.

## 패턴 2: 에너지 우위 상실을 동반한 저고도 추락

### 발생 조건
- 패턴 1 과 동일한 흐름에 에너지 우위 상실이 겹친다
- specific energy 차(내 SE − 적 SE)가 양 → 음으로 바뀜

### 관측된 순서
`HIGH_DESCENT → ENERGY_REVERSAL → LOW_ALTITUDE → ALTITUDE_CHECK_FAILED`

### 발생 빈도
- 전체 패배 중 **10.6%** (5 / 47)
- 대표 경기: `sac_mlp_obs8_v3_alt400/iter000280_ep00`,
  `sac_mlp_obs8_v3_alt400/iter000300_ep00`

### 보상 설계 후보
1. **에너지 우위 변화량 보상** — 현재 `my_reward.py` 의 `energy` 항이 이미 이것이다
   (`energy_weight: 0.02`). 가중치가 작아 기여가 거의 없을 가능성이 있다.
2. **에너지 우위 부호 전환 시점 패널티** — 희소하지만 신호가 뚜렷하다.

### 주의사항
- 에너지 보상을 키우면 정책이 상승만 반복하는 회피 전략으로 붕괴한다.
  `my_reward.py` 주석에도 같은 경고가 있다("크게 주면 회피 정책으로 붕괴한다").
- 패턴 1 과 이벤트가 겹치므로 두 패턴에 각각 패널티를 주면 이중 처벌이 된다.

## 패턴이 2종뿐인 이유

억지로 3종을 만들지 않았다. 실제로 발견된 패턴이 2종이다.

- 종료 원인이 `altitude_check_failed` 한 가지로 100% 쏠려 있어 패턴이 갈릴 여지가 없다.
- 승리 경기가 0건이라 "성공 경로"와 대조할 수 없다.
- BFM 축을 못 써서 전환 실패 계열 패턴이 아예 빠져 있다(아래 참조).

패턴을 더 얻으려면 다음이 필요하다.
- BT 조기 실패 수정 후 재수집(BFM 축 확보)
- 승리 경기가 나올 만큼의 학습량
- 임계값 조정(`--descent-rate`, `--speed-loss`)으로 세분화

## 계산하지 못한 축: BFM 전환 실패

BFM 모드는 `training_log.csv` / Tacview CSV / `summary.json` / `replay_index.jsonl`
어디에도 없다. BT(C++)가 `std::cout` 으로만 찍고 파일에 남기지 않는다.

추가로, 직접 확인한 결과 **BT 가 BFM 판정 노드에 도달하지 못한다**:

- `AIP_STIL.dll` 을 30 tick 직접 구동 → `[SetBFMMode_*]` 로그 0건, VP 가 `(0,0,0)` 에서
  갱신되지 않음, Roll/Rudder 가 ±1.0 포화
- 같은 DLL 에서 HP=0 분기의 `HP : 0 !!!!!!!!!` 는 정상 출력 → **stdout 캡처 문제가 아니다**

따라서 다음은 BT 수정 전까지 계산할 수 없다.
- 특정 모드 과체류 / 전환 조건 발생했으나 미변경
- 짧은 시간 두 모드 반복 전환
- 방어 필요 상황에서 공격 모드 유지

## 보상값 제안 (숫자 확정 아님)

| 항목 | 근거 | 관측값 | 발생 시점 | 밀도 | 중복 위험 | 탐색 범위 |
|---|---|---|---|---|---|---|
| 저고도 하강률 패널티 | 패턴 1 (89.4%) | 고도, 하강률 | 매 step | dense | `altitude` 항과 겹침 | 0.05 ~ 0.2 |
| 고도 마진 shaping | 패턴 1 | 고도 | 매 step | dense | 위와 동일 | potential-based 로 전환 |
| 저고도 회복 보상 | 패턴 1 | 고도 변화 | 이벤트 | sparse | 없음 | 1 ~ 5 |
| 에너지 부호 전환 패널티 | 패턴 2 (10.6%) | SE 차 | 이벤트 | sparse | `energy` 항과 겹침 | 0.5 ~ 2 |

ablation 순서 제안
1. 기준선: 현재 `my_reward.py` 에서 `altitude` 항 제거 (v1 구성)
2. `altitude` 항을 potential-based 로 바꿔 단독 추가
3. 저고도 하강률 패널티 단독 추가
4. 2 + 3 동시
5. 에너지 부호 전환 패널티 추가

각 단계는 초기 조건 무작위화를 켠 상태로 최소 200 iteration, 같은 seed 로 비교한다.

## 선행 조건

이 문서의 제안을 실험하기 전에 **보상 컴포넌트 로깅 문제를 먼저 풀어야 한다.**
`train_rllib.py:1112` 의 CSV 컬럼이 고정이라 `my_reward.py` 가 내보내는 9개 키 중
`pursuit` 하나만 기록된다. 대시보드 `metrics.jsonl` 에도 보상 컴포넌트가 없다.
어느 항이 얼마나 기여하는지 볼 수 없는 상태에서 보상을 튜닝하면 노이즈를 쫓게 된다.
