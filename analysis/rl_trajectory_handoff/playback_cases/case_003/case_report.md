# case_003 — F_TIMEOUT_DRAW

- episode_id: `sac_mlp_obs8_iter400/iter000100_ep00`
- 결과: **DRAW** / end_condition: `target altitude below min`
- outcome(원문): `draw`
- 핵심 timestamp: 9.45
- 길이: 9.45 초 / 샘플 568개 (stride 1)
- 체력: 아군 1.0 / 표적 1.0

## 선정 이유

outcome='draw', end_condition='target altitude below min'. 결착 없이 종료.

## 시계열 요약 (파생값)

- Own ATA: 최소 0.21도 / 최대 174.32도
- Target AA: 최소 4.37도 / 최대 179.95도
- 내가 WEZ 안이던 프레임: 0 / 568
- 표적이 나를 WEZ 안에 둔 프레임: 0 / 568
- GeoMathUtil 부호 붕괴 구간 프레임: 23 / 568 (플랫폼 결함 1 — 이 구간의 ATA 부호는 신뢰할 수 없다)

## 이벤트

| 시각(초) | 유형 | 내용 |
|---:|---|---|
| 9.450 | `EPISODE_END` | target altitude below min (outcome=draw) |

## 원본 파일

- ownship_log: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\iter_000100\episode_00\2026_8_4_0_46_53_ownship_(F-16)[Blue].csv`
- target_log: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\iter_000100\episode_00\2026_8_4_0_46_53_target_(F-16)[Red].csv`
- summary_json: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\iter_000100\episode_00\2026_8_4_0_46_53_summary.json`
- replay_index: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\replay_index.jsonl`

## 담지 못한 값

- PredictManeuver CSV(--predict-log)를 주지 않아 BFM 모드 / SCISSORS / avgDelta 를 담지 못했다. 유형 A(avgDelta 이상값)와 유형 B(SCISSORS)는 선정하지 않았다. 로그를 만들려면 PM_CSV_LOG 를 설정하고 교전을 실행하라.
