# case_001 — C_WEZ_OR_DEFENSE_FAILURE

- episode_id: `sac_mlp_obs8_iter400/iter000000_ep00`
- 결과: **LOSS** / end_condition: `ownship altitude below min`
- outcome(원문): `crash`
- 핵심 timestamp: 1.1666
- 길이: 6.9333 초 / 샘플 417개 (stride 1)
- 체력: 아군 0.9286207608866219 / 표적 1.0

## 선정 이유

표적이 나를 WEZ 안에 둔 시간 0.05s, 내 체력 손실 0.0714. WEZ/방어 실패로 끝난 패배.

## 시계열 요약 (파생값)

- Own ATA: 최소 0.18도 / 최대 179.11도
- Target AA: 최소 17.50도 / 최대 179.95도
- 내가 WEZ 안이던 프레임: 0 / 417
- 표적이 나를 WEZ 안에 둔 프레임: 3 / 417
- GeoMathUtil 부호 붕괴 구간 프레임: 23 / 417 (플랫폼 결함 1 — 이 구간의 ATA 부호는 신뢰할 수 없다)

## 이벤트

| 시각(초) | 유형 | 내용 |
|---:|---|---|
| 1.167 | `WEZ_ENTER_TARGET` | 표적이 나를 WEZ 안에 0.05s 유지 |
| 1.167 | `OWN_DAMAGE` | 체력 1.0000 -> 0.9887 |
| 1.183 | `OWN_DAMAGE` | 체력 0.9887 -> 0.9681 |
| 1.200 | `OWN_DAMAGE` | 체력 0.9681 -> 0.9398 |
| 1.217 | `OWN_DAMAGE` | 체력 0.9398 -> 0.9286 |
| 6.933 | `EPISODE_END` | ownship altitude below min (outcome=crash) |

## 원본 파일

- ownship_log: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\iter_000000\episode_00\2026_8_4_0_23_35_ownship_(F-16)[Blue].csv`
- target_log: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\iter_000000\episode_00\2026_8_4_0_23_35_target_(F-16)[Red].csv`
- summary_json: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\iter_000000\episode_00\2026_8_4_0_23_35_summary.json`
- replay_index: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_obs8_iter400\engagement_replays\replay_index.jsonl`

## 담지 못한 값

- PredictManeuver CSV(--predict-log)를 주지 않아 BFM 모드 / SCISSORS / avgDelta 를 담지 못했다. 유형 A(avgDelta 이상값)와 유형 B(SCISSORS)는 선정하지 않았다. 로그를 만들려면 PM_CSV_LOG 를 설정하고 교전을 실행하라.
