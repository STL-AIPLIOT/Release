# case_002 — D_LOW_ALTITUDE_CRASH

- episode_id: `sac_mlp_v1/iter000020_ep00`
- 결과: **LOSS** / end_condition: `ownship altitude below min`
- outcome(원문): `crash`
- 핵심 timestamp: 6.3333
- 길이: 6.3333 초 / 샘플 381개 (stride 1)
- 체력: 아군 1.0 / 표적 1.0

## 선정 이유

end_condition='ownship altitude below min'. 최저 고도 258.5 m 로 최소 고도 아래에서 종료.

## 시계열 요약 (파생값)

- Own ATA: 최소 0.10도 / 최대 178.18도
- Target AA: 최소 15.14도 / 최대 179.95도
- 내가 WEZ 안이던 프레임: 0 / 381
- 표적이 나를 WEZ 안에 둔 프레임: 0 / 381
- GeoMathUtil 부호 붕괴 구간 프레임: 74 / 381 (플랫폼 결함 1 — 이 구간의 ATA 부호는 신뢰할 수 없다)

## 이벤트

| 시각(초) | 유형 | 내용 |
|---:|---|---|
| 6.333 | `EPISODE_END` | ownship altitude below min (outcome=crash) |

## 원본 파일

- ownship_log: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_v1\engagement_replays\iter_000020\episode_00\2026_8_2_22_27_32_ownship_(F-16)[Blue].csv`
- target_log: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_v1\engagement_replays\iter_000020\episode_00\2026_8_2_22_27_32_target_(F-16)[Red].csv`
- summary_json: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_v1\engagement_replays\iter_000020\episode_00\2026_8_2_22_27_32_summary.json`
- replay_index: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs\stil\sac_mlp_v1\engagement_replays\replay_index.jsonl`

## 담지 못한 값

- PredictManeuver CSV(--predict-log)를 주지 않아 BFM 모드 / SCISSORS / avgDelta 를 담지 못했다. 유형 A(avgDelta 이상값)와 유형 B(SCISSORS)는 선정하지 않았다. 로그를 만들려면 PM_CSV_LOG 를 설정하고 교전을 실행하라.
