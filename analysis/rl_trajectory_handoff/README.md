# RL 담당자용 대표 궤적 전달 자료

생성: 2026-08-04T07:29:33

## 1. 분석 목적

PredictManeuver 의 각도 wrap-around 수정 이후 남은 패배 패턴을 확인하고,
보상·관측 설계에 반영할 이벤트를 고르기 위한 자료다. 코드를 실행하지 않아도
대표 경기의 궤적과 핵심 이벤트를 볼 수 있게 정리했다.

## 2. 사용한 실험과 checkpoint

- 로그 위치: `C:\AIP_LIB\DogFightEnv\Release\artifacts\logs`
- 실험(run): ['sac_mlp_obs8_iter400', 'sac_mlp_obs8_rand100', 'sac_mlp_obs8_smoke', 'sac_mlp_obs8_v3_alt400', 'sac_mlp_v1', 'smoke_custom']
- observation 설정: observation_module=student.my_observation, mode=stil8, size=8 (tools/check_observation_consistency.py 로 확인)

## 3. 대표 경기 목록과 선정 이유

| case | 유형 | episode_id | 결과 | 핵심 시각(초) | 선정 이유 |
|---|---|---|---|---:|---|
| case_001 | `C_WEZ_OR_DEFENSE_FAILURE` | `sac_mlp_obs8_iter400/iter000000_ep00` | LOSS | 1.17 | 표적이 나를 WEZ 안에 둔 시간 0.05s, 내 체력 손실 0.0714. WEZ/방어 실패로 끝난 패배. |
| case_002 | `D_LOW_ALTITUDE_CRASH` | `sac_mlp_v1/iter000020_ep00` | LOSS | 6.33 | end_condition='ownship altitude below min'. 최저 고도 258.5 m 로 최소 고도 아래에서 종료. |
| case_003 | `F_TIMEOUT_DRAW` | `sac_mlp_obs8_iter400/iter000100_ep00` | DRAW | 9.45 | outcome='draw', end_condition='target altitude below min'. 결착 없이 종료. |
| case_004 | `H_ENERGY_REVERSAL` | `sac_mlp_obs8_iter400/iter000280_ep00` | LOSS | 2.53 | t=2.53s 에 비에너지 우위가 역전됐고 최저 차이 -213582 J/kg 까지 벌어진 뒤 패배. |

## 4. 웹 뷰어 실행 방법

```powershell
# Release 에서 실행. 추가 의존성 없음(표준 라이브러리만 쓴다).
python tools/dashboard.py --playback-dir analysis\playback_cases --port 7860
# 브라우저: http://localhost:7860/  -> 상단 'Replay' 탭
```

학습 지표까지 함께 보려면:

```powershell
python tools/dashboard.py --logdir artifacts/logs/stil --playback-dir analysis\playback_cases --port 7860
```

## 5. 각 파일의 역할

| 파일 | 내용 |
|---|---|
| `trajectory_manifest.json` | 케이스 목록. 뷰어와 스크립트의 진입점 |
| `representative_episodes.csv` | 대표 경기 한 줄 요약 |
| `event_timeline.csv` | 모든 케이스의 이벤트를 한 파일에 모은 것 |
| `playback_cases/<case>/playback.json` | 시계열 + 이벤트 (뷰어가 읽는다) |
| `playback_cases/<case>/trajectory.csv` | 같은 내용의 CSV |
| `playback_cases/<case>/source_summary.json` | 원본 summary + replay_index 행 |
| `playback_cases/<case>/case_report.md` | 선정 이유와 핵심 시각 |

원본 Tacview CSV 는 복사하지 않았다. 각 케이스의 `source_files` 에 경로가 있다.

## 6. Own ATA / Target AA / WEZ 해석

- **own_ata_deg** — 내 기수와 표적 LOS 사이의 각. 부호 있음. 0 = 정조준, 양수/음수는 좌우. |ATA| 가 작을수록 유리하다.
- **target_aa_deg** — 표적 기준 aspect angle. **0 = 내가 표적의 6시**, 180 = 표적의 정면. GeoMathUtil 규약이다. BT 의 MyAspectAngle_Degree 는 반대 규약이니 섞지 말 것.
- **in_wez** — update_damage 와 같은 게이트: min_range <= 거리 <= max_range 이고 angle_deg/2 >= |ATA|. 기본 2.0도이므로 실제 원뿔은 **1도**다.

이 세 값은 **로그에 없어서 다시 계산한 파생값**이다. 계산식은 호스트의
`GeoMathUtil.GeometryInfo` / `single_agent_env.update_damage` 와 같다.
`derived_ata_sign_degenerate` 가 True 인 프레임은 GeoMathUtil 의 부호 규칙이
붕괴하는 구간이라(플랫폼 결함 1) ATA 부호를 믿으면 안 된다.

## 7. PredictManeuver 이상값 / SCISSORS 관찰

- PredictManeuver CSV(--predict-log)를 주지 않아 BFM 모드 / SCISSORS / avgDelta 를 담지 못했다. 유형 A(avgDelta 이상값)와 유형 B(SCISSORS)는 선정하지 않았다. 로그를 만들려면 PM_CSV_LOG 를 설정하고 교전을 실행하라.

## 8. 보상 / 관측 설계에 참고할 이벤트

- `WEZ_ENTER_TARGET` — 표적이 나를 조준한 구간. 이 구간의 진입 조건이
  방어 보상의 1차 후보다.
- `OWN_DAMAGE` — 실제 체력이 깎인 시점. WEZ 보상 가중치를 이 시점 기준으로
  검증할 수 있다.
- `EPISODE_END` — 종료 원인. 현재 로그에서는 고도 하한 위반이 지배적이다.
- `BFM_TRANSITION` / `SCISSORS_ENTER` — PredictManeuver CSV 를 붙였을 때만 존재.

## 9. 재현 명령

```powershell
# 대표 경기 데이터 재생성
python tools/export_playback_cases.py --logdir C:\AIP_LIB\DogFightEnv\Release\artifacts\logs \
    --output analysis\playback_cases --handoff analysis\rl_trajectory_handoff
```

> seed: 현재 로그(replay_index.jsonl / summary.json)에는 seed 필드가 없다.
> 정확한 재현이 필요하면 실험 YAML 과 commit hash 를 함께 고정해야 한다.

## 10. observation 설정

observation_module=student.my_observation, mode=stil8, size=8 (tools/check_observation_consistency.py 로 확인)

```powershell
python tools/check_observation_consistency.py \
    --config experiments/stil_sac_mlp_obs8_iter400.yaml \
    --metadata <bundle>/metadata.json \
    --bundle-weights <bundle>/policy_weights.pkl.gz
```
