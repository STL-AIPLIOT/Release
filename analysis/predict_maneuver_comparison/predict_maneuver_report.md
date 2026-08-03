# PredictManeuver wrap-around 수정 전/후 비교

- 생성 시각: 2026-08-04T07:23:20
- 판정: **INSUFFICIENT_DATA**
- 각도 단위: degree / 거리 meter / 시간 second

## 1. 입력

| 항목 | before | after |
|---|---|---|
| logdir | `logs\predict\before` | `logs\predict\after` |
| CSV 파일 수 | 0 | 0 |
| 프레임 수 | 0 | 0 |
| 경기 수 | 0 | 0 |
| episode 파생 여부 | False | False |
| 없는 컬럼 | - | - |

## 2. 판정 근거

- after 로그에 프레임이 없다.

## 3. 로그를 만드는 방법

```powershell
# 수정 전: wrap 보정이 없던 커밋으로 DLL 을 만든 뒤
$env:PM_CSV_RUNTYPE = "before"
$env:PM_CSV_LOG     = "logs/predict/before/run01.csv"
python run_local_dogfight.py --ownship-backend rl `
    --ownship-bundle-dir artifacts/models/stil/sac_mlp_obs8_iter400 `
    --observation-module student.my_observation `
    --target-backend bt --target-bt-dll AIP_STIL.dll --save-log

# 수정 후: 현재 트리로 만든 DLL 로 같은 인자 반복
$env:PM_CSV_RUNTYPE = "after"
$env:PM_CSV_LOG     = "logs/predict/after/run01.csv"
python run_local_dogfight.py ...
```

권장 경기 수: 그룹당 최소 20경기 (`--min-matches` 로 조정).

## 4. 같은 조건인지 확인할 항목

- [ ] seed
- [ ] 상대 정책 또는 상대 bundle (--target-backend / --target-bt-dll / --target-bundle-dir)
- [ ] aircraft 설정 (Release/aircraft, engine)
- [ ] scenario (initial_scenario: altitude_m / distance_m / heading)
- [ ] episode 제한시간 (--max-engage-time / --episode-step-limit)
- [ ] observation 설정 (--observation-mode / --observation-module)
- [ ] reward 설정 (reward_module / MY_REWARD_CONFIG)
- [ ] PredictManeuver 설정 (historySize=5, TURN_THRESHOLD_DEG=1.5)
- [ ] BFM 상태 전환 설정 (Rule.xml, SetBFMMode_* 임계값)
- [ ] 경기 수 (--min-matches 이상)
- [ ] checkpoint / bundle 디렉터리
- [ ] 코드 commit hash (Release / Behaviortree 각각)
