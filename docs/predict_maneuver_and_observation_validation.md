# PredictManeuver 각도 검증 & observation 설정 일관성

2026-08-04 작성. 이 문서 하나로 다음을 할 수 있다.

- PredictManeuver 의 wrap-around 수정이 실제로 동작하는지 확인
- 수정 전/후 로그를 만들고 `avgDelta` 이상값과 SCISSORS 진입 빈도를 비교
- 대표 패배 경기를 웹 뷰어에서 복기
- observation 설정 다섯 곳이 일치하는지 확인하고, 어긋나면 어느 값을 기준으로 고칠지 판단

명령은 전부 **`AIP_LIB\DogFightEnv\Release`** 에서 실행한다(팀 저장소 `Release/` 를 그쪽으로
복사한 상태). 분석 도구만 쓸 때는 팀 저장소 `Release/` 에서 그대로 실행해도 된다.

---

## 1. PredictManeuver wrap-around 문제

`PredictManeuver` 는 표적의 Yaw 를 5프레임 모아 인접 차이의 평균(`avgDelta`)으로
좌/우 선회를 판정한다. Yaw 는 주기가 360도라 단순 뺄셈을 하면 ±180 경계에서 값이 뒤집힌다.

```text
이전 +179도, 현재 -179도
  단순 차이   -358도      <- 이 값이 평균에 들어가면 avgDelta 가 -88도로 급등한다
  최소 회전량 +2도        <- 실제로는 왼쪽으로 2도 돈 것
```

`avgDelta` 가 뒤집히면 `PredictedTurnDirection` 이 LEFT ↔ RIGHT 로 반대가 되고,
그 값을 읽는 기동 노드가 표적의 반대편으로 선회한다. 경계를 지나는 순간에만 생기므로
평소에는 정상으로 보인다.

## 2. 수정한 계산 방식

공통 유틸리티 하나만 쓴다. 사본을 두면 두 구현이 갈라진다.

| 위치 | 함수 |
|---|---|
| `Behaviortree/BT_Content/AngleUtil.h` | `BTAngle::WrapDeltaDeg`, `BTAngle::SignedDeltaDeg` |
| `Release/tools/log_analysis/angles.py` | `wrap_angle_deg`, `signed_angle_delta_deg` (+ `*_rad`) |

```cpp
// Behaviortree/BT_Content/AngleUtil.h
inline float WrapDeltaDeg(float delta)
{
    if (!std::isfinite(delta)) { return delta; }   // NaN/inf 는 그대로 통과
    float folded = std::fmod(delta + 180.0f, 360.0f);
    if (folded < 0.0f) { folded += 360.0f; }
    return folded - 180.0f;
}
```

`PredictManeuver::tick()` 은 이 함수를 통해서만 차이를 만든다.

```cpp
const float normalizedDelta = BTAngle::SignedDeltaDeg(currentRecordedYaw, previousYaw);
```

## 3. 각도 단위와 반환 범위

- 이 저장소의 각도는 **전부 degree** 다 (`FighterSim.py:174-176`, `GeoMathUtil` 의 ATA/AA,
  `CPPBlackBoard` 의 `EulerAngle` / `*_Degree`). radian 로그는 없다.
- 반환 범위는 **`[-180, 180)`** 이다. Python 쪽 radian 대응 함수는 `[-pi, pi)`.
- **±180 경계 정책**: 반열린 구간이므로 정확히 180도 차이는 **항상 -180 으로 접는다**.
  `signed_angle_delta_deg(180, 0) == signed_angle_delta_deg(0, 180) == -180.0`.
  크기만 필요하면 `abs()` 를 쓰는 편이 안전하다.
- 입력은 0~360 으로 정규화되어 있을 필요가 없다. 음수, 360 초과, 여러 바퀴 돈 값 모두 허용한다.
- NaN/inf 는 보정하지 않고 그대로 돌려준다. 0 으로 바꾸면 이상 프레임을 정상과 구분할 수 없다.
  `avgDelta` 가 NaN 이면 두 비교가 모두 거짓이 되어 `PredictedTurnDirection` 은 `STRAIGHT` 로 남는다.
- 초기 프레임: 히스토리(5개)가 차기 전에는 판정하지 않는다. 차이 개수는 항상 표본보다 1 적다.
- `avgDelta` 는 **이미 보정된 차이들의 산술 평균**이다. circular mean 을 쓰지 않는다 —
  차이는 이미 작은 부호값이라 circular mean 은 오히려 방향 정보를 뭉갠다.
  각도 '자체'를 평균낼 때만 `circular_mean_deg` 를 쓴다.

### 단위 테스트

```powershell
# C++ (MSVC). 24개 검사.
cd <STIL_TOPGUN>\Behaviortree
.\tests\build_and_run_predict_angle.ps1

# Python. 같은 케이스 표 + 로더/집계까지 93개 검사.
cd <STIL_TOPGUN>\Release
python tests\tools\test_predict_maneuver.py
```

두 테스트는 **같은 케이스 표**를 검증한다(`DELTA_CASES`).

| Current | Previous | 기대값 |
|---:|---:|---:|
| 10 | 5 | +5 |
| 5 | 10 | -5 |
| -179 | 179 | +2 |
| 179 | -179 | -2 |
| 1 | 359 | +2 |
| 359 | 1 | -2 |
| 180 | 0 | -180 |
| 0 | 180 | -180 |
| 360 | 0 | 0 |
| 0 | 360 | 0 |

C++ 테스트는 여기에 더해 `PredictManeuver` 노드를 실제로 tick 해서, ±180 경계를 지나는
+2도/프레임 좌선회가 `LEFT` 로 판정되는지 확인한다(보정이 없으면 `RIGHT` 가 된다).
즉 공통 함수가 판정 경로에 실제로 쓰이는지를 동작으로 검증한다.

---

## 4. 수정 전/후 실험 실행 방법

`avgDelta` 는 **환경변수 `PM_CSV_LOG` 를 설정한 실행에서만** CSV 로 남는다.
설정하지 않으면 로깅 경로가 통째로 비활성이라 시뮬레이션에 영향이 없다.

| 환경변수 | 의미 |
|---|---|
| `PM_CSV_LOG` | 출력 CSV 경로. 상위 디렉터리는 자동 생성된다. |
| `PM_CSV_RUNTYPE` | `runType` 컬럼 값. `before` / `after` 로 그룹을 나눈다. |

CSV 컬럼(단위: degree / meter / second):

```text
runType,episode,frame,time,prevAngle,currAngle,rawDelta,normalizedDelta,avgDelta,
predictedTurn,bfmMode,scissorsEntered,distance_m,ownAta_deg,targetAa_deg,angleOff_deg,enemyInSight
```

- `episode` 는 `RunningTime` 이 되감기는 지점을 경계로 로거가 파생시킨다(BT 에는 경기 ID 가 없다).
- `scissorsEntered` 는 **전환 프레임만 1**이다. 체류 프레임은 0.
- `bfmMode` 는 한 프레임 지연 flush 로 기록된다. `Rule.xml` 상 BFM 모드가
  `PredictManeuver` 보다 뒤에서 정해지기 때문이다.
- WEZ 컬럼은 없다. WEZ 판정은 Python 환경(`update_damage`)이 하며 BT 블랙보드에 없다.
  분석 도구가 `distance_m` / `ownAta_deg` 로 파생시킨다.

### 수정 전 로그

`Behaviortree` 를 wrap 보정이 없던 시점으로 체크아웃해 DLL 을 만든다.

```powershell
cd <STIL_TOPGUN>\Behaviortree
git stash                       # 또는 해당 커밋 체크아웃
.\tools\build_bt.ps1 -HostRoot C:\AIP_LIB\AIP_DCS -ReleaseDir C:\AIP_LIB\DogFightEnv\Release -Deploy

cd C:\AIP_LIB\DogFightEnv\Release
$env:PM_CSV_RUNTYPE = "before"
$env:PM_CSV_LOG     = "logs/predict/before/run01.csv"
python run_local_dogfight.py --ownship-backend rl `
    --ownship-bundle-dir artifacts/models/stil/sac_mlp_obs8_iter400 `
    --observation-module student.my_observation `
    --target-backend bt --target-bt-dll AIP_STIL.dll `
    --save-log
```

### 수정 후 로그

```powershell
cd <STIL_TOPGUN>\Behaviortree
git stash pop
.\tools\build_bt.ps1 -HostRoot C:\AIP_LIB\AIP_DCS -ReleaseDir C:\AIP_LIB\DogFightEnv\Release -Deploy

cd C:\AIP_LIB\DogFightEnv\Release
$env:PM_CSV_RUNTYPE = "after"
$env:PM_CSV_LOG     = "logs/predict/after/run01.csv"
python run_local_dogfight.py ...    # 나머지 인자를 한 글자도 바꾸지 말 것
```

### 필요한 경기 수

그룹당 **최소 20경기**를 권장한다(`--min-matches` 로 조정). `avgDelta` 이상값은
±180 경계를 지날 때만 생기므로 짧은 교전 몇 판으로는 나타나지 않을 수 있다.
파일을 경기마다 나눠도 되고(`run01.csv`, `run02.csv` …) 디렉터리를 통째로 넘기면 된다.

### 같은 조건인지 확인할 체크리스트

도구가 자동으로 알 수 없는 항목이다. 두 실행에서 **반드시 동일**해야 한다.

- [ ] seed
- [ ] 상대 정책 또는 상대 bundle (`--target-backend` / `--target-bt-dll` / `--target-bundle-dir`)
- [ ] aircraft 설정 (`Release/aircraft`, `engine`)
- [ ] scenario (`initial_scenario`: `altitude_m` / `distance_m` / heading)
- [ ] episode 제한시간 (`--max-engage-time` / `--episode-step-limit`)
- [ ] observation 설정 (`--observation-mode` / `--observation-module`)
- [ ] reward 설정 (`reward_module` / `MY_REWARD_CONFIG`)
- [ ] PredictManeuver 설정 (`historySize=5`, `TURN_THRESHOLD_DEG=1.5`)
- [ ] BFM 상태 전환 설정 (`Rule.xml`, `SetBFMMode_*` 임계값)
- [ ] 경기 수
- [ ] checkpoint / bundle 디렉터리
- [ ] 코드 commit hash (`Release` / `Behaviortree` 각각)

**wrap 수정 외에 다른 것을 함께 바꾸지 말 것.** 바꿨다면 그 사실을 보고서에 적어야 한다.

---

## 5. avgDelta 이상값 분석

```powershell
python tools\analyze_predict_maneuver.py compare `
    --before logs\predict\before `
    --after  logs\predict\after `
    --output analysis\predict_maneuver_comparison

# 임계값 조정 (하드코딩 아님)
python tools\analyze_predict_maneuver.py compare `
    --before logs\predict\before --after logs\predict\after `
    --outlier-threshold-deg 170 --spike-threshold-deg 90 `
    --outlier-thresholds-deg 170 175 179 `
    --min-matches 20 `
    --output analysis\predict_maneuver_comparison
```

산출물:

```text
analysis/predict_maneuver_comparison/
  predict_maneuver_summary.json     기계 판독용. wraparound_verdict 가 여기 있다.
  avg_delta_statistics.csv          분포 + wrap 보정 증거
  avg_delta_outliers.csv            이상값 한 건씩. 전후 BFM/SCISSORS/ATA/AA/WEZ 포함
  scissors_entry_statistics.csv     SCISSORS 진입/체류 비교
  episode_comparison.csv            경기별 한 행
  representative_events.csv         보고서에 인용할 대표 이벤트
  predict_maneuver_report.md        사람이 읽는 보고서
```

집계하는 값: 최대/최소/평균/중앙값/표준편차/95·99 분위수/절댓값 최대,
`|avgDelta| >= 170/175/179` 발생 횟수, 프레임 간 급변 횟수, 경기별 이상값 수,
이상값 직전·직후 BFM 모드와 SCISSORS 상태, 이상값 시점의 Own ATA / Target AA / 파생 WEZ.

급변은 **같은 경기 안에서만** 센다. 경기 경계를 넘는 차이는 세지 않는다.

### 판정 기준

| 판정 | 조건 |
|---|---|
| `PASS` | after 로그에서 `\|normalizedDelta\| > 180` 인 프레임이 0, `normalizedDelta` 가 전부 `wrap(rawDelta)` 와 일치, `\|avgDelta\| >= 임계값` 인 프레임이 0 |
| `PARTIAL` | wrap 경계 급등은 사라졌으나 `\|avgDelta\|` 이상값이 남아 있다 (원인이 wrap 이 아니다) |
| `FAIL` | after 로그에도 보정되지 않은 값이 남아 있다 |
| `INSUFFICIENT_DATA` | 비교할 로그나 필수 컬럼이 없다 |

판정은 `rawDelta` 와 `normalizedDelta` 를 **함께 기록**한다는 점을 이용한다.
`normalizedDelta == wrap_angle_deg(rawDelta)` 가 모든 프레임에서 성립하면
보정이 실제로 적용된 것이고, `rawDelta` 가 ±180 을 넘은 프레임 수가
"수정이 없었다면 급등했을 프레임 수"다.

---

## 6. SCISSORS 진입 빈도 비교

같은 명령이 `scissors_entry_statistics.csv` 를 함께 만든다.

**진입은 상태 행 수가 아니라 transition 으로 센다.** 비-SCISSORS → SCISSORS 로 바뀐
프레임만 1회다. 5프레임 체류해도 진입은 1회다.

- 첫 프레임부터 SCISSORS 인 구간은 직전 상태를 모르므로 진입에서 **제외**한다.
- 경기 끝까지 SCISSORS 였던 구간은 종료 시각을 알 수 없어 체류시간을 `null` 로 두고
  `open_segment_count` 로 따로 센다. 0 으로 채우지 않는다.
- 경기당 평균 체류는 진입이 없던 경기도 분모에 넣는다(진입한 경기만 평균내면 과대평가된다).
- timestamp 간격이 일정하지 않아도 체류시간은 `end - start` 라 영향이 없다.

집계 항목: 전체 경기 수, 진입 경기 수/비율, 총 진입 수, 경기당 평균·중앙값,
총·경기당·1회당 체류시간, 최장 연속 체류, 재진입 수, 이상값 직후 진입 수,
진입 직전 BFM 모드 분포, 종료 후 전환 모드 분포. 비교는 절대 차이 / percentage point /
상대 변화율 / 표본 수를 함께 낸다.

> **해석 주의.** SCISSORS 진입 빈도가 변했다는 것만으로 성능이 좋아졌다고 볼 수 없다.
> PredictManeuver CSV 에는 경기 결과가 없다. 승률·패배율·타임아웃 비율·WEZ 상태와
> 함께 읽어야 한다.
>
> ```powershell
> python tools\count_end_conditions.py --logdir artifacts\logs\stil\<tag>
> ```

---

## 7. DogFight Log Playback 웹 뷰어 실행

> 벤더 드롭(2026-06-16)에는 `DogFightEnv/tools/dogfight_dashboard` 패키지가 **없다**.
> 그래서 `tools/web_log_viewer.py` 와 `tools/training_dashboard/server.py` 는 둘 다
> `ImportError` 로 실행되지 않는다. 팀의 `tools/dashboard.py` 가 그 자리를 대신한다
> (벤더 패키지가 있으면 그쪽을 그대로 호출하므로 기존 동작은 유지된다).

```powershell
# 대표 경기 복기만
python tools\dashboard.py --playback-dir analysis\playback_cases --port 7860

# 학습 지표까지 함께
python tools\dashboard.py --logdir artifacts\logs\stil `
    --playback-dir analysis\playback_cases --port 7860
```

브라우저에서 `http://localhost:7860/` → 상단 **Replay** 탭.

- 의존성: 없음(표준 라이브러리만 쓴다). 기본 포트 7860.
- 입력 형식: `export_playback_cases.py` 가 만든 `manifest.json` + `case_*/playback.json`.
  Tacview CSV 를 직접 넣지는 못한다 — ATA/AA/WEZ 가 CSV 에 없어서 변환 단계가 필요하다.
  그 변환이 `export_playback_cases.py` 다.
- 기능: 재생 / 일시정지 / 배속(0.25~8×) / 슬라이더 시간 이동 / 차트 클릭 이동 /
  hover 상세 / `playback.json`·`trajectory.csv` 내려받기.
- `case_id` 는 manifest 에 있는 것만 열린다. 경로 탈출 요청은 404 다.

---

## 8. 대표 패배 경기 만들고 여는 법

```powershell
python tools\export_playback_cases.py `
    --logdir artifacts\logs `
    --output analysis\playback_cases `
    --handoff analysis\rl_trajectory_handoff

# PredictManeuver 로그가 있으면 유형 A(avgDelta 이상값) / B(SCISSORS) 도 선정된다
python tools\export_playback_cases.py --logdir artifacts\logs `
    --predict-log logs\predict\after `
    --predict-match-map analysis\pm_match_map.json `
    --output analysis\playback_cases
```

유형:

| 코드 | 내용 | 필요한 로그 |
|---|---|---|
| `A_AVG_DELTA_OUTLIER` | `avgDelta` 이상값이 남은 경기 | PredictManeuver CSV |
| `B_SCISSORS` | SCISSORS 진입/체류가 많았던 경기 | PredictManeuver CSV |
| `C_WEZ_OR_DEFENSE_FAILURE` | 표적이 나를 WEZ 안에 둔 채 끝난 패배 | 현재 로그로 가능 |
| `D_LOW_ALTITUDE_CRASH` | 최소 고도 위반으로 끝난 패배 | 현재 로그로 가능 |
| `E_COLLISION` | 최근접 거리가 임계값 이하 | 현재 로그로 가능 |
| `F_TIMEOUT_DRAW` | 결착 없이 종료 | 현재 로그로 가능 |
| `G_HABFM_STUCK` | HABFM 고착 | PredictManeuver CSV |
| `H_ENERGY_REVERSAL` | 비에너지 우위 역전 후 패배 | 현재 로그로 가능 |

`--predict-match-map` 은 PM 경기 ID(`runType/epNNN`)와 RL 경기 ID(`run/iterNNNNNN_epNN`)를
잇는 JSON 이다. 두 체계가 다르므로 이 매핑 없이는 **자동으로 이어붙이지 않는다**
(없는 대응을 지어내지 않기 위해서다).

```json
{ "after/ep000": "sac_mlp_obs8_iter400/iter000000_ep00" }
```

경기 결과 판정은 **환경이 기록한 `outcome` 이 최우선**이다.
`outcome="draw", end_condition="target altitude below min"` 을 승리로 읽으면 안 된다 —
환경은 그것을 draw 로 판정한다.

---

## 9. Own ATA / Target AA / WEZ 해석

이 세 값은 **어떤 로그에도 없다.** Tacview CSV 에 있는 것은
`Time, Longitude, Latitude, Altitude, Roll, Pitch, Yaw, Health` 뿐이다.
두 기체의 위치·자세에서 다시 계산한 파생값이며, 산출물에서 `derived_` 접두사가 붙는다.
계산식은 호스트의 `GeoMathUtil.GeometryInfo` / `single_agent_env.update_damage` 와 같다.

| 값 | 규약 |
|---|---|
| `derived_own_ata_deg` | 내 기수와 표적 LOS 사이의 각. **부호 있음**, 0 = 정조준. 작을수록 유리. |
| `derived_target_aa_deg` | 표적 기준 aspect angle. **0 = 내가 표적의 6시**, 180 = 표적의 정면. |
| `derived_own_in_wez` | 내가 표적을 WEZ 안에 두었는가. `min<=거리<=max` **그리고** `angle_deg/2 >= \|ATA\|`. |
| `derived_target_in_wez` | 표적이 나를 WEZ 안에 두었는가. |

주의할 점 두 가지.

1. **WEZ 각도는 `angle_deg` 가 아니라 `angle_deg/2` 다.** 기본 `angle_deg: 2.0` 이므로
   실제 피해 원뿔은 **1도**다. `update_damage` 의 비교식과 비트 단위로 같게 두었고
   각도 epsilon 을 넣지 않았다.
2. **BT 의 `MyAspectAngle_Degree` 는 반대 규약이다.** `AspectAngleUpdate` 는 부호 없는
   0~180 을 쓰고 0 이 "내가 표적의 기수 앞"이다. `GeoMathUtil` 의 AA 와 섞지 말 것.
   PredictManeuver CSV 의 `targetAa_deg` 는 BT 규약, playback 의 `derived_target_aa_deg` 는
   GeoMathUtil 규약이다.

`derived_ata_sign_degenerate` 가 `true` 인 프레임은 `GeoMathUtil` 의 부호 규칙이 붕괴하는
구간이다(플랫폼 결함 1: `np.sign(0.0) == 0.0`). 그 구간의 ATA **부호**는 믿으면 안 된다.
뷰어 HUD 에도 경고로 표시된다.

### Tacview `Time` 컬럼은 실제 경과 시간이 아니다 (2026-08-04 확인)

호스트 로깅의 결함이다. 팀 코드와 무관하다.

```python
single_agent_env.py:405   self._append_logs()          # env step 1회마다 1행
single_agent_env.py:289   for _ in range(step_ratio)   # 1 env step = step_ratio 내부 스텝
single_agent_env.py:996   time_value = self._delta_t * step      # _delta_t = 1/sim_hz = 1/60
```

한 행의 실제 경과는 `step_ratio / 60` 초인데 `Time` 은 `1 / 60` 초만 증가한다.
즉 **`Time` 은 실제보다 `step_ratio`(=6) 배 느리게 흐른다.**

물리적 확인 — 7000 m → 300 m 강하를 `Time` 기준으로 읽으면:

| 경기 | Time 기준 길이 | 평균 강하율 (Time) | ×6 보정 후 |
|---|---:|---:|---:|
| case_001 | 6.93 s | 969 m/s | 162 m/s |
| case_002 | 6.33 s | 1065 m/s | 177 m/s |
| case_004 | 8.10 s | 829 m/s | 138 m/s |

`Time` 을 그대로 쓰면 F-16 이 낼 수 없는 값이 나오고, 보정하면 타당해진다.
속도도 마찬가지로 1800 m/s → 300 m/s 가 된다.

**영향 범위**

| 값 | 영향 |
|---|---|
| 속도, 강하율, specific energy | `step_ratio` 배 부풀려진다 → 보정 필요 |
| ATA / AA / 거리 / WEZ | **영향 없음.** 시간과 무관한 순간값이다. |
| 경기 길이 | `Time` 기준 값은 실제의 1/6 |

**대응**

- `log_analysis/metrics.py` 의 `speed_series` / `descent_rate_series` 에 `time_scale`
  인자를 추가했다. **기본값 1.0 은 기존 호출부 동작을 그대로 유지**한다.
- `export_playback_cases.py --step-ratio` (기본 6) 가 보정을 적용하고,
  `playback.json` 의 각 프레임에 `derived_real_time_sec` 를 함께 담는다.
  원본 `time_sec` 은 손대지 않는다.
- `analyze_loss_patterns.py --step-ratio` (기본 6) 도 같은 보정을 쓴다.
  이 값을 바꾸면 `--descent-rate` / `--speed-loss` 임계값의 의미도 함께 바뀐다.
- 뷰어는 실제 시각을 먼저 보여주고 원본 `Time` 을 괄호로 덧붙이며, 재생 속도도
  실제 시각에 맞춘다.

`step_ratio` 를 6 이 아닌 값으로 학습했다면 `--step-ratio` 를 그 값으로 바꿔야 한다.

파생 결과가 맞는지에 대한 교차 확인: `case_001` 에서 `WEZ_ENTER_TARGET` 시각과
원본 체력이 처음 깎인 `OWN_DAMAGE` 시각이 같은 프레임에서 일치한다. 파생 WEZ 판정이
환경의 `update_damage` 와 같은 시점에 켜졌다는 뜻이다.

---

## 10. RL 담당자에게 궤적 공유

```powershell
python tools\export_playback_cases.py --logdir artifacts\logs `
    --output analysis\playback_cases --handoff analysis\rl_trajectory_handoff
```

```text
analysis/rl_trajectory_handoff/
  README.md                    목적·대표 경기·뷰어 실행법·해석 방법
  trajectory_manifest.json     케이스 목록
  representative_episodes.csv  경기 한 줄 요약
  event_timeline.csv           모든 케이스의 이벤트
  playback_cases/              뷰어가 읽는 데이터
```

원본 Tacview CSV 는 복사하지 않는다. 각 케이스의 `source_files` 에 절대경로가 들어 있다.

> `replay_index.jsonl` / `summary.json` 에는 **seed 필드가 없다.** 정확한 재현이 필요하면
> 실험 YAML 과 commit hash 를 함께 고정해야 한다. 없는 값을 지어내지 않았다.

---

## 11. observation 설정의 source of truth

| 상황 | 기준 |
|---|---|
| 새 학습을 시작할 때 | **실험 YAML** (`experiments/*.yaml`) |
| 기존 checkpoint / bundle 을 실행·제출할 때 | **그 bundle 의 `metadata.json` + 모델 입력 shape** |

결정 우선순위:

1. `observation_module` 이 비어 있지 않으면 훅의 `OBSERVATION_MODE` 가 선언된
   `observation_mode` 를 **이긴다**. `train_rllib.py` / `run_local_dogfight.py` /
   `my_submission.py` 세 곳 모두 같은 규칙이다.
   → YAML 이 `custom` 인데 `metadata.json` 이 `stil8` 인 것은 **정상**이다.
2. CLI 인자는 YAML 값을 이긴다 (`run_experiment.py` 가 YAML 을 CLI 로 변환).
3. bundle 을 로드할 때 모델 입력 차원을 정하는 것은
   `metadata.json` 의 `algorithm_config.env_config.observation_size` 다
   (`src/dogfight/ai/inference_env.py:21-22`).
4. 그 키가 없으면 `observation_size(mode)` 가 **12** 를 돌려주어 조용히 틀린 크기가 된다.

**다섯 곳이 반드시 같아야 하는 값은 `observation_module` 이다.** `observation_mode` 를
서로 비교하면 정상 번들이 깨진 것으로 잘못 보인다.

## 12. 다섯 개 설정 위치

| # | 위치 | 무엇이 들어 있나 |
|---|---|---|
| A | `experiments/*.yaml` | `env.observation_mode`, `env.observation_module` |
| B | `train_rllib.py` | CLI 기본값 + 훅 모드로 덮어쓰는 규칙 |
| C | `<bundle>/metadata.json` | `metadata.obs_mode`, `metadata.observation_module`, `env_config.observation_size`, `env_config.observation_summary` |
| D | `run_local_dogfight.py` | CLI 기본값 (**metadata 를 읽지 않는다**) |
| E | `student/my_submission.py` | 모듈 상수 `OBSERVATION_MODE` / `OBSERVATION_MODULE` |

여기에 두 곳을 더 본다.

| # | 위치 | 무엇이 들어 있나 |
|---|---|---|
| F | `student/my_observation.py` | `OBSERVATION_MODE`, `OBSERVATION_SIZE` — 실행 시 실제로 나올 크기 |
| G | `<bundle>/policy_weights.pkl.gz` | `pi_encoder.net.mlp.0.weight` 의 입력 차원 = 모델이 실제로 받는 크기 |

`my_train.py` 는 B 로 들어가는 또 하나의 진입점이다. observation 을 바꾸면 여기도 함께 고쳐야 한다.

## 13. consistency checker 실행

```powershell
python tools\check_observation_consistency.py `
    --config experiments\stil_sac_mlp_obs8_iter400.yaml `
    --metadata artifacts\models\stil\sac_mlp_obs8_iter400\metadata.json `
    --train-script train_rllib.py `
    --local-runner run_local_dogfight.py `
    --submission student\my_submission.py `
    --observation-module-file student\my_observation.py `
    --bundle-weights artifacts\models\stil\sac_mlp_obs8_iter400\policy_weights.pkl.gz `
    --output analysis\observation_consistency
```

옵션:

- `--source-of-truth auto|yaml|metadata` — `auto` 는 `--metadata` 를 주면 metadata.
- `--import-check` — observation 모듈을 실제로 import 해 크기를 확인한다(호스트 트리 필요).
- `--strict` — `risk` 항목도 실패로 승격한다. CI 용.
- `--json` — 표준출력을 JSON 으로.

종료 코드:

| 코드 | 의미 |
|---|---|
| 0 | 필수 설정이 모두 일치 |
| 1 | 불일치 (`mismatch`) |
| 2 | 필수 파일 또는 필드 누락 (`missing`) |
| 3 | 파일 파싱 실패 |

Python 파일은 **AST** 로 읽는다(문자열 검색이 아니다). 실행 중 CLI 인자로 바뀌는 값은
정적 분석으로 알 수 없으므로 `UNVERIFIED` / `risk` 로 표시하고 결정 규칙을 함께 보고한다.

가중치 파일은 제한 `Unpickler` 로 **shape 만** 읽는다. torch 도 numpy 도 필요 없고
임의 코드가 실행되지 않는다.

상태 값: `MATCH` / `MISMATCH` / `MISSING` / `HARDCODED` / `OVERRIDDEN` / `UNVERIFIED`.

## 14. 불일치가 나왔을 때

한 값을 다른 값으로 덮어쓰기 전에 순서대로 확인한다.

1. 학습 YAML 이 의도한 observation 설정
2. 해당 checkpoint 의 metadata
3. 실제 모델 입력 크기 (`--bundle-weights`)
4. observation 모듈의 `OBSERVATION_SIZE` / 실제 반환 shape
5. 로컬 실행 설정
6. 제출 설정
7. 기존 bundle 계약

지켜야 할 것:

- observation 크기와 모델 입력 크기가 다르면 **강제로 실행하지 말 것.**
- 크기를 바꿨으면 기존 checkpoint 를 재사용할 수 있다고 가정하지 말 것.
- 8차원 → 다른 차원으로 바뀌었으면 **새 output tag** 를 쓸 것.
- 기존 실험 결과 디렉터리를 덮어쓰지 말 것.
- 로컬 실행과 제출 실행이 같은 모듈을 import 하도록 할 것.
- 조용히 다른 observation 으로 fallback 하는 동작을 없애거나 명확히 경고할 것.
- 실행 시작 시 최종 설정을 로그로 남길 것.

`my_submission.py` 는 실행 직전에 계약을 검증하고 아래를 출력한다.

```text
[ObservationConfig]
mode=stil8
module=student.my_observation
size=8
source=metadata
checkpoint=...\artifacts\models\stil\sac_mlp_obs8_iter400
[ObservationConfig] bundle 검증 통과 (size=8, 출처 algorithm_config.env_config.observation_size)
```

어긋나면 `ObservationContractError` 로 **즉시 멈춘다.** padding / truncation / 조용한
fallback 을 하지 않는다 — 그러면 정책이 돌긴 하면서 성능만 무너지는 silent failure 가 된다.

```text
Observation configuration mismatch:
- bundle metadata observation_size : 8
- loaded module observation_size   : 10

bundle: ...\metadata.json

This checkpoint cannot be reused with the selected observation module.
Use the matching module or train a new checkpoint with a new output tag.
```

`run_local_dogfight.py` 는 벤더 호스트 파일이라 팀 저장소에 없다. 대신 교전 전에
checker 를 preflight 로 돌린다.

```powershell
python tools\check_observation_consistency.py --config <yaml> `
    --metadata <bundle>\metadata.json --bundle-weights <bundle>\policy_weights.pkl.gz
if ($LASTEXITCODE -ne 0) { throw "observation 설정 불일치" }
python run_local_dogfight.py --observation-module student.my_observation ...
```

> `run_local_dogfight.py` 는 `--observation-module` 을 주지 않으면 기본값
> `tactical16` 으로 조용히 실행된다. 8차원 bundle 을 그렇게 돌리면 로드 자체가 실패하거나
> 엉뚱한 입력을 먹는다. **인자를 항상 명시하라.** checker 가 이것을 `risk` 로 보고한다.

## 15. checkpoint 재사용 가능 여부 판단

checker 의 `checkpoint_reuse` 절이 판정한다.

| 판정 | 의미 |
|---|---|
| `COMPATIBLE` | 모델 입력 차원 == 실행 시 observation 크기. 그대로 재사용 가능. |
| `INCOMPATIBLE` | 다르다. 이 bundle 로는 그 모듈을 쓸 수 없다. 새 tag 로 재학습. |
| `UNVERIFIED` | 가중치를 읽지 못했다. 재사용 가능하다고 단정하지 않는다. |

비교 기준은 **metadata 의 크기가 아니라 observation 모듈이 실제로 돌려줄 크기**다.
metadata 가 10 이라 적혀 있어도 모듈이 8을 돌려주면 그 조합은 못 쓴다.

`observation_size` 키가 metadata 에 없는 옛 bundle 은 값을 지어내지 않고 `missing` 으로
보고한다. 값 자체는 `observation_summary.size` 에 이미 들어 있으므로 아래로 채운다.

```powershell
python student\tools\fix_bundle_obs_size.py artifacts\models\stil\<tag>          # dry-run
python student\tools\fix_bundle_obs_size.py artifacts\models\stil\<tag> --apply  # 반영
```

원본은 `metadata.json.bak` 으로 1회 백업된다. 여러 번 실행해도 안전하다.

## 16. 새 observation 으로 학습할 때

`OBSERVATION_SIZE`, 모듈 경로, **feature 순서** 중 하나라도 바꾸면 기존 checkpoint 와
bundle 은 전부 무효다. 다음 순서로 진행한다.

1. `student/my_observation.py` 를 고치고 `OBSERVATION_MODE` 를 새 이름으로 바꾼다
   (예: `stil8` → `stil10`). 이름을 그대로 두면 metadata 만으로는 구분할 수 없다.
2. 실험 YAML 을 **새 파일로 복사**하고 `output.tag` 를 새로 준다. 템플릿을 그 자리에서
   고치지 않는다 — YAML 이 실험 기록이다.
3. `student/my_train.py` 의 `TRAINING_CONFIG` / `ENV_CONFIG` 도 같은 값으로 맞춘다.
4. `my_submission.py` 의 `OBSERVATION_MODULE` 과 `BUNDLE_DIR` 을 새 tag 로 바꾼다.
5. 학습 후 새 bundle 에 대해 checker 를 돌리고, 필요하면 `fix_bundle_obs_size.py` 를 적용한다.
6. 기존 tag 디렉터리는 **덮어쓰지 않는다.**

```powershell
copy experiments\stil_sac_mlp_obs8_iter400.yaml experiments\stil_sac_mlp_obs10_v1.yaml
# 파일 안에서 output.tag 를 obs10_v1 로 바꾼다 (runtime.iterations 도 YAML 에서만 바꿀 수 있다)
python scripts\run_experiment.py experiments\stil_sac_mlp_obs10_v1.yaml --dry-run
python scripts\run_experiment.py experiments\stil_sac_mlp_obs10_v1.yaml
python tools\check_observation_consistency.py --config experiments\stil_sac_mlp_obs10_v1.yaml `
    --metadata artifacts\models\stil\obs10_v1\metadata.json `
    --bundle-weights artifacts\models\stil\obs10_v1\policy_weights.pkl.gz
```

---

## 17. 테스트 한 번에 돌리기

```powershell
cd <STIL_TOPGUN>\Release
python tests\tools\test_log_analysis.py              # 46건
python tests\tools\test_predict_maneuver.py          # 93건
python tests\tools\test_observation_consistency.py   # 61건
python student\tests\check_student_contracts.py      # 193건

cd <STIL_TOPGUN>\Behaviortree
.\tests\build_and_run_predict_angle.ps1              # 24건
```
