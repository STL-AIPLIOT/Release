<#
.SYNOPSIS
    BT 상대 교전을 N판 모으고 곧바로 분석까지 돌린다 (수집 + 분석 한 번에).

.DESCRIPTION
    보상/정책을 바꾼 뒤 같은 조건에서 재측정하려면 수집 조건이 **한 글자도 달라지면
    안 된다.** 손으로 foreach 루프를 다시 쓰면 `--max-engage-time` 하나가 어긋나도
    비교가 무의미해지므로, 프로토콜을 이 스크립트에 고정한다.

    기본값은 2026-08-04 의 30판 측정과 동일하다:
        --max-engage-time 60 / --target-backend bt / AIP_STIL.dll / --save-log
        PM_CSV_RUNTYPE=after, PM_CSV_LOG=<BatchDir>\predict\runNN.csv

    벤더 `AIP_BASE_target.dll` 은 기본값이 아니다. Release 루트의 `Rule.xml` 이
    팀 트리용이면 벤더 DLL 이 `CreateBehaviorTree` 에서 C++ 예외(0xe06d7363)로 죽는다.
    (Behaviortree/tools/README.md 참조)

.NOTES
    학습이 도는 동안 같이 돌리지 말 것. JSBSim DLL 과 CPU 를 두고 경합해
    교전 결과가 흔들린다. 학습이 끝난 뒤에 실행한다.

    이 스크립트는 **읽기 전용이 아니다** — 교전을 실제로 돌려 artifacts\logs 에 쓴다.
    기존 번들/checkpoint 는 건드리지 않는다.

.EXAMPLE
    # 학습 후 재측정 (구 30판과 같은 조건)
    cd C:\AIP_LIB\DogFightEnv\Release
    powershell -ExecutionPolicy Bypass -File tools\run_bt_batch.ps1 `
        -BundleDir artifacts\models\stil\sac_mlp_obs8_rangegate_bt_v1 `
        -BatchDir artifacts\logs_bt30_after -Output analysis\bt30_after -Count 30
#>
[CmdletBinding()]
param(
    # 평가할 정책 번들 (metadata.json + policy_weights.pkl.gz 가 있는 디렉터리)
    [Parameter(Mandatory = $true)][string]$BundleDir,
    # 교전 산출물(predict CSV, BT stdout)을 모을 곳
    [Parameter(Mandatory = $true)][string]$BatchDir,
    # 분석 결과를 쓸 곳
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$Count = 30,
    [string]$TargetDll = "AIP_STIL.dll",
    [double]$MaxEngageTime = 60,
    [string]$ObservationModule = "student.my_observation",
    # 수집을 건너뛰고 이미 모아둔 BatchDir 을 분석만 한다
    [switch]$AnalyzeOnly,
    # 관측 훅이 학습 시점과 달라도 진행한다 (기본은 중단). 의도한 경우에만 쓴다.
    [switch]$AllowHookDrift,
    # 분석 대상 로그의 하한 시각 "YYYY-MM-DD HH:mm".
    # 수집을 하면 자동으로 수집 시작 시각이 들어간다. -AnalyzeOnly 로 이미 모아둔 배치를
    # 분석할 때는 비워 두어야 한다 — "지금" 을 넣으면 대상이 0판이 된다.
    [string]$Since = ""
)

$ErrorActionPreference = "Stop"

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

if (-not (Test-Path $BundleDir)) { throw "번들이 없다: $BundleDir" }

# 관측 훅 드리프트 차단.
# 번들은 가중치만 갖고, 관측은 지금 디스크의 student/my_observation.py 를 import 해서
# 만든다. 학습 뒤에 그 파일을 고치면 옛 가중치에 새 관측이 들어가는데, 크기와 feature
# 이름이 그대로면 기존 일관성 검사는 통과한다. 즉 여기서 막지 않으면 아무도 못 잡는다.
& $python tools\check_hook_drift.py --bundle-dir $BundleDir
$driftCode = $LASTEXITCODE
if ($driftCode -eq 1 -and -not $AllowHookDrift) {
    throw "관측 훅이 학습 이후 바뀌었다. 이 상태로 측정하면 결과가 무의미하다. " +
          "훅을 되돌리거나, 의도한 것이면 -AllowHookDrift 를 주고 다시 실행하라."
}
if ($driftCode -eq 2) {
    Write-Warning "학습 시점 훅 사본이 없어 드리프트를 검사하지 못했다. 결과 해석에 주의."
}

# 재측정 시작 시각. 분석 단계에서 이 시각 이후 로그만 골라 이전 배치와 섞이지 않게 한다.
# 수집을 건너뛰면 이 시각을 만들면 안 된다(대상이 0판이 된다). 호출자가 준 값만 쓴다.
$since = $Since
if (-not $AnalyzeOnly) {
    $since = (Get-Date).ToString("yyyy-MM-dd HH:mm")
}

if (-not $AnalyzeOnly) {
    New-Item -ItemType Directory -Force -Path (Join-Path $BatchDir "predict") | Out-Null
    $stdout = Join-Path $BatchDir "bt_stdout.log"

    Write-Host ("== 교전 {0}판 수집 (상대 {1}, {2}초) ==" -f $Count, $TargetDll, $MaxEngageTime)
    Write-Host ("   기준 시각: {0}" -f $since)

    # native exe 의 stderr 는 $ErrorActionPreference="Stop" 아래에서 **종료 오류로 승격**된다.
    # run_local_dogfight 은 정상 동작 중에도 Ray FutureWarning 을 stderr 로 뿜으므로,
    # 그대로 두면 첫 판에서 루프가 죽는다(2026-08-05 실측). 성패는 $LASTEXITCODE 로만 본다.
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    for ($i = 1; $i -le $Count; $i++) {
        $env:PM_CSV_RUNTYPE = "after"
        $env:PM_CSV_LOG = Join-Path $BatchDir ("predict\run{0:d2}.csv" -f $i)

        & $python run_local_dogfight.py --ownship-backend rl `
            --ownship-bundle-dir $BundleDir `
            --observation-module $ObservationModule `
            --target-backend bt --target-bt-dll $TargetDll `
            --max-engage-time $MaxEngageTime --save-log *>> $stdout

        if ($LASTEXITCODE -ne 0) {
            Write-Warning ("{0}판째 실패 (exit {1}). {2} 확인." -f $i, $LASTEXITCODE, $stdout)
        }
        if ($i % 10 -eq 0) { Write-Host ("   {0}/{1}" -f $i, $Count) }
    }

    $ErrorActionPreference = $savedEAP
    Remove-Item Env:\PM_CSV_RUNTYPE, Env:\PM_CSV_LOG -ErrorAction SilentlyContinue

    $collected = @(Get-ChildItem -Path (Join-Path $BatchDir "predict") -Filter *.csv -File).Count
    if ($collected -eq 0) {
        throw "교전이 한 판도 수집되지 않았다. $stdout 를 확인하라."
    }
    if ($collected -lt $Count) {
        Write-Warning ("{0}/{1} 판만 수집됐다. 분석은 진행하되 표본 부족으로 읽어라." -f $collected, $Count)
    }
}

Write-Host "`n== 분석 =="
& powershell -NoProfile -ExecutionPolicy Bypass -File tools\analyze_bt_batch.ps1 `
    -BatchDir $BatchDir -Output $Output -Since $since -MinMatches $Count -StepRatio 1

# WEZ 밴드 통과 실태. 판정 표의 핵심 숫자(|ATA| 최선, 피해 조건 충족)가 여기서 나온다.
& $python tools\analyze_wez_window.py --logdir artifacts\logs --release-root . `
    --step-ratio 1 --run logs --output (Join-Path $Output "wez")

Write-Host ("`n완료: {0}" -f $Output)
Write-Host  "구 측정과 비교하려면:"
Write-Host ("  {0} tools\compare_bt_batches.py --before analysis\bt30\wez --after {1}\wez" -f $python, $Output)
