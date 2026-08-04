<#
.SYNOPSIS
    BT 상대 교전 배치(run_local_dogfight 다판)를 한 번에 분석한다.

.DESCRIPTION
    `--target-backend bt` 로 모은 산출물 3종을 각각의 분석 도구에 연결한다.

        <BatchDir>\bt_stdout.log   -> tools\extract_bfm_log.py        (BFM 전환/차단 집계)
        <BatchDir>\predict\*.csv   -> tools\analyze_predict_maneuver.py (avgDelta wrap 검증)
        artifacts\logs\*           -> tools\analyze_loss_last5s.py     (패배 5초 패턴)

    셋 다 읽기 전용이다. 원본 로그·checkpoint·bundle 은 건드리지 않는다.

.NOTES
    시간 축 주의: run_local_dogfight 로그는 step_ratio=1(60 Hz)이라 Time 컬럼이 실제
    시간이다. 학습 로그(step_ratio=6)와 섞어 쓰면 안 되므로 -StepRatio 기본값이 1이다.

    -Since 를 주지 않으면 artifacts\logs 전체가 대상이 된다. 이전 배치가 같은 곳에
    쌓여 있으면 표본이 섞이므로, 배치 시작 시각을 넘겨 주는 편이 안전하다.

.EXAMPLE
    cd C:\AIP_LIB\DogFightEnv\Release
    powershell -File tools\analyze_bt_batch.ps1 `
        -BatchDir artifacts\logs_bt30 -Output analysis\bt30 -Since "2026-08-04 21:53"
#>
[CmdletBinding()]
param(
    # 교전 배치 산출물 디렉터리 (bt_stdout.log 과 predict\ 를 포함)
    [Parameter(Mandatory = $true)][string]$BatchDir,
    # 분석 결과를 쓸 디렉터리
    [Parameter(Mandatory = $true)][string]$Output,
    # run_local_dogfight 의 --save-log 산출물 위치
    [string]$LogDir = "artifacts\logs",
    # 이 시각 이후 로그만 대상으로 한다 ("YYYY-MM-DD HH:MM")
    [string]$Since = "",
    # 로그 1행이 몇 sim step 인지. run_local_dogfight 은 1, 학습 로그는 6.
    [int]$StepRatio = 1,
    # predict CSV 비교에 요구할 최소 경기 수
    [int]$MinMatches = 20
)

$ErrorActionPreference = "Stop"

function Resolve-Under([string]$path) {
    if ([System.IO.Path]::IsPathRooted($path)) { return $path }
    return (Join-Path (Get-Location) $path)
}

$batch   = Resolve-Under $BatchDir
$outDir  = Resolve-Under $Output
$predict = Join-Path $batch "predict"
$stdout  = Join-Path $batch "bt_stdout.log"

if (-not (Test-Path $batch)) { throw "배치 디렉터리가 없다: $batch" }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# --- 표본 수를 먼저 센다. 부족하면 분석 전에 알린다. -------------------------
$csvCount = 0
if (Test-Path $predict) {
    $csvCount = @(Get-ChildItem -Path $predict -Filter *.csv -File).Count
}
Write-Host "== 표본 =="
Write-Host ("  predict CSV : {0} 판" -f $csvCount)
if ($csvCount -lt $MinMatches) {
    Write-Warning ("표본 부족: {0} 판 < 요구 {1} 판. 아래 결과는 참고용이다." -f $csvCount, $MinMatches)
}

$failed = @()

# --- 1) BFM 전환/차단 ---------------------------------------------------------
Write-Host "`n== 1/3 BFM 전환 집계 =="
if (Test-Path $stdout) {
    & python tools\extract_bfm_log.py --stdout $stdout --output (Join-Path $outDir "bfm")
    if ($LASTEXITCODE -ne 0) { $failed += "extract_bfm_log" }
} else {
    Write-Warning "bt_stdout.log 없음 -> BFM 집계 건너뜀 (INSUFFICIENT_DATA)"
    $failed += "extract_bfm_log(no input)"
}

# --- 2) PredictManeuver wrap 검증 --------------------------------------------
# before/after 를 같은 디렉터리로 준다: 이 배치는 wrap 보정이 이미 들어간 'after'
# 빌드 하나뿐이라, A/B 가 아니라 after 단독 무결성 검증이 목적이다.
Write-Host "`n== 2/3 PredictManeuver 각도 검증 =="
if ($csvCount -gt 0) {
    & python tools\analyze_predict_maneuver.py compare `
        --before $predict --after $predict `
        --min-matches $MinMatches --output (Join-Path $outDir "predict")
    if ($LASTEXITCODE -ne 0) { $failed += "analyze_predict_maneuver" }
} else {
    Write-Warning "predict CSV 없음 -> 각도 검증 건너뜀 (INSUFFICIENT_DATA)"
    $failed += "analyze_predict_maneuver(no input)"
}

# --- 3) 패배 5초 패턴 ---------------------------------------------------------
Write-Host "`n== 3/3 패배 5초 패턴 =="
$indexArgs = @("tools\index_local_dogfights.py", "--logdir", $LogDir, "--run", "logs")
if ($Since) { $indexArgs += @("--since", $Since) }
& python @indexArgs
if ($LASTEXITCODE -ne 0) { $failed += "index_local_dogfights" }

$lossArgs = @(
    "tools\analyze_loss_last5s.py",
    "--logdir", $LogDir, "--release-root", ".",
    "--step-ratio", $StepRatio, "--run", "logs",
    "--output", (Join-Path $outDir "loss5s")
)
if (Test-Path (Join-Path $outDir "bfm")) {
    $lossArgs += @("--bfm-events", (Join-Path $outDir "bfm"))
}
& python @lossArgs
if ($LASTEXITCODE -ne 0) { $failed += "analyze_loss_last5s" }

# --- 정리 ---------------------------------------------------------------------
Write-Host "`n== 결과 =="
Write-Host ("  출력: {0}" -f $outDir)
if ($failed.Count -gt 0) {
    Write-Warning ("실패/건너뜀: {0}" -f ($failed -join ", "))
    exit 1
}
Write-Host "  3/3 완료"
exit 0
