param(
    [ValidateRange(0, 16)]
    [int]$Workers = 2,
    [switch]$PreflightOnly,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$loggedRunner = Join-Path $projectRoot "run_logged.py"
$dataRoot = Join-Path $projectRoot "data"
$baselineRunRoot = Join-Path $projectRoot "runs_fives_v2_screen"
$baselineWeightDir = Join-Path $projectRoot "weight_files_fives_v2_screen"
$runRoot = Join-Path $projectRoot "runs_fives_color_controls"
$weightDir = Join-Path $projectRoot "weight_files_fives_color_controls"
$resultRoot = Join-Path $projectRoot "result_fives_color_controls"
$referenceResultRoot = Join-Path $projectRoot "reference_result_fives_carer_official_test"
$outputDir = Join-Path $projectRoot "paper_outputs_fives_color_controls"
$statusFile = Join-Path $projectRoot "fives_color_controls_status.txt"
$consoleLog = Join-Path $projectRoot "fives_color_controls_console.log"
$seeds = @(42, 123, 3407)
$models = @(
    [PSCustomObject]@{ Name = "green_channel_unet"; InputMode = "gray_rgb" },
    [PSCustomObject]@{ Name = "gray_clahe_unet"; InputMode = "gray_rgb_clahe" },
    [PSCustomObject]@{ Name = "rgb_1x1_projection_unet"; InputMode = "gray_rgb" },
    [PSCustomObject]@{ Name = "carer_rgb_only_unet"; InputMode = "gray_rgb" },
    [PSCustomObject]@{ Name = "carer_contrast_only_unet"; InputMode = "gray_rgb" }
)
$modelNames = @($models | ForEach-Object { $_.Name })

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class FivesColorControlsSleepControl {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_AWAYMODE_REQUIRED = [uint32]0x00000040
$previousCublasConfig = $env:CUBLAS_WORKSPACE_CONFIG

function Write-Status([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Add-Content -LiteralPath $statusFile -Value $line -Encoding UTF8
    Write-Output $line
}

function Invoke-Python([string[]]$arguments) {
    $runnerArguments = @($loggedRunner, "--log-file", $consoleLog, "--") + $arguments
    & $python @runnerArguments
    $pythonExitCode = $LASTEXITCODE
    if ($pythonExitCode -ne 0) {
        throw "Python command failed with exit code ${pythonExitCode}: $($arguments -join ' ')"
    }
}

foreach ($requiredPath in @(
    $python,
    $loggedRunner,
    (Join-Path $projectRoot "preflight_fives_color_controls.py"),
    (Join-Path $projectRoot "train_carer_ablation.py"),
    (Join-Path $projectRoot "test.py"),
    (Join-Path $projectRoot "summarize_fives_color_controls.py")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required experiment file not found: $requiredPath"
    }
}

foreach ($seed in $seeds) {
    $baselineRun = Join-Path $baselineRunRoot "FIVES\res_attention_unet_gn\seed_$seed"
    $splitFile = Join-Path $baselineRun "split.json"
    $initialCheckpoint = Join-Path $baselineWeightDir "FIVES_best_ResAttentionUNetGN_seed$seed.pth"
    foreach ($requiredPath in @($splitFile, $initialCheckpoint)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Required seed-$seed GN reference file not found: $requiredPath"
        }
    }
    foreach ($referenceModel in @("gray_rgb_control_30_unet", "color_detail_only_unet")) {
        $referenceDirectory = Join-Path $referenceResultRoot "FIVES\$referenceModel\seed_$seed"
        foreach ($fileName in @("metrics.json", "per_image_metrics.csv")) {
            $requiredPath = Join-Path $referenceDirectory $fileName
            if (-not (Test-Path -LiteralPath $requiredPath)) {
                throw "Bundled reference result not found: $requiredPath"
            }
        }
    }
}

try {
    [FivesColorControlsSleepControl]::SetThreadExecutionState(
        $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED
    ) | Out-Null
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

    Set-Content -LiteralPath $statusFile -Value @() -Encoding UTF8
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Add-Content -LiteralPath $consoleLog -Value "`r`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') FIVES predefined color controls =====" -Encoding UTF8
    Write-Status "STARTED: five predefined controls x three seeds; same historical splits and GN checkpoints; fixed test threshold 0.5"

    Invoke-Python @(
        "-u", (Join-Path $projectRoot "preflight_fives_color_controls.py"),
        "--project-root", $projectRoot,
        "--data-root", $dataRoot,
        "--require-cuda"
    )
    Write-Status "PREFLIGHT PASSED: GPU, 600 FIVES training pairs, three splits/checkpoints, CLAHE, and five model arms validated; test set untouched"

    if ($PreflightOnly) {
        Write-Status "PREFLIGHT-ONLY COMPLETE: no training or official-test inference was started"
        return
    }

    foreach ($seed in $seeds) {
        $baselineRun = Join-Path $baselineRunRoot "FIVES\res_attention_unet_gn\seed_$seed"
        $splitFile = Join-Path $baselineRun "split.json"
        $initialCheckpoint = Join-Path $baselineWeightDir "FIVES_best_ResAttentionUNetGN_seed$seed.pth"
        $allComplete = $true
        foreach ($modelName in $modelNames) {
            $resultPath = Join-Path $runRoot "FIVES\$modelName\seed_$seed\best_result.json"
            if (-not (Test-Path -LiteralPath $resultPath)) {
                $allComplete = $false
            }
        }
        if ($allComplete -and -not $Overwrite) {
            Write-Status "SKIPPED TRAINING seed ${seed}: all five arms already have best_result.json"
            continue
        }

        Write-Status "TRAINING seed ${seed}: green, CLAHE, generic 1x1, RGB-only residual, contrast-only residual"
        $trainArguments = @(
            "-u", (Join-Path $projectRoot "train_carer_ablation.py"),
            "--dataset", "FIVES",
            "--data-root", $dataRoot,
            "--split-file", $splitFile,
            "--initial-checkpoint", $initialCheckpoint,
            "--run-root", $runRoot,
            "--weight-dir", $weightDir,
            "--epochs", "30",
            "--warmup-epochs", "5",
            "--head-learning-rate", "0.0001",
            "--tail-learning-rate", "0.00001",
            "--train-seed", "$seed",
            "--variants"
        ) + $modelNames + @(
            "--experiment-name", "FIVES predefined color controls",
            "--workers", "$Workers",
            "--device", "cuda"
        )
        if ($Overwrite) {
            $trainArguments += "--overwrite"
        }
        Invoke-Python $trainArguments
    }

    Write-Status "TRAINING COMPLETE: all fifteen predefined validation runs are available; starting fixed official-test inference without model selection"
    foreach ($seed in $seeds) {
        foreach ($model in $models) {
            $checkpoint = Join-Path $weightDir "FIVES_best_$($model.Name)_seed$seed.pth"
            $metricsFile = Join-Path $resultRoot "FIVES\$($model.Name)\seed_$seed\metrics.json"
            if (-not (Test-Path -LiteralPath $checkpoint)) {
                throw "Trained checkpoint not found: $checkpoint"
            }
            if ((Test-Path -LiteralPath $metricsFile) -and -not $Overwrite) {
                Write-Status "SKIPPED TEST: model=$($model.Name) seed=$seed already has metrics.json"
                continue
            }
            Write-Status "TESTING: model=$($model.Name) seed=$seed threshold=0.5"
            Invoke-Python @(
                "-u", (Join-Path $projectRoot "test.py"),
                "--dataset", "FIVES",
                "--model", $model.Name,
                "--checkpoint", $checkpoint,
                "--data-root", $dataRoot,
                "--result-root", $resultRoot,
                "--image-size", "512",
                "--input-mode", $model.InputMode,
                "--threshold", "0.5",
                "--fov", "derived",
                "--seed", "$seed",
                "--workers", "$Workers",
                "--device", "cuda"
            )
        }
    }

    Invoke-Python @(
        "-u", (Join-Path $projectRoot "summarize_fives_color_controls.py"),
        "--run-root", $runRoot,
        "--result-root", $resultRoot,
        "--reference-result-root", $referenceResultRoot,
        "--output-dir", $outputDir
    )

    Write-Status "EXPERIMENT COMPLETE: five predefined arms x three seeds trained and tested; summaries include frozen grayscale control and CARER-Core references"
    Write-Status "COPY BACK: runs_fives_color_controls, result_fives_color_controls, paper_outputs_fives_color_controls, fives_color_controls_status.txt, fives_color_controls_console.log; large weight_files_fives_color_controls may remain on the 4070 PC"
}
catch {
    Write-Status "EXPERIMENT FAILED: $($_.Exception.Message)"
    throw
}
finally {
    [FivesColorControlsSleepControl]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    if ($null -eq $previousCublasConfig) {
        Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue
    }
    else {
        $env:CUBLAS_WORKSPACE_CONFIG = $previousCublasConfig
    }
}
