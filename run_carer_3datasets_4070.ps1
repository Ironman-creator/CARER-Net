param(
    [ValidateRange(0, 16)]
    [int]$Workers = 2,
    [switch]$Overwrite,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$loggedRunner = Join-Path $projectRoot "run_logged.py"
$dataRoot = Join-Path $projectRoot "data"
$splitRoot = Join-Path $projectRoot "reference_splits_carer_3datasets"
$decisionFile = Join-Path $projectRoot "frozen_protocol_carer_3datasets\FIVES_carer_multiseed_decision.json"
$gnRunRoot = Join-Path $projectRoot "runs_carer_3datasets_gn"
$gnWeightDir = Join-Path $projectRoot "weight_files_carer_3datasets_gn"
$carerRunRoot = Join-Path $projectRoot "runs_carer_3datasets_confirm"
$carerWeightDir = Join-Path $projectRoot "weight_files_carer_3datasets_confirm"
$resultRoot = Join-Path $projectRoot "result_carer_3datasets_official_test"
$outputDir = Join-Path $projectRoot "paper_outputs_carer_3datasets"
$statusFile = Join-Path $projectRoot "carer_3datasets_status.txt"
$consoleLog = Join-Path $projectRoot "carer_3datasets_console.log"
$preflightReport = Join-Path $outputDir "CARER_3datasets_preflight.json"
$checkpointManifest = Join-Path $outputDir "CARER_3datasets_checkpoint_manifest.json"
$seeds = @(42, 123, 3407)
$datasets = @("DRIVE", "STARE", "CHASE")
$carerModels = @("gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet")
$testModels = @(
    [PSCustomObject]@{ Name = "res_attention_unet_gn"; InputMode = "grayscale"; Source = "gn"; FileName = "{0}_best_ResAttentionUNetGN_seed{1}.pth" },
    [PSCustomObject]@{ Name = "gray_rgb_control_30_unet"; InputMode = "gray_rgb"; Source = "carer"; FileName = "{0}_best_gray_rgb_control_30_unet_seed{1}.pth" },
    [PSCustomObject]@{ Name = "color_detail_only_unet"; InputMode = "gray_rgb"; Source = "carer"; FileName = "{0}_best_color_detail_only_unet_seed{1}.pth" },
    [PSCustomObject]@{ Name = "carer_unet"; InputMode = "gray_rgb"; Source = "carer"; FileName = "{0}_best_carer_unet_seed{1}.pth" }
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class CarerThreeDatasetSleepControl {
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
    $decisionFile,
    (Join-Path $projectRoot "train.py"),
    (Join-Path $projectRoot "train_carer_ablation.py"),
    (Join-Path $projectRoot "test.py"),
    (Join-Path $projectRoot "preflight_carer_3datasets.py"),
    (Join-Path $projectRoot "verify_carer_3datasets_checkpoints.py"),
    (Join-Path $projectRoot "summarize_carer_3datasets.py")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required file not found: $requiredPath"
    }
}

try {
    [CarerThreeDatasetSleepControl]::SetThreadExecutionState(
        $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED
    ) | Out-Null
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

    if (-not $PreflightOnly) {
        Set-Content -LiteralPath $statusFile -Value @() -Encoding UTF8
    }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Add-Content -LiteralPath $consoleLog -Value "`r`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') CARER DRIVE/STARE/CHASE frozen confirmation =====" -Encoding UTF8
    Write-Status "STARTED: frozen CARER confirmation on DRIVE, STARE, CHASE; 3 seeds; fixed threshold 0.5"

    Invoke-Python @(
        "-u", (Join-Path $projectRoot "preflight_carer_3datasets.py"),
        "--project-root", $projectRoot,
        "--data-root", $dataRoot,
        "--split-root", $splitRoot,
        "--decision-file", $decisionFile,
        "--output", $preflightReport,
        "--require-cuda"
    )
    Write-Status "PREFLIGHT PASSED: datasets, 9 historical splits, frozen decision, CUDA, and model forwards are valid"
    if ($PreflightOnly) {
        Write-Status "PREFLIGHT-ONLY COMPLETE: no training or testing was started"
        exit 0
    }

    foreach ($dataset in $datasets) {
        if ($dataset -eq "STARE") {
            $baseEpochs = "100"
            $baseMinEpochs = "80"
            $basePatience = "25"
        }
        else {
            $baseEpochs = "50"
            $baseMinEpochs = "30"
            $basePatience = "15"
        }

        foreach ($seed in $seeds) {
            $splitFile = Join-Path $splitRoot "$dataset\seed_$seed.json"
            $gnResult = Join-Path $gnRunRoot "$dataset\res_attention_unet_gn\seed_$seed\best_result.json"
            $gnLog = Join-Path $gnRunRoot "$dataset\res_attention_unet_gn\seed_$seed\training_log.csv"
            $gnCheckpoint = Join-Path $gnWeightDir "${dataset}_best_ResAttentionUNetGN_seed$seed.pth"

            if ((Test-Path -LiteralPath $gnResult) -and (Test-Path -LiteralPath $gnCheckpoint) -and -not $Overwrite) {
                Write-Status "SKIPPED GN: dataset=$dataset seed=$seed already complete"
            }
            else {
                Write-Status "TRAINING GN: dataset=$dataset seed=$seed epochs=$baseEpochs min_epochs=$baseMinEpochs"
                $baseArguments = @(
                    "-u", (Join-Path $projectRoot "train.py"),
                    "--dataset", $dataset,
                    "--model", "res_attention_unet_gn",
                    "--data-root", $dataRoot,
                    "--run-root", $gnRunRoot,
                    "--weight-dir", $gnWeightDir,
                    "--split-file", $splitFile,
                    "--image-size", "512",
                    "--input-mode", "grayscale",
                    "--epochs", $baseEpochs,
                    "--batch-size", "1",
                    "--learning-rate", "0.0001",
                    "--loss", "bce_dice",
                    "--bce-weight", "1.0",
                    "--dice-weight", "1.0",
                    "--selection-metric", "soft_dice",
                    "--min-epochs", $baseMinEpochs,
                    "--patience", $basePatience,
                    "--seed", "$seed",
                    "--workers", "$Workers",
                    "--device", "cuda",
                    "--deterministic"
                )
                if ($Overwrite -or (Test-Path -LiteralPath $gnLog)) {
                    $baseArguments += "--overwrite"
                }
                Invoke-Python $baseArguments
            }

            foreach ($requiredPath in @($gnResult, $gnCheckpoint)) {
                if (-not (Test-Path -LiteralPath $requiredPath)) {
                    throw "GN training did not produce required file: $requiredPath"
                }
            }

            $allCarerComplete = $true
            foreach ($modelName in $carerModels) {
                $resultPath = Join-Path $carerRunRoot "$dataset\$modelName\seed_$seed\best_result.json"
                $weightPath = Join-Path $carerWeightDir "${dataset}_best_${modelName}_seed$seed.pth"
                if (-not ((Test-Path -LiteralPath $resultPath) -and (Test-Path -LiteralPath $weightPath))) {
                    $allCarerComplete = $false
                }
            }
            if ($allCarerComplete -and -not $Overwrite) {
                Write-Status "SKIPPED CARER arms: dataset=$dataset seed=$seed all three arms complete"
            }
            else {
                Write-Status "TRAINING CARER arms: dataset=$dataset seed=$seed; control, color-only, full CARER"
                $carerArguments = @(
                    "-u", (Join-Path $projectRoot "train_carer_ablation.py"),
                    "--dataset", $dataset,
                    "--data-root", $dataRoot,
                    "--split-file", $splitFile,
                    "--initial-checkpoint", $gnCheckpoint,
                    "--run-root", $carerRunRoot,
                    "--weight-dir", $carerWeightDir,
                    "--image-size", "512",
                    "--batch-size", "1",
                    "--epochs", "30",
                    "--warmup-epochs", "5",
                    "--head-learning-rate", "0.0001",
                    "--tail-learning-rate", "0.00001",
                    "--base-loss-weight", "0.25",
                    "--error-loss-weight", "0.10",
                    "--train-seed", "$seed",
                    "--variants", "gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet",
                    "--experiment-name", "Frozen CARER three-dataset confirmation",
                    "--workers", "$Workers",
                    "--device", "cuda"
                )
                if ($Overwrite) {
                    $carerArguments += "--overwrite"
                }
                Invoke-Python $carerArguments
            }
        }
    }

    Invoke-Python @(
        "-u", (Join-Path $projectRoot "verify_carer_3datasets_checkpoints.py"),
        "--gn-weight-dir", $gnWeightDir,
        "--carer-weight-dir", $carerWeightDir,
        "--output", $checkpointManifest
    )
    Write-Status "CHECKPOINT VERIFICATION PASSED: all 36 checkpoints strictly loaded and hashed"

    foreach ($dataset in $datasets) {
        foreach ($seed in $seeds) {
            foreach ($model in $testModels) {
                $weightDir = if ($model.Source -eq "gn") { $gnWeightDir } else { $carerWeightDir }
                $checkpoint = Join-Path $weightDir ($model.FileName -f $dataset, $seed)
                $metricsFile = Join-Path $resultRoot "$dataset\$($model.Name)\seed_$seed\metrics.json"
                $perImageFile = Join-Path $resultRoot "$dataset\$($model.Name)\seed_$seed\per_image_metrics.csv"
                if ((Test-Path -LiteralPath $metricsFile) -and (Test-Path -LiteralPath $perImageFile) -and -not $Overwrite) {
                    Write-Status "SKIPPED TEST: dataset=$dataset model=$($model.Name) seed=$seed already complete"
                    continue
                }
                Write-Status "TESTING: dataset=$dataset model=$($model.Name) seed=$seed"
                Invoke-Python @(
                    "-u", (Join-Path $projectRoot "test.py"),
                    "--dataset", $dataset,
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
    }

    Invoke-Python @(
        "-u", (Join-Path $projectRoot "summarize_carer_3datasets.py"),
        "--result-root", $resultRoot,
        "--manifest", $checkpointManifest,
        "--gn-run-root", $gnRunRoot,
        "--carer-run-root", $carerRunRoot,
        "--output-dir", $outputDir
    )
    Write-Status "COMPLETE: DRIVE, STARE, CHASE frozen CARER training, testing, and statistical summaries finished"
    Write-Status "COPY BACK: runs_carer_3datasets_gn, runs_carer_3datasets_confirm, result_carer_3datasets_official_test, paper_outputs_carer_3datasets, carer_3datasets_status.txt, carer_3datasets_console.log; weights may remain on the 4070 PC"
}
catch {
    Write-Status "PIPELINE FAILED: $($_.Exception.Message)"
    throw
}
finally {
    [CarerThreeDatasetSleepControl]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    if ($null -eq $previousCublasConfig) {
        Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue
    }
    else {
        $env:CUBLAS_WORKSPACE_CONFIG = $previousCublasConfig
    }
}
