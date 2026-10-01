param(
    [ValidateRange(0, 16)][int]$Workers = 2,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "run_logged.py"
$dataRoot = Join-Path $projectRoot "data"
$splitRoot = Join-Path $projectRoot "reference_splits_jiim_stage2"
$runRoot = Join-Path $projectRoot "runs_jiim_protocol_baselines"
$weightRoot = Join-Path $projectRoot "weight_files_jiim_protocol_baselines"
$resultRoot = Join-Path $projectRoot "result_jiim_protocol_baselines"
$outputRoot = Join-Path $projectRoot "paper_outputs_jiim_protocol_baselines"
$log = Join-Path $projectRoot "jiim_protocol_baselines_console.log"
$status = Join-Path $projectRoot "jiim_protocol_baselines_status.txt"
$datasets = @("DRIVE", "STARE", "CHASE", "FIVES")
$models = @("unet", "attention_unet", "unet_plus_plus_gn", "r2_unet_gn", "sa_unet_gn")
$referenceModels = @(
    [PSCustomObject]@{Name="res_attention_unet_gn"; InputMode="grayscale"},
    [PSCustomObject]@{Name="gray_rgb_control_30_unet"; InputMode="gray_rgb"},
    [PSCustomObject]@{Name="color_detail_only_unet"; InputMode="gray_rgb"},
    [PSCustomObject]@{Name="carer_unet"; InputMode="gray_rgb"}
)
$seeds = @(42, 123, 3407)

function Write-Status([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Add-Content -LiteralPath $status -Value $line -Encoding UTF8
    Write-Host $line
}

function Invoke-Python([string[]]$arguments) {
    $runnerArguments = @($runner, "--log-file", $log, "--") + $arguments
    & $python @runnerArguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($arguments -join ' ')" }
}

function Resolve-ReferenceCheckpoint([string]$dataset, [string]$model, [int]$seed) {
    if ($dataset -eq "FIVES") {
        if ($model -eq "res_attention_unet_gn") {
            return Join-Path $projectRoot "weight_files_fives_v2_screen\FIVES_best_ResAttentionUNetGN_seed$seed.pth"
        }
        if ($seed -eq 42) {
            if ($model -eq "gray_rgb_control_30_unet" -or $model -eq "color_detail_only_unet") {
                return Join-Path $projectRoot "weight_files_fives_carer_ablation\FIVES_best_${model}_seed42.pth"
            }
            return Join-Path $projectRoot "weight_files_fives_carer_paired_screen\FIVES_best_carer_unet_seed42.pth"
        }
        return Join-Path $projectRoot "weight_files_fives_carer_multiseed_confirm\FIVES_best_${model}_seed$seed.pth"
    }
    if ($model -eq "res_attention_unet_gn") {
        return Join-Path $projectRoot "weight_files_carer_3datasets_gn\${dataset}_best_ResAttentionUNetGN_seed$seed.pth"
    }
    return Join-Path $projectRoot "weight_files_carer_3datasets_confirm\${dataset}_best_${model}_seed$seed.pth"
}

foreach ($path in @($python, $runner, $dataRoot, $splitRoot, (Join-Path $projectRoot "train.py"), (Join-Path $projectRoot "test.py"))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}
Set-Content -LiteralPath $status -Value @() -Encoding UTF8
Write-Status "STARTED: protocol-matched baselines; 4 datasets; 5 models; 3 seeds"

# Re-evaluate the frozen CARER family with the exact same test implementation
# and output layout, so pairwise comparisons never merge incompatible tables.
foreach ($dataset in $datasets) {
    foreach ($seed in $seeds) {
        foreach ($reference in $referenceModels) {
            $checkpoint = Resolve-ReferenceCheckpoint $dataset $reference.Name $seed
            if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing reference checkpoint: $checkpoint" }
            $metrics = Join-Path $resultRoot "$dataset\$($reference.Name)\seed_$seed\metrics.json"
            if ((Test-Path -LiteralPath $metrics) -and -not $Overwrite) {
                Write-Status "SKIPPED REFERENCE TEST: $dataset $($reference.Name) seed=$seed"
                continue
            }
            Write-Status "REFERENCE TEST: $dataset $($reference.Name) seed=$seed"
            Invoke-Python @(
                "-u", (Join-Path $projectRoot "test.py"),
                "--dataset", $dataset,
                "--model", $reference.Name,
                "--checkpoint", $checkpoint,
                "--data-root", $dataRoot,
                "--result-root", $resultRoot,
                "--image-size", "512",
                "--input-mode", $reference.InputMode,
                "--threshold", "0.5",
                "--fov", "derived",
                "--seed", "$seed",
                "--workers", "$Workers",
                "--device", "cuda",
                "--save-probabilities"
            )
        }
    }
}

foreach ($dataset in $datasets) {
    if ($dataset -eq "STARE") {
        $epochs=100; $minEpochs=80; $patience=25; $batch=1
    } elseif ($dataset -eq "FIVES") {
        $epochs=50; $minEpochs=30; $patience=15; $batch=1
    } else {
        $epochs=50; $minEpochs=30; $patience=15; $batch=1
    }
    foreach ($model in $models) {
        foreach ($seed in $seeds) {
            $split = Join-Path $splitRoot "$dataset\seed_$seed.json"
            $tag = switch ($model) {
                "unet" { "UNet" }
                "attention_unet" { "AttentionUNet" }
                "unet_plus_plus_gn" { "UNetPlusPlusGN" }
                "r2_unet_gn" { "R2UNetGN" }
                "sa_unet_gn" { "SAUNetGN" }
            }
            $checkpoint = Join-Path $weightRoot "${dataset}_best_${tag}_seed$seed.pth"
            $best = Join-Path $runRoot "$dataset\$model\seed_$seed\best_result.json"
            if ((Test-Path -LiteralPath $checkpoint) -and (Test-Path -LiteralPath $best) -and -not $Overwrite) {
                Invoke-Python @(
                    "-u", (Join-Path $projectRoot "validate_completed_training_run.py"),
                    "--run-dir", (Split-Path -Parent $best),
                    "--expected-split", $split,
                    "--checkpoint", $checkpoint,
                    "--dataset", $dataset,
                    "--model", $model,
                    "--seed", "$seed",
                    "--epochs", "$epochs",
                    "--batch-size", "$batch",
                    "--min-epochs", "$minEpochs",
                    "--patience", "$patience"
                )
                Write-Status "SKIPPED TRAIN: $dataset $model seed=$seed"
            } else {
                Write-Status "TRAINING: $dataset $model seed=$seed"
                $arguments = @(
                    "-u", (Join-Path $projectRoot "train.py"),
                    "--dataset", $dataset,
                    "--model", $model,
                    "--data-root", $dataRoot,
                    "--run-root", $runRoot,
                    "--weight-dir", $weightRoot,
                    "--split-file", $split,
                    "--image-size", "512",
                    "--input-mode", "grayscale",
                    "--epochs", "$epochs",
                    "--batch-size", "$batch",
                    "--learning-rate", "0.0001",
                    "--loss", "bce_dice",
                    "--bce-weight", "1.0",
                    "--dice-weight", "1.0",
                    "--selection-metric", "soft_dice",
                    "--min-epochs", "$minEpochs",
                    "--patience", "$patience",
                    "--seed", "$seed",
                    "--workers", "$Workers",
                    "--device", "cuda",
                    "--deterministic"
                )
                if ($Overwrite -or (Test-Path -LiteralPath (Split-Path -Parent $best))) { $arguments += "--overwrite" }
                Invoke-Python $arguments
            }
            if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint: $checkpoint" }

            $metrics = Join-Path $resultRoot "$dataset\$model\seed_$seed\metrics.json"
            if ((Test-Path -LiteralPath $metrics) -and -not $Overwrite) {
                Write-Status "SKIPPED TEST: $dataset $model seed=$seed"
            } else {
                Write-Status "TESTING: $dataset $model seed=$seed"
                Invoke-Python @(
                    "-u", (Join-Path $projectRoot "test.py"),
                    "--dataset", $dataset,
                    "--model", $model,
                    "--checkpoint", $checkpoint,
                    "--data-root", $dataRoot,
                    "--result-root", $resultRoot,
                    "--image-size", "512",
                    "--input-mode", "grayscale",
                    "--threshold", "0.5",
                    "--fov", "derived",
                    "--seed", "$seed",
                    "--workers", "$Workers",
                    "--device", "cuda",
                    "--save-probabilities"
                )
            }
        }
    }
}

Invoke-Python @(
    "-u", (Join-Path $projectRoot "summarize_jiim_stage2.py"),
    "--result-root", $resultRoot,
    "--output-dir", $outputRoot,
    "--datasets", "DRIVE", "STARE", "CHASE", "FIVES",
    "--models", "res_attention_unet_gn", "gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet", "unet", "attention_unet", "unet_plus_plus_gn", "r2_unet_gn", "sa_unet_gn",
    "--seeds", "42", "123", "3407"
)
Invoke-Python @(
    "-u", (Join-Path $projectRoot "analyze_probability_calibration.py"),
    "--result-root", $resultRoot,
    "--data-root", $dataRoot,
    "--output-dir", (Join-Path $outputRoot "calibration"),
    "--datasets", "DRIVE", "STARE", "CHASE", "FIVES",
    "--models", "res_attention_unet_gn", "gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet", "unet", "attention_unet", "unet_plus_plus_gn", "r2_unet_gn", "sa_unet_gn",
    "--seeds", "42", "123", "3407"
)
Write-Status "COMPLETE: protocol-matched baseline training, testing, probability export and summary finished"
Write-Status "COPY BACK: runs_jiim_protocol_baselines, result_jiim_protocol_baselines, paper_outputs_jiim_protocol_baselines, status and console log; weights can remain on the 4070 PC"
