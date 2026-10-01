param(
    [ValidateRange(0, 16)][int]$Workers = 2,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "run_logged.py"
$dataRoot = Join-Path $projectRoot "data"
$resultRoot = Join-Path $projectRoot "result_jiim_external_validation"
$outputRoot = Join-Path $projectRoot "paper_outputs_jiim_external_validation"
$log = Join-Path $projectRoot "jiim_external_validation_console.log"
$status = Join-Path $projectRoot "jiim_external_validation_status.txt"
$datasets = @("DRIVE", "STARE", "CHASE")
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

function Resolve-Checkpoint([string]$model, [int]$seed) {
    if ($model -eq "res_attention_unet_gn") {
        return Join-Path $projectRoot "weight_files_fives_v2_screen\FIVES_best_ResAttentionUNetGN_seed$seed.pth"
    }
    if ($seed -eq 42) {
        if ($model -eq "gray_rgb_control_30_unet") {
            return Join-Path $projectRoot "weight_files_fives_carer_ablation\FIVES_best_gray_rgb_control_30_unet_seed42.pth"
        }
        if ($model -eq "color_detail_only_unet") {
            return Join-Path $projectRoot "weight_files_fives_carer_ablation\FIVES_best_color_detail_only_unet_seed42.pth"
        }
        if ($model -eq "carer_unet") {
            return Join-Path $projectRoot "weight_files_fives_carer_paired_screen\FIVES_best_carer_unet_seed42.pth"
        }
    }
    return Join-Path $projectRoot "weight_files_fives_carer_multiseed_confirm\FIVES_best_${model}_seed$seed.pth"
}

$models = @(
    [PSCustomObject]@{Name="res_attention_unet_gn"; InputMode="grayscale"},
    [PSCustomObject]@{Name="gray_rgb_control_30_unet"; InputMode="gray_rgb"},
    [PSCustomObject]@{Name="color_detail_only_unet"; InputMode="gray_rgb"},
    [PSCustomObject]@{Name="carer_unet"; InputMode="gray_rgb"}
)

foreach ($path in @($python, $runner, $dataRoot, (Join-Path $projectRoot "test.py"))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}
Set-Content -LiteralPath $status -Value @() -Encoding UTF8
Write-Status "STARTED: FIVES-trained zero-shot validation on DRIVE, STARE, CHASE"

foreach ($dataset in $datasets) {
    foreach ($seed in $seeds) {
        foreach ($model in $models) {
            $checkpoint = Resolve-Checkpoint $model.Name $seed
            if (-not (Test-Path -LiteralPath $checkpoint)) {
                throw "FIVES checkpoint not found: $checkpoint"
            }
            $metrics = Join-Path $resultRoot "$dataset\$($model.Name)\seed_$seed\metrics.json"
            if ((Test-Path -LiteralPath $metrics) -and -not $Overwrite) {
                Write-Status "SKIPPED: $dataset $($model.Name) seed=$seed"
                continue
            }
            Write-Status "TESTING: FIVES -> $dataset model=$($model.Name) seed=$seed"
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
                "--device", "cuda",
                "--save-probabilities"
            )
        }
    }
}

Invoke-Python @(
    "-u", (Join-Path $projectRoot "summarize_jiim_stage2.py"),
    "--result-root", $resultRoot,
    "--output-dir", $outputRoot,
    "--datasets", "DRIVE", "STARE", "CHASE",
    "--models", "res_attention_unet_gn", "gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet",
    "--seeds", "42", "123", "3407"
)
Invoke-Python @(
    "-u", (Join-Path $projectRoot "analyze_probability_calibration.py"),
    "--result-root", $resultRoot,
    "--data-root", $dataRoot,
    "--output-dir", (Join-Path $outputRoot "calibration"),
    "--datasets", "DRIVE", "STARE", "CHASE",
    "--models", "res_attention_unet_gn", "gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet",
    "--seeds", "42", "123", "3407"
)
Write-Status "COMPLETE: zero-shot external validation and probability export finished"
Write-Status "COPY BACK: result_jiim_external_validation, paper_outputs_jiim_external_validation, status and console log"
