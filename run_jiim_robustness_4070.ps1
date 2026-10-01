param(
    [ValidateRange(0, 16)][int]$Workers = 2,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "run_logged.py"
$dataRoot = Join-Path $projectRoot "data"
$resultRoot = Join-Path $projectRoot "result_jiim_robustness"
$outputRoot = Join-Path $projectRoot "paper_outputs_jiim_robustness"
$log = Join-Path $projectRoot "jiim_robustness_console.log"
$status = Join-Path $projectRoot "jiim_robustness_status.txt"
$corruptions = @("clean", "dark_0.65", "bright_1.35", "contrast_0.65", "gamma_1.50", "gaussian_blur_1.5")
$models = @("gray_rgb_control_30_unet", "color_detail_only_unet", "carer_unet")

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

function Resolve-Checkpoint([string]$model) {
    if ($model -eq "carer_unet") {
        return Join-Path $projectRoot "weight_files_fives_carer_paired_screen\FIVES_best_carer_unet_seed42.pth"
    }
    return Join-Path $projectRoot "weight_files_fives_carer_ablation\FIVES_best_${model}_seed42.pth"
}

Set-Content -LiteralPath $status -Value @() -Encoding UTF8
Write-Status "STARTED: exploratory deterministic FIVES robustness, frozen seed=42"
foreach ($corruption in $corruptions) {
    foreach ($model in $models) {
        $checkpoint = Resolve-Checkpoint $model
        if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
        $metrics = Join-Path $resultRoot "$corruption\$model\seed_42\metrics.json"
        if ((Test-Path -LiteralPath $metrics) -and -not $Overwrite) {
            Write-Status "SKIPPED: $corruption $model"
            continue
        }
        Write-Status "TESTING: $corruption $model"
        Invoke-Python @(
            "-u", (Join-Path $projectRoot "evaluate_robustness.py"),
            "--model", $model,
            "--checkpoint", $checkpoint,
            "--data-root", $dataRoot,
            "--result-root", $resultRoot,
            "--corruption", $corruption,
            "--input-mode", "gray_rgb",
            "--seed", "42",
            "--image-size", "512",
            "--batch-size", "2",
            "--workers", "$Workers",
            "--device", "cuda"
        )
    }
}
Invoke-Python @(
    "-u", (Join-Path $projectRoot "summarize_robustness.py"),
    "--result-root", $resultRoot,
    "--output-dir", $outputRoot
)
Write-Status "COMPLETE: robustness evaluation and paired summaries finished"
Write-Status "COPY BACK: result_jiim_robustness, paper_outputs_jiim_robustness, status and console log"
