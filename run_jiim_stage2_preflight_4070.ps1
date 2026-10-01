param([ValidateRange(0, 16)][int]$Workers = 2)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataRoot = Join-Path $projectRoot "data"
$splitRoot = Join-Path $projectRoot "reference_splits_jiim_stage2"
$output = Join-Path $projectRoot "paper_outputs_jiim_stage2\preflight.json"

foreach ($path in @($python, $dataRoot, $splitRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}

& $python -u (Join-Path $projectRoot "preflight_jiim_stage2.py") `
    --project-root $projectRoot --data-root $dataRoot --split-root $splitRoot `
    --output $output --require-cuda
if ($LASTEXITCODE -ne 0) { throw "JIIM stage-2 preflight failed" }
Write-Host "PREFLIGHT PASSED. No training was started."
