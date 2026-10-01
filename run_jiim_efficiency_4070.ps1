param()

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$output = Join-Path $projectRoot "paper_outputs_jiim_efficiency"
$log = Join-Path $projectRoot "jiim_efficiency_console.log"
$status = Join-Path $projectRoot "jiim_efficiency_status.txt"
$runner = Join-Path $projectRoot "run_logged.py"
Set-Content -LiteralPath $status -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') STARTED: uniform efficiency benchmark" -Encoding UTF8
$runnerArguments = @(
    $runner,
    "--log-file", $log,
    "--",
    "-u", (Join-Path $projectRoot "benchmark_jiim_models.py"),
    "--output-dir", $output
)
& $python @runnerArguments
if ($LASTEXITCODE -ne 0) { throw "Efficiency benchmark failed" }
Add-Content -LiteralPath $status -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') COMPLETE: copy back paper_outputs_jiim_efficiency, status and console" -Encoding UTF8
