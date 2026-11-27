# Stops all main_italy.py runs of the current user, and the run_parallel.ps1 queues
# that start them (so no new case starts). Shows what it stops first.
#
# Usage (any folder):
#   powershell -ExecutionPolicy Bypass -File .\stop_runs.ps1
$me = $env:USERNAME
$targets = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -like '*main_italy.py*' -or $_.CommandLine -like '*run_parallel.ps1*') -and
    $_.ProcessId -ne $PID -and
    (Invoke-CimMethod -InputObject $_ -MethodName GetOwner).User -eq $me
})
if ($targets.Count -eq 0) {
    Write-Host "No main_italy.py runs or queues of $me found."
    exit 0
}
# queues first, so they do not start the next case when a run is stopped
$targets = @($targets | Sort-Object { if ($_.CommandLine -like '*run_parallel.ps1*') { 0 } else { 1 } })
foreach ($p in $targets) {
    Write-Host ("stopping pid {0}: {1}" -f $p.ProcessId, $p.CommandLine)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host "Stopped $($targets.Count) processes."
