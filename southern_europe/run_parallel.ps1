# Runs the cases of main_italy.py's SCENARIOS as a queue on Windows: at most
# MaxParallel cases at a time, each a separate python process with Threads Gurobi
# threads; when one finishes, the next one starts. Each case warm starts from its
# own latest finished run (see SCENARIOS), so the order only matters for
# c_inj_min95 (last in the list: by the time it starts, it may pick up the new
# c_inj run's design). Windows version of run_parallel.sh.
#
# Usage (from southern_europe\):
#   powershell -ExecutionPolicy Bypass -File .\run_parallel.ps1        # all 7 at once, 5 threads each
#   powershell -ExecutionPolicy Bypass -File .\run_parallel.ps1 -MaxParallel 2 -Threads 8
#   powershell -ExecutionPolicy Bypass -File .\run_parallel.ps1 -Python <venv>\Scripts\python.exe
# Without -Python, the server's venv ..\..\Adopt_NET0_Xiao\venv is used if it exists,
# otherwise the first python on the PATH. Stop all runs with .\stop_runs.ps1.
#
# A run needs ~7 GB of memory at its peak (measured on the server, 2026-11-27), so
# all 7 cases fit at once: 7 x 5 threads = 35 of the server's 40 cores.
#   powershell -ExecutionPolicy Bypass -File .\run_parallel.ps1 -MaxParallel 1 -Cases tax175_costs_technology_selection,tax200_costs_c_inj
#   (several cases: comma-separated, no spaces)
#
# The cases run as their own processes: they keep running if this window is
# closed (only the queue stops starting new ones). Disconnect from the remote
# desktop session instead of signing out -- signing out ends all processes.
#
# Output: logs\<case>.log (and <case>.err.log); the queue's own messages,
# including each case's peak memory, go to logs\queue.log.
param(
    [int]$MaxParallel = 7,
    [int]$Threads = 5,
    [string]$Python = "",
    [int]$StaggerSeconds = 120,
    [string[]]$Cases = @(
        "tax175_costs_technology_selection",
        "tax200_costs_technology_selection",
        "tax250_costs_technology_selection",
        "tax250_emissions_minC_technology_selection",
        "tax250_emissions_minC_technology_selection_wasteCaL",
        "tax200_costs_c_inj",
        "tax200_costs_c_inj_min95"
    )
)

# "-Cases a,b,c" arrives as one string when the script is run with -File
$Cases = @($Cases | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

Set-Location $PSScriptRoot
if (-not $Python) {
    $serverVenv = Join-Path $PSScriptRoot "..\..\Adopt_NET0_Xiao\venv\Scripts\python.exe"
    $Python = if (Test-Path $serverVenv) { (Resolve-Path $serverVenv).Path } else { "python" }
}
New-Item -ItemType Directory -Force logs | Out-Null
$queueLog = Join-Path $PSScriptRoot "logs\queue.log"

function Write-QueueLog([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Write-Host $line
    # the network share sometimes refuses the write (e.g. while queue.log is open
    # elsewhere): retry briefly, and never let it stop the queue
    for ($try = 1; $try -le 5; $try++) {
        try { Add-Content -Path $queueLog -Value $line -Encoding utf8 -ErrorAction Stop; break }
        catch { Start-Sleep -Seconds 2 }
    }
}

# Inherited by the python processes
$env:GUROBI_THREADS = "$Threads"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
# This repository's root first on the path (as PyCharm does): main_italy.py imports
# adopt_net0 and southern_europe.* from here, not from whatever the venv has installed
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot;$env:PYTHONPATH" } else { $repoRoot }

$pythonExe = (Get-Command $Python -ErrorAction Stop).Source
Write-QueueLog "queue: $($Cases.Count) cases, $MaxParallel at a time, $Threads threads each, python $pythonExe"
$adoptPath = & $pythonExe -c "import adopt_net0, gurobipy; print(adopt_net0.__file__)"
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "this python cannot import adopt_net0 and gurobipy -- pass the project's venv with -Python <path to python.exe>"
    exit 1
}
Write-QueueLog "adopt_net0 from $adoptPath"

$running = @{}   # case -> @{ Process; PeakGB }

function Update-Running {
    foreach ($case in @($running.Keys)) {
        $entry = $running[$case]
        $proc = $entry.Process
        if ($proc.HasExited) {
            Write-QueueLog ("finished {0}: exit code {1}, peak memory {2:N1} GB" -f `
                $case, $proc.ExitCode, $entry.PeakGB)
            $running.Remove($case)
        } else {
            # A venv's python.exe is a launcher that runs the real python as its
            # child process: take the largest peak working set of the two.
            try {
                $procs = @(Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id) OR ParentProcessId=$($proc.Id)")
                $gb = ($procs | Measure-Object -Property PeakWorkingSetSize -Maximum).Maximum * 1KB / 1GB
                if ($gb -gt $entry.PeakGB) { $entry.PeakGB = $gb }
            } catch {}
        }
    }
}

foreach ($case in $Cases) {
    while ($running.Count -ge $MaxParallel) {
        Start-Sleep -Seconds 30
        Update-Running
    }
    $proc = Start-Process -FilePath $pythonExe -ArgumentList "main_italy.py", $case `
        -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru `
        -RedirectStandardOutput "logs\$case.log" -RedirectStandardError "logs\$case.err.log"
    $null = $proc.Handle   # keeps the handle, so ExitCode is still readable after exit
    $running[$case] = @{ Process = $proc; PeakGB = 0.0 }
    Write-QueueLog "started $case (pid $($proc.Id)) -> logs\$case.log"
    # stagger the starts: the data preparation and model build need the most memory
    Start-Sleep -Seconds $StaggerSeconds
    Update-Running
}

while ($running.Count -gt 0) {
    Start-Sleep -Seconds 30
    Update-Running
}
Write-QueueLog "all cases finished"
