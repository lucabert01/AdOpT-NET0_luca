#!/usr/bin/env bash
# Runs only the two "constant_injection" scenarios of main_italy.py (CO2 tax 200
# and 250), one after the other, in the background. Survives logging out of the
# server. Usage, from anywhere:
#     bash southern_europe/run_constant_injection.sh
# Follow progress with:
#     tail -f southern_europe/logs/constant_injection_<timestamp>.log
#
# Case-study inputs are written to southern_europe/Italy_CaseStudy/constant_injection/
# and results to southern_europe/Results_CCSchainOptimization/constant_injection/
# (both created by main_italy.py). Needs internet access for the JRC climate data API.

set -euo pipefail

# main_italy.py uses paths relative to southern_europe/
cd "$(dirname "$0")"

mkdir -p logs Italy_CaseStudy Results_CCSchainOptimization
log_file="logs/constant_injection_$(date +%Y%m%d_%H%M%S).log"

nohup python -u main_italy.py constant_injection > "$log_file" 2>&1 &

echo "Started constant_injection scenarios (PID $!)"
echo "Log: southern_europe/$log_file"
