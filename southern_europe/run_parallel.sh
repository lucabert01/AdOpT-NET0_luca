#!/usr/bin/env bash
# Runs the cases of main_italy.py's SCENARIOS as a queue: at most MAX_PARALLEL
# cases at a time, each a separate process with THREADS Gurobi threads; when one
# finishes, the next one starts. Each case warm starts from its own latest finished
# run (see SCENARIOS), so the order only matters for c_inj_min95 (last in the list:
# by the time it starts, it may pick up the new c_inj run's design).
#
# Usage (from southern_europe/, in the Python environment of the project):
#   bash run_parallel.sh                                  # 4 at a time, 4 threads each
#   MAX_PARALLEL=3 THREADS=12 bash run_parallel.sh
#   MAX_PARALLEL=1 bash run_parallel.sh tax175_costs_technology_selection
#
# Keep it running after logging out:
#   nohup bash run_parallel.sh > logs/queue.log 2>&1 &
#
# Output goes to logs/<case>.log. When a case ends, its peak memory ("Maximum
# resident set size", in kB) is at the end of that log (needs /usr/bin/time);
# use it to decide whether more cases fit at once:
#   grep "Maximum resident" logs/*.log
set -uo pipefail
cd "$(dirname "$0")"

MAX_PARALLEL="${MAX_PARALLEL:-4}"
THREADS="${THREADS:-4}"
PYTHON="${PYTHON:-python}"
# this repository first on the path (as PyCharm does), not the venv's adopt_net0
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
    CASES=(
        tax175_costs_technology_selection
        tax200_costs_technology_selection
        tax250_costs_technology_selection
        tax250_emissions_minC_technology_selection
        tax250_emissions_minC_technology_selection_wasteCaL
        tax200_costs_c_inj
        tax200_costs_c_inj_min95
    )
fi

TIME=()
if [ -x /usr/bin/time ]; then TIME=(/usr/bin/time -v); fi

mkdir -p logs
for case in "${CASES[@]}"; do
    # wait for a free slot
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
    done
    echo "$(date '+%F %T') start $case ($THREADS threads) -> logs/$case.log"
    GUROBI_THREADS="$THREADS" PYTHONUNBUFFERED=1 \
        "${TIME[@]}" "$PYTHON" main_italy.py "$case" > "logs/$case.log" 2>&1 &
    # stagger the starts: the data preparation and model build need the most memory
    sleep 120
done
wait
echo "$(date '+%F %T') all cases finished"
grep -H "Maximum resident" logs/*.log 2>/dev/null
