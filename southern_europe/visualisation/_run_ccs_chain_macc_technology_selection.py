"""
Ad-hoc driver: runs ccs_chain_emitter_cost_ranking.py's emitter cost-ranking +
MACC figures for every (scenario_name, objective, carbon_tax) combination in
main_italy.py's SCENARIOS, mirroring _run_ccs_chain_macc_multi.py /
_run_ccs_chain_technology_selection.py.
"""
from pathlib import Path

import ccs_chain_emitter_cost_ranking as m
import ccs_chain_plots as plots

# carbon_tax disambiguates the three same-name/same-objective "costs" runs
# (main_italy.py's tax sweep); leave None for scenarios that don't vary it.
SCENARIOS = [
    ("technology_selection", "costs", 150),
    ("technology_selection", "costs", 200),
    ("technology_selection", "costs", 250),
    ("technology_selection", "emissions_minC", None),
    ("technology_selection_wasteCaL", "emissions_minC", None),
]

for scenario_name, objective, carbon_tax in SCENARIOS:
    run_key = f"{scenario_name}_{objective}" + (f"_tax{carbon_tax}" if carbon_tax is not None else "")
    print("\n" + "=" * 80)
    print(f"SCENARIO: {run_key}")
    print("=" * 80)

    try:
        h5_path = plots.find_run_h5(scenario_name, objective, carbon_tax=carbon_tax)
    except FileNotFoundError as e:
        print(f"  SKIPPED - {e}")
        continue

    m.RESULTS_H5 = h5_path
    out_dir = Path(f"ccs_chain_results/{run_key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    m.OUT_DIR = out_dir

    df, storage_node = m.build_emitter_cost_table(m.RESULTS_H5)

    out_csv = out_dir / "ccs_chain_emitter_cost_ranking.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} emitters)")

    m.plot_emitter_cost_ranking(df, storage_node)
    m.plot_macc(df, storage_node)

print("\nDone.")
