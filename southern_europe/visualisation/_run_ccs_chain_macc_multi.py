"""
Ad-hoc driver: runs ccs_chain_emitter_cost_ranking.py's emitter cost-ranking +
MACC figures for multiple scenario result folders without touching the
module's own hardcoded RESULTS_H5/OUT_DIR (monkeypatches them per scenario,
mirroring _run_ccs_chain_multi.py).
"""
from pathlib import Path

import ccs_chain_emitter_cost_ranking as m

RUNS = {
    "mea": "../Results_CCSchainOptimization/mea/20260821181652_emissions_minC_mea-1/optimization_results.h5",
    "mea_timeless": "../Results_CCSchainOptimization/mea_timeless/20260822180441_emissions_minC_mea_timeless-1/optimization_results.h5",
    "oxy": "../Results_CCSchainOptimization/oxy/20260821191359_emissions_minC_oxy-1/optimization_results.h5",
    "cal": "../Results_CCSchainOptimization/cal/20260821214449_emissions_minC_cal-1/optimization_results.h5",
    "oxyCal": "../Results_CCSchainOptimization/oxyCal/20260822075757_emissions_minC_oxyCal-1/optimization_results.h5",
}

for scenario, h5_rel in RUNS.items():
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario}")
    print("=" * 80)

    m.RESULTS_H5 = Path(h5_rel)
    out_dir = Path(f"ccs_chain_results/{scenario}")
    out_dir.mkdir(parents=True, exist_ok=True)
    m.OUT_DIR = out_dir

    df, storage_node = m.build_emitter_cost_table(m.RESULTS_H5)

    out_csv = out_dir / "ccs_chain_emitter_cost_ranking.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} emitters)")

    m.plot_emitter_cost_ranking(df, storage_node)
    m.plot_macc(df, storage_node)

print("\nDone.")
