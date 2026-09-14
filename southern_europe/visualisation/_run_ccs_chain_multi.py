"""
Ad-hoc driver: runs ccs_chain_plots.py's figure set for multiple scenario
result folders without touching the module's own hardcoded RESULTS_H5/OUT_DIR
(monkeypatches them per scenario, since the plotting functions read those as
module globals at call time).
"""
import sys
from pathlib import Path

import ccs_chain_plots as m

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

    nodes_gdf = m.gpd.read_file(m.GIS_NODES)
    built_arcs = m.load_built_arcs(m.RESULTS_H5)
    built_arcs = m.attach_route_geometries(built_arcs, nodes_gdf)
    ccs_df = m.load_ccs_status(m.RESULTS_H5)
    summary = m.load_summary(m.RESULTS_H5)
    cost_breakdown = m.compute_cost_breakdown(m.RESULTS_H5)

    m.plot_main_map(built_arcs, nodes_gdf, ccs_df, summary)
    m.plot_cost_breakdown(cost_breakdown, per_tonne=True)
    m.plot_cost_breakdown(cost_breakdown, per_tonne=False)
    m.plot_summary_dashboard(built_arcs, ccs_df, cost_breakdown)

    print(f"\n--- {scenario} summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v:,.0f}")
    n_ccs = int((ccs_df["ccs_installed"] > 0).sum()) if "ccs_installed" in ccs_df.columns else None
    print(f"  built arcs: {len(built_arcs)}")

print("\nDone.")
