"""
Ad-hoc driver: runs ccs_chain_plots.py's figure set for the two "technology
selection" scenario result folders (each sector can pick its own capture
technology, so nodes like Ferrara/Piacenza host more than one distinct
emitter -- see ccs_chain_plots.py's _draw_node_emitters). Same monkeypatch
pattern as _run_ccs_chain_multi.py.

Each scenario folder holds two solved runs; the smaller-objective one in
each (~1e6-1e7) is a design-days/clustering pre-solve, not the real
multi-period result -- picked the larger-objective run (~1e9, same order of
magnitude as the single-family mea/cal runs' total_cost) in each folder.
"""
from pathlib import Path

import ccs_chain_plots as m

RUNS = {
    "technology_selection": "../Results_CCSchainOptimization/technology_selection/20260915182519_costs_technology_selection-1/optimization_results.h5",
    "technology_selection_wasteCaL": "../Results_CCSchainOptimization/technology_selection_wasteCaL/20260916043843_costs_technology_selection_wasteCaL-1/optimization_results.h5",
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

    print("\n--- emitters per node (families present) ---")
    for node, grp in ccs_df.groupby("node"):
        if len(grp) > 1:
            print(f"  {node}: " + ", ".join(f"{t}[{fam}]" for t, fam in zip(grp['tech'], grp['family'])))

    m.plot_main_map(built_arcs, nodes_gdf, ccs_df, summary)
    m.plot_map_sized_by_capacity(built_arcs, nodes_gdf, ccs_df, summary)
    m.plot_network_map_cost_factor(built_arcs, nodes_gdf, ccs_df, summary)
    m.plot_trunk_highlight(built_arcs, nodes_gdf, ccs_df, summary)
    m.plot_cost_breakdown(cost_breakdown, per_tonne=True)
    m.plot_cost_breakdown(cost_breakdown, per_tonne=False)
    m.plot_summary_dashboard(built_arcs, ccs_df, cost_breakdown)

    print(f"\n--- {scenario} summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v:,.0f}")
    print(f"  built arcs: {len(built_arcs)}")
    print(f"  emitters (tech rows): {len(ccs_df)}  |  CCS installed: {int(ccs_df['ccs_installed'].sum())}")

print("\nDone.")
