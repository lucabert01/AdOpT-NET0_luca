"""
Ad-hoc driver: runs ccs_chain_plots.py's figure set for every (scenario_name,
objective, carbon_tax) combination in main_italy.py's SCENARIOS (each sector
can pick its own capture technology, so nodes like Ferrara/Piacenza host more
than one distinct emitter -- see ccs_chain_plots.py's _draw_node_emitters).
Same monkeypatch pattern as _run_ccs_chain_multi.py.

main_italy.py runs "technology_selection" cost-minimizing at three carbon_tax
levels (150/200/250 EUR/tonne) plus once emissions-minimizing, and
"technology_selection_wasteCaL" once emissions-minimizing -- see
main_italy.py's SCENARIOS list -- so this iterates all five combinations
instead of hardcoding one h5 path per scenario name.
ccs_chain_plots.find_run_h5 resolves each (name, objective, carbon_tax)
combination to its result folder and picks the real multi-period solve over
any leftover design-days/clustering pre-solve (see that function's
docstring); carbon_tax disambiguates the three same-name/same-objective cost
runs from each other.
"""
from pathlib import Path

import ccs_chain_plots as m

# carbon_tax is only needed to disambiguate multiple runs sharing the same
# (scenario_name, objective) -- e.g. the "costs" tax sweep below. Leave it
# None for scenarios that don't vary carbon_tax (emissions_minC runs).
SCENARIOS = [
    ("technology_selection", "costs", 150),
    ("technology_selection", "costs", 200),
    ("technology_selection", "costs", 250),
    ("technology_selection", "emissions_minC", None),
    ("technology_selection_wasteCaL", "emissions_minC", None),
]

# Node used for the per-emitter zoom-in and the downstream-of-storage inflow
# check, run for every scenario so the same two plants/nodes are directly
# comparable across scenarios and objectives.
EMITTER_ZOOM_NODE = "SILLA 2"
INFLOW_NODE = "Eni S.p.A Casalborsetti"

for scenario_name, objective, carbon_tax in SCENARIOS:
    run_key = f"{scenario_name}_{objective}" + (f"_tax{carbon_tax}" if carbon_tax is not None else "")
    print("\n" + "=" * 80)
    print(f"SCENARIO: {run_key}")
    print("=" * 80)

    try:
        h5_path = m.find_run_h5(scenario_name, objective, carbon_tax=carbon_tax)
    except FileNotFoundError as e:
        print(f"  SKIPPED - {e}")
        continue

    m.RESULTS_H5 = h5_path
    out_dir = Path(f"ccs_chain_results/{run_key}")
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

    if EMITTER_ZOOM_NODE in ccs_df["node"].values:
        m.plot_emitter_zoom(m.RESULTS_H5, node_name=EMITTER_ZOOM_NODE)
    else:
        print(f"  SKIPPED emitter zoom - '{EMITTER_ZOOM_NODE}' not present in this run")

    if INFLOW_NODE in built_arcs["to"].values or INFLOW_NODE in built_arcs["from"].values:
        m.plot_node_inflow(m.RESULTS_H5, node_name=INFLOW_NODE)
    else:
        print(f"  SKIPPED inflow plot - '{INFLOW_NODE}' not present in this run's built network")

    print(f"\n--- {run_key} summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v:,.0f}")
    print(f"  built arcs: {len(built_arcs)}")
    print(f"  emitters (tech rows): {len(ccs_df)}  |  CCS installed: {int(ccs_df['ccs_installed'].sum())}")

print("\nDone.")
