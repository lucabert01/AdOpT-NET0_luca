"""
Exports everything the interactive technology-selection dashboard (an HTML
artifact) needs as one JSON file: Italy basemap polygons, node positions,
built transport corridors (with annualized flow + load factor), and capture
units (one row per genuine real-world emitter, plus its underlying candidate
technologies) for all four (scenario_name, objective) combinations in
main_italy.py's SCENARIOS: technology_selection and
technology_selection_wasteCaL, each run once minimizing "costs" and once
minimizing "emissions_minC".

Reuses ccs_chain_plots.py's loaders so the dashboard's numbers are exactly
the ones already validated in the static PNG figures (same materiality
filters, same real-emitter collapsing, same annualized network flow).
"""
import json
from pathlib import Path

import numpy as np

import ccs_chain_plots as m

SCENARIOS = [
    ("technology_selection", "costs"),
    ("technology_selection", "emissions_minC"),
    ("technology_selection_wasteCaL", "costs"),
    ("technology_selection_wasteCaL", "emissions_minC"),
]

SCENARIO_LABELS = {
    "technology_selection_costs": "Technology selection (cost-minimizing)",
    "technology_selection_emissions_minC": "Technology selection (emissions-minimizing)",
    "technology_selection_wasteCaL_costs": "Technology selection + calcium looping (cost-minimizing)",
    "technology_selection_wasteCaL_emissions_minC": "Technology selection + calcium looping (emissions-minimizing)",
}

HOURS_PER_YEAR = 8760


def geom_to_paths(geom):
    """[[ [lon,lat], ... ], ...] -- a list of one or more line-strings (a
    route can be a MultiLineString), each a list of [lon, lat] pairs."""
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        return [[[round(x, 5), round(y, 5)] for x, y, *_ in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[[round(x, 5), round(y, 5)] for x, y, *_ in line.coords] for line in geom.geoms]
    return []


def polygon_to_paths(geom, tolerance=0.01):
    """Simplified [[ [lon,lat], ... ], ...] boundary rings for the basemap --
    tolerance in degrees is generous (~1km) since this is a backdrop, not a
    survey map."""
    geom = geom.simplify(tolerance, preserve_topology=True)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return [
        [[round(x, 4), round(y, 4)] for x, y in poly.exterior.coords]
        for poly in polys
    ]


def build_scenario(h5_path: Path) -> dict:
    nodes_gdf = m.gpd.read_file(m.GIS_NODES)
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")

    built_arcs = m.load_built_arcs(h5_path)
    built_arcs = m.attach_route_geometries(built_arcs, nodes_gdf)
    ccs_df = m.load_ccs_status(h5_path)
    summary = m.load_summary(h5_path)
    cost_breakdown = m.compute_cost_breakdown(h5_path)

    # ---- nodes ----
    nodes = []
    for _, row in nodes_unique.iterrows():
        nodes.append({
            "name": row["node_name"],
            "lon": round(row.geometry.x, 5),
            "lat": round(row.geometry.y, 5),
            "type": row["node_type"],
        })

    # ---- arcs ----
    arcs = []
    for _, r in built_arcs.iterrows():
        annual_max = r["size"] * HOURS_PER_YEAR
        load_factor = (r["total_flow"] / annual_max) if annual_max > 0 else 0.0
        arcs.append({
            "mode": r["mode"],
            "network": r["network"],
            "from": r["from"],
            "to": r["to"],
            "size_t_h": round(r["size"], 2),
            "annual_flow_t_yr": round(r["total_flow"], 1),
            "load_factor": round(load_factor, 4),
            "capex_eur": round(r["capex"], 0),
            "paths": geom_to_paths(r["geometry"]),
        })

    # ---- capture units: one row per genuine real-world emitter, each
    # carrying the candidate technologies collapsed into it ----
    KNOWN_SECTORS = {"Cement", "Waste", "Refining", "Lime", "FertilizersCombustion", "FertilizersSMR"}

    def display_sector(tech_name: str) -> str:
        # ccs_chain_plots._tech_sector falls back to the raw tech name for
        # anything it doesn't recognize (e.g. UnspecifiedEmitter_existing at
        # Ravenna) -- fine for its own internal grouping key, not fine as a
        # user-facing sector label, so normalize any unrecognized value to
        # "Other" here (matches node_metrics_paper's "Other" node_type).
        sector = m._tech_sector(tech_name)
        return sector if sector in KNOWN_SECTORS else "Other"

    name_to_point = {row["node_name"]: row.geometry for _, row in nodes_unique.iterrows()}
    real_df = m.real_emitters_df(ccs_df)
    emitters = []
    for node_name in ccs_df["node"].unique():
        node_rows = ccs_df[ccs_df["node"] == node_name]
        real_rows = m._real_emitter_rows(ccs_df, node_name)
        point = name_to_point.get(node_name)
        if point is None:
            continue
        positions = m._node_marker_positions(point, len(real_rows))
        tagged = node_rows.copy()
        tagged["_sector"] = tagged["tech"].map(m._tech_sector)
        for (x, y), (_, real_row) in zip(positions, real_rows.iterrows()):
            sector = display_sector(real_row["tech"])
            # A node with a single real emitter (the common case) attributes
            # ALL of its candidate technologies to that one emitter,
            # regardless of sector tag -- only a genuinely multi-sector node
            # (Ferrara, Piacenza) needs the sector filter to split them.
            candidates = tagged if len(real_rows) == 1 else tagged[tagged["_sector"] == sector]
            # capture_capacity_t_h (peak captured-CO2 rate), NOT the design
            # "size" field -- "size" means a different thing per family and
            # is frequently the HOST emitter's own production capacity, not
            # the capture equipment's (see load_ccs_status's docstring: one
            # real run showed a 98.45 t/h host plant with a 72.53 t/h
            # capture retrofit built to match a 72.53 t/h pipeline -- using
            # "size" here would have wrongly read as a 26 t/h shortfall).
            capacity = float(real_row["capture_capacity_t_h"]) if "capture_capacity_t_h" in real_row else float(candidates["capture_capacity_t_h"].sum())
            total_emissions = float(real_row["total_emissions"])
            captured_annual = float(real_row["captured_annual"])
            # Utilization = how hard the CAPTURE equipment itself runs
            # relative to its own peak rate (captured / capacity*8760), not
            # against total_emissions -- total_emissions includes the
            # vented share the capture equipment was never sized to handle.
            utilization = (captured_annual / (capacity * HOURS_PER_YEAR)) if capacity > 0 else 0.0
            emitters.append({
                "node": node_name,
                "lon": round(x, 5),
                "lat": round(y, 5),
                "sector": sector,
                "family": real_row["family"],
                "dominant_tech": real_row["tech"],
                "installed": bool(real_row["ccs_installed"]),
                "size_t_h": round(capacity, 2),
                "captured_annual_t_yr": round(captured_annual, 1),
                "total_emissions_t_yr": round(total_emissions, 1),
                "captured_fraction": round(captured_annual / total_emissions, 4) if total_emissions > 0 else 0.0,
                "utilization": round(min(utilization, 1.0), 4),
                "n_candidate_techs": int(len(candidates)),
                "candidate_techs": [
                    {"tech": t, "family": f, "installed": bool(inst), "total_emissions_t_yr": round(float(te), 1)}
                    for t, f, inst, te in zip(
                        candidates["tech"], candidates["family"], candidates["ccs_installed"], candidates["total_emissions"]
                    )
                ],
            })

    n_installed = int(real_df["ccs_installed"].sum())
    n_total = len(real_df)
    total_capture = float(real_df["captured_annual"].sum())

    family_counts = real_df[real_df["ccs_installed"]]["family"].value_counts().to_dict()

    return {
        "summary": {k: float(v) for k, v in summary.items()},
        "kpi": {
            "n_installed": n_installed,
            "n_total": n_total,
            "total_capture_t_yr": round(total_capture, 0),
            "network_capex_eur": summary["cost_capex_netws"],
        },
        "family_counts": family_counts,
        "cost_breakdown": {
            "capture": cost_breakdown["capture"],
            "transport": cost_breakdown["transport"],
            "storage": cost_breakdown["storage"],
            "capture_by_family": cost_breakdown["capture_by_family"],
            "total_stored_t": cost_breakdown["total_stored_t"],
        },
        "nodes": nodes,
        "arcs": arcs,
        "emitters": emitters,
    }


def main():
    italy = m.gpd.read_file(m.ITALY_SHP)
    generated_from = {}
    data = {
        "generated_from": generated_from,
        "scenario_labels": SCENARIO_LABELS,
        "italy_boundary": polygon_to_paths(italy.geometry.iloc[0]),
        "map_bounds": m.MAP_BOUNDS,
        "scenarios": {},
    }
    for scenario_name, objective in SCENARIOS:
        run_key = f"{scenario_name}_{objective}"
        try:
            h5_path = m.find_run_h5(scenario_name, objective)
        except FileNotFoundError as e:
            print(f"Building {run_key}... SKIPPED - {e}")
            continue
        print(f"Building {run_key} ({h5_path})...")
        generated_from[run_key] = str(h5_path)
        data["scenarios"][run_key] = build_scenario(h5_path)

    out_path = Path("dashboard_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, allow_nan=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
