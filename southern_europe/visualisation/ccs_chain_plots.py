"""
Conference-ready visualizations for the CCS chain optimization case study.

Produces five figures from a solved optimization_results.h5:

  1. ccs_chain_network_map.png       - Italy map: built CO2 transport network
                                        (pipeline/truck/railway), emitters colored
                                        by whether CCS was installed, transport
                                        hubs and the storage site.
  2. ccs_chain_emitter_zoom_<node>.png - captured vs. emitted CO2 for one
                                        CCS-equipped waste-to-energy plant.
  2b. ccs_chain_inflow_<node>.png     - hourly CO2 received at a node (e.g. a
                                        transport hub just upstream of storage).
  3. ccs_chain_cost_breakdown_per_tonne.png / _per_year.png
                                      - levelized cost of capture/transport/storage,
                                        each split into capex, opex (fixed/variable),
                                        electricity and heat. Carbon tax excluded.
  4. ccs_chain_summary_dashboard.png - headline numbers: CCS adoption, cost by
                                        chain stage, transport-mode split.
"""

import h5py
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString
from pathlib import Path
import cmcrameri.cm as cmc

# ============================================================
# Palette -- batlow-derived categorical palette used throughout
# ============================================================
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
LAND = "#f2f1ec"

# 7 evenly spaced samples from the actual cmcrameri "batlow" colormap
BATLOW = [cmc.batlow(x) for x in np.linspace(0, 1, 7)]

MODE_COLORS = {
    "CO2_Pipeline": BATLOW[0],  # dark blue
    "CO2Truck": BATLOW[2],      # green
    "CO2Railway": BATLOW[4],    # yellow
}
MODE_LABELS = {"CO2_Pipeline": "Pipeline", "CO2Truck": "Truck", "CO2Railway": "Railway"}

STATUS_GOOD = BATLOW[2]    # CCS installed
STATUS_MUTED = "#9a988f"   # no CCS (kept neutral -- absence, not a category)
STATUS_CRITICAL = BATLOW[5]

TRANSPORT_COLOR = BATLOW[1]  # transport hub marker
STORAGE_COLOR = BATLOW[6]    # storage marker

# CO2 captured vs. emitted -- used for the emitter zoom plot
CAPTURED_COLOR = BATLOW[0]
EMITTED_COLOR = BATLOW[4]

# Cost-component palette, shared by every stage (capture/transport/storage) so
# a given component (e.g. "Capex") reads as the same color everywhere.
COMPONENT_COLORS = {
    "Capex": BATLOW[0],
    "Opex (fixed)": BATLOW[1],
    "Opex (variable)": BATLOW[2],
    "Electricity": BATLOW[4],
    "Heat": BATLOW[5],
}
STAGE_LABELS = ["Capture", "Transport", "Storage"]

# ============================================================
# Paths
# ============================================================
RESULTS_H5 = Path("../Results_CCSchainOptimization/20260710104256-1/optimization_results.h5")

path_data_case_study = Path("../italy_data")
path_files_gis = path_data_case_study / "raw_data/gis_data"

GIS_NODES = path_files_gis / "all_nodes_italy.shp"
ITALY_SHP = path_files_gis / "italy_WGS1984.shp"
ROUTES = {
    "CO2_Pipeline": path_files_gis / "routes_distances_pipelines.shp",
    "CO2Truck": path_files_gis / "truck_italy_150.shp",
    "CO2Railway": path_files_gis / "routes_distances_railway.shp",
}
OUT_DIR = Path(__file__).resolve().parent


# ============================================================
# Data loading
# ============================================================
def load_built_arcs(h5_path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(h5_path, "r") as f:
        net = f["design"]["networks"]["period1"]
        for ntype in net.keys():
            for arc_name in net[ntype].keys():
                g = net[ntype][arc_name]
                rows.append(
                    {
                        "network": ntype,
                        "from": g["fromNode"][()].decode(),
                        "to": g["toNode"][()].decode(),
                        "size": float(g["size"][()]),
                        "total_flow": float(g["total_flow"][()]),
                        "capex": float(g["capex"][()]),
                    }
                )
    df = pd.DataFrame(rows)
    return df[df["size"] > 0].reset_index(drop=True)


def load_ccs_status(h5_path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(h5_path, "r") as f:
        nodes = f["design"]["nodes"]["period1"]
        for node_name in nodes.keys():
            size_ccs = 0.0
            capex_ccs = 0.0
            has_emitter_tech = False
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                keys = list(g.keys())
                if "size_ccs" in keys:
                    has_emitter_tech = True
                    size_ccs += float(g["size_ccs"][()][0])
                    capex_ccs += float(g["capex_ccs"][()][0])
            if has_emitter_tech:
                rows.append(
                    {
                        "node": node_name,
                        "size_ccs": size_ccs,
                        "capex_ccs": capex_ccs,
                        "ccs_installed": size_ccs > 0,
                    }
                )
    return pd.DataFrame(rows)


def load_summary(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        s = f["summary"]
        keys = [
            "cost_capex_tecs", "cost_capex_netws", "cost_opex_tecs",
            "cost_opex_netws", "cost_imports", "carbon_cost", "total_cost", "emissions_pos",
        ]
        return {k: float(s[k][()]) for k in keys}


def compute_cost_breakdown(h5_path: Path, storage_node: str = "Porto Corsini") -> dict:
    """
    Splits system cost into capture / transport / storage, each broken down by
    component (capex, opex fixed, opex variable, electricity, heat). Carbon tax
    is deliberately excluded -- this is the cost of running the CCS chain, not
    the cost of not running it.

    Electricity/heat import cost is not in technology opex (technology
    opex_variable is 0 for both the emitter and the MEA CCS component; energy
    is priced at the node's carrier balance instead -- see construct_balances.py
    :func:`construct_import_costs`). It is attributed to whichever node
    consumes it: the storage node's own electricity draw counts as "storage",
    everything else counts as "capture".

    Carrier-balance arrays (operation/energy_balance) are stored at the
    design-days (clustered) resolution; they are expanded back to the full
    8760-hour year via k_means_specs/sequence before summing, exactly like the
    model's own full-resolution linking constraint does.

    :return: dict with 'capture', 'transport', 'storage' (each a dict of
        component -> EUR/year) and 'total_stored_t' (t CO2 stored per year).
    """
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]

        def carrier_import_cost(node_name, carrier):
            eb = f["operation"]["energy_balance"]["period1"]
            if node_name not in eb or carrier not in eb[node_name]:
                return 0.0
            imp = eb[node_name][carrier]["import"][()][seq - 1]
            price = eb[node_name][carrier]["import_price"][()][seq - 1]
            return float((imp * price).sum())

        nodes = f["design"]["nodes"]["period1"]
        capture = {k: 0.0 for k in COMPONENT_COLORS}
        storage = {k: 0.0 for k in COMPONENT_COLORS}

        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                keys = list(g.keys())
                if "size_ccs" in keys:
                    capture["Capex"] += float(g["capex_ccs"][()][0])
                    capture["Opex (fixed)"] += float(g["opex_fixed_ccs"][()][0])
                    capture["Opex (variable)"] += float(g["opex_variable_ccs"][()][0])
                elif tech == "PermanentStorage_CO2_simple":
                    storage["Capex"] += float(g["capex_tot"][()][0])
                    storage["Opex (fixed)"] += float(g["opex_fixed"][()][0])
                    storage["Opex (variable)"] += float(g["opex_variable"][()][0])

            elec_cost = carrier_import_cost(node_name, "electricity")
            heat_cost = carrier_import_cost(node_name, "heat")
            if node_name == storage_node:
                storage["Electricity"] += elec_cost
                storage["Heat"] += heat_cost
            else:
                capture["Electricity"] += elec_cost
                capture["Heat"] += heat_cost

        net = f["design"]["networks"]["period1"]
        transport = {k: 0.0 for k in COMPONENT_COLORS}
        for ntype in net.keys():
            for arc_name in net[ntype].keys():
                g = net[ntype][arc_name]
                if float(g["size"][()]) <= 0:
                    continue
                transport["Capex"] += float(g["capex"][()])
                of = g["opex_fixed"][()]
                transport["Opex (fixed)"] += float(of[0]) if hasattr(of, "__len__") else float(of)
                ov = g["opex_variable"][()]
                transport["Opex (variable)"] += float(ov[0]) if hasattr(ov, "__len__") else float(ov)

        co2_in = f["operation"]["technology_operation"]["period1"][storage_node][
            "PermanentStorage_CO2_simple"
        ]["CO2captured_input"][()][seq - 1]
        total_stored_t = float(co2_in.sum())

    return {
        "capture": capture,
        "transport": transport,
        "storage": storage,
        "total_stored_t": total_stored_t,
    }


def _oriented_coords(geom, from_point):
    """Return line coords ordered so the first point is nearest from_point."""
    coords = list(geom.coords)
    coords = [(x, y) for x, y, *_ in coords]
    start, end = coords[0], coords[-1]
    d_start = (start[0] - from_point.x) ** 2 + (start[1] - from_point.y) ** 2
    d_end = (end[0] - from_point.x) ** 2 + (end[1] - from_point.y) ** 2
    if d_end < d_start:
        coords = coords[::-1]
    return coords


def _point_and_tangent_at_fraction(coords, fraction):
    cum = [0.0]
    total = 0.0
    for i in range(1, len(coords)):
        d = ((coords[i][0] - coords[i - 1][0]) ** 2 + (coords[i][1] - coords[i - 1][1]) ** 2) ** 0.5
        total += d
        cum.append(total)
    if total == 0:
        return coords[0], (1.0, 0.0)
    target = fraction * total
    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg_start, seg_end = coords[i - 1], coords[i]
            seg_len = cum[i] - cum[i - 1]
            seg_frac = (target - cum[i - 1]) / seg_len if seg_len > 0 else 0.0
            point = (
                seg_start[0] + seg_frac * (seg_end[0] - seg_start[0]),
                seg_start[1] + seg_frac * (seg_end[1] - seg_start[1]),
            )
            dx = seg_end[0] - seg_start[0]
            dy = seg_end[1] - seg_start[1]
            return point, (dx, dy)
    dx = coords[-1][0] - coords[-2][0]
    dy = coords[-1][1] - coords[-2][1]
    return coords[-1], (dx, dy)


def attach_route_geometries(built_arcs: pd.DataFrame, nodes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Match each built arc to its real route geometry (pipeline/truck/railway
    corridor), falling back to a straight line if no geometry is found."""
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_id = dict(zip(nodes_unique["node_name"], nodes_unique["node_id"]))
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))

    route_lookup = {}
    for ntype, route_path in ROUTES.items():
        route_gdf = gpd.read_file(route_path).to_crs(nodes_gdf.crs)
        pair_to_geom = {}
        for _, row in route_gdf.iterrows():
            parts = str(row["Node"]).strip().split(",")
            if len(parts) == 2:
                try:
                    a, b = int(parts[0]), int(parts[1])
                    pair_to_geom[frozenset((a, b))] = row.geometry
                except ValueError:
                    continue
        route_lookup[ntype] = pair_to_geom

    geometries = []
    for _, r in built_arcs.iterrows():
        from_id, to_id = name_to_id.get(r["from"]), name_to_id.get(r["to"])
        geom = None
        if from_id is not None and to_id is not None:
            geom = route_lookup.get(r["network"], {}).get(frozenset((int(from_id), int(to_id))))
        if geom is None:
            p1, p2 = name_to_point.get(r["from"]), name_to_point.get(r["to"])
            if p1 is not None and p2 is not None:
                geom = LineString([p1, p2])
        geometries.append(geom)

    built_arcs = built_arcs.copy()
    built_arcs["geometry"] = geometries
    return built_arcs


# ============================================================
# PLOT 1 - Main network map
# ============================================================
def plot_main_map(built_arcs, nodes_gdf, ccs_df, summary):
    italy = gpd.read_file(ITALY_SHP)
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))
    ccs_map = dict(zip(ccs_df["node"], ccs_df["ccs_installed"]))

    fig, ax = plt.subplots(figsize=(12.5, 12))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    italy.plot(ax=ax, color=LAND, alpha=0.9, zorder=0)
    italy.boundary.plot(ax=ax, color=INK_SECONDARY, linewidth=1, alpha=0.8, zorder=1)

    # --- routes, drawn truck/rail first so pipeline (usually dominant) sits on top ---
    draw_order = ["CO2Truck", "CO2Railway", "CO2_Pipeline"]
    max_size_by_net = built_arcs.groupby("network")["size"].max().to_dict()

    for ntype in draw_order:
        sub = built_arcs[built_arcs["network"] == ntype]
        color = MODE_COLORS[ntype]
        max_size = max_size_by_net.get(ntype, 1) or 1
        for _, r in sub.iterrows():
            geom = r["geometry"]
            if geom is None:
                continue
            from_point = name_to_point.get(r["from"])
            coords = _oriented_coords(geom, from_point) if from_point else list(geom.coords)
            lw = 1.3 + 2.7 * (r["size"] / max_size)
            gpd.GeoSeries([LineString(coords)]).plot(ax=ax, color=color, linewidth=lw, alpha=0.9, zorder=5)

            point, (dx, dy) = _point_and_tangent_at_fraction(coords, 0.55)
            norm = (dx ** 2 + dy ** 2) ** 0.5
            if norm > 0:
                ux, uy = dx / norm, dy / norm
                arrow_len = 0.10
                start = (point[0] - ux * arrow_len / 2, point[1] - uy * arrow_len / 2)
                end = (point[0] + ux * arrow_len / 2, point[1] + uy * arrow_len / 2)
                ax.annotate(
                    "", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=13),
                    zorder=6,
                )

    # --- nodes ---
    for _, row in nodes_unique.iterrows():
        name, ntype_raw, point = row["node_name"], row["node_type"], row.geometry
        if ntype_raw == "Storage":
            ax.scatter(point.x, point.y, marker="*", s=480, color=STORAGE_COLOR,
                       edgecolor="white", linewidth=1.4, zorder=25)
        elif ntype_raw == "Transport":
            ax.scatter(point.x, point.y, marker="s", s=100, color=TRANSPORT_COLOR,
                       edgecolor="white", linewidth=1.2, zorder=20)
        else:
            installed = ccs_map.get(name, False)
            if installed:
                ax.scatter(point.x, point.y, marker="o", s=100, color=STATUS_GOOD,
                           edgecolor="white", linewidth=1.2, zorder=22)
            else:
                ax.scatter(point.x, point.y, marker="o", s=90, facecolors="none",
                           edgecolor=STATUS_MUTED, linewidth=1.8, zorder=21)

    minx, miny, maxx, maxy = nodes_unique.total_bounds
    pad = 0.4
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_axis_off()
    ax.set_title(
        "Optimized CO$_2$ Capture, Transport & Storage Network — Northern Italy",
        fontsize=16, weight="bold", color=INK_PRIMARY, pad=16,
    )

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STATUS_GOOD, markeredgecolor="white",
               markersize=11, label="Emitter — CCS installed", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=STATUS_MUTED,
               markersize=11, markeredgewidth=1.8, label="Emitter — no CCS", linestyle="None"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
    )

    n_installed = int(ccs_df["ccs_installed"].sum())
    n_total = len(ccs_df)
    total_capture = ccs_df["size_ccs"].sum()
    kpi_text = (
        f"{n_installed}/{n_total} emitters equipped with CCS\n"
        f"{total_capture:,.0f} t/h captured CO$_2$ capacity\n"
        f"Network capex: €{summary['cost_capex_netws'] / 1e6:,.0f}M"
    )
    ax.text(
        0.02, 0.02, kpi_text, transform=ax.transAxes, fontsize=11, color=INK_PRIMARY,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=GRIDLINE, alpha=0.92),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_file = OUT_DIR / "ccs_chain_network_map.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 2 - Emitter zoom-in: captured vs. emitted CO2
# ============================================================
def plot_emitter_zoom(h5_path: Path, node_name: str = "SILLA 2", tech_name: str = "WasteToEnergyEmitter"):
    with h5py.File(h5_path, "r") as f:
        op = f["operation"]["technology_operation"]["period1"][node_name][tech_name]
        captured = op["CO2captured_var_output_ccs"][()]
        emitted = op["emissions_pos"][()]
        seq = f["k_means_specs"]["period1"]["sequence"][()]

    captured_full = captured[seq - 1]
    emitted_full = emitted[seq - 1]
    hours = np.arange(len(captured_full))

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(hours, 0, captured_full, color=CAPTURED_COLOR, alpha=0.9, linewidth=0,
                     label="Captured CO$_2$", zorder=3)
    ax.fill_between(hours, captured_full, captured_full + emitted_full, color=EMITTED_COLOR, alpha=0.9,
                     linewidth=0, label="Emitted CO$_2$", zorder=3)

    ax.set_xlim(0, hours[-1])
    ax.set_ylim(0, (captured_full + emitted_full).max() * 1.1)
    ax.set_xlabel("Hours [h]", fontsize=11)
    ax.set_ylabel("CO$_2$ rate (t/h)", fontsize=11)
    ax.set_title(f"{node_name} — captured vs. emitted CO$_2$", fontsize=14, weight="bold", color=INK_PRIMARY)
    ax.grid(True, alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True,
              fontsize=10.5, edgecolor=GRIDLINE)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_file = OUT_DIR / f"ccs_chain_emitter_zoom_{node_name.replace(' ', '_')}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 2b - CO2 received at a node (hourly)
# ============================================================
def plot_node_inflow(h5_path: Path, node_name: str = "Eni S.p.A Casalborsetti"):
    """CO2 arriving at a node via the network (network_inflow on the
    CO2captured carrier balance), as an hourly profile."""
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        inflow_clustered = f["operation"]["energy_balance"]["period1"][node_name][
            "CO2captured"
        ]["network_inflow"][()]

    inflow_full = inflow_clustered[seq - 1]
    hours = np.arange(len(inflow_full))

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(hours, inflow_full, color=BATLOW[0], linewidth=0.6)
    ax.set_xlim(0, hours[-1])
    ax.set_ylim(0, inflow_full.max() * 1.1)
    ax.set_xlabel("Hours [h]", fontsize=11)
    ax.set_ylabel("CO$_2$ received (t/h)", fontsize=11)
    ax.set_title(f"CO$_2$ received at {node_name}", fontsize=14, weight="bold", color=INK_PRIMARY)

    ax.grid(True, alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)

    fig.tight_layout()
    out_file = OUT_DIR / f"ccs_chain_inflow_{node_name.replace(' ', '_').replace('.', '')}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 3 - Levelized cost breakdown: capture / transport / storage
# ============================================================
def plot_cost_breakdown(cost_breakdown: dict, per_tonne: bool = True):
    """Stacked bar of capture/transport/storage cost by component. Carbon tax
    is excluded on purpose -- this is the cost of running the chain, not the
    cost of not running it."""
    stages = {
        "Capture": cost_breakdown["capture"],
        "Transport": cost_breakdown["transport"],
        "Storage": cost_breakdown["storage"],
    }
    divisor = cost_breakdown["total_stored_t"] if per_tonne else 1e6
    unit = "€/t CO$_2$ stored" if per_tonne else "Million €/year"

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(stages))
    bottoms = np.zeros(len(stages))
    stage_totals = np.array([sum(v.values()) for v in stages.values()]) / divisor

    for component, color in COMPONENT_COLORS.items():
        heights = np.array([stages[s][component] for s in stages]) / divisor
        bars = ax.bar(x, heights, bottom=bottoms, color=color, width=0.55,
                      label=component, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        # direct-label segments that are large enough to read
        for i, h in enumerate(heights):
            if h / stage_totals[i] > 0.06:
                ax.text(x[i], bottoms[i] + h / 2, f"{h:,.1f}" if per_tonne else f"{h:,.0f}",
                        ha="center", va="center", fontsize=9, color="white", weight="bold", zorder=4)
        bottoms += heights

    for i, total in enumerate(stage_totals):
        ax.text(x[i], total * 1.02, f"{total:,.1f}" if per_tonne else f"€{total:,.0f}M",
                ha="center", va="bottom", fontsize=12.5, weight="bold", color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(list(stages.keys()), fontsize=12)
    ax.set_ylabel(unit, fontsize=11.5)
    ax.set_ylim(0, stage_totals.max() * 1.18)
    ax.set_title(
        "Levelized cost of the CO$_2$ chain — capture, transport & storage\n"
        "(carbon tax excluded)",
        fontsize=14.5, weight="bold", color=INK_PRIMARY,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=True,
              fontsize=10, edgecolor=GRIDLINE)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    suffix = "per_tonne" if per_tonne else "per_year"
    out_file = OUT_DIR / f"ccs_chain_cost_breakdown_{suffix}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 4 - Summary dashboard
# ============================================================
def plot_summary_dashboard(built_arcs: pd.DataFrame, ccs_df: pd.DataFrame, cost_breakdown: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    fig.patch.set_facecolor(SURFACE)

    # Panel A: CCS adoption
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    n_installed = int(ccs_df["ccs_installed"].sum())
    n_not = len(ccs_df) - n_installed
    bars = ax.bar(["CCS\ninstalled", "No\nCCS"], [n_installed, n_not],
                   color=[STATUS_GOOD, STATUS_MUTED], width=0.55)
    for rect, v in zip(bars, [n_installed, n_not]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, str(v),
                ha="center", fontsize=12, color=INK_PRIMARY, weight="bold")
    ax.set_title("CCS adoption across emitters", fontsize=12.5, weight="bold", color=INK_PRIMARY)
    ax.set_ylabel("Number of emitters", fontsize=10.5)
    ax.set_ylim(0, max(n_installed, n_not) * 1.25)

    # Panel B: cost breakdown by chain stage (carbon tax excluded -- see the
    # dedicated ccs_chain_cost_breakdown_* figures for the component-level split)
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    cats = STAGE_LABELS
    vals = [
        sum(cost_breakdown["capture"].values()),
        sum(cost_breakdown["transport"].values()),
        sum(cost_breakdown["storage"].values()),
    ]
    colors = ["#4a3aa7", MODE_COLORS["CO2_Pipeline"], STORAGE_COLOR]
    bars = ax.bar(cats, [v / 1e6 for v in vals], color=colors, width=0.55)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v / 1e6 * 1.02, f"€{v / 1e6:,.0f}M",
                ha="center", fontsize=10.5, color=INK_PRIMARY)
    ax.set_title("Cost by chain stage (excl. carbon tax)", fontsize=12.5, weight="bold", color=INK_PRIMARY)
    ax.set_ylabel("Million €/year", fontsize=10.5)
    ax.set_ylim(0, max(vals) / 1e6 * 1.25)

    # Panel C: transport mode split
    ax = axes[2]
    ax.set_facecolor(SURFACE)
    mode_counts = built_arcs.groupby("network").size()
    modes = ["CO2_Pipeline", "CO2Truck", "CO2Railway"]
    counts = [int(mode_counts.get(m, 0)) for m in modes]
    colors = [MODE_COLORS[m] for m in modes]
    labels = [MODE_LABELS[m] for m in modes]
    bars = ax.bar(labels, counts, color=colors, width=0.55)
    for rect, v in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, v + max(counts) * 0.02, str(v),
                ha="center", fontsize=12, color=INK_PRIMARY, weight="bold")
    ax.set_title("Built connections by transport mode", fontsize=12.5, weight="bold", color=INK_PRIMARY)
    ax.set_ylabel("Number of arcs built", fontsize=10.5)
    ax.set_ylim(0, max(counts) * 1.25 if counts else 1)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRIDLINE)
        ax.spines["bottom"].set_color(GRIDLINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=10)
        ax.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("CCS Chain Optimization — Summary", fontsize=16, weight="bold", y=1.05, color=INK_PRIMARY)
    fig.tight_layout()
    out_file = OUT_DIR / "ccs_chain_summary_dashboard.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# Main
# ============================================================
def main():
    nodes_gdf = gpd.read_file(GIS_NODES)

    built_arcs = load_built_arcs(RESULTS_H5)
    built_arcs = attach_route_geometries(built_arcs, nodes_gdf)

    ccs_df = load_ccs_status(RESULTS_H5)
    summary = load_summary(RESULTS_H5)
    cost_breakdown = compute_cost_breakdown(RESULTS_H5)

    plot_main_map(built_arcs, nodes_gdf, ccs_df, summary)
    plot_emitter_zoom(RESULTS_H5, node_name="SILLA 2", tech_name="WasteToEnergyEmitter")
    plot_node_inflow(RESULTS_H5, node_name="Eni S.p.A Casalborsetti")
    plot_cost_breakdown(cost_breakdown, per_tonne=True)
    plot_cost_breakdown(cost_breakdown, per_tonne=False)
    plot_summary_dashboard(built_arcs, ccs_df, cost_breakdown)


if __name__ == "__main__":
    main()
