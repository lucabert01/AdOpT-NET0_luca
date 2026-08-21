"""
Ranks every CO2-capturing emitter (technology, really - see below) in a solved
main_italy.py case study by its total levelized cost per tonne of CO2
(capture + transport + storage), and plots it as a horizontal ranking bar
chart.

One row per (node, technology), not per node - a node can host more than one
emitter technology at once (e.g. "Piacenza" runs both a CementEmitter and a
WasteToEnergyEmitter with their own separate CCS retrofits), and each is its
own cost/capture accounting unit. Rows are labelled "<node> (<sector>)" when
a node has more than one.

Capture cost is each technology's own capture cost (capex + opex + its own
electricity/heat draw, attributed via that technology's own consumption
series x the node's import price - exact, since node-level import cost is
just the sum of every technology's own draw at that node) divided by its own
annual captured CO2.

Storage cost is the single storage site's total annual cost divided by total
annual tonnes stored - a flat €/t added to every emitter, since storage is a
shared, non-allocable resource.

Transport cost is allocated by CAPACITY SHARE, not by flow share: for every
arc on an emitter's path to the storage node, the emitter pays
    arc's annual cost x (emitter's own max captured CO2, t/h) / (arc's built size, t/h)
i.e. a 15 t/h emitter on a 150 t/h pipeline pays 10% of that arc's annual
cost; on a 15 t/h pipeline it pays 100%. This is summed over every arc on the
emitter's path (not just the first arc out of its node) and then divided by
the emitter's own annual captured tonnes. Two emitters with identical paths
and capture technology therefore end up with different €/t if one runs at a
lower load factor (same nameplate share of the pipeline cost, spread over
fewer actual tonnes) - which is the point: a poorly-utilized reservation
costs more per tonne, same as an oversized capture unit does.

Output:
  - ccs_chain_emitter_cost_ranking.png  (ranked stacked-bar chart)
  - ccs_chain_emitter_cost_ranking.csv  (per-emitter table backing the chart)
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

import cmcrameri.cm as cmc

from ccs_chain_plots import (
    classify_capture_family,
    captured_co2_operation_key,
    FAMILY_LABELS,
    INK_PRIMARY,
    INK_SECONDARY,
    GRIDLINE,
    SURFACE,
    MODE_COLORS,
    STORAGE_COLOR,
)

# ============================================================
# Paths
# ============================================================
# Point this at whichever solved case study you want to rank. Defaults to the
# most recently solved "mea" scenario run under Results_CCSchainOptimization/.
RESULTS_H5 = Path(
    "../Results_CCSchainOptimization/mea/20260820183756_emissions_minC_mea-1/optimization_results.h5"
)
OUT_DIR = Path(__file__).resolve().parent

STAGE_COLORS = {
    "Capture": "#4a3aa7",
    "Transport": MODE_COLORS["CO2_Pipeline"],
    "Storage": STORAGE_COLOR,
}

SECTOR_ORDER = ["Cement", "Waste", "Refining", "Other"]
_BATLOW = [cmc.batlow(x) for x in np.linspace(0, 1, 7)]
SECTOR_COLORS = {
    "Cement": _BATLOW[0],
    "Waste": _BATLOW[2],
    "Refining": _BATLOW[4],
    "Other": _BATLOW[6],
}

# Below this annual-tonnes / (peak-rate x 8760h) ratio, an emitter is flagged
# on the MACC plot as running at a low load factor - it reserves pipeline
# capacity sized for its peak rate but rarely uses all of it, so it pays a
# capacity-share transport cost spread over comparatively few actual tonnes.
LOW_CAPACITY_FACTOR_THRESHOLD = 0.75

# Each capture technology family reports its own electricity/heat draw under
# a different dataset name (see classify_capture_family in ccs_chain_plots.py
# for why: generic bolt-on MEA retrofit vs. the two self-contained
# technologies). WasteCaL_CCS has neither - the calcium-looping process is
# self-sufficient (no purchased electricity/heat operation variable at all).
ELEC_INPUT_KEY_BY_FAMILY = {
    "mea_retrofit": "electricity_var_input_ccs",
    "oxyfuel_hybrid": "electricity_input",
    "calcium_looping": None,
}
HEAT_INPUT_KEY_BY_FAMILY = {
    "mea_retrofit": "heat_var_input_ccs",
    "oxyfuel_hybrid": None,
    "calcium_looping": None,
}


def _sector_from_tech(tech_name: str) -> str:
    t = tech_name.lower()
    if "cement" in t:
        return "Cement"
    if "waste" in t:
        return "Waste"
    if "refin" in t:
        return "Refining"
    return "Other"


# ============================================================
# Data loading / cost allocation
# ============================================================
def build_emitter_cost_table(h5_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, storage_node). df has one row per (node, technology) with
    CCS actually running (annual captured CO2 > 0), columns: node, tech,
    sector, family, captured_annual_t, max_captured_t_h, capture_eur_per_t,
    transport_eur_per_t, storage_eur_per_t, total_eur_per_t - sorted
    descending by total_eur_per_t.
    """
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        eb = f["operation"]["energy_balance"]["period1"]

        def expand(dataset):
            """Clustered-typical-days series -> true annual total. HDF5 only
            stores the representative-day hours; every per-timestep series
            (flows, emissions, imports) has to be re-expanded to the full
            8760h year via k_means_specs/sequence before summing, or totals
            come out far too low (off by the day-expansion factor)."""
            return float(dataset[()][seq - 1].sum())

        def node_carrier_cost(node_name, carrier):
            if node_name not in eb or carrier not in eb[node_name]:
                return 0.0
            imp = eb[node_name][carrier]["import"][()][seq - 1]
            price = eb[node_name][carrier]["import_price"][()][seq - 1]
            return float((imp * price).sum())

        def tech_carrier_cost(node_name, tech_name, op_keys, key, carrier):
            """Same node-level import price, but only this technology's own
            consumption - exact, not an approximation, since the node-level
            import is just the sum of every technology's own draw there."""
            if key is None or key not in op_keys or node_name not in eb or carrier not in eb[node_name]:
                return 0.0
            consumption = op_tech[node_name][tech_name][key][()]
            price = eb[node_name][carrier]["import_price"][()]
            return float((consumption * price)[seq - 1].sum())

        nodes = f["design"]["nodes"]["period1"]
        op_tech = f["operation"]["technology_operation"]["period1"]

        storage_node, storage_tech = None, None
        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                if "PermanentStorage" in tech:
                    storage_node, storage_tech = node_name, tech
        if storage_node is None:
            raise RuntimeError("No PermanentStorage technology found in design/nodes")

        # ---- per-(node, technology) capture cost, captured tonnes, and peak rate ----
        capture_rows = {}
        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                family = classify_capture_family(list(g.keys()))
                if family is None:
                    continue
                op_keys = list(op_tech[node_name][tech].keys())
                captured_key = captured_co2_operation_key(op_keys)
                if captured_key is None:
                    continue
                captured_series = op_tech[node_name][tech][captured_key][()]
                captured_annual = expand(op_tech[node_name][tech][captured_key])
                if captured_annual <= 1e-6:
                    continue  # CCS technology present in the design but not actually run
                max_captured_t_h = float(np.max(captured_series))

                if family == "mea_retrofit":
                    capex = float(g["capex_ccs"][()][0])
                    opex_fixed = float(g["opex_fixed_ccs"][()][0])
                    opex_variable = float(g["opex_variable_ccs"][()][0])
                else:
                    capex = float(g["capex_tot"][()][0])
                    opex_fixed = float(g["opex_fixed"][()][0])
                    opex_variable = float(g["opex_variable"][()][0])

                elec_cost = tech_carrier_cost(
                    node_name, tech, op_keys, ELEC_INPUT_KEY_BY_FAMILY[family], "electricity"
                )
                heat_cost = tech_carrier_cost(
                    node_name, tech, op_keys, HEAT_INPUT_KEY_BY_FAMILY[family], "heat"
                )

                capture_cost = capex + opex_fixed + opex_variable + elec_cost + heat_cost
                capture_rows[(node_name, tech)] = {
                    "family": family,
                    "sector": _sector_from_tech(tech),
                    "captured_annual_t": captured_annual,
                    "max_captured_t_h": max_captured_t_h,
                    "capture_eur_per_t": capture_cost / captured_annual,
                }

        # ---- storage cost: one flat €/t added to every emitter ----
        g_store = nodes[storage_node][storage_tech]
        storage_cost = (
            float(g_store["capex_tot"][()][0])
            + float(g_store["opex_fixed"][()][0])
            + float(g_store["opex_variable"][()][0])
            + node_carrier_cost(storage_node, "electricity")
            + node_carrier_cost(storage_node, "heat")
        )
        total_stored_t = expand(op_tech[storage_node][storage_tech]["CO2captured_input"])
        storage_eur_per_t = storage_cost / total_stored_t

        # ---- built arcs: annual cost, true annual flow, and built size (t/h) ----
        # design/networks/.../total_flow is NOT annual - it's the raw
        # unweighted sum over the clustered representative hours (same
        # pitfall as emissions_pos, see ccs_chain_plots.py load_ccs_status).
        # Re-derive the real annual flow from operation/networks/.../flow -
        # used here only to pick each node's dominant downstream arc, not for
        # cost allocation itself (that's by capacity share, see module docstring).
        net_design = f["design"]["networks"]["period1"]
        net_op = f["operation"]["networks"]["period1"]
        edge_rows = []
        for ntype in net_design.keys():
            for arc_name in net_design[ntype].keys():
                gd = net_design[ntype][arc_name]
                size = float(gd["size"][()])
                if size <= 0:
                    continue
                of = gd["opex_fixed"][()]
                ov = gd["opex_variable"][()]
                opex_fixed = float(of[0]) if hasattr(of, "__len__") else float(of)
                opex_variable = float(ov[0]) if hasattr(ov, "__len__") else float(ov)
                annual_cost = float(gd["capex"][()]) + opex_fixed + opex_variable

                annual_flow = expand(net_op[ntype][arc_name]["flow"])
                if annual_flow <= 0:
                    continue

                edge_rows.append(
                    {
                        "from": gd["fromNode"][()].decode(),
                        "to": gd["toNode"][()].decode(),
                        "cost": annual_cost,
                        "flow": annual_flow,
                        "size": size,
                    }
                )

        # collapse parallel arcs on the same physical corridor (e.g. a route
        # built as two pipeline size classes at once) into one cost/size pool
        # - "the pipeline's size" for capacity-share purposes is the whole
        # corridor's combined built capacity, however many classes make it up
        combined = (
            pd.DataFrame(edge_rows)
            .groupby(["from", "to"], as_index=False)[["cost", "flow", "size"]]
            .sum()
        )
        out_edges: dict[str, list[tuple[str, float, float, float]]] = {}
        for _, r in combined.iterrows():
            out_edges.setdefault(r["from"], []).append((r["to"], r["cost"], r["flow"], r["size"]))

        # ---- deterministic path to storage: dominant (highest-flow) arc out
        # of every node. Real branching (a node's flow genuinely split across
        # more than one built corridor) is rare in this network - it's
        # essentially a tree converging on the storage node - but flag it
        # if a secondary arc carries more than 1% of the dominant one, since
        # the capacity-share allocation below only follows the dominant arc.
        path_cache: dict[str, list[tuple[str, str, float, float]] | None] = {}

        def path_to_storage(start_node):
            if start_node in path_cache:
                return path_cache[start_node]
            path, current, visited = [], start_node, set()
            while current != storage_node:
                if current in visited:
                    raise RuntimeError(f"Cycle detected in built network reaching '{current}'")
                visited.add(current)
                edges_out = out_edges.get(current)
                if not edges_out:
                    path_cache[start_node] = None
                    return None
                edges_out = sorted(edges_out, key=lambda e: e[2], reverse=True)
                to_node, cost, flow, size = edges_out[0]
                if len(edges_out) > 1 and edges_out[1][2] > 0.01 * flow:
                    print(
                        f"Warning: '{current}' splits flow across multiple built arcs "
                        f"(dominant {flow:,.0f} t/yr -> '{to_node}', secondary "
                        f"{edges_out[1][2]:,.0f} t/yr -> '{edges_out[1][0]}'); only the "
                        f"dominant arc is used for capacity-share allocation downstream."
                    )
                path.append((current, to_node, cost, size))
                current = to_node
            path_cache[start_node] = path
            return path

        rows = []
        for (node_name, tech), cap in capture_rows.items():
            path = [] if node_name == storage_node else path_to_storage(node_name)
            if path is None:
                print(f"Warning: '{node_name}' ({tech}) captures CO2 but has no built path to '{storage_node}' - excluded")
                continue

            transport_cost_eur = 0.0
            for _, _, arc_cost, arc_size_t_h in path:
                share = min(1.0, cap["max_captured_t_h"] / arc_size_t_h) if arc_size_t_h > 0 else 0.0
                transport_cost_eur += arc_cost * share
            transport_eur_per_t = transport_cost_eur / cap["captured_annual_t"]

            rows.append(
                {
                    "node": node_name,
                    "tech": tech,
                    "sector": cap["sector"],
                    "family": FAMILY_LABELS.get(cap["family"], cap["family"]),
                    "captured_annual_t": cap["captured_annual_t"],
                    "max_captured_t_h": cap["max_captured_t_h"],
                    "capacity_factor": cap["captured_annual_t"] / (cap["max_captured_t_h"] * 8760),
                    "capture_eur_per_t": cap["capture_eur_per_t"],
                    "transport_eur_per_t": transport_eur_per_t,
                    "storage_eur_per_t": storage_eur_per_t,
                    "total_eur_per_t": cap["capture_eur_per_t"] + transport_eur_per_t + storage_eur_per_t,
                }
            )

        df = pd.DataFrame(rows).sort_values("total_eur_per_t", ascending=False).reset_index(drop=True)
        return df, storage_node


# ============================================================
# Plot
# ============================================================
def plot_emitter_cost_ranking(df: pd.DataFrame, storage_node: str):
    n = len(df)
    plot_df = df.iloc[::-1]  # highest cost at the top of the chart

    node_counts = df["node"].value_counts()
    show_family = df["family"].nunique() > 1

    def make_label(row):
        parts = []
        if node_counts[row["node"]] > 1:
            parts.append(row["sector"])
        if show_family:
            parts.append(row["family"])
        return f"{row['node']} ({', '.join(parts)})" if parts else row["node"]

    labels = plot_df.apply(make_label, axis=1)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * n)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = np.arange(n)
    left = np.zeros(n)
    for stage, col in [
        ("Storage", "storage_eur_per_t"),
        ("Transport", "transport_eur_per_t"),
        ("Capture", "capture_eur_per_t"),
    ]:
        vals = plot_df[col].to_numpy()
        ax.barh(y, vals, left=left, color=STAGE_COLORS[stage], label=stage, height=0.68, zorder=3)
        left += vals

    for i, total in enumerate(plot_df["total_eur_per_t"]):
        ax.text(total + left.max() * 0.01, i, f"{total:,.0f}", va="center", fontsize=8.5, color=INK_PRIMARY)

    # Extra right-hand padding guarantees a bar-free column (no bar ever
    # reaches past left.max()) to anchor the legend/KPI box in, regardless of
    # how the cost distribution happens to look for this particular run.
    ax.set_xlim(0, left.max() * 1.35)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel("€/t CO$_2$ (storage + transport + capture)", fontsize=11)
    ax.set_title(
        f"Emitter Cost Ranking — Levelized €/t CO$_2$ to {storage_node}",
        fontsize=14, weight="bold", color=INK_PRIMARY,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis="x", alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    # Both sit in the guaranteed bar-free right-hand padding column (see
    # set_xlim above), stacked vertically - safe regardless of which bar
    # happens to be longest for this particular run.
    ax.legend(
        loc="upper right", bbox_to_anchor=(0.99, 0.98), frameon=True,
        fontsize=10, framealpha=0.95, edgecolor=GRIDLINE,
    )

    weighted_avg = np.average(df["total_eur_per_t"], weights=df["captured_annual_t"])
    kpi_text = (
        f"{n} emitters with CCS running\n"
        f"{df['captured_annual_t'].sum():,.0f} t/yr captured\n"
        f"Capture-weighted average: €{weighted_avg:,.0f}/t"
    )
    ax.text(
        0.99, 0.80, kpi_text, transform=ax.transAxes, fontsize=10, color=INK_PRIMARY,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRIDLINE, alpha=0.92),
    )

    fig.tight_layout()
    out_file = OUT_DIR / "ccs_chain_emitter_cost_ranking.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_macc(df: pd.DataFrame, storage_node: str):
    """
    A MACC-style (marginal abatement cost curve) view of the same per-emitter
    costs: bars sorted left-to-right by ascending total €/t, bar WIDTH is that
    emitter's own annual captured tonnes (so cheap, high-volume tonnes read as
    wide-and-low, expensive/small ones as narrow-and-tall), bar HEIGHT is
    total €/t, and color is sector rather than cost stage - individual
    emitter names are dropped, this is about the shape of the system-wide
    cost curve, not any one plant.

    Emitters running at a low load factor (annual tonnes far below what
    their own peak captured rate x 8760h would allow) are flagged - see
    module docstring: they still pay their full nameplate share of every
    downstream arc's cost, so a low load factor inflates their €/t just like
    an oversized capture unit would.
    """
    plot_df = df.sort_values("total_eur_per_t").reset_index(drop=True)
    captured_mt = (plot_df["captured_annual_t"] / 1e6).to_numpy()
    left = np.concatenate([[0.0], np.cumsum(captured_mt)[:-1]])
    heights = plot_df["total_eur_per_t"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, row in plot_df.iterrows():
        ax.bar(
            left[i], heights[i], width=captured_mt[i], align="edge",
            color=SECTOR_COLORS[row["sector"]], edgecolor="white", linewidth=0.5, zorder=3,
        )

    sectors_present = [s for s in SECTOR_ORDER if s in set(plot_df["sector"])]
    legend_handles = [Patch(facecolor=SECTOR_COLORS[s], edgecolor="white", label=s) for s in sectors_present]
    ax.legend(
        handles=legend_handles, loc="upper left", frameon=True, fontsize=10.5,
        framealpha=0.95, edgecolor=GRIDLINE, title="Sector", title_fontsize=10.5,
    )

    # --- flag low-load-factor emitters with an arrow, ordered left-to-right
    # and stacked upward so nearby bars' callouts don't collide ---
    low_cf_idx = [i for i in plot_df.index if plot_df.loc[i, "capacity_factor"] < LOW_CAPACITY_FACTOR_THRESHOLD]
    low_cf_idx.sort(key=lambda i: left[i])

    y_top = heights.max()
    for rank, i in enumerate(low_cf_idx):
        cx = left[i] + captured_mt[i] / 2
        cy = heights[i]
        cf_pct = plot_df.loc[i, "capacity_factor"] * 100
        ax.annotate(
            f"Low capacity factor ({cf_pct:.0f}%)",
            xy=(cx, cy), xycoords="data",
            xytext=(cx, y_top * 1.10 + rank * y_top * 0.09), textcoords="data",
            ha="center", va="bottom", fontsize=9.5, color=INK_PRIMARY,
            arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, lw=1.3, connectionstyle="arc3,rad=0.15"),
            zorder=6,
        )

    ax.set_ylim(0, y_top * (1.16 + max(0, len(low_cf_idx) - 1) * 0.09))
    ax.set_xlim(0, left[-1] + captured_mt[-1])

    ax.set_xlabel("Cumulative CO$_2$ captured (Mt/yr)", fontsize=11)
    ax.set_ylabel("€/t CO$_2$ (storage + transport + capture)", fontsize=11)
    ax.set_title(
        f"CO$_2$ Capture Cost Curve — {storage_node}",
        fontsize=14, weight="bold", color=INK_PRIMARY,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_file = OUT_DIR / "ccs_chain_emitter_macc.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


def main():
    df, storage_node = build_emitter_cost_table(RESULTS_H5)

    out_csv = OUT_DIR / "ccs_chain_emitter_cost_ranking.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} emitters)")

    plot_emitter_cost_ranking(df, storage_node)
    plot_macc(df, storage_node)


if __name__ == "__main__":
    main()
