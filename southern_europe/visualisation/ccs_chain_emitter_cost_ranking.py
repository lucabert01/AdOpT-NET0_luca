"""
Ranks every CO2-capturing emitter (technology, really - see below) in a solved
main_italy.py case study by its total levelized cost per tonne of CO2
(capture + transport + storage), and plots it as a horizontal ranking bar
chart.

One row per genuinely distinct real-world emitter, not per (node, technology)
row main_italy.py's design exposes - a node can host more than one real
emitter (e.g. "Piacenza" genuinely runs both a CementEmitter and a
WasteToEnergyEmitter, confirmed against node_metrics_paper.xlsx's per-node
sector list), but a node offering more than one CANDIDATE capture technology
for the SAME plant (e.g. every cement node's MEA-retrofit vs CementHybridCCS
choice) is one emitter, not two - see _keep_real_emitters, which MERGES
(sums captured tonnes and cost, never drops either candidate's numbers) the
candidates per real sector into one bar. Rows are labelled "<node> (<sector>)"
when a node has more than one real emitter.

Capture cost is each technology's own capture cost (capex + opex + its own
electricity/heat draw, attributed via that technology's own consumption
series x the node's import price - exact, since it uses only that
technology's own operation dataset, not the node's total import) divided by
its own annual captured CO2. This deliberately excludes any electricity/heat
also drawn at that node by a network passing through it (pipeline
compression/pumping power) - that portion is transport cost, see below.

FertilizerSMREmitter (FertilizersSMR sector, family "direct_capture" - see
classify_capture_family in ccs_chain_plots.py) has no separate CCS retrofit
component at all: it directly outputs CO2captured as one of its own base
output carriers. Its capture cost is therefore its own capex_tot/opex_* (the
whole technology, same convention as CementHybridCCS/WasteCaL_CCS) plus its
own electricity input draw (its only input carrier - see
ELEC_INPUT_KEY_BY_FAMILY) - both read via the same per-technology operation
dataset as every other family, so pipeline compression power at its node is
excluded exactly the same way (it is a separate node-level
network_consumption/compressor_input entry, never part of any technology's
own operation dataset).

Storage cost is the single storage site's total annual cost (capex + opex +
its own electricity/heat draw, net of any network-consumption draw at that
node - see transport, below) divided by total annual tonnes stored - a flat
€/t added to every emitter, since storage is a shared, non-allocable
resource.

Transport cost is allocated by CAPACITY SHARE, not by flow share: for every
arc on an emitter's path to the storage node, the emitter pays
    arc's annual cost x (emitter's own max captured CO2, t/h) / (arc's built size, t/h)
i.e. a 15 t/h emitter on a 150 t/h pipeline pays 10% of that arc's annual
cost; on a 15 t/h pipeline it pays 100%. An arc's annual cost is its
capex + opex, PLUS the electricity/heat cost of the network flowing along it
- pipeline pumping/compression power is a per-arc send/receive consumption
(operation/networks/.../consumption_send<carrier> and
consumption_receive<carrier>, priced at the sending/receiving node's own
import price - see genericNetworks/fluid.py's _define_energyconsumption_arc)
that would otherwise fall through the cracks: it isn't any technology's own
draw (excluded from capture above) and, except at the storage node, isn't
counted in any node-level total this module reads either. This is summed
over every arc on the emitter's path (not just the first arc out of its
node) and then divided by the emitter's own annual captured tonnes. Two
emitters with identical paths and capture technology therefore end up with
different €/t if one runs at a lower load factor (same nameplate share of
the pipeline cost, spread over fewer actual tonnes) - which is the point: a
poorly-utilized reservation costs more per tonne, same as an oversized
capture unit does.

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
    ELEC_INPUT_KEY_BY_FAMILY,
    HEAT_INPUT_KEY_BY_FAMILY,
    INK_PRIMARY,
    INK_SECONDARY,
    GRIDLINE,
    SURFACE,
    MODE_COLORS,
    STORAGE_COLOR,
    _load_node_sectors,
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

SECTOR_ORDER = ["Cement", "Waste", "Refining", "Lime", "FertilizersCombustion", "FertilizersSMR", "Other"]
# FertilizersCombustion and FertilizersSMR are two technologies at the SAME
# physical fertilizer complex (see Ferrara in ccs_chain_plots.py's
# _load_node_sectors) -- one color group ("Fertilizers"), not two, so the
# MACC/ranking charts read them as one sector at a glance; per-bar
# annotations (see FERTILIZER_BAR_NOTE / plot_macc) carry the SMR-vs-not
# distinction instead of color.
_COLOR_GROUPS = ["Cement", "Waste", "Refining", "Lime", "Fertilizers", "Other"]
_BATLOW = dict(zip(_COLOR_GROUPS, [cmc.batlow(x) for x in np.linspace(0, 1, len(_COLOR_GROUPS))]))
_SECTOR_TO_COLOR_GROUP = {
    "FertilizersCombustion": "Fertilizers", "FertilizersSMR": "Fertilizers",
}
SECTOR_COLORS = {s: _BATLOW[_SECTOR_TO_COLOR_GROUP.get(s, s)] for s in SECTOR_ORDER}
# Legend label for a sector -- collapses the two fertilizer sectors to one
# shared "Fertilizers" legend entry, since they now share a color.
SECTOR_LEGEND_LABEL = {s: _SECTOR_TO_COLOR_GROUP.get(s, s) for s in SECTOR_ORDER}
# Per-bar annotation clarifying WHICH fertilizer technology a bar is, since
# color no longer does (see above) -- keyed by the `sector` column value.
FERTILIZER_BAR_NOTE = {
    "FertilizersSMR": "SMR only",
    "FertilizersCombustion": "Non-SMR emissions",
}

# Below this annual-tonnes / (peak-rate x 8760h) ratio, an emitter is flagged
# on the MACC plot as running at a low load factor - it reserves pipeline
# capacity sized for its peak rate but rarely uses all of it, so it pays a
# capacity-share transport cost spread over comparatively few actual tonnes.
LOW_CAPACITY_FACTOR_THRESHOLD = 0.75


def _sector_from_tech(tech_name: str) -> str:
    t = tech_name.lower()
    if "cement" in t:
        return "Cement"
    if "waste" in t:
        return "Waste"
    if "refin" in t:
        return "Refining"
    if "lime" in t:
        return "Lime"
    # Check "smr" before the generic "fertilizer" substring -- both
    # FertilizerCombustionEmitter and FertilizerSMREmitter contain
    # "fertilizer", so SMR must be matched first to not fall through.
    if "smr" in t:
        return "FertilizersSMR"
    if "fertilizer" in t:
        return "FertilizersCombustion"
    return "Other"


def _merge_candidate_group(group: list[tuple[str, dict]]) -> tuple[str, dict]:
    """
    Merges one real emitter's candidate-technology entries (see
    _keep_real_emitters) into a single MACC bar: captured_annual_t and
    max_captured_t_h are SUMMED (every candidate's reported tonnes are real
    in the solved instance -- see _keep_real_emitters), capture_eur_per_t is
    re-derived as the tonnage-weighted average (total capture cost / total
    captured tonnes), and family/sector/tech are taken from whichever
    candidate captured the most (the dominant one, for labeling/coloring).
    Summing max_captured_t_h is an upper-bound approximation when a plant's
    two technologies don't peak in the same hour (they still draw on the
    same outgoing pipeline, so a shared, conservative capacity-share number
    is preferable to picking just one candidate's peak) -- consistent with
    the rest of this module's transport allocation already being a
    capacity-share approximation, not an exact hourly one.
    """
    tech, info = max(group, key=lambda ti: ti[1]["captured_annual_t"])
    if len(group) == 1:
        return tech, info
    total_captured = sum(i["captured_annual_t"] for _, i in group)
    total_cost = sum(i["capture_eur_per_t"] * i["captured_annual_t"] for _, i in group)
    merged = dict(info)
    merged["captured_annual_t"] = total_captured
    merged["max_captured_t_h"] = sum(i["max_captured_t_h"] for _, i in group)
    merged["capture_eur_per_t"] = total_cost / total_captured if total_captured > 0 else 0.0
    return tech, merged


def _keep_real_emitters(capture_rows: dict) -> dict:
    """
    Collapses capture_rows (keyed by (node, tech), one entry per candidate
    capture technology) down to one entry per genuinely distinct real-world
    emitter, using node_metrics_paper.xlsx as ground truth for how many
    physically separate plants share a node -- the same rule
    ccs_chain_plots.py's _real_emitter_rows applies to the network map.

    main_italy.py wires up EVERY capture technology a sector COULD use as a
    candidate at each node (e.g. a cement plant gets both the MEA-retrofit
    path and CementHybridCCS) so the optimizer can pick between them -- they
    are not two separate plants, just two capture routes for the one plant,
    and only node_metrics_paper.xlsx's per-node sector count tells you which
    nodes (Ferrara, Piacenza) genuinely DO host more than one plant.

    IMPORTANT: candidates are MERGED (see _merge_candidate_group), never
    dropped. Dropping the non-dominant candidate's numbers was tried (on the
    theory that its non-negligible captured_annual_t is just a main_italy.py
    formulation bug, piecewise capex nonzero at size=0) and it silently
    undercounted total system capture: summing every technology's own
    captured_annual across the whole scenario reconciles exactly against the
    model's own summary/emissions_pos (total_emissions - captured ==
    emissions_pos, to rounding); keeping only the dominant candidate did not
    -- it undercounted by ~2.4 Mt/yr on the 2026-09-14 technology_selection
    rerun. So regardless of whether the dual-technology split is itself a
    modeling bug, the tonnes each candidate reports captured are real in the
    solved instance and this MACC's total captured tonnage must match them.
    """
    node_sectors = _load_node_sectors()
    by_node: dict[str, list[tuple[str, dict]]] = {}
    for (node, tech), info in capture_rows.items():
        by_node.setdefault(node, []).append((tech, info))

    kept = {}
    for node, techs in by_node.items():
        sectors = node_sectors.get(node)
        if not sectors or len(sectors) <= 1:
            groups = {"_single_": techs}
        else:
            groups = {}
            for tech, info in techs:
                groups.setdefault(info["sector"], []).append((tech, info))
        for group in groups.values():
            tech, info = _merge_candidate_group(group)
            kept[(node, tech)] = info
    return kept


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

        def network_consumption_cost(node_name, carrier):
            """EUR/year of `carrier` import at this node drawn by a network
            passing through it, not by any technology - see module
            docstring. Only relevant here at the storage node (the only node
            whose electricity/heat cost is read via the node-level total,
            node_carrier_cost, rather than per-technology); it must be
            carved out of that total since it is accounted for instead via
            arc_energy_cost below, on whichever arc actually carries it."""
            if node_name not in eb or carrier not in eb[node_name]:
                return 0.0
            g = eb[node_name][carrier]
            price = g["import_price"][()][seq - 1]
            total = 0.0
            for key in ("network_consumption", "compressor_input"):
                if key in g:
                    total += float((g[key][()][seq - 1] * price).sum())
            return total

        def arc_energy_cost(ntype, arc_name, from_node, to_node):
            """EUR/year this specific arc cost in electricity/heat to move
            CO2 along it (e.g. pipeline compression power) - the per-arc
            send/receive consumption from genericNetworks/fluid.py, priced
            at the sending/receiving node's own import price. This is the
            network-consumption counterpart of tech_carrier_cost above: it
            isolates the network's own draw from everything else importing
            at that node."""
            g = net_op[ntype][arc_name]
            total = 0.0
            for carrier in ("electricity", "heat"):
                send_key, receive_key = f"consumption_send{carrier}", f"consumption_receive{carrier}"
                if send_key in g and from_node in eb and carrier in eb[from_node]:
                    price = eb[from_node][carrier]["import_price"][()]
                    total += float((g[send_key][()] * price)[seq - 1].sum())
                if receive_key in g and to_node in eb and carrier in eb[to_node]:
                    price = eb[to_node][carrier]["import_price"][()]
                    total += float((g[receive_key][()] * price)[seq - 1].sum())
            return total

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
                op_keys = list(op_tech[node_name][tech].keys())
                family = classify_capture_family(list(g.keys()), op_keys)
                if family is None:
                    continue
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

        capture_rows = _keep_real_emitters(capture_rows)

        # ---- storage cost: one flat €/t added to every emitter ----
        # node_carrier_cost is storage_node's FULL electricity/heat import,
        # which (unlike the per-technology draws above) may also include a
        # network's own consumption at that node (e.g. an arriving pipeline's
        # receive-side pumping power) - that portion is carved out here and
        # picked up instead by arc_energy_cost on the incoming arc, below.
        g_store = nodes[storage_node][storage_tech]
        storage_cost = (
            float(g_store["capex_tot"][()][0])
            + float(g_store["opex_fixed"][()][0])
            + float(g_store["opex_variable"][()][0])
            + node_carrier_cost(storage_node, "electricity")
            - network_consumption_cost(storage_node, "electricity")
            + node_carrier_cost(storage_node, "heat")
            - network_consumption_cost(storage_node, "heat")
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
                from_node = gd["fromNode"][()].decode()
                to_node = gd["toNode"][()].decode()
                annual_cost = (
                    float(gd["capex"][()])
                    + opex_fixed
                    + opex_variable
                    + arc_energy_cost(ntype, arc_name, from_node, to_node)
                )

                annual_flow = expand(net_op[ntype][arc_name]["flow"])
                if annual_flow <= 0:
                    continue

                edge_rows.append(
                    {
                        "from": from_node,
                        "to": to_node,
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
            # FertilizersCombustion/FertilizersSMR share a sector color on
            # the MACC chart (see SECTOR_COLORS) - use the same clarifying
            # note here instead of the raw sector name, so a node's two
            # rows read as distinguishable without re-deriving it.
            parts.append(FERTILIZER_BAR_NOTE.get(row["sector"], row["sector"]))
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
    n = len(plot_df)
    captured_mt = (plot_df["captured_annual_t"] / 1e6).to_numpy()
    left = np.concatenate([[0.0], np.cumsum(captured_mt)[:-1]])
    heights = plot_df["total_eur_per_t"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Low-capacity-factor bars get a hatch instead of a text callout -- with
    # dozens of technologies in the technology-selection scenarios (vs. a
    # handful in the single-family runs this was designed for), one arrow
    # per flagged bar stops being readable (arrows/labels for 20-30 bars
    # overlap into an unreadable tangle). A hatch scales to any count and
    # keeps the exact number available in the CSV / row-count callouts for
    # anyone who needs it.
    low_cf_mask = (plot_df["capacity_factor"] < LOW_CAPACITY_FACTOR_THRESHOLD).to_numpy()
    for i, row in plot_df.iterrows():
        ax.bar(
            left[i], heights[i], width=captured_mt[i], align="edge",
            color=SECTOR_COLORS[row["sector"]], edgecolor="white", linewidth=0.5, zorder=3,
            hatch="////" if low_cf_mask[i] else None,
        )

    # De-duplicated by legend label, not sector, so FertilizersCombustion and
    # FertilizersSMR (same color, see SECTOR_COLORS) collapse to one
    # "Fertilizers" swatch instead of two identical-looking entries.
    seen_labels = set()
    legend_handles = []
    for s in SECTOR_ORDER:
        label = SECTOR_LEGEND_LABEL[s]
        if s not in set(plot_df["sector"]) or label in seen_labels:
            continue
        seen_labels.add(label)
        legend_handles.append(Patch(facecolor=SECTOR_COLORS[s], edgecolor="white", label=label))
    if low_cf_mask.any():
        legend_handles.append(Patch(
            facecolor="none", edgecolor=INK_SECONDARY, hatch="////",
            label=f"Capacity factor < {LOW_CAPACITY_FACTOR_THRESHOLD:.0%} ({int(low_cf_mask.sum())} of {n})",
        ))
    # upper right, not left -- the cheapest (leftmost, shortest) bars are
    # exactly where the fertilizer callouts below sit, and MACC bars rise
    # left-to-right, so the top-right corner is the one reliably empty
    # region regardless of how many bars this run has.
    ax.legend(
        handles=legend_handles, loc="upper right", frameon=True, fontsize=10.5,
        framealpha=0.95, edgecolor=GRIDLINE, title="Sector", title_fontsize=10.5,
    )

    # --- per-bar callouts: which fertilizer technology a bar is, since
    # color alone no longer distinguishes FertilizersCombustion from
    # FertilizersSMR (see SECTOR_COLORS) -- ordered left-to-right and
    # stacked upward so nearby bars' callouts don't collide.
    notes = {
        i: [FERTILIZER_BAR_NOTE[plot_df.loc[i, "sector"]]]
        for i in plot_df.index if plot_df.loc[i, "sector"] in FERTILIZER_BAR_NOTE
    }
    note_idx = sorted(notes, key=lambda i: left[i])

    y_top = heights.max()
    x_total = left[-1] + captured_mt[-1]
    # Text anchored a little right of the y-axis (these are always the
    # cheapest/leftmost bars) so it doesn't sit on top of the tick labels.
    text_x = max(x_total * 0.035, captured_mt[note_idx].max() / 2 if len(note_idx) else 0)
    for rank, i in enumerate(note_idx):
        cx = left[i] + captured_mt[i] / 2
        cy = heights[i]
        ax.annotate(
            "\n".join(notes[i]),
            xy=(cx, cy), xycoords="data",
            xytext=(text_x, y_top * 1.10 + rank * y_top * 0.09), textcoords="data",
            ha="left", va="bottom", fontsize=9.5, color=INK_PRIMARY,
            arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, lw=1.3, connectionstyle="arc3,rad=0.15"),
            zorder=6,
        )

    ax.set_ylim(0, y_top * (1.16 + max(0, len(note_idx) - 1) * 0.09))
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
