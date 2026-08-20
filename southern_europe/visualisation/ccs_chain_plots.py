"""
Conference-ready visualizations for the CCS chain optimization case study.

Produces five figures from a solved optimization_results.h5:

  1. ccs_chain_network_map.png       - Italy map: built CO2 transport network
                                        (pipeline/truck/railway), emitters colored
                                        by whether CCS was installed, transport
                                        hubs and the storage site.
  1b. ccs_chain_network_map_sized_by_capacity.png - same map, but emitter bubble
                                        area scales with technology capacity
                                        (design "size" field, t/h).
  1c. ccs_chain_network_map_cost_factor.png - same built network + CCS status,
                                        on the grayscale integrated cost-factor
                                        grid (as in cost_factor_grid_map_italy.py).
  1d. ccs_chain_network_map_trunk_highlight.png - one specific chain of built
                                        arcs (default: the Piacenza -> Modena-H ->
                                        HERAMBIENTE Spa -> Ravenna -> Casalborsetti
                                        -> Porto Corsini trunk) highlighted in bold
                                        against the rest of the network, muted.

  All maps share the same base style (Italy fill/boundary, fixed lon/lat extent,
  axis labels) as routes_connection.py, via setup_base_map().
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

# main_italy.py builds three separate pipeline network technologies -
# CO2_Pipeline_{small,medium,large} - instead of one CO2_Pipeline (see
# pipeline_capex_per_arc_calculator.py::SIZE_CLASS_MASSFLOW_RANGES_KG_S).
# They share the same physical corridors/route shapefile and should read as
# one visual "Pipeline" mode here (one legend entry, one color, one linewidth
# scale across all three) rather than three separate untracked network names
# that would otherwise silently vanish from every map (built_arcs["network"]
# would never match the bare "CO2_Pipeline" key used throughout this file).
NETWORK_TYPE_TO_MODE = {
    "CO2_Pipeline": "CO2_Pipeline",
    "CO2_Pipeline_small": "CO2_Pipeline",
    "CO2_Pipeline_medium": "CO2_Pipeline",
    "CO2_Pipeline_large": "CO2_Pipeline",
    "CO2Truck": "CO2Truck",
    "CO2Railway": "CO2Railway",
}

STATUS_GOOD = BATLOW[2]    # CCS installed
STATUS_MUTED = "#9a988f"   # no CCS (kept neutral -- absence, not a category)
STATUS_CRITICAL = BATLOW[5]

TRANSPORT_COLOR = BATLOW[1]  # transport hub marker
STORAGE_COLOR = BATLOW[6]    # storage marker

# ------------------------------------------------------------------
# Capture technology families
# ------------------------------------------------------------------
# The case study lets each sector pick between a generic bolt-on MEA
# retrofit (CementEmitter/WasteToEnergyEmitter/RefineryEmitter/
# UnspecifiedEmitter, tec_type CONV4 with a Performance.ccs block) and two
# self-contained capture technologies with their own tec_type and their own
# design/operation variable names (see main_italy.py SCENARIOS and
# data_process/utilities/defined_functions.py:update_emitter_ccs_references,
# which explicitly skips MEA-retrofit wiring for these):
#   - CementHybridCCS   (oxyfuel + MEA polish, cement sector)
#   - WasteCaL_CCS      (calcium looping, waste sector; tec_type WasteToEnergyCaLCCS)
# Encoding capture technology as marker SHAPE (rather than a 4th hue) keeps
# it legible under the same batlow-derived color language already used for
# CCS status (green=captured, muted=not) and avoids adding low-contrast
# batlow hues for identity (validated: adjacent-pair contrast among mid-tone
# batlow stops falls below the CVD-safe floor for small filled markers).
FAMILY_ORDER = ["mea_retrofit", "oxyfuel_hybrid", "calcium_looping"]
FAMILY_LABELS = {
    "mea_retrofit": "MEA retrofit",
    "oxyfuel_hybrid": "Oxyfuel + MEA hybrid",
    "calcium_looping": "Calcium looping",
}
FAMILY_MARKERS = {
    "mea_retrofit": "o",
    "oxyfuel_hybrid": "^",
    "calcium_looping": "D",
}
# matplotlib scatter `s` is a bounding-box area, so a triangle/diamond of the
# same `s` as a circle reads as visually smaller (lower fill ratio) -- scale
# up so the three shapes read as equal-weight on the map.
FAMILY_MARKER_SCALE = {"mea_retrofit": 1.0, "oxyfuel_hybrid": 1.35, "calcium_looping": 1.2}

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
RESULTS_H5 = Path("../Results_CCSchainOptimization/20260710184058_emissions_minC-1/optimization_results.h5")

path_data_case_study = Path("../italy_data")
path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_grids = path_data_case_study / "geographical_feature"
path_cost_factor_table = Path(
    "../../adopt_net0/database/data/networks/enhanced_co2_transport_cost_model/cost_factor_table.xlsx"
)

GIS_NODES = path_files_gis / "all_nodes_italy.shp"
ITALY_SHP = path_files_gis / "italy_WGS1984.shp"
ROUTES = {
    "CO2_Pipeline": path_files_gis / "routes_distances_pipelines.shp",
    "CO2Truck": path_files_gis / "truck_italy_150.shp",
    "CO2Railway": path_files_gis / "routes_distances_railway.shp",
}
OUT_DIR = Path(__file__).resolve().parent

# Same map extent used throughout routes_connection.py, so every map in this
# project reads as one consistent system.
MAP_BOUNDS = {"minx": 6.5, "maxx": 14.0, "miny": 43.5, "maxy": 47.0}
COST_FACTOR_PIPELINE_CATEGORY = 300
BW_CMAP = plt.cm.Greys


def setup_base_map(ax, italy: gpd.GeoDataFrame, title: str):
    """Base map styling matching routes_connection.py exactly: same Italy
    boundary/fill, same fixed extent, same axis labels/grid."""
    italy.boundary.plot(ax=ax, color="black", linewidth=1, alpha=0.7)
    italy.plot(ax=ax, color="lightgray", alpha=0.2)
    ax.set_xlim(MAP_BOUNDS["minx"], MAP_BOUNDS["maxx"])
    ax.set_ylim(MAP_BOUNDS["miny"], MAP_BOUNDS["maxy"])
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")


def compute_cost_factor_grid(italy: gpd.GeoDataFrame, category: float = COST_FACTOR_PIPELINE_CATEGORY):
    """Integrated cost factor fishnet grid, same calculation as
    cost_factor_grid_map_italy.py / routes_connection.py."""
    fishnet = gpd.read_file(path_files_gis / "fishnet_italy_5km.shp").reset_index().rename(
        columns={"index": "GRID_OID"})
    soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
    anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
    morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")
    cost_factor_table = pd.read_excel(path_cost_factor_table)

    table = cost_factor_table.sort_values("pipeline_category").reset_index(drop=True)
    exact_match = table.loc[table["pipeline_category"] == category]
    if not exact_match.empty:
        coeffs = exact_match.iloc[0].to_dict()
    else:
        coeffs = {"pipeline_category": category}
        for col in table.columns:
            if col == "pipeline_category":
                continue
            coeffs[col] = float(np.interp(category, table["pipeline_category"], table[col]))

    fishnet = (fishnet
               .merge(soil_data, on="GRID_OID")
               .merge(anthro_data, on="GRID_OID")
               .merge(morpho_data, on="GRID_OID"))

    fishnet["SOIL_FACTOR"] = (coeffs["k_soil_non_rock"] * fishnet["NON_ROCK_S"]
                               + coeffs["k_soil_rock"] * fishnet["ROCK_S"])
    fishnet["ANTHRO_FACTOR"] = (coeffs["k_anthro_non_anthropised"] * fishnet["NON_ANTHROPISED_A"]
                                 + coeffs["k_anthro_anthropised"] * fishnet["ANTHROPISED_A"])
    fishnet["MORPH_FACTOR"] = (coeffs["k_morpho_plain"] * fishnet["PLAIN_M"]
                                + coeffs["k_morpho_hill"] * fishnet["HILL_M"]
                                + coeffs["k_morpho_mountain"] * fishnet["MOUNTAIN_M"])
    fishnet["COST_FACTOR"] = fishnet[["SOIL_FACTOR", "ANTHRO_FACTOR", "MORPH_FACTOR"]].sum(axis=1)

    return gpd.clip(fishnet, italy)


def classify_capture_family(design_keys) -> str | None:
    """
    Identifies which capture technology a node's technology block belongs to,
    from the dataset names present in its design/nodes/period1/<node>/<tech>
    HDF5 group -- robust to technology naming/suffixes ("_existing" etc.),
    since it keys off variables each technology CLASS writes unconditionally:

      - "size_ccs"  -- generic bolt-on MEA retrofit (technology.py's shared
                       CCS mixin, gated on Performance.ccs.possible)
      - "size_mea"  -- CementHybridCCS (oxyfuel + MEA polish); written by
                       cement_hybrid_ccs.py:write_results_tec_design
      - "size_cal"  -- WasteCaL_CCS / WasteToEnergyCaLCCS (calcium looping);
                       written by wasteToEnergy_CaL_ccs.py:write_results_tec_design

    Returns None for technologies with none of these keys (transport/storage
    technologies, or an emitter with no capture at all).
    """
    keys = set(design_keys)
    if "size_ccs" in keys:
        return "mea_retrofit"
    if "size_mea" in keys:
        return "oxyfuel_hybrid"
    if "size_cal" in keys:
        return "calcium_looping"
    return None


def captured_co2_operation_key(operation_keys) -> str | None:
    """
    The operation-group dataset name holding hourly captured CO2 (t/h),
    which differs by capture family: the generic MEA-retrofit CCS component
    writes to a "_var_output_ccs"-suffixed carrier dataset (technology.py
    write_results_tec_operation), while the self-contained CementHybridCCS/
    WasteCaL_CCS technologies emit CO2captured as a plain output carrier
    (their own output_carrier lists include "CO2captured" directly) and so
    write the unsuffixed "CO2captured_output" via the base class.
    """
    keys = set(operation_keys)
    if "CO2captured_var_output_ccs" in keys:
        return "CO2captured_var_output_ccs"
    if "CO2captured_output" in keys:
        return "CO2captured_output"
    return None


def _scatter_emitter(ax, x, y, family, installed, s, zorder,
                      edgecolor=None, linewidth=None, alpha=None, area_scale=True):
    """Draws one emitter marker: shape encodes capture technology (family),
    fill encodes whether it actually captures anything -- see FAMILY_MARKERS
    docstring above. edgecolor/linewidth let callers (e.g. the trunk-highlight
    map) override the ring styling without touching fill/shape logic.
    area_scale=False skips the shape fill-ratio correction, for maps where
    `s` is itself a data-driven magnitude encoding (e.g. plot_map_sized_by_
    capacity) and must stay strictly proportional across markers."""
    marker = FAMILY_MARKERS.get(family, "o")
    scale = FAMILY_MARKER_SCALE.get(family, 1.0) if area_scale else 1.0
    kwargs = dict(
        marker=marker,
        s=s * scale,
        edgecolor=edgecolor if edgecolor is not None else ("white" if installed else STATUS_MUTED),
        linewidth=linewidth if linewidth is not None else (1.2 if installed else 1.8),
        zorder=zorder,
    )
    if alpha is not None:
        kwargs["alpha"] = alpha
    if installed:
        kwargs["color"] = STATUS_GOOD
    else:
        kwargs["facecolors"] = "none"
    ax.scatter(x, y, **kwargs)


def _capture_legend_handles(ccs_df: pd.DataFrame) -> list:
    """Shape/status legend entries for whichever capture families actually
    appear in this result file, in a fixed canonical order (FAMILY_ORDER) so
    the same shape always means the same technology across different runs."""
    present = set(ccs_df["family"].dropna()) if len(ccs_df) else set()
    handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[f], color="w", markerfacecolor=STATUS_GOOD,
               markeredgecolor="white", markersize=11, linestyle="None", label=FAMILY_LABELS[f])
        for f in FAMILY_ORDER if f in present
    ]
    if len(ccs_df) and (~ccs_df["ccs_installed"]).any():
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=STATUS_MUTED,
                   markersize=11, markeredgewidth=1.8, linestyle="None", label="No CCS installed")
        )
    return handles


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
                        "mode": NETWORK_TYPE_TO_MODE.get(ntype, ntype),
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
    """
    Per-emitter capture technology (family), CCS status, and total annual
    emissions (captured + vented).

    Handles all three capture technology families the case study can select
    per sector (see classify_capture_family docstring): the generic bolt-on
    MEA retrofit, and the self-contained CementHybridCCS/WasteCaL_CCS
    technologies, which use different design/operation variable names.
    "Installed" is judged generically from annual captured CO2 > 0 rather
    than a family-specific design variable, since e.g. CementHybridCCS's
    oxyfuel front-end captures CO2 unconditionally whenever it's producing
    clinker (size_mea, the optional MEA-polish add-on, can be 0 even though
    the technology is actively capturing).

    Total emissions is the true measure of a plant's scale -- unlike the
    design "size" field (a free technology-capacity decision variable that,
    for "existing=0" generic techs, is NOT tied to real historical output).

    IMPORTANT: design/nodes/.../emissions_pos is NOT an annual total -- it's
    the raw unweighted sum over the 360 clustered representative hours (confirmed
    by comparing it directly to the node's carrier "demand", which sums to the
    correct real annual value only after expansion). Both emissions_pos and
    captured CO2 must be pulled from operation/technology_operation (the
    per-timestep series) and expanded back to the full 8760-hour year via
    k_means_specs/sequence before summing -- same treatment for both, or the
    two terms end up on inconsistent scales.
    """
    rows = []
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        nodes = f["design"]["nodes"]["period1"]
        op = f["operation"]["technology_operation"]["period1"]
        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                design_keys = list(g.keys())
                family = classify_capture_family(design_keys)
                if family is None:
                    continue

                captured_key = captured_co2_operation_key(op[node_name][tech].keys())
                if captured_key is None:
                    continue

                size = float(g["size"][()][0])
                emitted_clustered = op[node_name][tech]["emissions_pos"][()]
                emitted_annual = float(emitted_clustered[seq - 1].sum())
                captured_clustered = op[node_name][tech][captured_key][()]
                captured_annual = float(captured_clustered[seq - 1].sum())

                rows.append(
                    {
                        "node": node_name,
                        "tech": tech,
                        "family": family,
                        "size": size,
                        "captured_annual": captured_annual,
                        "ccs_installed": captured_annual > 1e-6,
                        "total_emissions": captured_annual + emitted_annual,
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

    Capture cost accounting differs by technology family (see
    classify_capture_family): for the generic bolt-on MEA retrofit, only the
    CCS component's own capex_ccs/opex_*_ccs count -- the host emitter's own
    production cost is out of scope for a "cost of the CCS chain" breakdown.
    For the self-contained CementHybridCCS/WasteCaL_CCS technologies, capture
    is inseparable from production (one technology block, no retrofit split
    available), so their full capex_tot/opex_fixed/opex_variable is the
    capture-chain cost -- the same convention already used below for the
    storage technology.

    Electricity/heat import cost is not in technology opex (technology
    opex_variable is 0 for both the emitter and the MEA CCS component; energy
    is priced at the node's carrier balance instead -- see construct_balances.py
    :func:`construct_import_costs`). It is attributed to whichever node
    consumes it: the storage node's own electricity draw counts as "storage",
    everything else counts as "capture" (and, within that, its node's capture
    family).

    Carrier-balance arrays (operation/energy_balance) are stored at the
    design-days (clustered) resolution; they are expanded back to the full
    8760-hour year via k_means_specs/sequence before summing, exactly like the
    model's own full-resolution linking constraint does.

    :return: dict with 'capture', 'transport', 'storage' (each a dict of
        component -> EUR/year), 'capture_by_family' (dict of family label ->
        same component dict, one entry per capture family actually present),
        and 'total_stored_t' (t CO2 stored per year).
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
        capture_by_family = {}
        storage = {k: 0.0 for k in COMPONENT_COLORS}

        for node_name in nodes.keys():
            node_family = None
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                keys = list(g.keys())
                family = classify_capture_family(keys)
                if family is not None:
                    node_family = family
                    if family == "mea_retrofit":
                        comp_capex = float(g["capex_ccs"][()][0])
                        comp_opex_fixed = float(g["opex_fixed_ccs"][()][0])
                        comp_opex_variable = float(g["opex_variable_ccs"][()][0])
                    else:
                        comp_capex = float(g["capex_tot"][()][0])
                        comp_opex_fixed = float(g["opex_fixed"][()][0])
                        comp_opex_variable = float(g["opex_variable"][()][0])
                    capture["Capex"] += comp_capex
                    capture["Opex (fixed)"] += comp_opex_fixed
                    capture["Opex (variable)"] += comp_opex_variable
                    fam_costs = capture_by_family.setdefault(family, {k: 0.0 for k in COMPONENT_COLORS})
                    fam_costs["Capex"] += comp_capex
                    fam_costs["Opex (fixed)"] += comp_opex_fixed
                    fam_costs["Opex (variable)"] += comp_opex_variable
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
                if node_family is not None:
                    capture_by_family[node_family]["Electricity"] += elec_cost
                    capture_by_family[node_family]["Heat"] += heat_cost

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
        "capture_by_family": {
            FAMILY_LABELS.get(fam, fam): costs
            for fam, costs in sorted(
                capture_by_family.items(), key=lambda kv: FAMILY_ORDER.index(kv[0])
            )
        },
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
            geom = route_lookup.get(r["mode"], {}).get(frozenset((int(from_id), int(to_id))))
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
    family_map = dict(zip(ccs_df["node"], ccs_df["family"]))

    fig, ax = plt.subplots(figsize=(12.5, 12))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    setup_base_map(ax, italy, "Optimized CO$_2$ Capture, Transport & Storage Network — Northern Italy")

    # --- routes, drawn truck/rail first so pipeline (usually dominant) sits on top ---
    draw_order = ["CO2Truck", "CO2Railway", "CO2_Pipeline"]
    max_size_by_net = built_arcs.groupby("mode")["size"].max().to_dict()

    for ntype in draw_order:
        sub = built_arcs[built_arcs["mode"] == ntype]
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
            _scatter_emitter(ax, point.x, point.y, family_map.get(name), installed,
                              s=100 if installed else 90, zorder=22 if installed else 21)

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *_capture_legend_handles(ccs_df),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=4, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
    )

    n_installed = int(ccs_df["ccs_installed"].sum())
    n_total = len(ccs_df)
    total_capture = ccs_df["captured_annual"].sum()
    kpi_text = (
        f"{n_installed}/{n_total} emitters equipped with CCS\n"
        f"{total_capture:,.0f} t/yr captured CO$_2$\n"
        f"Network capex: €{summary['cost_capex_netws'] / 1e6:,.0f}M"
    )
    ax.text(
        0.02, 0.02, kpi_text, transform=ax.transAxes, fontsize=11, color=INK_PRIMARY,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=GRIDLINE, alpha=0.92),
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    out_file = OUT_DIR / "ccs_chain_network_map.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 1b - Network map with emitter bubble size ~ emitter capacity
# ============================================================
def plot_map_sized_by_capacity(built_arcs, nodes_gdf, ccs_df, summary,
                                size_range=(25, 900)):
    """Same map as plot_main_map, but each emitter's marker area scales with
    its total annual emissions (captured + vented, t/year) instead of a fixed
    radius -- the true measure of plant scale (see load_ccs_status). Route
    linewidth also scales with each arc's built size (e.g. pipeline capacity),
    same as plot_main_map."""
    italy = gpd.read_file(ITALY_SHP)
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))
    ccs_map = dict(zip(ccs_df["node"], ccs_df["ccs_installed"]))
    family_map = dict(zip(ccs_df["node"], ccs_df["family"]))
    size_map = dict(zip(ccs_df["node"], ccs_df["total_emissions"]))

    s_min, s_max = size_range
    max_capacity = ccs_df["total_emissions"].max() if len(ccs_df) else 1
    max_capacity = max_capacity if max_capacity > 0 else 1

    def marker_area(capacity):
        # sqrt-compressed: emissions here span ~3,900 to ~874,000 t/yr (~220x).
        # Pure linear-area scaling crushes everything below ~10,000 t/yr to
        # within a couple of points of s_min, making them look identical.
        # Compressing by sqrt keeps the ordering monotonic while spreading out
        # the small end enough to stay distinguishable.
        frac = (capacity / max_capacity) ** 0.5
        return s_min + (s_max - s_min) * frac

    fig, ax = plt.subplots(figsize=(12.5, 12))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    setup_base_map(ax, italy, "CO$_2$ Network — Emitter Bubble Size ~ Total Annual Emissions")

    # routes, linewidth scaled by built size (e.g. pipeline capacity) within each mode
    max_size_by_net = built_arcs.groupby("mode")["size"].max().to_dict()
    for ntype in ["CO2Truck", "CO2Railway", "CO2_Pipeline"]:
        sub = built_arcs[built_arcs["mode"] == ntype]
        color = MODE_COLORS[ntype]
        max_size = max_size_by_net.get(ntype, 1) or 1
        for _, r in sub.iterrows():
            geom = r["geometry"]
            if geom is None:
                continue
            lw = 1.0 + 2.5 * (r["size"] / max_size)
            gpd.GeoSeries([geom]).plot(ax=ax, color=color, linewidth=lw, alpha=0.75, zorder=5)

    for _, row in nodes_unique.iterrows():
        name, ntype_raw, point = row["node_name"], row["node_type"], row.geometry
        if ntype_raw == "Storage":
            ax.scatter(point.x, point.y, marker="*", s=480, color=STORAGE_COLOR,
                       edgecolor="white", linewidth=1.4, zorder=25)
        elif ntype_raw == "Transport":
            ax.scatter(point.x, point.y, marker="s", s=100, color=TRANSPORT_COLOR,
                       edgecolor="white", linewidth=1.2, zorder=20)
        else:
            area = marker_area(size_map.get(name, 0))
            installed = ccs_map.get(name, False)
            _scatter_emitter(ax, point.x, point.y, family_map.get(name), installed, s=area,
                              zorder=22 if installed else 21, alpha=0.85 if installed else None,
                              area_scale=False)

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *_capture_legend_handles(ccs_df),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    legend1 = ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.35, -0.13),
        ncol=2, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
        title="Mode / node type", title_fontsize=11,
    )
    ax.add_artist(legend1)

    # --- bubble-size legend (reference capacities) ---
    ref_fracs = [0.25, 0.6, 1.0]
    ref_vals = [round(max_capacity * f) for f in ref_fracs]
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK_MUTED, markeredgecolor="white",
               markeredgewidth=1.0, linestyle="None",
               markersize=2 * (marker_area(v) / 3.14159) ** 0.5, label=f"{v:,.0f} t/yr")
        for v in ref_vals
    ]
    ax.legend(
        handles=size_handles, loc="upper center", bbox_to_anchor=(0.75, -0.13),
        ncol=1, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
        title="Total annual emissions", title_fontsize=11, labelspacing=1.6, borderpad=1.1,
    )

    n_installed = int(ccs_df["ccs_installed"].sum())
    n_total = len(ccs_df)
    kpi_text = (
        f"{n_installed}/{n_total} emitters equipped with CCS\n"
        f"Network capex: €{summary['cost_capex_netws'] / 1e6:,.0f}M"
    )
    ax.text(
        0.02, 0.02, kpi_text, transform=ax.transAxes, fontsize=11, color=INK_PRIMARY,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=GRIDLINE, alpha=0.92),
    )

    fig.tight_layout(rect=[0, 0.14, 1, 1])
    out_file = OUT_DIR / "ccs_chain_network_map_sized_by_capacity.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 1c - Network map with grayscale integrated cost-factor background
# ============================================================
def plot_network_map_cost_factor(built_arcs, nodes_gdf, ccs_df, summary):
    """Same built network + CCS-status map as plot_main_map, but with the
    integrated cost-factor grid (grayscale) as background, matching the
    pipeline-connections plot in cost_factor_grid_map_italy.py."""
    italy = gpd.read_file(ITALY_SHP)
    fishnet_clipped = compute_cost_factor_grid(italy)
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))
    ccs_map = dict(zip(ccs_df["node"], ccs_df["ccs_installed"]))
    family_map = dict(zip(ccs_df["node"], ccs_df["family"]))

    fig, ax = plt.subplots(figsize=(12.5, 12))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    fishnet_clipped.plot(column="COST_FACTOR", ax=ax, cmap=BW_CMAP, legend=False, zorder=0)
    fishnet_clipped.boundary.plot(ax=ax, color="gray", linewidth=0.3, alpha=0.5, zorder=1)
    setup_base_map(ax, italy, f"CO$_2$ Network on Integrated Cost Factor (pipeline category {COST_FACTOR_PIPELINE_CATEGORY})")

    draw_order = ["CO2Truck", "CO2Railway", "CO2_Pipeline"]
    max_size_by_net = built_arcs.groupby("mode")["size"].max().to_dict()
    for ntype in draw_order:
        sub = built_arcs[built_arcs["mode"] == ntype]
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
            _scatter_emitter(ax, point.x, point.y, family_map.get(name), installed,
                              s=100 if installed else 90, zorder=22 if installed else 21)

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *_capture_legend_handles(ccs_df),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=4, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
    )

    sm = plt.cm.ScalarMappable(
        cmap=BW_CMAP,
        norm=plt.Normalize(fishnet_clipped["COST_FACTOR"].min(), fishnet_clipped["COST_FACTOR"].max()),
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cost Factor Value", fontsize=11)

    n_installed = int(ccs_df["ccs_installed"].sum())
    n_total = len(ccs_df)
    total_capture = ccs_df["captured_annual"].sum()
    kpi_text = (
        f"{n_installed}/{n_total} emitters equipped with CCS\n"
        f"{total_capture:,.0f} t/yr captured CO$_2$\n"
        f"Network capex: €{summary['cost_capex_netws'] / 1e6:,.0f}M"
    )
    ax.text(
        0.02, 0.02, kpi_text, transform=ax.transAxes, fontsize=11, color=INK_PRIMARY,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=GRIDLINE, alpha=0.92),
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    out_file = OUT_DIR / "ccs_chain_network_map_cost_factor.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 1d - Trunk line highlighted against the rest of the network
# ============================================================
TRUNK_PATH_DEFAULT = [
    "Piacenza",
    "Modena-H",
    "HERAMBIENTE Spa -Termovalorizzatore",
    "Ravenna",
    "Eni S.p.A Casalborsetti",
    "Porto Corsini",
]
TRUNK_COLOR = "#d03b3b"


def plot_trunk_highlight(built_arcs, nodes_gdf, ccs_df, summary, trunk_path: list = None):
    """Highlights one specific chain of built arcs (e.g. the main trunk line
    carrying most flow toward storage) in bold against the rest of the network,
    which is shown in its normal per-mode colors (no dimming/fading)."""
    trunk_path = trunk_path or TRUNK_PATH_DEFAULT
    trunk_pairs = set(zip(trunk_path[:-1], trunk_path[1:]))

    italy = gpd.read_file(ITALY_SHP)
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))
    ccs_map = dict(zip(ccs_df["node"], ccs_df["ccs_installed"]))
    family_map = dict(zip(ccs_df["node"], ccs_df["family"]))

    fig, ax = plt.subplots(figsize=(12.5, 12))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    setup_base_map(ax, italy, "CO$_2$ Network — Trunk Line to Storage Highlighted")

    # --- context: every built arc, normal per-mode color (no dimming) ---
    for _, r in built_arcs.iterrows():
        geom = r["geometry"]
        if geom is None or (r["from"], r["to"]) in trunk_pairs:
            continue
        gpd.GeoSeries([geom]).plot(ax=ax, color=MODE_COLORS[r["mode"]], linewidth=1.2, alpha=0.9, zorder=3)

    # --- trunk: bold, on top, with arrows ---
    for a, b in zip(trunk_path[:-1], trunk_path[1:]):
        match = built_arcs[(built_arcs["from"] == a) & (built_arcs["to"] == b)]
        if match.empty:
            print(f"Warning: no built arc found for trunk leg {a} -> {b}")
            continue
        r = match.iloc[0]
        geom = r["geometry"]
        from_point = name_to_point.get(a)
        coords = _oriented_coords(geom, from_point) if from_point else list(geom.coords)
        gpd.GeoSeries([LineString(coords)]).plot(ax=ax, color=TRUNK_COLOR, linewidth=4.5, alpha=0.95, zorder=8)

        point, (dx, dy) = _point_and_tangent_at_fraction(coords, 0.55)
        norm = (dx ** 2 + dy ** 2) ** 0.5
        if norm > 0:
            ux, uy = dx / norm, dy / norm
            arrow_len = 0.12
            start = (point[0] - ux * arrow_len / 2, point[1] - uy * arrow_len / 2)
            end = (point[0] + ux * arrow_len / 2, point[1] + uy * arrow_len / 2)
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=TRUNK_COLOR, lw=2.2, mutation_scale=20),
                zorder=9,
            )

    # --- nodes: usual CCS-status styling, with a highlight ring + label on the trunk ---
    for _, row in nodes_unique.iterrows():
        name, ntype_raw, point = row["node_name"], row["node_type"], row.geometry
        on_trunk = name in trunk_path

        if ntype_raw == "Storage":
            ax.scatter(point.x, point.y, marker="*", s=520 if on_trunk else 480, color=STORAGE_COLOR,
                       edgecolor=(TRUNK_COLOR if on_trunk else "white"), linewidth=(2.5 if on_trunk else 1.4),
                       zorder=25)
        elif ntype_raw == "Transport":
            ax.scatter(point.x, point.y, marker="s", s=130 if on_trunk else 90, color=TRANSPORT_COLOR,
                       edgecolor=(TRUNK_COLOR if on_trunk else "white"), linewidth=(2.5 if on_trunk else 1.2),
                       alpha=(1.0 if on_trunk else 0.5), zorder=20)
        else:
            installed = ccs_map.get(name, False)
            edge = TRUNK_COLOR if on_trunk else None
            _scatter_emitter(ax, point.x, point.y, family_map.get(name), installed,
                              s=130 if on_trunk else 80, zorder=22 if on_trunk else 18,
                              edgecolor=edge, linewidth=2.8 if on_trunk else None,
                              alpha=1.0 if on_trunk else 0.55)

    legend_handles = [
        Line2D([0], [0], color=TRUNK_COLOR, lw=4.5, label="Trunk line"),
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=2, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=2, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=2, label="Railway"),
        *_capture_legend_handles(ccs_df),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=4, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
    )

    trunk_arcs = built_arcs[
        built_arcs.apply(lambda r: (r["from"], r["to"]) in trunk_pairs, axis=1)
    ]
    kpi_text = (
        f"Trunk: {' → '.join(trunk_path)}\n"
        f"Trunk capacity: {trunk_arcs['size'].min():,.0f}–{trunk_arcs['size'].max():,.0f} t/h\n"
        f"Trunk capex: €{trunk_arcs['capex'].sum() / 1e6:,.0f}M"
    )
    ax.text(
        0.02, 0.02, kpi_text, transform=ax.transAxes, fontsize=10, color=INK_PRIMARY,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=TRUNK_COLOR, alpha=0.92),
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    out_file = OUT_DIR / "ccs_chain_network_map_trunk_highlight.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ============================================================
# PLOT 2 - Emitter zoom-in: captured vs. emitted CO2
# ============================================================
def plot_emitter_zoom(h5_path: Path, node_name: str = "SILLA 2", tech_name: str | None = None):
    """
    Captured vs. emitted CO2 for one node's capture technology.

    tech_name can be left unset -- the node's capture-capable technology
    (and its family, for the title) is auto-detected the same way
    load_ccs_status does, so this works for any of the three capture
    families without hardcoding a variable name (see classify_capture_family
    / captured_co2_operation_key docstrings for why the underlying HDF5
    dataset names differ by technology).
    """
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        design_node = f["design"]["nodes"]["period1"][node_name]
        op_node = f["operation"]["technology_operation"]["period1"][node_name]

        if tech_name is None:
            for candidate in design_node.keys():
                if classify_capture_family(design_node[candidate].keys()) is not None:
                    tech_name = candidate
                    break
            if tech_name is None:
                raise ValueError(f"No capture-capable technology found at node '{node_name}'")

        family = classify_capture_family(design_node[tech_name].keys())
        op = op_node[tech_name]
        captured_key = captured_co2_operation_key(op.keys())
        captured = op[captured_key][()]
        emitted = op["emissions_pos"][()]

    captured_full = captured[seq - 1]
    emitted_full = emitted[seq - 1]
    hours = np.arange(len(captured_full))

    fig, ax = plt.subplots(figsize=(7, 4))
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
    family_label = FAMILY_LABELS.get(family, "capture technology")
    ax.set_title(f"{node_name} ({family_label}) — captured vs. emitted CO$_2$",
                 fontsize=14, weight="bold", color=INK_PRIMARY)
    ax.grid(True, alpha=0.6, linestyle="--", linewidth=0.5, color=GRIDLINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True,
              fontsize=10.5, edgecolor=GRIDLINE)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
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
    """Stacked bar of chain-stage cost by component. Carbon tax is excluded
    on purpose -- this is the cost of running the chain, not the cost of not
    running it.

    When the scenario mixes more than one capture technology (see
    compute_cost_breakdown's 'capture_by_family'), the single "Capture" bar
    is split into one bar per family instead, so the cost impact of each
    capture technology is directly comparable -- the whole point of the
    scenario matrix in main_italy.py. A single-family run (e.g. the
    MEA-retrofit-only baseline) keeps the original single "Capture" bar."""
    by_family = cost_breakdown["capture_by_family"]
    stages = {}
    if len(by_family) > 1:
        for fam_label, costs in by_family.items():
            stages[f"Capture\n({fam_label})"] = costs
        # capture_by_family only accumulates electricity/heat for nodes with a
        # classified capture technology (see compute_cost_breakdown), while
        # cost_breakdown["capture"] includes every non-storage node
        # unconditionally - a non-storage node with no capture tech at all
        # would otherwise silently vanish from the stacked total instead of
        # being visible as its own bar. max(0, ...) guards only against
        # floating-point noise; capture_by_family is a strict subset sum of
        # capture by construction, so a large negative residual here would
        # itself indicate a different bug upstream.
        other = {
            component: max(0.0, cost_breakdown["capture"][component]
                            - sum(costs[component] for costs in by_family.values()))
            for component in COMPONENT_COLORS
        }
        if sum(other.values()) > 1e-6:
            stages["Capture\n(Other)"] = other
    else:
        stages["Capture"] = cost_breakdown["capture"]
    stages["Transport"] = cost_breakdown["transport"]
    stages["Storage"] = cost_breakdown["storage"]

    divisor = cost_breakdown["total_stored_t"] if per_tonne else 1e6
    unit = "€/t CO$_2$ stored" if per_tonne else "Million €/year"

    fig, ax = plt.subplots(figsize=(max(9.5, 2.3 * len(stages) + 3.5), 6.2))
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

    # Panel A: CCS adoption by capture technology
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    installed_df = ccs_df[ccs_df["ccs_installed"]]
    family_counts = installed_df["family"].value_counts()
    n_not = int((~ccs_df["ccs_installed"]).sum())
    present_families = [f for f in FAMILY_ORDER if family_counts.get(f, 0) > 0]
    labels = [FAMILY_LABELS[f].replace(" ", "\n", 1) for f in present_families]
    counts = [int(family_counts[f]) for f in present_families]
    colors = [STATUS_GOOD] * len(present_families)
    if n_not > 0:
        labels.append("No\nCCS")
        counts.append(n_not)
        colors.append(STATUS_MUTED)
    bars = ax.bar(labels, counts, color=colors, width=0.55)
    for rect, v in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, str(v),
                ha="center", fontsize=12, color=INK_PRIMARY, weight="bold")
    ax.set_title("CCS adoption by capture technology", fontsize=12.5, weight="bold", color=INK_PRIMARY)
    ax.set_ylabel("Number of emitters", fontsize=10.5)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.tick_params(axis="x", labelsize=9.5)

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
    mode_counts = built_arcs.groupby("mode").size()
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
    plot_map_sized_by_capacity(built_arcs, nodes_gdf, ccs_df, summary)
    plot_network_map_cost_factor(built_arcs, nodes_gdf, ccs_df, summary)
    plot_trunk_highlight(built_arcs, nodes_gdf, ccs_df, summary)
    plot_emitter_zoom(RESULTS_H5, node_name="SILLA 2", tech_name="WasteToEnergyEmitter_existing")
    plot_node_inflow(RESULTS_H5, node_name="Eni S.p.A Casalborsetti")
    plot_cost_breakdown(cost_breakdown, per_tonne=True)
    plot_cost_breakdown(cost_breakdown, per_tonne=False)
    plot_summary_dashboard(built_arcs, ccs_df, cost_breakdown)


if __name__ == "__main__":
    main()
