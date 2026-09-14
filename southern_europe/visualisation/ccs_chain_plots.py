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
#   - FertilizerSMREmitter (direct capture, FertilizersSMR sector) -- a plain
#     CONV3 technology with no Performance.ccs block at all (main_italy.py
#     deliberately skips MEA-retrofit wiring for this sector -- see
#     assign_mea_technology's skip for "FertilizersSMR"); it outputs
#     CO2captured directly as one of its base output carriers, so its own
#     capex_tot/opex_* (there is no separate retrofit component to split out)
#     IS the capture cost, same accounting convention as the other two
#     self-contained technologies -- see classify_capture_family below.
# Encoding capture technology as marker SHAPE (rather than a 5th hue) keeps
# it legible under the same batlow-derived color language already used for
# CCS status (green=captured, muted=not) and avoids adding low-contrast
# batlow hues for identity (validated: adjacent-pair contrast among mid-tone
# batlow stops falls below the CVD-safe floor for small filled markers).
FAMILY_ORDER = ["mea_retrofit", "oxyfuel_hybrid", "calcium_looping", "direct_capture"]
FAMILY_LABELS = {
    "mea_retrofit": "MEA retrofit",
    "oxyfuel_hybrid": "Oxyfuel + MEA hybrid",
    "calcium_looping": "Calcium looping",
    "direct_capture": "Direct capture (integrated)",
}
FAMILY_MARKERS = {
    "mea_retrofit": "o",
    "oxyfuel_hybrid": "^",
    "calcium_looping": "D",
    # "P" (filled plus), not "s" (square) -- transport hubs already use a
    # square marker (TRANSPORT_COLOR) elsewhere on the same maps, and reusing
    # it here for a technology family would be ambiguous with node type.
    "direct_capture": "P",
}
# matplotlib scatter `s` is a bounding-box area, so a triangle/diamond/plus of
# the same `s` as a circle reads as visually smaller (lower fill ratio) -- scale
# up so the shapes read as equal-weight on the map.
FAMILY_MARKER_SCALE = {
    "mea_retrofit": 1.0, "oxyfuel_hybrid": 1.35, "calcium_looping": 1.2, "direct_capture": 1.3,
}

# In the technology-selection scenarios, main_italy.py wires up EVERY capture
# family a sector could use as a candidate technology at each node (e.g. a
# waste-to-energy node gets both the MEA-retrofit path and a WasteCaL_CCS
# candidate) -- so load_ccs_status's HDF5 scan finds a design/operation group
# for a technology the optimizer priced out entirely (size > 0 as a leftover
# free variable, but essentially zero throughput -- see load_ccs_status
# docstring on why "size" alone can't tell real from candidate). Observed
# gap in this case study's results: every genuinely-operating technology
# clears ~47,000 t/yr of total_emissions; every never-really-used candidate
# sits below ~10 t/yr (see _run_ccs_chain_technology_selection.py's node
# dump) -- three orders of magnitude of headroom, so 1,000 t/yr is a safe,
# generously-clear cutoff rather than a tight one.
EMITTER_MATERIALITY_THRESHOLD_T = 1000.0

# Each capture technology family reports its own electricity/heat draw under a
# different operation dataset name -- canonical home for this mapping (shared
# with ccs_chain_emitter_cost_ranking.py, which imports it from here rather
# than keeping its own copy) since it's tied one-to-one to classify_capture_family
# above:
#   - "mea_retrofit"    -- the generic CCS mixin's own suffixed variables
#                          (technology.py's _define_ccs_performance)
#   - "oxyfuel_hybrid"  -- CementHybridCCS is a plain generic technology as far
#                          as operation reporting goes (no override), so its
#                          main input carrier is written under the base
#                          Technology class's unsuffixed "{carrier}_input"
#   - "calcium_looping" -- WasteCaL_CCS draws neither (the calcium-looping
#                          process is self-sufficient -- no purchased
#                          electricity/heat operation variable at all)
#   - "direct_capture"  -- FertilizerSMREmitter is likewise a plain generic
#                          technology (no specificTechnologies override), so
#                          its electricity draw is under the same unsuffixed
#                          "electricity_input" convention as oxyfuel_hybrid
#                          (electricity is its only input_carrier -- see
#                          FertilizerSMREmitter.json); it has no heat input.
ELEC_INPUT_KEY_BY_FAMILY = {
    "mea_retrofit": "electricity_var_input_ccs",
    "oxyfuel_hybrid": "electricity_input",
    "calcium_looping": None,
    "direct_capture": "electricity_input",
}
HEAT_INPUT_KEY_BY_FAMILY = {
    "mea_retrofit": "heat_var_input_ccs",
    "oxyfuel_hybrid": None,
    "calcium_looping": None,
    "direct_capture": None,
}

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

NODE_METRICS_PAPER = path_files_grids / "node_metrics_paper.xlsx"
GIS_NODES = path_files_gis / "all_nodes_italy.shp"
ITALY_SHP = path_files_gis / "italy_WGS1984.shp"
ROUTES = {
    "CO2_Pipeline": path_files_gis / "routes_distances_pipeline.shp",
    # Real road-network paths for all 30 directed arcs in node_metrics_paper.xlsx's
    # 'truck' sheet (see italy_data/geographical_feature/update_truck_arcs_from_routes_gpkg.py)
    # - supersedes the older, partial truck_italy_150.shp.
    "CO2Truck": path_files_gis / "truck_routes.gpkg",
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


def classify_capture_family(design_keys, operation_keys=None) -> str | None:
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

    A fourth family, "direct_capture", has NO distinguishing design key at all
    -- FertilizerSMREmitter (FertilizersSMR sector) is a plain CONV3
    technology with no Performance.ccs block (main_italy.py deliberately
    skips the generic MEA-retrofit wiring for it), so its design group looks
    like any ordinary non-capturing technology's (just the base
    capex_tot/opex_fixed/opex_variable technology.py writes for everyone).
    The only way to tell it apart from a genuinely non-capturing technology
    is that it still directly outputs "CO2captured" as one of its own output
    carriers (technology.py:write_results_tec_operation writes an unsuffixed
    "CO2captured_output" for any technology whose output_carrier list
    includes it) -- hence the operation_keys parameter. Pass None only when
    the caller already knows design_keys alone is sufficient (e.g. a context
    where a False positive for "direct_capture" is impossible).

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
    if operation_keys is not None and "CO2captured_output" in set(operation_keys):
        return "direct_capture"
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


def _node_marker_positions(point, n: int, radius: float = 0.055):
    """Coordinates for `n` emitter markers to be drawn at a shared node
    location. n<=1 sits exactly on the node; n>1 spreads them evenly on a
    small ring around it (radius in degrees -- every map in this file uses
    ax.set_aspect("equal") on raw lon/lat, so the same radius reads as the
    same on-screen distance in both directions)."""
    if n <= 1:
        return [(point.x, point.y)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return [(point.x + radius * np.cos(a), point.y + radius * np.sin(a)) for a in angles]


def _node_rows_or_default(ccs_df: pd.DataFrame, node_name: str) -> pd.DataFrame:
    """ccs_df rows for one node, or a synthetic single 'no capture tech at
    all' row -- the fallback the old ccs_map.get(name, False) / family_map.get
    dicts used to give any node with zero capture-classified technologies."""
    rows = ccs_df[ccs_df["node"] == node_name]
    if rows.empty:
        rows = pd.DataFrame([{"family": None, "ccs_installed": False, "total_emissions": 0.0}])
    return rows


_NODE_SECTOR_CACHE: dict | None = None


def _load_node_sectors() -> dict:
    """
    Per-node set of genuinely distinct real-world emitter sectors, from
    node_metrics_paper.xlsx's 'nodes' sheet -- the authoritative source for
    how many physically separate plants share one optimization node.

    Almost every node lists exactly one sector here even when ccs_df lists
    more than one CANDIDATE technology for it -- e.g. every cement node's
    CementEmitter_existing + CementHybridCCS pair are two alternative
    capture routes the optimizer can pick between for the SAME plant, not
    two plants. Only a couple of nodes genuinely aggregate more than one
    co-located plant and appear as multiple rows here, one per sector:
    Ferrara (Waste + FertilizersCombustion + FertilizersSMR) and Piacenza
    (Cement + Waste). See _real_emitter_rows, which uses this to decide how
    many markers a node actually needs.
    """
    global _NODE_SECTOR_CACHE
    if _NODE_SECTOR_CACHE is None:
        nodes_sheet = pd.read_excel(NODE_METRICS_PAPER, sheet_name="nodes")
        non_emitter_types = {"Transport", "Storage", "Other"}
        _NODE_SECTOR_CACHE = {
            node_name: set(group["node_type"]) - non_emitter_types
            for node_name, group in nodes_sheet.groupby("node_name")
        }
        _NODE_SECTOR_CACHE = {k: v for k, v in _NODE_SECTOR_CACHE.items() if v}
    return _NODE_SECTOR_CACHE


def _tech_sector(tech_name: str) -> str:
    """Maps a technology name to the sector label node_metrics_paper.xlsx
    uses in its 'nodes' sheet, so alternative capture-technology candidates
    for the same physical plant (e.g. CementEmitter_existing and
    CementHybridCCS -- both just different capture routes for one cement
    plant) tag as the same sector and collapse to one emitter -- see
    _real_emitter_rows. Order matters: the Fertilizer checks must precede
    any generic check, since both fertilizer technology names could
    otherwise collide with a broader substring."""
    if "FertilizerCombustion" in tech_name:
        return "FertilizersCombustion"
    if "FertilizerSMR" in tech_name:
        return "FertilizersSMR"
    if "Cement" in tech_name:
        return "Cement"
    if "Waste" in tech_name:
        return "Waste"
    if "Lime" in tech_name:
        return "Lime"
    if "Refinery" in tech_name:
        return "Refining"
    return tech_name


def _real_emitter_rows(ccs_df: pd.DataFrame, node_name: str) -> pd.DataFrame:
    """
    Collapses ccs_df's per-technology rows for one node down to one row per
    genuinely distinct real-world emitter, so the map draws one marker per
    physical plant rather than one per candidate technology.

    Uses node_metrics_paper.xlsx as ground truth (_load_node_sectors): a
    node it lists only one sector for (the common case) collapses
    unconditionally to a single marker, keeping whichever technology has the
    larger total_emissions as the displayed one (the dominant/real operating
    choice -- e.g. FANNA cement plant genuinely runs both
    CementEmitter_existing and CementHybridCCS at once, splitting production
    across the old and new capacity, but node_metrics_paper says this is one
    cement plant, so it draws as one marker for the dominant technology).
    installed/captured_annual/total_emissions on the returned row are summed
    across whatever collapsed into it, so KPI text built from these rows
    still reflects the node's true total. Only a node node_metrics_paper
    lists more than one sector for (Ferrara, Piacenza) keeps multiple rows,
    grouped by sector via _tech_sector -- see _draw_node_emitters for how
    those are then rendered as a linked cluster.
    """
    rows = _node_rows_or_default(ccs_df, node_name)
    if len(rows) <= 1:
        return rows

    def _collapse(group: pd.DataFrame) -> pd.DataFrame:
        winner = group.loc[[group["total_emissions"].idxmax()]].copy()
        winner["ccs_installed"] = bool(group["ccs_installed"].any())
        winner["captured_annual"] = group["captured_annual"].sum()
        winner["total_emissions"] = group["total_emissions"].sum()
        return winner

    sectors = _load_node_sectors().get(node_name)
    if not sectors or len(sectors) <= 1:
        return _collapse(rows).reset_index(drop=True)

    tagged = rows.copy()
    tagged["_sector"] = tagged["tech"].map(_tech_sector)
    return pd.concat(
        [_collapse(group) for _, group in tagged.groupby("_sector")], ignore_index=True
    )


def real_emitters_df(ccs_df: pd.DataFrame) -> pd.DataFrame:
    """
    ccs_df collapsed to one row per genuinely distinct real-world emitter
    (see _real_emitter_rows) -- the correct basis for any "how many
    emitters" count: KPI text, the capture-family legend, and the
    adoption-by-technology dashboard panel. Raw ccs_df has one row per
    CANDIDATE capture technology and would otherwise double-count every
    node offering more than one route to the same physical plant (e.g.
    every cement node's MEA-retrofit + CementHybridCCS pair). Per-technology
    cost accounting (compute_cost_breakdown) is unaffected by this and
    intentionally keeps counting each technology separately -- both routes'
    capex/opex are real money spent even when only one is the "dominant"
    one shown on the map.
    """
    if ccs_df.empty:
        return ccs_df
    return pd.concat(
        [_real_emitter_rows(ccs_df, node) for node in ccs_df["node"].unique()],
        ignore_index=True,
    )


def _draw_node_emitters(ax, point, node_rows: pd.DataFrame, style_fn, area_scale=True,
                         crowd_scale=True, link_zorder=15):
    """
    Draws one marker per capture-classified technology present at a shared
    node location (node_rows = ccs_df filtered to this node -- see
    load_ccs_status). A handful of nodes host more than one distinct emitter
    at the same physical site -- e.g. Ferrara (WasteToEnergyEmitter +
    FertilizerCombustionEmitter + FertilizerSMREmitter) or Piacenza
    (CementEmitter + WasteToEnergyEmitter), see compute_cost_breakdown's
    docstring -- which a plain node -> value dict (the old ccs_map/family_map
    built with dict(zip(...))) silently collapses to whichever technology
    HDF5 happened to be iterated last, dropping the rest from every map.

    Instead, co-located emitters are spread evenly on a small ring around
    the true node coordinate and linked back to it with a thin spoke + a
    small center dot, so the cluster reads as "one place, several plants"
    rather than either losing information or looking like several unrelated
    nearby nodes.

    style_fn(row) -> dict of kwargs forwarded to _scatter_emitter (s,
    zorder, and optionally edgecolor/linewidth/alpha), so each caller keeps
    its own per-marker logic (e.g. trunk highlight's on_trunk styling).
    crowd_scale shrinks markers slightly at multi-emitter nodes so the ring
    doesn't overwhelm neighboring nodes; set False where `s` is itself a
    proportional data encoding (plot_map_sized_by_capacity) that must not be
    rescaled per node.
    """
    n = len(node_rows)
    positions = _node_marker_positions(point, n)
    scale = 0.82 if (n > 1 and crowd_scale) else 1.0
    if n > 1:
        for x, y in positions:
            ax.plot([point.x, x], [point.y, y], color=STATUS_MUTED,
                    linewidth=0.6, alpha=0.55, zorder=link_zorder)
        ax.scatter(point.x, point.y, marker="o", s=9, color=INK_SECONDARY,
                   edgecolor="white", linewidth=0.4, zorder=link_zorder)
    for (x, y), (_, row) in zip(positions, node_rows.iterrows()):
        kwargs = style_fn(row)
        kwargs["s"] = kwargs.get("s", 90) * scale
        _scatter_emitter(ax, x, y, row.get("family"), bool(row.get("ccs_installed", False)),
                          area_scale=area_scale, **kwargs)


# ============================================================
# Data loading
# ============================================================
def load_built_arcs(h5_path: Path) -> pd.DataFrame:
    """
    One row per built network arc (pipeline/truck/railway segment).

    "total_flow" is the true annual flow (t/yr), NOT design/networks/.../
    total_flow -- that field is the raw unweighted sum over the ~240-360
    clustered representative hours (same pitfall as design/nodes/.../
    emissions_pos, see load_ccs_status's docstring), off by roughly
    8760/n_representative_hours (~20-35x in this case study) from the real
    annual figure, while "size" (design capacity, t/h) is unaffected by
    clustering. Comparing the two without expanding total_flow via
    k_means_specs/sequence first makes a fully-utilized arc look like it is
    carrying a tiny fraction of a node's real output.
    """
    rows = []
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        net = f["design"]["networks"]["period1"]
        op_net = f["operation"]["networks"]["period1"]
        for ntype in net.keys():
            for arc_name in net[ntype].keys():
                g = net[ntype][arc_name]
                flow_clustered = op_net[ntype][arc_name]["flow"][()]
                rows.append(
                    {
                        "network": ntype,
                        "mode": NETWORK_TYPE_TO_MODE.get(ntype, ntype),
                        "from": g["fromNode"][()].decode(),
                        "to": g["toNode"][()].decode(),
                        "size": float(g["size"][()]),
                        "total_flow": float(flow_clustered[seq - 1].sum()),
                        "capex": float(g["capex"][()]),
                    }
                )
    df = pd.DataFrame(rows)
    df = df[df["size"] > 1].reset_index(drop=True)
    return _merge_parallel_pipeline_segments(df)


def _merge_parallel_pipeline_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    main_italy.py builds three separate pipeline network technologies --
    CO2_Pipeline_{small,medium,large} -- sharing the same physical corridors
    (see NETWORK_TYPE_TO_MODE above), and the optimizer routinely lays more
    than one of them in parallel on the same corridor+direction to reach a
    capacity no single discrete size class covers alone -- seen on every
    scenario's storage-bound trunk (e.g. Ravenna -> Eni S.p.A Casalborsetti
    combines the large + small classes). Drawing each such technology as its
    own overlapping line with its own arrowhead reads as accidental
    duplication rather than what it physically is: one wider corridor.
    Merges same-mode segments sharing a corridor+direction into a single row
    (summed size/flow/capex) so every consumer of load_built_arcs -- the
    maps and the transport-mode-split panel alike -- sees one row per real
    built corridor. Different modes on the same corridor (e.g. a truck route
    riding alongside a pipeline) are a separate phenomenon and are not
    merged here -- each is its own physically distinct built asset.
    """
    if df.empty:
        return df
    merged = df.groupby(["mode", "from", "to"], as_index=False).agg(
        network=("network", lambda s: "+".join(sorted(set(s)))),
        size=("size", "sum"),
        total_flow=("total_flow", "sum"),
        capex=("capex", "sum"),
    )
    return merged[["network", "mode", "from", "to", "size", "total_flow", "capex"]]


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

    Rows with total_emissions below EMITTER_MATERIALITY_THRESHOLD_T are
    dropped -- in the technology-selection scenarios, a sector's non-chosen
    candidate capture technology still has its own design/operation group
    (just essentially zero throughput), and would otherwise show up
    everywhere as a phantom "emitter" that never actually ran.
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
                op_keys = list(op[node_name][tech].keys())
                family = classify_capture_family(design_keys, op_keys)
                if family is None:
                    continue

                captured_key = captured_co2_operation_key(op_keys)
                if captured_key is None:
                    continue

                size = float(g["size"][()][0])
                emitted_clustered = op[node_name][tech]["emissions_pos"][()]
                emitted_annual = float(emitted_clustered[seq - 1].sum())
                captured_clustered = op[node_name][tech][captured_key][()]
                captured_annual = float(captured_clustered[seq - 1].sum())

                if captured_annual + emitted_annual < EMITTER_MATERIALITY_THRESHOLD_T:
                    continue

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
    For the self-contained CementHybridCCS/WasteCaL_CCS technologies AND for
    FertilizerSMREmitter ("direct_capture" -- see classify_capture_family),
    capture is inseparable from production (one technology block, no retrofit
    split available), so their full capex_tot/opex_fixed/opex_variable is the
    capture-chain cost -- the same convention already used below for the
    storage technology. EXCEPTION: WasteCaL_CCS's "size" is the host
    waste-to-energy plant's own capacity, not the capture add-on ("size_cal"
    is) -- when size_cal is 0, no calcium-looping capture equipment was
    actually built, so capex_tot/opex_* there is host-plant cost, not
    capture-chain cost, and is zeroed out.

    Electricity/heat import cost is not in technology opex (technology
    opex_variable is 0 for both the emitter and the MEA CCS component; energy
    is priced at the node's carrier balance instead -- see construct_balances.py
    :func:`construct_import_costs`). The AGGREGATE 'capture'/'storage' totals
    are attributed to whichever node consumes it: the storage node's own
    electricity draw counts as "storage", everything else counts as "capture"
    -- EXCEPT for the portion of a node's electricity/heat import that is
    drawn by a network passing through it (pipeline compression/pumping
    power, stored per node/carrier as operation/energy_balance's
    "network_consumption" and, if the pressure model is on, "compressor_input"
    -- see construct_balances.py's const_energybalance and
    genericNetworks/fluid.py's const_netw_consumption), which is transport
    cost and is carved out of the node's capture/storage bucket into
    "transport" instead. Both draws are met out of the same node-level import
    and priced at the same import_price, so the split is exact, not an
    approximation.

    The PER-FAMILY 'capture_by_family' breakdown is, separately, attributed
    per TECHNOLOGY rather than per node: each capture-classified technology's
    own electricity/heat draw is read from its own operation dataset (see
    ELEC_INPUT_KEY_BY_FAMILY/HEAT_INPUT_KEY_BY_FAMILY), the same exact
    per-technology mechanism ccs_chain_emitter_cost_ranking.py uses -- so a
    node hosting more than one capture technology at once (e.g. "Ferrara"
    running WasteToEnergyEmitter + FertilizerCombustionEmitter +
    FertilizerSMREmitter, or "Piacenza" running CementEmitter +
    WasteToEnergyEmitter) splits its electricity/heat correctly across each
    technology's own family instead of dumping the whole node's draw onto
    whichever technology happened to be iterated last. This does NOT feed
    into the aggregate 'capture' total above (kept as the node-level
    computation, deliberately independent of any technology's own reporting,
    so the grand total stays exactly right regardless of any per-technology
    edge case); by construction, in this case study only capture technologies
    and networks draw electricity/heat (a plain host emitter with no CCS has
    no input_carrier at all), so the two should reconcile to within
    floating-point noise. plot_cost_breakdown's existing "Capture (Other)"
    fallback bar (aggregate capture minus the per-family sum) is what would
    surface a real discrepancy, should that assumption ever not hold.

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

        def network_consumption_cost(node_name, carrier):
            """EUR/year of `carrier` import at this node that was drawn by a
            network passing through it (pipeline pumping/compression power),
            not by the node's own technologies -- transport cost, not
            capture/storage. Priced at the same import_price as
            carrier_import_cost, since it is met out of the same import."""
            eb = f["operation"]["energy_balance"]["period1"]
            if node_name not in eb or carrier not in eb[node_name]:
                return 0.0
            g = eb[node_name][carrier]
            price = g["import_price"][()][seq - 1]
            total = 0.0
            for key in ("network_consumption", "compressor_input"):
                if key in g:
                    total += float((g[key][()][seq - 1] * price).sum())
            return total

        def tech_carrier_cost(node_name, tech_name, op_keys, key, carrier):
            """EUR/year this specific technology's own `carrier` consumption
            cost, at the node's import price -- exact, not an approximation,
            since op_tech[node][tech][key] is that technology's own operation
            series (see ELEC_INPUT_KEY_BY_FAMILY/HEAT_INPUT_KEY_BY_FAMILY),
            never another technology's or the network's. Same helper as
            ccs_chain_emitter_cost_ranking.py's tech_carrier_cost."""
            eb = f["operation"]["energy_balance"]["period1"]
            if key is None or key not in op_keys or node_name not in eb or carrier not in eb[node_name]:
                return 0.0
            consumption = op_tech[node_name][tech_name][key][()]
            price = eb[node_name][carrier]["import_price"][()]
            return float((consumption * price)[seq - 1].sum())

        nodes = f["design"]["nodes"]["period1"]
        op_tech = f["operation"]["technology_operation"]["period1"]
        capture = {k: 0.0 for k in COMPONENT_COLORS}
        capture_by_family = {}
        storage = {k: 0.0 for k in COMPONENT_COLORS}
        transport = {k: 0.0 for k in COMPONENT_COLORS}

        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                keys = list(g.keys())
                op_keys = list(op_tech[node_name][tech].keys())
                family = classify_capture_family(keys, op_keys)
                if family is not None:
                    if family == "mea_retrofit":
                        comp_capex = float(g["capex_ccs"][()][0])
                        comp_opex_fixed = float(g["opex_fixed_ccs"][()][0])
                        comp_opex_variable = float(g["opex_variable_ccs"][()][0])
                    elif family == "calcium_looping" and float(g["size_cal"][()][0]) <= 0:
                        # WasteCaL_CCS's "size" is the host waste-to-energy
                        # plant's own throughput capacity, separate from
                        # "size_cal" (the calcium-looping capture add-on
                        # itself) -- unlike CementHybridCCS, whose oxyfuel
                        # front-end captures unconditionally, no capture
                        # equipment exists here at all when size_cal is 0, so
                        # capex_tot/opex_* (sized off the host plant, not the
                        # capture add-on) is not a capture-chain cost.
                        comp_capex = comp_opex_fixed = comp_opex_variable = 0.0
                    else:
                        comp_capex = float(g["capex_tot"][()][0])
                        comp_opex_fixed = float(g["opex_fixed"][()][0])
                        comp_opex_variable = float(g["opex_variable"][()][0])
                    # This technology's OWN electricity/heat draw, not the
                    # whole node's -- see docstring's "PER-FAMILY" paragraph.
                    comp_elec = tech_carrier_cost(
                        node_name, tech, op_keys, ELEC_INPUT_KEY_BY_FAMILY[family], "electricity"
                    )
                    comp_heat = tech_carrier_cost(
                        node_name, tech, op_keys, HEAT_INPUT_KEY_BY_FAMILY[family], "heat"
                    )
                    capture["Capex"] += comp_capex
                    capture["Opex (fixed)"] += comp_opex_fixed
                    capture["Opex (variable)"] += comp_opex_variable
                    fam_costs = capture_by_family.setdefault(family, {k: 0.0 for k in COMPONENT_COLORS})
                    fam_costs["Capex"] += comp_capex
                    fam_costs["Opex (fixed)"] += comp_opex_fixed
                    fam_costs["Opex (variable)"] += comp_opex_variable
                    fam_costs["Electricity"] += comp_elec
                    fam_costs["Heat"] += comp_heat
                elif tech == "PermanentStorage_CO2_simple":
                    storage["Capex"] += float(g["capex_tot"][()][0])
                    storage["Opex (fixed)"] += float(g["opex_fixed"][()][0])
                    storage["Opex (variable)"] += float(g["opex_variable"][()][0])

            elec_cost = carrier_import_cost(node_name, "electricity")
            heat_cost = carrier_import_cost(node_name, "heat")
            transport_elec_cost = network_consumption_cost(node_name, "electricity")
            transport_heat_cost = network_consumption_cost(node_name, "heat")
            transport["Electricity"] += transport_elec_cost
            transport["Heat"] += transport_heat_cost
            elec_cost -= transport_elec_cost
            heat_cost -= transport_heat_cost
            if node_name == storage_node:
                storage["Electricity"] += elec_cost
                storage["Heat"] += heat_cost
            else:
                # Aggregate total stays the robust node-level computation
                # (independent of any per-technology reporting quirk) -- see
                # docstring. The precise per-technology split above only
                # feeds capture_by_family, not this total.
                capture["Electricity"] += elec_cost
                capture["Heat"] += heat_cost

        net = f["design"]["networks"]["period1"]
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
        # truck_routes.gpkg carries from_id/to_id columns instead of a
        # pre-built "from,to" Node string (which the pipeline/railway route
        # shapefiles already have) - build it on the fly so the lookup below
        # works the same way for every mode.
        if "Node" not in route_gdf.columns and {"from_id", "to_id"}.issubset(route_gdf.columns):
            route_gdf["Node"] = (
                route_gdf["from_id"].astype(int).astype(str) + "," + route_gdf["to_id"].astype(int).astype(str)
            )
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
    real_df = real_emitters_df(ccs_df)

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
            node_rows = _real_emitter_rows(ccs_df, name)
            _draw_node_emitters(
                ax, point, node_rows,
                style_fn=lambda row: dict(s=100 if row["ccs_installed"] else 90,
                                           zorder=22 if row["ccs_installed"] else 21),
            )

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *_capture_legend_handles(real_df),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=4, frameon=True, fontsize=11, framealpha=0.95, edgecolor=GRIDLINE,
    )

    n_installed = int(real_df["ccs_installed"].sum())
    n_total = len(real_df)
    total_capture = real_df["captured_annual"].sum()
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
    real_df = real_emitters_df(ccs_df)

    s_min, s_max = size_range
    max_capacity = real_df["total_emissions"].max() if len(real_df) else 1
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
            node_rows = _real_emitter_rows(ccs_df, name)
            _draw_node_emitters(
                ax, point, node_rows, area_scale=False, crowd_scale=False,
                style_fn=lambda row: dict(
                    s=marker_area(row["total_emissions"]),
                    zorder=22 if row["ccs_installed"] else 21,
                    alpha=0.85 if row["ccs_installed"] else None,
                ),
            )

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *_capture_legend_handles(real_df),
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

    n_installed = int(real_df["ccs_installed"].sum())
    n_total = len(real_df)
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
    real_df = real_emitters_df(ccs_df)

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
            node_rows = _real_emitter_rows(ccs_df, name)
            _draw_node_emitters(
                ax, point, node_rows,
                style_fn=lambda row: dict(s=100 if row["ccs_installed"] else 90,
                                           zorder=22 if row["ccs_installed"] else 21),
            )

    legend_handles = [
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *_capture_legend_handles(real_df),
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

    n_installed = int(real_df["ccs_installed"].sum())
    n_total = len(real_df)
    total_capture = real_df["captured_annual"].sum()
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
    real_df = real_emitters_df(ccs_df)

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
            edge = TRUNK_COLOR if on_trunk else None
            node_rows = _real_emitter_rows(ccs_df, name)
            _draw_node_emitters(
                ax, point, node_rows,
                style_fn=lambda row: dict(
                    s=130 if on_trunk else 80, zorder=22 if on_trunk else 18,
                    edgecolor=edge, linewidth=2.8 if on_trunk else None,
                    alpha=1.0 if on_trunk else 0.55,
                ),
            )

    legend_handles = [
        Line2D([0], [0], color=TRUNK_COLOR, lw=4.5, label="Trunk line"),
        Line2D([0], [0], color=MODE_COLORS["CO2_Pipeline"], lw=2, label="Pipeline"),
        Line2D([0], [0], color=MODE_COLORS["CO2Truck"], lw=2, label="Truck"),
        Line2D([0], [0], color=MODE_COLORS["CO2Railway"], lw=2, label="Railway"),
        *_capture_legend_handles(real_df),
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
                if classify_capture_family(design_node[candidate].keys(), op_node[candidate].keys()) is not None:
                    tech_name = candidate
                    break
            if tech_name is None:
                raise ValueError(f"No capture-capable technology found at node '{node_name}'")

        family = classify_capture_family(design_node[tech_name].keys(), op_node[tech_name].keys())
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

    # Panel A: CCS adoption by capture technology -- counted per genuine
    # real-world emitter (real_emitters_df), not per candidate technology
    # row, so a node offering two alternative routes to the same physical
    # plant (e.g. every cement node's MEA-retrofit + CementHybridCCS pair)
    # is counted once, under whichever route is actually dominant there.
    real_df = real_emitters_df(ccs_df)
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    installed_df = real_df[real_df["ccs_installed"]]
    family_counts = installed_df["family"].value_counts()
    n_not = int((~real_df["ccs_installed"]).sum())
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
