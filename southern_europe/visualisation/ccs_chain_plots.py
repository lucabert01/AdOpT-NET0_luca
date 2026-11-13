"""
Conference-ready visualizations for the CCS chain optimization case study.

Produces five figures from a solved optimization_results.h5:

  1. network_map.png       - Italy map: built CO2 transport network
                                        (pipeline/truck/railway), emitters colored
                                        by whether CCS was installed, transport
                                        hubs and the storage site.
  1b. network_map_sized_by_capacity.png - same map, but emitter bubble
                                        area scales with technology capacity
                                        (design "size" field, t/h).
  1c. network_map_cost_factor.png - same built network + CCS status,
                                        on the grayscale integrated cost-factor
                                        grid (as in cost_factor_grid_map_italy.py).
  1e. italy_cost_factor_emitters_storage.png/.pdf - input-data map: the same
                                        cost-factor grid with every emitter
                                        marked by sector, transport hubs and
                                        the storage site (no results needed).
  1f. italy_{pipeline_network_<small|medium|large>,truck_network,
      railway_network}.png/.pdf - input-data maps of the candidate arcs
                                        per transport mode / pipeline size
                                        class, same node styling as 1e.
  1d. network_map_trunk_highlight.png - one specific chain of built
                                        arcs (default: the Piacenza -> Modena-H ->
                                        Ferrara -> Ravenna -> Casalborsetti
                                        -> Porto Corsini trunk) highlighted in bold
                                        against the rest of the network, muted.

  All maps share the same base style (Italy fill/boundary, fixed lon/lat extent,
  axis labels) as routes_connection.py, via setup_base_map().
  2. emitter_zoom_<node>.png - captured vs. emitted CO2 for one
                                        CCS-equipped waste-to-energy plant.
  2b. inflow_<node>.png     - hourly CO2 received at a node (e.g. a
                                        transport hub just upstream of storage),
                                        with the built inbound pipeline capacity
                                        as a reference line and the most
                                        significant sustained low-load period (if
                                        any) annotated -- see plot_node_inflow.
  3. cost_breakdown_per_tonne.png / _per_year.png
                                      - levelized cost of capture/transport/storage,
                                        each split into capex, opex (fixed/variable),
                                        electricity and heat. Carbon tax excluded.
  4. summary_dashboard.png - headline numbers: CCS adoption, cost by
                                        chain stage, transport-mode split.
"""

import importlib.util
import h5py
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point
from pathlib import Path
from datetime import datetime, timedelta
import cmcrameri.cm as cmc
import matplotlib.colors as matplotlib_colors

# ============================================================
# Palette -- all identity encodings live in figure_style.py
# ============================================================
# One variable = one channel across every figure: colour = sector, shape =
# capture technology, line style = transport mode (neutral grey), see
# figure_style.py. Re-exported here so existing imports keep working.
import figure_style as fs
from figure_style import (  # noqa: F401
    SURFACE, LAND, COASTLINE, GRIDLINE, INK_PRIMARY, INK_SECONDARY, INK_MUTED,
    MARK_EDGE, NO_CAPTURE_GREY, MODE_COLORS, MODE_LABELS, MODE_LINESTYLES, MODE_ORDER,
    HUB_STYLE, STORAGE_STYLE, save_figure,
)

fs.apply_rc()

# 7 evenly spaced samples from cmcrameri "batlow" -- only for charts whose
# categories are NOT sectors/technologies/modes (cost components, captured
# vs. emitted time series), which never share a figure with sector colours.
BATLOW = [cmc.batlow(x) for x in np.linspace(0, 1, 7)]

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

STATUS_MUTED = NO_CAPTURE_GREY   # no capture / muted context
STATUS_CRITICAL = fs.HIGHLIGHT_INK  # reference lines / flagged periods (emphasis by weight, not hue)

# chain-stage colours for cost charts -- neutral greys (colour = sector only);
# hub/storage MAP markers use figure_style.HUB_STYLE / STORAGE_STYLE
TRANSPORT_COLOR = fs.NEUTRAL_RAMP[2]
STORAGE_COLOR = fs.NEUTRAL_RAMP[3]
TEMPORARY_STORAGE_COLOR = fs.NEUTRAL_RAMP[4]

# Name prefix of the temporary CO2 buffer storage technology main_italy.py adds
# at a scenario's "temporary_storage_nodes" (TEMPORARY_STORAGE_TECHNOLOGY =
# "TemporaryStorage_CO2", constant_injection scenarios only). Matched as a
# prefix so an "_existing" suffix would still be picked up.
TEMPORARY_STORAGE_TECH_PREFIX = "TemporaryStorage"


def is_temporary_storage(tech_name: str) -> bool:
    return tech_name.startswith(TEMPORARY_STORAGE_TECH_PREFIX)

# User-facing name for a node_metrics_paper.xlsx node_type / sector, used by
# every map/legend that shows sectors -- the only "Other" node is Ravenna's
# industrial cluster, so say so instead of a generic "Other".
SECTOR_DISPLAY_LABELS = dict(fs.SECTOR_LABELS)


def sector_display_label(sector: str) -> str:
    return fs.sector_label(sector)


# Input-data maps: every emitter is a circle coloured by sector (shape is
# reserved for capture technology on the result maps). FertilizersCombustion/
# FertilizersSMR are one physical complex (Ferrara) -> one "Fertilizers" group.
SECTOR_MAP_ORDER = list(fs.SECTOR_ORDER)
SECTOR_TO_MAP_GROUP = {"FertilizersCombustion": "Fertilizers", "FertilizersSMR": "Fertilizers"}

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
#   - CementHybridCCS   (oxyfuel + optional MEA polish, cement sector) -- two
#     families: "oxyfuel_hybrid" when the MEA polish is built (size_mea > 0),
#     "oxyfuel" when the optimizer leaves it out (size_mea == 0 is an explicit
#     option in cement_hybrid_ccs.py's MEA installed/not-installed disjunction)
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
FAMILY_ORDER = ["mea_retrofit", "oxyfuel", "oxyfuel_hybrid", "calcium_looping", "direct_capture"]
FAMILY_LABELS = {
    "mea_retrofit": "MEA",
    "oxyfuel": "Oxyfuel",
    "oxyfuel_hybrid": "Oxyfuel + MEA hybrid",
    "calcium_looping": "Calcium looping",
    "direct_capture": "Pure process CO$_2$",
}
FAMILY_MARKERS = fs.FAMILY_MARKERS
FAMILY_MARKER_SCALE = fs.FAMILY_MARKER_SCALE
# CementHybridCCS's MEA polish counts as built from this size_mea (t CO2/h)
# up -- the same threshold cement_hybrid_ccs.py's "MEA is installed"
# disjunct enforces (var_size_mea >= 0.001); below it, the technology runs as
# oxyfuel only.
MEA_INSTALLED_MIN_SIZE = 0.001

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
#   - "oxyfuel"/"oxyfuel_hybrid" -- CementHybridCCS is a plain generic technology as far
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
    "oxyfuel": "electricity_input",
    "oxyfuel_hybrid": "electricity_input",
    "calcium_looping": None,
    "direct_capture": "electricity_input",
}
HEAT_INPUT_KEY_BY_FAMILY = {
    "mea_retrofit": "heat_var_input_ccs",
    "oxyfuel": None,
    "oxyfuel_hybrid": None,
    "calcium_looping": None,
    "direct_capture": None,
}

# CO2 captured vs. emitted -- used for the emitter zoom plot
CAPTURED_COLOR = fs.NEUTRAL_RAMP[0]
EMITTED_COLOR = fs.NEUTRAL_RAMP[3]

# Cost-component palette, shared by every stage (capture/transport/storage) so
# a given component (e.g. "Capex") reads as the same color everywhere.
COMPONENT_COLORS = dict(zip(["Capex", "Opex (fixed)", "Opex (variable)", "Electricity", "Heat"],
                            fs.NEUTRAL_RAMP))
# chain-stage colours (cost charts): neutral too -- see figure_style.py
STAGE_COLORS_NEUTRAL = {"Capture": fs.NEUTRAL_RAMP[0], "Transport": fs.NEUTRAL_RAMP[2],
                        "Storage": fs.NEUTRAL_RAMP[3], "Temporary storage": fs.NEUTRAL_RAMP[4]}


def text_color_on(fill) -> str:
    """White or black label for a fill, by its luminance."""
    r, g, b = matplotlib_colors.to_rgb(fill)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK_PRIMARY
STAGE_LABELS = ["Capture", "Transport", "Storage"]

# ============================================================
# Paths
# ============================================================
RESULTS_H5 = Path("../Results_CCSchainOptimization/20260710184058_emissions_minC-1/optimization_results.h5")
RESULTS_ROOT = Path("../Results_CCSchainOptimization")


# Folders from the SAME modelhub.solve() call that land within this long of
# each other are treated as one run's internal stages (see find_run_h5) --
# comfortably longer than the ~40-90s gap observed between a run's own
# stages, comfortably shorter than the hours/days between separate reruns.
_SAME_RUN_GAP = timedelta(minutes=10)


def find_run_h5(scenario_name: str, objective: str, carbon_tax: int | None = None) -> Path:
    """
    Resolves scenario_name + objective (see main_italy.py's SCENARIOS, where
    each entry is now one (name, objective, carbon_tax) combination -- e.g.
    "technology_selection" is run with objective="costs" at three different
    carbon_tax levels (150/200/250) and once with objective="emissions_minC")
    to that run's optimization_results.h5, instead of hand-editing a timestamp
    string here every time main_italy.py is re-run.

    :param carbon_tax: disambiguates between multiple runs that share the same
        scenario_name AND objective but differ in carbon_tax (main_italy.py's
        cost-minimizing tax sweep at 150/200/250 EUR/tonne all land in
        Results_CCSchainOptimization/<scenario_name>/ with objective="costs").
        When given, only that exact tax level's folder
        (tax<carbon_tax>_<objective>_<scenario_name>-*) is matched. When
        omitted (the default), ANY tax level matches -- correct for scenarios
        that don't vary carbon_tax (e.g. emissions_minC runs), but ambiguous
        for a tax sweep, where it silently falls back to "most recently run"
        (see below) rather than picking a specific level.

    Globs Results_CCSchainOptimization/<scenario_name>/ for every folder
    matching *_<objective>_<scenario_name>-* (or, with carbon_tax given,
    *_tax<carbon_tax>_<objective>_<scenario_name>-*) -- main_italy.py's
    case_name convention (case_name itself is always further prefixed with a
    run timestamp by the reporting layer, hence the leading "*" in both
    patterns). A single main_italy.py run can call modelhub's
    _call_solver() -- which writes a fresh timestamped results folder every
    time it's invoked -- more than once, e.g.:
      - objective="emissions_minC" runs modelhub._optimize_costs_minE(),
        which ALWAYS solves twice: first a pure emissions-net minimization
        (cost is irrelevant to that stage -- it can and does build a wildly
        more expensive, cost-blind network), then a second solve minimizing
        cost subject to emissions <= 1.001x the first stage's result. Both
        solves get their own results folder, timestamped ~40-90s apart.
      - main_italy.py can also leave behind a design-days/clustering
        pre-solve in a separate, nearly-identically-timestamped folder ahead
        of the real full-resolution solve (see the 2026-09-14
        technology_selection case this was first spotted in).
    In both cases the LAST folder written is the one that actually reflects
    the requested objective (the final cost-minimizing solve; the real
    full-resolution solve) -- so candidates are first clustered by timestamp
    proximity (gaps <= _SAME_RUN_GAP = same modelhub.solve() call) and only
    the latest-timestamped folder per cluster is kept. This matters a lot for
    emissions_minC specifically: picking by total_cost alone (the previous
    approach) picks the WRONG stage there, since the cost-blind first-stage
    solve's total_cost is reliably the larger of the two (confirmed via
    cost_netws: e.g. 4.7e9 for stage 1 vs 0.26e9 for stage 2 on the
    2026-09-17 technology_selection/emissions_minC run) -- exactly backwards
    from the design-days-pre-solve case total_cost was originally meant to
    filter for.

    Finally, across distinct runs (genuinely separate main_italy.py
    executions, not just internal stages of one), the MOST RECENT one is
    kept (previously: the one with the largest total_cost - but reruns exist
    to pick up code/data fixes, and fixes generally REDUCE cost, e.g. the
    2026-09-16 pipeline-electricity-consumption fix dropped
    technology_selection/costs's total_cost from 3.29e9 (2026-09-15 run) to
    2.87e9 (2026-09-16 run); picking by cost there silently preferred the
    stale pre-fix run over the corrected one every time).
    """
    scenario_dir = RESULTS_ROOT / scenario_name
    if carbon_tax is not None:
        glob_pattern = f"*_tax{carbon_tax}_{objective}_{scenario_name}-*"
    else:
        glob_pattern = f"*_{objective}_{scenario_name}-*"
    candidates = sorted(scenario_dir.glob(glob_pattern))

    timestamped = []
    for cand in candidates:
        h5_path = cand / "optimization_results.h5"
        if not h5_path.exists():
            continue
        try:
            ts = datetime.strptime(cand.name.split("_", 1)[0], "%Y%m%d%H%M%S")
        except ValueError:
            continue
        timestamped.append((ts, h5_path))
    timestamped.sort(key=lambda x: x[0])

    run_finalists = []
    cluster = []
    for ts, h5_path in timestamped:
        if cluster and ts - cluster[-1][0] > _SAME_RUN_GAP:
            run_finalists.append(cluster[-1])
            cluster = []
        cluster.append((ts, h5_path))
    if cluster:
        run_finalists.append(cluster[-1])

    best_path = None
    for _, h5_path in reversed(run_finalists):
        try:
            with h5py.File(h5_path, "r") as f:
                float(f["summary"]["total_cost"][()])  # sanity check: file is readable
        except (OSError, KeyError):
            continue
        best_path = h5_path
        break
    if best_path is None:
        raise FileNotFoundError(
            f"No usable optimization_results.h5 found for scenario_name="
            f"'{scenario_name}', objective='{objective}', carbon_tax={carbon_tax} "
            f"under {scenario_dir} (looked for {glob_pattern})."
        )
    return best_path


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

# Appended to every figure/table file name, e.g. the scenario's run key set by
# the _run_* drivers, so the files stay unique once copied out of their
# scenario folder (e.g. into the paper)
FIG_TAG = ""


def out_path(name: str, ext: str = "png") -> Path:
    """OUT_DIR/<name>_<FIG_TAG>.<ext> (no suffix while FIG_TAG is empty)."""
    return OUT_DIR / f"{name}{'_' + FIG_TAG if FIG_TAG else ''}.{ext}"

# Same map extent used throughout routes_connection.py, so every map in this
# project reads as one consistent system.
MAP_BOUNDS = {"minx": 6.5, "maxx": 14.0, "miny": 43.5, "maxy": 47.0}
COST_FACTOR_PIPELINE_CATEGORY = 300


def setup_base_map(ax, italy: gpd.GeoDataFrame, title: str = "", *, land: str | None = LAND,
                   lon_labels: bool = True, lat_labels: bool = True):
    """Shared base map for every Italy map: white background (sea and
    figure), light-grey Italy, thin grey boundary, dashed 1-degree
    graticule, fixed extent (MAP_BOUNDS). land=None skips the fill (when a
    raster such as the cost-factor grid is drawn underneath)."""
    ax.set_facecolor(SURFACE)
    if land is not None:
        italy.plot(ax=ax, color=land, linewidth=0, zorder=0)
    italy.boundary.plot(ax=ax, color=COASTLINE, linewidth=0.5, zorder=1)
    ax.set_xlim(MAP_BOUNDS["minx"], MAP_BOUNDS["maxx"])
    ax.set_ylim(MAP_BOUNDS["miny"], MAP_BOUNDS["maxy"])
    ax.set_xticks(np.arange(7, 14.01, 1))
    ax.set_yticks(np.arange(44, 47.01, 1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}°N"))
    ax.tick_params(axis="both", length=2, width=0.5, pad=1.5, labelsize=fs.TICK_SIZE - 0.5,
                   labelbottom=lon_labels, labelleft=lat_labels, colors="black", labelcolor="black")
    ax.grid(True, color=GRIDLINE, linewidth=0.4, linestyle=(0, (3, 2)), zorder=0.5)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(0.6)
        sp.set_zorder(50)  # above the graticule, which coincides with the frame at 14°E / 47°N
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, loc="left", fontsize=fs.PANEL_TITLE_SIZE, color=INK_PRIMARY)
    ax.set_aspect("equal")


def map_figure(width_in: float = fs.FULL_WIDTH_IN, legend_in: float = 1.0, top_in: float = 0.05):
    """Single-map figure at print size: map on top (exact equal-aspect
    height, so no dead space), `legend_in` inches underneath for the legend
    (returned as an axis-off Axes to anchor legends in)."""
    lon = MAP_BOUNDS["maxx"] - MAP_BOUNDS["minx"]
    lat = MAP_BOUNDS["maxy"] - MAP_BOUNDS["miny"]
    left, right = 0.055, 0.985
    map_h = width_in * (right - left) * lat / lon
    gap_in = 0.22
    height = top_in + map_h + gap_in + legend_in
    fig = plt.figure(figsize=(width_in, height))
    ax = fig.add_axes([left, (legend_in + gap_in) / height, right - left, map_h / height])
    legend_ax = fig.add_axes([left, 0.0, right - left, legend_in / height])
    legend_ax.axis("off")
    return fig, ax, legend_ax


def compute_cost_factor_grid(italy: gpd.GeoDataFrame, category: float = COST_FACTOR_PIPELINE_CATEGORY):
    """Integrated cost factor fishnet grid, same calculation as
    cost_factor_grid_map_italy.py / routes_connection.py."""
    fishnet = gpd.read_file(path_files_gis / "fishnet_italy_5km.shp")
    fishnet["GRID_OID"] = fishnet.index + 1  # 1-based ArcGIS object ID of the raster tables
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


def classify_capture_family(design, operation_keys=None) -> str | None:
    """
    Identifies which capture technology a node's technology block belongs to,
    from its design/nodes/period1/<node>/<tech> HDF5 group (`design`) -- robust
    to technology naming/suffixes ("_existing" etc.), since it keys off
    variables each technology CLASS writes unconditionally:

      - "size_ccs"  -- generic bolt-on MEA retrofit (technology.py's shared
                       CCS mixin, gated on Performance.ccs.possible)
      - "size_mea"  -- CementHybridCCS; written by
                       cement_hybrid_ccs.py:write_results_tec_design. Its
                       MEA polish is optional, so this is "oxyfuel_hybrid"
                       when size_mea >= MEA_INSTALLED_MIN_SIZE and "oxyfuel"
                       (oxyfuel only) otherwise
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

    Storage technologies (STOR/SINK, identified by the "storage_level" series
    stor.py/sink.py write to their operation group) are excluded from
    "direct_capture" explicitly: the temporary CO2 buffer storage of the
    constant_injection scenarios (TemporaryStorage_CO2) has CO2captured as
    its output carrier too, so it writes "CO2captured_output" exactly like
    FertilizerSMREmitter, but it only releases CO2 captured elsewhere.

    Returns None for technologies with none of these keys (transport/storage
    technologies, or an emitter with no capture at all).
    """
    keys = set(design.keys())
    if "size_ccs" in keys:
        return "mea_retrofit"
    if "size_mea" in keys:
        size_mea = float(design["size_mea"][()][0])
        return "oxyfuel_hybrid" if size_mea >= MEA_INSTALLED_MIN_SIZE else "oxyfuel"
    if "size_cal" in keys:
        return "calcium_looping"
    if operation_keys is not None:
        op_keys = set(operation_keys)
        if "CO2captured_output" in op_keys and "storage_level" not in op_keys:
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


EMITTER_S = 26          # marker area (pt^2) of a capturing emitter on a full-width map
NO_CAPTURE_S = 9        # hollow grey circle of a non-capturing emitter
HUB_S = 20
STORAGE_S = 115


def _row_sector(row) -> str:
    """Colour group (figure_style.SECTOR_ORDER) of a ccs_df row, from its
    technology name (same mapping _real_emitter_rows collapses by)."""
    tech = row.get("tech") if hasattr(row, "get") else None
    return fs.sector_group(_tech_sector(tech)) if isinstance(tech, str) else "Other"


def _scatter_emitter(ax, x, y, family, installed, s, zorder,
                      edgecolor=None, linewidth=None, alpha=None, area_scale=True, sector=None):
    """Draws one emitter on a result map.

    Capturing emitter: fill = SECTOR colour, shape = capture technology
    (FAMILY_MARKERS), thin dark outline. Emitter without capture: small
    hollow grey circle (no technology to show), so it recedes.
    edgecolor/linewidth let callers (e.g. the trunk-highlight map) override
    the outline. area_scale=False skips the shape fill-ratio correction, for
    maps where `s` is itself a magnitude encoding (plot_map_sized_by_capacity)."""
    if not installed:
        kwargs = dict(marker="o", s=s * NO_CAPTURE_S / EMITTER_S if area_scale else s,
                      facecolors="none", edgecolors=NO_CAPTURE_GREY,
                      linewidths=0.7, zorder=zorder - 2)
        if alpha is not None:
            kwargs["alpha"] = alpha
        ax.scatter(x, y, **kwargs)
        return
    marker = FAMILY_MARKERS.get(family, "o")
    scale = FAMILY_MARKER_SCALE.get(family, 1.0) if area_scale else 1.0
    kwargs = dict(
        marker=marker, s=s * scale,
        facecolors=fs.sector_color(sector),
        edgecolors=edgecolor if edgecolor is not None else MARK_EDGE,
        linewidths=linewidth if linewidth is not None else 0.45,
        zorder=zorder,
    )
    if alpha is not None:
        kwargs["alpha"] = alpha
    ax.scatter(x, y, **kwargs)


def draw_hub(ax, x, y, s=HUB_S, zorder=20, **override):
    style = {**HUB_STYLE, **override}
    ax.scatter(x, y, marker=style["marker"], s=s, facecolors=style["facecolor"],
               edgecolors=style["edgecolor"], linewidths=style["linewidth"], zorder=zorder)


def draw_storage(ax, x, y, s=STORAGE_S, zorder=25, **override):
    style = {**STORAGE_STYLE, **override}
    ax.scatter(x, y, marker=style["marker"], s=s, facecolors=style["facecolor"],
               edgecolors=style["edgecolor"], linewidths=style["linewidth"], zorder=zorder)


def route_linewidth(size_t_h: float, ref_capacity_t_h: float) -> float:
    """Line width (pt) proportional to built capacity, one scale for all modes."""
    return fs.LW_MIN + fs.LW_SPAN * size_t_h / ref_capacity_t_h


def draw_routes(ax, built_arcs: pd.DataFrame, name_to_point: dict, ref_capacity_t_h: float,
                arrows: bool = True, skip_pairs=frozenset(), color: str = fs.MODE_COLOR,
                zorder: float = 5, lw_override: float | None = None):
    """Built arcs in neutral grey: line style = mode, width = capacity (t/h)
    on one shared scale (ref_capacity_t_h), small arrowhead = flow direction.
    Truck/rail first so pipelines (usually dominant) sit on top."""
    for mode in ["CO2Truck", "CO2Railway", "CO2_Pipeline"]:
        sub = built_arcs[built_arcs["mode"] == mode]
        for _, r in sub.iterrows():
            geom = r["geometry"]
            if geom is None or (r["from"], r["to"]) in skip_pairs:
                continue
            from_point = name_to_point.get(r["from"])
            coords = _oriented_coords(geom, from_point) if from_point else list(geom.coords)
            lw = lw_override if lw_override is not None else route_linewidth(r["size"], ref_capacity_t_h)
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=MODE_LINESTYLES[mode],
                    solid_capstyle="round", dash_capstyle="butt", zorder=zorder)
            if arrows:
                _draw_arrow(ax, coords, color, 0.55, zorder=zorder + 0.5,
                            mutation_scale=4.5 + 1.0 * lw, lw=0.5)


def reference_capacity(*built_arcs_frames) -> float:
    """Largest built arc capacity (t/h) across the given frames -- the line
    width reference, shared by all panels of one figure."""
    sizes = [df["size"].max() for df in built_arcs_frames if len(df)]
    return float(max(sizes)) if sizes else 1.0


def _present_families(real_df: pd.DataFrame) -> set:
    if not len(real_df):
        return set()
    return set(real_df.loc[real_df["ccs_installed"], "family"].dropna())


def _present_sectors(real_df: pd.DataFrame) -> set:
    if not len(real_df):
        return set()
    return {_row_sector(r) for _, r in real_df[real_df["ccs_installed"]].iterrows()}


def add_map_legend(target, real_df: pd.DataFrame, ref_capacity_t_h: float | None = None, *, modes=None,
                   sectors=None, y: float = 1.0, x0: float = 0.0,
                   widths=(0.25, 0.215, 0.12, 0.17, 0.245),
                   extra_transport_handles: list = (), sector_ncol: int = 1, include_tech: bool = True,
                   emission_ref_t: float | None = None, s_range=None):
    """The one result-map legend, as titled columns: Sector (colour) |
    Capture technology (shape) | Transport (line style) | Infrastructure |
    Annual CO2 emissions (marker size, when emission_ref_t is given).
    `target` is an axis-off legend Axes or a Figure; x0/y and widths are
    fractions of it. Only items present are listed. (ref_capacity_t_h is
    accepted for backward compatibility; line widths are not keyed.)"""
    if sectors is None:
        sectors = _present_sectors(real_df)
    if modes is None:
        modes = set(MODE_ORDER)
    fams = _present_families(real_df)
    has_no_capture = bool(len(real_df)) and bool((~real_df["ccs_installed"]).any())
    columns = [("Sector (colour)", fs.sector_handles(sectors), sector_ncol, {})]
    if include_tech:
        columns.append(("Capture technology (shape)",
                        fs.technology_handles(fams, include_no_capture=has_no_capture), 1, {}))
    columns += [
        ("Transport", [*extra_transport_handles, *fs.mode_handles(modes)], 1, {}),
        ("Infrastructure", fs.node_handles(), 1, {}),
    ]
    if emission_ref_t is not None:
        columns.append(("Annual CO$_2$ emissions (size)",
                        fs.size_handles(emission_ref_t, lambda v: emitter_area(v, emission_ref_t, s_range)),
                        1, {"labelspacing": 0.75}))
    transform = target.transAxes if isinstance(target, plt.Axes) else target.transFigure
    x = x0
    legends = []
    for (title, handles, ncol, kw), w in zip(columns, widths):
        leg = fs.legend_block(target, handles, title, loc="upper left", bbox=(x, y), ncol=ncol,
                              bbox_transform=transform, **kw)
        if isinstance(target, plt.Axes):
            target.add_artist(leg)
        legends.append(leg)
        x += w
    return legends


def _capture_legend_handles(ccs_df: pd.DataFrame) -> list:
    """(kept for external callers) technology shape key for this result."""
    has_no = bool(len(ccs_df)) and bool((~ccs_df["ccs_installed"]).any())
    return fs.technology_handles(_present_families(ccs_df), include_no_capture=has_no)


# Same helper main_italy.py uses to turn the hourly emission profiles into the
# emitters' annual emissions (and from there their product demand).
_DEFINED_FUNCTIONS_PATH = Path(__file__).resolve().parent.parent / "data_process/utilities/defined_functions.py"
_BASELINE_EMISSIONS_T: float | None = None


def load_baseline_emissions_t() -> float:
    """
    Total annual CO2 emissions (t/yr) of every emitter in the source data,
    before any capture -- the reference for the "fraction avoided" reported
    on the maps (see fraction_avoided).

    Taken from the source data, NOT from the results: a technology's
    emissions in a solved run depend on the technology chosen (e.g. a cement
    plant switched to CementHybridCCS emits more CO2 per tonne of clinker
    than the CementEmitter reference its demand was derived from), so
    summing captured + vented CO2 from the results would not give the fixed
    baseline. Computed exactly as main_italy.py does: node_metrics_paper.xlsx's
    emitter nodes, each with its annual emission = the sum of its hourly
    profile in emission_profile_emitters.xlsx (calculate_annual_emission_values).
    This differs from node_metrics_paper.xlsx's own annual_flux column for some
    nodes (e.g. SILLA 2), which main_italy.py does not use.
    """
    global _BASELINE_EMISSIONS_T
    if _BASELINE_EMISSIONS_T is None:
        spec = importlib.util.spec_from_file_location("_data_process_defined_functions", _DEFINED_FUNCTIONS_PATH)
        defined_functions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(defined_functions)

        nodes = pd.read_excel(NODE_METRICS_PAPER, sheet_name="nodes")
        nodes["node_type"] = nodes["node_type"].str.strip()
        nodes["node_name"] = nodes["node_name"].str.strip()
        nodes = defined_functions.calculate_annual_emission_values(nodes, path_files_grids)
        emitters = nodes[~nodes["node_type"].isin(["Transport", "Storage"])]
        _BASELINE_EMISSIONS_T = float(emitters["annual_emission"].sum())
    return _BASELINE_EMISSIONS_T


def fraction_avoided(emissions_pos_t: float) -> float:
    """
    Fraction of the baseline emissions (load_baseline_emissions_t) avoided:
    (baseline - CO2 still emitted) / baseline, with the CO2 still emitted
    being the run's summary emissions_pos (t/yr, see load_summary).

    Deliberately NOT captured / baseline: capture technologies that produce
    more CO2 than the reference technology (e.g. oxyfuel cement) also capture
    that extra CO2, which captured / baseline would count as avoided although
    it was never part of the baseline -- making a run that captures more but
    also emits more look better than one that emits less.
    """
    baseline = load_baseline_emissions_t()
    return (baseline - emissions_pos_t) / baseline


def captured_kpi_text(total_capture_t: float, emissions_pos_t: float) -> str:
    """'<captured> t/yr captured CO2' plus the fraction of baseline emissions
    avoided (see fraction_avoided), for the maps' KPI boxes."""
    return (f"{total_capture_t:,.0f} t/y captured CO$_2$\n"
            f"{fraction_avoided(emissions_pos_t):.1%} of baseline emissions avoided")


def _node_marker_positions(point, n: int, radius: float = 0.075):
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
        rows = pd.DataFrame([{"family": None, "ccs_installed": False, "total_emissions": 0.0,
                              "capture_capacity_t_h": 0.0}])
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
    larger total_emissions as the DISPLAYED one (shape/family/tech label),
    while installed/captured_annual/total_emissions on the returned row are
    SUMMED across every candidate that collapsed into it.

    IMPORTANT: do not change this to keep only the dominant candidate's own
    numbers -- that was tried (on the theory that a cement node's non-chosen
    CementHybridCCS candidate running with non-negligible captured_annual is
    just a main_italy.py formulation bug, piecewise capex nonzero at size=0,
    and should be zeroed out) and it silently undercounted total system
    capture: summing EVERY technology row's captured_annual across the whole
    scenario reconciles exactly against the model's own summary/emissions_pos
    (total_emissions_sum - captured_sum == emissions_pos, to rounding), while
    keeping only the dominant row per node does not -- it was found to
    undercount by ~2.4 Mt/yr on the 2026-09-14 technology_selection rerun.
    So regardless of whether the dual-technology split is itself a
    modeling bug, the tonnes each candidate reports captured are real in the
    solved instance and must be counted. Only a node node_metrics_paper
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
        winner["capture_capacity_t_h"] = group["capture_capacity_t_h"].sum()
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
            ax.plot([point.x, x], [point.y, y], color=INK_MUTED,
                    linewidth=0.4, zorder=link_zorder)
        ax.scatter(point.x, point.y, marker="o", s=2.5, color=INK_SECONDARY,
                   linewidth=0, zorder=link_zorder)
    for (x, y), (_, row) in zip(positions, node_rows.iterrows()):
        kwargs = style_fn(row)
        kwargs["s"] = kwargs.get("s", EMITTER_S) * scale
        _scatter_emitter(ax, x, y, row.get("family"), bool(row.get("ccs_installed", False)),
                          area_scale=area_scale, sector=_row_sector(row), **kwargs)


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

    "capture_capacity_t_h" -- NOT the design "size" field -- is the right
    number to compare against a downstream pipeline's built size (t/h) or to
    call this technology's "CO2 capture capacity." "size" means a different
    thing per family and is frequently NOT the capture rate: for
    mea_retrofit, "size" is the HOST emitter's own production capacity
    (t/h of throughput, e.g. waste incinerated) while "size_ccs" is the
    retrofit's own capture capacity (e.g. one real run: host size 98.45 t/h,
    size_ccs/built pipeline both 72.53 t/h -- the retrofit only ever
    captures a fraction of the host's throughput, by design, so the smaller
    pipeline is correct, not a sign of stranded capacity); for
    calcium_looping (WasteCaL_CCS), "size" is likewise the host plant's
    capacity while "size_cal" is the capture reactor's own -- see
    compute_cost_breakdown's docstring for the same distinction on the cost
    side. capture_capacity_t_h sidesteps needing a per-family lookup for
    this by reading it directly off what actually matters: the technology's
    own peak captured-CO2 operating rate (max of the clustered captured
    series -- taking the max of the clustered representative hours instead
    of the full 8760h expansion is equivalent here since expansion only
    repeats existing hourly values, never creates a new peak).

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
                op_keys = list(op[node_name][tech].keys())
                family = classify_capture_family(g, op_keys)
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
                capture_capacity_t_h = float(np.max(captured_clustered))

                if captured_annual + emitted_annual < EMITTER_MATERIALITY_THRESHOLD_T:
                    continue

                rows.append(
                    {
                        "node": node_name,
                        "tech": tech,
                        "family": family,
                        "size": size,
                        "capture_capacity_t_h": capture_capacity_t_h,
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

    The temporary CO2 buffer storage of the constant_injection scenarios
    (see is_temporary_storage) is its own 'temporary_storage' stage: capex +
    fixed opex only (it has no variable opex and draws no electricity/heat,
    see TemporaryStorage_CO2.json).

    :return: dict with 'capture', 'transport', 'storage' (each a dict of
        component -> EUR/year), 'temporary_storage' (same component dict, or
        None when the run has no temporary storage technology, i.e. any
        scenario other than constant_injection), 'capture_by_family' (dict of
        family label -> same component dict, one entry per capture family
        actually present), and 'total_stored_t' (t CO2 stored per year).
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
        temporary_storage = None

        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                g = nodes[node_name][tech]
                op_keys = list(op_tech[node_name][tech].keys())
                family = classify_capture_family(g, op_keys)
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
                elif is_temporary_storage(tech):
                    if temporary_storage is None:
                        temporary_storage = {k: 0.0 for k in COMPONENT_COLORS}
                    temporary_storage["Capex"] += float(g["capex_tot"][()][0])
                    temporary_storage["Opex (fixed)"] += float(g["opex_fixed"][()][0])

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
        "temporary_storage": temporary_storage,
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
# Marker AREA range (pt^2) for emitter size = total annual CO2 emissions
EMITTER_S_RANGE_FULL = (6, 120)   # single full-width map
EMITTER_S_RANGE_PANEL = (3, 62)   # one panel of a 2x2 grid


def emitter_area(total_emissions_t: float, ref_t: float, s_range=EMITTER_S_RANGE_FULL) -> float:
    """Marker area for an emitter's total annual CO2 emissions (captured +
    vented, t/yr -- load_ccs_status's total_emissions, summed per real
    emitter by _real_emitter_rows). sqrt-compressed, as in
    plot_map_sized_by_capacity: emissions span ~4 kt to ~1.6 Mt (~400x), and
    a purely linear area would shrink every small plant to a dot."""
    s_range = s_range or EMITTER_S_RANGE_FULL
    s_min, s_max = s_range
    frac = (max(total_emissions_t, 0.0) / ref_t) ** 0.5 if ref_t > 0 else 0.0
    return s_min + (s_max - s_min) * min(frac, 1.0)


def emission_reference(*real_dfs) -> float:
    """Largest real-emitter total annual emissions (t/yr) across the given
    frames -- the marker-size reference shared by all panels of a figure."""
    vals = [df["total_emissions"].max() for df in real_dfs if len(df)]
    return float(max(vals)) if vals else 1.0


def connected_nodes(built_arcs: pd.DataFrame) -> set:
    """Names of every node touched by at least one built arc."""
    if built_arcs is None or built_arcs.empty:
        return set()
    return set(built_arcs["from"]) | set(built_arcs["to"])


def _draw_nodes_on_ax(ax, nodes_unique, ccs_df, style_fn=None, *, emission_ref_t: float | None = None,
                      s_range=EMITTER_S_RANGE_FULL, connected: set | None = None, **emitter_kwargs):
    """Storage (black star), transport hubs (white squares) and emitters
    (sector colour x technology shape, see _scatter_emitter) of a result map.
    If `connected` is given, transport hubs not in it (no built arc) are
    not drawn.
    By default each emitter's marker area encodes its total annual CO2
    emissions (emitter_area, on the emission_ref_t scale); smaller markers
    are drawn on top of larger ones so none is hidden."""
    if style_fn is None:
        if emission_ref_t is None:
            emission_ref_t = emission_reference(real_emitters_df(ccs_df))

        def style_fn(row):
            frac = min(1.0, row["total_emissions"] / emission_ref_t) if emission_ref_t > 0 else 0.0
            return dict(s=emitter_area(row["total_emissions"], emission_ref_t, s_range),
                        zorder=23 - frac)

        emitter_kwargs.setdefault("area_scale", False)
        emitter_kwargs.setdefault("crowd_scale", False)
    for _, row in nodes_unique.iterrows():
        name, ntype_raw, point = row["node_name"], row["node_type"], row.geometry
        if ntype_raw == "Storage":
            draw_storage(ax, point.x, point.y)
        elif ntype_raw == "Transport":
            if connected is None or name in connected:
                draw_hub(ax, point.x, point.y)
        else:
            _draw_node_emitters(ax, point, _real_emitter_rows(ccs_df, name), style_fn=style_fn, **emitter_kwargs)


def _draw_ccs_network_on_ax(ax, built_arcs, nodes_gdf, ccs_df, ref_capacity_t_h: float | None = None,
                            emission_ref_t: float | None = None, s_range=EMITTER_S_RANGE_FULL):
    """
    Draws the built CO2 transport network (neutral grey; line style = mode,
    width scaled by built capacity, ref_capacity_t_h) and the emitters
    (sector colour x capture-technology shape, area = total annual CO2
    emissions on the emission_ref_t scale) onto an existing ax -- the shared
    core of plot_main_map and the tax-sweep panels. Pass the same
    ref_capacity_t_h / emission_ref_t to every panel of a multi-panel figure.

    Returns real_emitters_df(ccs_df) so the caller can build a legend/KPIs.
    """
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))
    real_df = real_emitters_df(ccs_df)
    if ref_capacity_t_h is None:
        ref_capacity_t_h = reference_capacity(built_arcs)
    if emission_ref_t is None:
        emission_ref_t = emission_reference(real_df)
    draw_routes(ax, built_arcs, name_to_point, ref_capacity_t_h)
    _draw_nodes_on_ax(ax, nodes_unique, ccs_df, emission_ref_t=emission_ref_t, s_range=s_range,
                      connected=connected_nodes(built_arcs))
    return real_df


def kpi_caption(real_df: pd.DataFrame, summary: dict) -> str:
    """One-line caption above single-scenario maps (the avoided / captured
    KPIs are in the corner box instead, see kpi_box)."""
    n_installed = int(real_df["ccs_installed"].sum())
    return (f"{n_installed}/{len(real_df)} emitters capture   ·   "
            f"network capex €{summary['cost_capex_netws'] / 1e6:,.0f}M")


def kpi_box(ax, avoided: float, captured_mt: float, *, fontsize: float = fs.PANEL_TITLE_SIZE):
    """Black-framed box in the map's upper-right corner: % of baseline
    emissions avoided (fraction_avoided) and Mt/yr captured."""
    ax.annotate(f"{avoided:.0%} CO$_2$ avoided\n{captured_mt:.1f} Mt/y captured",
                xy=(1, 1), xycoords="axes fraction", xytext=(-4, -4), textcoords="offset points",
                ha="right", va="top", fontsize=fontsize, color="black", linespacing=1.12, zorder=60,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.5, boxstyle="square,pad=0.35"))


def _map_kpi_box(ax, real_df: pd.DataFrame, summary: dict):
    kpi_box(ax, fraction_avoided(summary["emissions_pos"]), real_df["captured_annual"].sum() / 1e6)


def _map_caption(ax, text):
    ax.text(0.0, 1.015, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fs.TICK_SIZE, color=INK_SECONDARY)


def plot_main_map(built_arcs, nodes_gdf, ccs_df, summary):
    italy = gpd.read_file(ITALY_SHP)
    fig, ax, lax = map_figure(legend_in=0.95, top_in=0.2)
    setup_base_map(ax, italy)
    ref = reference_capacity(built_arcs)
    real_df = _draw_ccs_network_on_ax(ax, built_arcs, nodes_gdf, ccs_df, ref)
    _map_caption(ax, kpi_caption(real_df, summary))
    _map_kpi_box(ax, real_df, summary)
    add_map_legend(lax, real_df, modes=set(built_arcs["mode"]), emission_ref_t=emission_reference(real_df))
    save_figure(fig, out_path("network_map"))
    plt.close(fig)


# ============================================================
# PLOT 1b - Network map with emitter bubble size ~ emitter capacity
# ============================================================
def plot_map_sized_by_capacity(built_arcs, nodes_gdf, ccs_df, summary,
                                size_range=(6, 160)):
    """Same map as plot_main_map, but each emitter's marker area scales with
    its total annual emissions (captured + vented, t/year) instead of a fixed
    radius -- the true measure of plant scale (see load_ccs_status). Route
    width scales with each arc's built capacity, as on every map."""
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
        frac = (capacity / max_capacity) ** 0.5
        return s_min + (s_max - s_min) * frac

    fig, ax, lax = map_figure(legend_in=1.05, top_in=0.2)
    setup_base_map(ax, italy)
    ref = reference_capacity(built_arcs)
    draw_routes(ax, built_arcs, name_to_point, ref, arrows=False)
    _draw_nodes_on_ax(
        ax, nodes_unique, ccs_df, area_scale=False, crowd_scale=False,
        connected=connected_nodes(built_arcs),
        style_fn=lambda row: dict(s=marker_area(row["total_emissions"]),
                                  zorder=22 if row["ccs_installed"] else 21),
    )
    _map_caption(ax, kpi_caption(real_df, summary))
    _map_kpi_box(ax, real_df, summary)
    add_map_legend(lax, real_df, modes=set(built_arcs["mode"]), widths=(0.25, 0.215, 0.12, 0.17))

    ref_vals = [round(max_capacity * f, -4) for f in (0.1, 0.4, 1.0)]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor=INK_SECONDARY,
               markeredgewidth=0.6, markersize=(marker_area(v)) ** 0.5, label=f"{v / 1e6:.2f} Mt/y")
        for v in ref_vals
    ]
    fs.legend_block(lax, size_handles, "Annual emissions (area)", loc="upper left", bbox=(0.86, 1.0),
                    labelspacing=0.9)
    save_figure(fig, out_path("network_map_sized_by_capacity"))
    plt.close(fig)


# ============================================================
# PLOT 1c - Network map with grayscale integrated cost-factor background
# ============================================================
def draw_cost_factor_grid(ax, fig, fishnet_clipped, *, label="Integrated pipeline cost factor (–)", cmap=None):
    """Cost-factor raster (no cell outlines) + slim colour bar matched to the
    map height. Default cmap is the light half of Greys only, so the dark
    network lines and sector markers stay legible on top; pass another cmap
    (e.g. cmcrameri navia_r) for a coloured map without overlays."""
    from matplotlib.colors import LinearSegmentedColormap
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("cost_factor_greys", plt.cm.Greys(np.linspace(0.05, 0.6, 256)))
    norm = plt.Normalize(fishnet_clipped["COST_FACTOR"].min(), fishnet_clipped["COST_FACTOR"].max())
    fishnet_clipped.plot(column="COST_FACTOR", ax=ax, cmap=cmap, norm=norm, legend=False,
                         linewidth=0, edgecolor="none", zorder=0, rasterized=True)
    ax.set_xlabel("")  # geopandas adds CRS axis labels; the degree ticks suffice
    ax.set_ylabel("")
    cax = ax.inset_axes([1.012, 0.0, 0.014, 1.0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=fs.TICK_SIZE - 0.5, length=2, width=0.5)
    cbar.set_label(label, fontsize=fs.TICK_SIZE)
    return cbar


def plot_network_map_cost_factor(built_arcs, nodes_gdf, ccs_df, summary):
    """Same built network map as plot_main_map, on the integrated
    cost-factor grid (grayscale), as in cost_factor_grid_map_italy.py."""
    italy = gpd.read_file(ITALY_SHP)
    fishnet_clipped = compute_cost_factor_grid(italy)
    fig, ax, lax = map_figure(width_in=fs.FULL_WIDTH_IN - 0.35, legend_in=0.95, top_in=0.2)
    setup_base_map(ax, italy, land=None)
    draw_cost_factor_grid(ax, fig, fishnet_clipped)
    ref = reference_capacity(built_arcs)
    real_df = _draw_ccs_network_on_ax(ax, built_arcs, nodes_gdf, ccs_df, ref)
    _map_caption(ax, kpi_caption(real_df, summary)
                 + f"   ·   cost factor for DN {COST_FACTOR_PIPELINE_CATEGORY:g}")
    _map_kpi_box(ax, real_df, summary)
    add_map_legend(lax, real_df, modes=set(built_arcs["mode"]), emission_ref_t=emission_reference(real_df))
    save_figure(fig, out_path("network_map_cost_factor"))
    plt.close(fig)


# ============================================================
# PLOT 1e - Input data: cost-factor grid + emitters by sector + storage
# ============================================================
def _load_input_nodes() -> pd.DataFrame:
    """node_metrics_paper.xlsx 'nodes' sheet with a map-group column
    (fertilizer sectors collapsed, see SECTOR_TO_MAP_GROUP)."""
    nodes = pd.read_excel(NODE_METRICS_PAPER, sheet_name="nodes")
    nodes["group"] = nodes["node_type"].map(lambda t: SECTOR_TO_MAP_GROUP.get(t, t))
    return nodes


INPUT_EMITTER_S = 30


def _draw_input_nodes(ax, nodes: pd.DataFrame, connected_ids: set | None = None,
                      emission_ref_t: float | None = None) -> set:
    """Draws every case-study node for the input-data maps: each emitter a
    CIRCLE in its sector colour (shape is reserved for capture technology on
    result maps), transport hubs as white squares, the storage site as a
    black star. Nodes hosting more than one sector (Ferrara, Piacenza) are a
    small linked ring around the true location. If `connected_ids` is
    given, transport hubs whose node_id is not in it are not drawn. If
    `emission_ref_t` is given, each circle's area encodes the sector's
    annual emissions at that node ('annual_emission' column, see
    _input_annual_emissions) on the result maps' emitter_area scale.
    Returns the sector groups present (for the legend)."""
    present_groups = set()
    for name, g in nodes.groupby("node_name", sort=False):
        point = Point(g["longitude"].iloc[0], g["latitude"].iloc[0])
        groups = list(dict.fromkeys(g["group"]))
        if "Storage" in groups:
            draw_storage(ax, point.x, point.y)
            continue
        if "Transport" in groups:
            if connected_ids is None or int(g["node_id"].iloc[0]) in connected_ids:
                draw_hub(ax, point.x, point.y)
            continue
        positions = _node_marker_positions(point, len(groups), radius=0.08)
        if len(groups) > 1:
            for x, y in positions:
                ax.plot([point.x, x], [point.y, y], color=INK_MUTED, linewidth=0.4, zorder=15)
            ax.scatter(point.x, point.y, marker="o", s=2.5, color=INK_SECONDARY, linewidth=0, zorder=16)
        for (x, y), grp in zip(positions, groups):
            present_groups.add(fs.sector_group(grp))
            if emission_ref_t is not None:
                e = float(g.loc[g["group"] == grp, "annual_emission"].fillna(0).sum())
                s = emitter_area(e, emission_ref_t, EMITTER_S_RANGE_FULL)
                z = 23 - min(1.0, e / emission_ref_t)  # small markers on top of large ones
            else:
                s, z = INPUT_EMITTER_S * (0.8 if len(groups) > 1 else 1.0), 22
            ax.scatter(x, y, marker="o", s=s,
                       facecolors=fs.sector_color(grp), edgecolors=MARK_EDGE, linewidths=0.45, zorder=z)
    return present_groups


def _input_annual_emissions(nodes: pd.DataFrame) -> pd.DataFrame:
    """Adds 'annual_emission' (t CO2/yr) exactly as main_italy.py computes it:
    calculate_annual_emission_values, i.e. the sum of each emitter's hourly
    profile in emission_profile_emitters.xlsx (real or synthetic). Node
    names/types are stripped first, as main_italy.py does."""
    import contextlib, io, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_process.utilities.defined_functions import calculate_annual_emission_values
    nodes = nodes.copy()
    nodes["node_type"] = nodes["node_type"].str.strip()
    nodes["node_name"] = nodes["node_name"].str.strip()
    with contextlib.redirect_stdout(io.StringIO()):
        return calculate_annual_emission_values(nodes, path_data_case_study / "geographical_feature")


def add_input_legend(target, sectors, *, extra_columns=(), y=1.0, widths=(0.52, 0.22, 0.26)):
    """Legend for input-data maps: Sector (colour) | [extra columns] | Infrastructure."""
    columns = [("Sector (colour)", fs.sector_handles(sectors), 2), *extra_columns,
               ("Infrastructure", fs.node_handles(), 1)]
    transform = target.transAxes if isinstance(target, plt.Axes) else target.transFigure
    x = 0.0
    for (title, handles, ncol), w in zip(columns, widths):
        leg = fs.legend_block(target, handles, title, loc="upper left", bbox=(x, y), ncol=ncol,
                              bbox_transform=transform)
        if isinstance(target, plt.Axes):
            target.add_artist(leg)
        x += w


def plot_emitters_by_sector(out_name: str = "italy_emitters_by_sector"):
    """Input map: every emitter (circle, sector colour), the transport hubs
    and the CO2 storage site, on the plain base map."""
    italy = gpd.read_file(ITALY_SHP)
    fig, ax, lax = map_figure(legend_in=0.62)
    setup_base_map(ax, italy)
    sectors = _draw_input_nodes(ax, _load_input_nodes())
    add_input_legend(lax, sectors, widths=(0.47, 0.53))
    save_figure(fig, OUT_DIR / out_name)
    plt.close(fig)


def plot_emitters_by_sector_sized(out_name: str = "italy_emitters_by_sector_sized"):
    """As plot_emitters_by_sector, but each circle's area is the emitter's
    annual CO2 emissions as used in main_italy.py (_input_annual_emissions),
    with the same sqrt-compressed area scale as the result maps."""
    italy = gpd.read_file(ITALY_SHP)
    nodes = _input_annual_emissions(_load_input_nodes())
    per_group = nodes.groupby(["node_name", "group"])["annual_emission"].sum()
    e_ref = float(per_group.max())
    fig, ax, lax = map_figure(legend_in=0.62)
    setup_base_map(ax, italy)
    sectors = _draw_input_nodes(ax, nodes, emission_ref_t=e_ref)
    size_keys = fs.size_handles(e_ref, lambda v: emitter_area(v, e_ref, EMITTER_S_RANGE_FULL))
    add_input_legend(lax, sectors, extra_columns=[("Annual CO$_2$ emissions (size)", size_keys, 1)],
                     widths=(0.47, 0.25, 0.28))
    print(f"{len(per_group[per_group > 0])} emitters, total {per_group.sum() / 1e6:.2f} Mt/y, "
          f"largest {e_ref / 1e6:.2f} Mt/y")
    save_figure(fig, OUT_DIR / out_name)
    plt.close(fig)


def plot_cost_factor_grid_only_map(out_name: str, category: float = COST_FACTOR_PIPELINE_CATEGORY,
                                   cmap=None):
    """Integrated pipeline cost-factor grid alone (no emitters, hubs or
    storage), coloured with cmap -- default cmcrameri navia_r, the green
    scale of cost_factor_grid_map_italy.py."""
    if cmap is None:
        import cmcrameri.cm as cmc
        cmap = cmc.navia_r
    italy = gpd.read_file(ITALY_SHP)
    fishnet_clipped = compute_cost_factor_grid(italy, category)
    fig, ax, _lax = map_figure(width_in=fs.FULL_WIDTH_IN - 0.35, legend_in=0.0)
    setup_base_map(ax, italy, land=None)
    draw_cost_factor_grid(ax, fig, fishnet_clipped, cmap=cmap)
    save_figure(fig, OUT_DIR / out_name)
    plt.close(fig)


def plot_cost_factor_input_map(out_name: str = "italy_cost_factor_emitters_storage",
                               category: float = COST_FACTOR_PIPELINE_CATEGORY):
    """Case-study input map (no optimization results needed): the integrated
    pipeline cost-factor grid (grayscale) with every emitter by sector, the
    transport hubs and the CO2 storage site -- see _draw_input_nodes.
    category is the pipeline DN (mm) whose cost-factor coefficients are
    mapped (see compute_cost_factor_grid)."""
    italy = gpd.read_file(ITALY_SHP)
    fishnet_clipped = compute_cost_factor_grid(italy, category)
    fig, ax, lax = map_figure(width_in=fs.FULL_WIDTH_IN - 0.35, legend_in=0.62)
    setup_base_map(ax, italy, land=None)
    draw_cost_factor_grid(ax, fig, fishnet_clipped)
    sectors = _draw_input_nodes(ax, _load_input_nodes())
    add_input_legend(lax, sectors, widths=(0.47, 0.53))
    save_figure(fig, OUT_DIR / out_name)
    plt.close(fig)


# ============================================================
# PLOT 1f - Input data: candidate connections per transport mode
# ============================================================
NODE_METRICS_SHEET_BY_MODE = {"CO2_Pipeline": "pipeline", "CO2Truck": "truck", "CO2Railway": "railway"}
PIPELINE_CLASS_CONNECTIONS = path_data_case_study / "network_capex_metrics/pipeline_size_class_connections.xlsx"


def _excluded_pipeline_routes() -> list:
    """main_italy.py's EXCLUDED_PIPELINE_ROUTES, read from its source (importing
    main_italy.py runs its whole setup)."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parent.parent / "main_italy.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EXCLUDED_PIPELINE_ROUTES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("EXCLUDED_PIPELINE_ROUTES not found in main_italy.py")


def _candidate_connection_matrix(mode: str, size_class: str | None = None) -> pd.DataFrame:
    """node_id x node_id matrix of the arcs the optimizer may build for one
    transport mode (>0 = allowed, directed), exactly as main_italy.py reads
    it: node_metrics_paper.xlsx's per-mode sheet, for pipelines without
    main_italy.py's EXCLUDED_PIPELINE_ROUTES (both directions), and for a
    pipeline size class additionally masked by
    pipeline_size_class_connections.xlsx via the same
    load_pipeline_class_connection_matrix main_italy.py uses."""
    matrix = pd.read_excel(NODE_METRICS_PAPER, index_col=0, sheet_name=NODE_METRICS_SHEET_BY_MODE[mode])
    if mode == "CO2_Pipeline":
        nodes = _load_input_nodes().drop_duplicates("node_id")
        name_to_id = dict(zip(nodes["node_name"].str.strip(), nodes["node_id"]))
        for name_a, name_b in _excluded_pipeline_routes():
            id_a, id_b = name_to_id[name_a], name_to_id[name_b]
            matrix.at[id_a, id_b] = 0
            matrix.at[id_b, id_a] = 0
    if size_class is None:
        return matrix
    import contextlib, io, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_process.utilities.defined_functions import load_pipeline_class_connection_matrix
    # silence its emoji progress print (crashes a cp1252 Windows console)
    with contextlib.redirect_stdout(io.StringIO()):
        return load_pipeline_class_connection_matrix(PIPELINE_CLASS_CONNECTIONS, size_class, matrix)


def _route_geometries_by_pair(mode: str) -> dict:
    """{frozenset((id_a, id_b)): geometry} for one mode's route file -- same
    lookup as attach_route_geometries."""
    route_gdf = gpd.read_file(ROUTES[mode]).to_crs("EPSG:4326")
    if "Node" not in route_gdf.columns and {"from_id", "to_id"}.issubset(route_gdf.columns):
        route_gdf["Node"] = route_gdf["from_id"].astype(int).astype(str) + "," + route_gdf["to_id"].astype(int).astype(str)
    pair_to_geom = {}
    for _, row in route_gdf.iterrows():
        parts = str(row["Node"]).strip().split(",")
        if len(parts) == 2:
            try:
                pair_to_geom[frozenset((int(parts[0]), int(parts[1])))] = row.geometry
            except ValueError:
                continue
    return pair_to_geom


def _draw_arrow(ax, coords, color, fraction, reverse=False, zorder=6, mutation_scale=6, lw=0.5):
    """Small arrowhead along a route at `fraction` of its length."""
    point, (dx, dy) = _point_and_tangent_at_fraction(coords, fraction)
    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm == 0:
        return
    ux, uy = dx / norm, dy / norm
    if reverse:
        ux, uy = -ux, -uy
    arrow_len = 0.06
    start = (point[0] - ux * arrow_len / 2, point[1] - uy * arrow_len / 2)
    end = (point[0] + ux * arrow_len / 2, point[1] + uy * arrow_len / 2)
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation_scale,
                                shrinkA=0, shrinkB=0), zorder=zorder)


def plot_candidate_network_map(mode: str, size_class: str | None = None, out_name: str | None = None):
    """Case-study input map of the candidate arcs for one transport mode
    (pipeline per size class, truck, railway) on the same base map and node
    styling as plot_emitters_by_sector. Each corridor is drawn once along
    its real route geometry (straight line if none) in the mode's neutral
    line style, with one arrowhead per allowed direction."""
    italy = gpd.read_file(ITALY_SHP)
    nodes = _load_input_nodes()
    id_to_point = {
        int(r["node_id"]): Point(r["longitude"], r["latitude"])
        for _, r in nodes.drop_duplicates("node_id").iterrows()
    }
    matrix = _candidate_connection_matrix(mode, size_class)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    pair_to_geom = _route_geometries_by_pair(mode)
    color = fs.MODE_COLOR

    fig, ax, lax = map_figure(legend_in=0.62)
    setup_base_map(ax, italy)

    n_arcs, drawn_pairs = 0, set()
    for a in matrix.index:
        for b in matrix.columns:
            if a == b or not matrix.loc[a, b] > 0:
                continue
            n_arcs += 1
            pair = frozenset((a, b))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            backward = b in matrix.index and a in matrix.columns and matrix.loc[b, a] > 0
            p_a, p_b = id_to_point.get(a), id_to_point.get(b)
            if p_a is None or p_b is None:
                continue
            geom = pair_to_geom.get(pair) or LineString([p_a, p_b])
            coords = _oriented_coords(geom, p_a)  # oriented a -> b
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=0.8, linestyle=MODE_LINESTYLES[mode],
                    dash_capstyle="butt", zorder=5)
            if backward:
                _draw_arrow(ax, coords, color, 0.4, reverse=True, mutation_scale=5.5)
                _draw_arrow(ax, coords, color, 0.6, mutation_scale=5.5)
            else:
                _draw_arrow(ax, coords, color, 0.5, mutation_scale=5.5)

    connected_ids = {n for pair in drawn_pairs for n in pair}
    sectors = _draw_input_nodes(ax, nodes, connected_ids=connected_ids)
    mode_label = MODE_LABELS[mode] + (f" ({size_class})" if size_class else "")
    arc_handle = Line2D([0], [0], color=color, lw=0.9, linestyle=MODE_LINESTYLES[mode],
                        label=f"{mode_label}\n{n_arcs} directed arcs")
    add_input_legend(lax, sectors, extra_columns=[("Candidate arcs", [arc_handle], 1)],
                     widths=(0.50, 0.25, 0.25))
    if out_name is None:
        out_name = {"CO2_Pipeline": "italy_pipeline_network", "CO2Truck": "italy_truck_network",
                    "CO2Railway": "italy_railway_network"}[mode] + (f"_{size_class}" if size_class else "")
    save_figure(fig, OUT_DIR / out_name)
    plt.close(fig)


def plot_all_candidate_network_maps():
    for size_class in ("small", "medium", "large"):
        plot_candidate_network_map("CO2_Pipeline", size_class)
    plot_candidate_network_map("CO2Truck")
    plot_candidate_network_map("CO2Railway")


# ============================================================
# PLOT 1d - Trunk line highlighted against the rest of the network
# ============================================================
TRUNK_PATH_DEFAULT = [
    "Piacenza",
    "Modena-H",
    "Ferrara",
    "Ravenna",
    "Eni S.p.A Casalborsetti",
    "Porto Corsini",
]
TRUNK_COLOR = fs.HIGHLIGHT_INK  # emphasis by weight (thick black), never by hue


def plot_trunk_highlight(built_arcs, nodes_gdf, ccs_df, summary, trunk_path: list = None):
    """Highlights one specific chain of built arcs (e.g. the main trunk line
    carrying most flow toward storage) as a thick black line; the rest of the
    network is drawn thin and light grey so the trunk reads first."""
    trunk_path = trunk_path or TRUNK_PATH_DEFAULT
    trunk_pairs = set(zip(trunk_path[:-1], trunk_path[1:]))

    italy = gpd.read_file(ITALY_SHP)
    nodes_unique = nodes_gdf.drop_duplicates(subset="node_name")
    name_to_point = dict(zip(nodes_unique["node_name"], nodes_unique.geometry))
    real_df = real_emitters_df(ccs_df)

    fig, ax, lax = map_figure(legend_in=0.95, top_in=0.2)
    setup_base_map(ax, italy)
    ref = reference_capacity(built_arcs)
    # context network: muted grey, normal mode line styles / widths
    draw_routes(ax, built_arcs, name_to_point, ref, arrows=False, skip_pairs=trunk_pairs,
                color="#b3b3b3", zorder=3)
    trunk_arcs = built_arcs[built_arcs.apply(lambda r: (r["from"], r["to"]) in trunk_pairs, axis=1)]
    for a, b in zip(trunk_path[:-1], trunk_path[1:]):
        if trunk_arcs[(trunk_arcs["from"] == a) & (trunk_arcs["to"] == b)].empty:
            print(f"Warning: no built arc found for trunk leg {a} -> {b}")
    draw_routes(ax, trunk_arcs, name_to_point, ref, color=TRUNK_COLOR, zorder=8, lw_override=3.2)

    _draw_nodes_on_ax(ax, nodes_unique, ccs_df)
    _map_caption(ax, f"Trunk: {' → '.join(trunk_path)}   ·   "
                     f"{trunk_arcs['size'].min():,.0f}–{trunk_arcs['size'].max():,.0f} t/h   ·   "
                     f"capex €{trunk_arcs['capex'].sum() / 1e6:,.0f}M")
    add_map_legend(lax, real_df, modes=set(built_arcs["mode"]), emission_ref_t=emission_reference(real_df),
                   extra_transport_handles=[Line2D([0], [0], color=TRUNK_COLOR, lw=3.2, label="Trunk line")])
    save_figure(fig, out_path("network_map_trunk_highlight"))
    plt.close(fig)


# ============================================================
# PLOT 2 - Emitter zoom-in: captured vs. emitted CO2
# ============================================================
def plot_emitter_zoom(h5_path: Path, node_name: str = "SILLA 2", tech_name: str | None = None):
    """
    Captured vs. emitted CO2 for one node's capture technology.

    tech_name can be left unset -- the node's DOMINANT capture-capable
    technology (largest captured+emitted, i.e. total_emissions -- same
    "winner" convention as _real_emitter_rows._collapse, so this always
    matches whichever candidate the rest of the figures/dashboard treat as
    the real emitter) is auto-detected, so this works for any of the three
    capture families without hardcoding a variable name (see
    classify_capture_family / captured_co2_operation_key docstrings for why
    the underlying HDF5 dataset names differ by technology). Picking merely
    the FIRST capture-capable candidate (in HDF5 key order) instead of the
    dominant one would silently show an uninstalled, near-zero-throughput
    alternative technology at any node offering more than one route (e.g.
    SILLA 2 in a technology_selection run also lists WasteCaL_CCS, which
    HDF5 happens to iterate before the actually-installed MEA retrofit).
    """
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        design_node = f["design"]["nodes"]["period1"][node_name]
        op_node = f["operation"]["technology_operation"]["period1"][node_name]

        if tech_name is None:
            best_total = -1.0
            for candidate in design_node.keys():
                if classify_capture_family(design_node[candidate], op_node[candidate].keys()) is None:
                    continue
                cand_op = op_node[candidate]
                cand_captured_key = captured_co2_operation_key(cand_op.keys())
                total = float(cand_op[cand_captured_key][()][seq - 1].sum()) + float(
                    cand_op["emissions_pos"][()][seq - 1].sum()
                )
                if total > best_total:
                    best_total, tech_name = total, candidate
            if tech_name is None:
                raise ValueError(f"No capture-capable technology found at node '{node_name}'")

        family = classify_capture_family(design_node[tech_name], op_node[tech_name].keys())
        op = op_node[tech_name]
        captured_key = captured_co2_operation_key(op.keys())
        captured = op[captured_key][()]
        emitted = op["emissions_pos"][()]

    captured_full = captured[seq - 1]
    emitted_full = emitted[seq - 1]
    hours = np.arange(len(captured_full))

    fig, ax = plt.subplots(figsize=fs.TS_SIZE_IN)
    ax.fill_between(hours, 0, captured_full, color=fs.TS_MAIN, linewidth=0, step="post",
                    label="CO$_2$ captured", zorder=3)
    ax.fill_between(hours, captured_full, captured_full + emitted_full, color=fs.TS_LIGHT, linewidth=0,
                    step="post", label="CO$_2$ emitted", zorder=3)
    ax.set_xlim(0, hours[-1])
    ax.set_ylim(0, (captured_full + emitted_full).max() * 1.18)
    ax.set_ylabel("CO$_2$ (t/h)")
    style_timeseries_axes(ax)
    # no title or plant label (the caption names the plant and technology)
    ax.legend(loc="upper right", ncol=2, frameon=False, handlelength=1.2, borderaxespad=0.3, fontsize=fs.TS_LEGEND_SIZE)
    fig.tight_layout(pad=0.3)
    save_figure(fig, out_path(f"emitter_zoom_{node_name.replace(' ', '_')}"))
    plt.close(fig)


def style_timeseries_axes(ax):
    """Shared axes for the hourly figures: hour-of-year x axis, black axes
    with a full box frame, no grid."""
    ax.set_xlabel("Hour of the year (h)", color="black", fontsize=fs.TS_LABEL_SIZE)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2000))
    ax.yaxis.label.set_color("black")
    ax.yaxis.label.set_fontsize(fs.TS_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", colors="black", labelsize=fs.TS_TICK_SIZE, width=0.8, length=3)
    ax.grid(False)
    ax.set_axisbelow(True)


# ============================================================
# PLOT 2b - CO2 received at a node (hourly)
# ============================================================
# Below this fraction of the feeding pipeline's built capacity, an hour counts
# as "reduced load" for the purposes of flagging a sustained drop.
INFLOW_LOW_LOAD_THRESHOLD = 0.85
# Minimum length of a below-threshold stretch to bother annotating (avoids
# flagging single clustered-day artifacts or short dips as if they mattered).
INFLOW_MIN_DROP_HOURS = 24 * 10
# Below-threshold runs separated by a gap no longer than this are merged into
# one drop before measuring duration -- k-means expansion (see
# load_built_arcs' docstring) can otherwise chop one real seasonal dip into a
# sawtooth of short runs.
INFLOW_MERGE_GAP_HOURS = 48
# Only this many of the deepest flagged dips get a shaded band + annotation.
INFLOW_N_ANNOTATED = 2


def _find_significant_load_drop(
    load_factor: np.ndarray, threshold: float = INFLOW_LOW_LOAD_THRESHOLD,
    min_hours: int = INFLOW_MIN_DROP_HOURS, merge_gap_hours: int = INFLOW_MERGE_GAP_HOURS,
) -> dict | None:
    """
    Finds the most severe sustained below-threshold stretch in an hourly
    load-factor series, if any qualifies.

    Adjacent below-threshold runs separated by a short above-threshold gap are
    merged first (see INFLOW_MERGE_GAP_HOURS), then every merged run shorter
    than min_hours is discarded. Among what's left, "most severe" = lowest
    mean load factor during the run (not longest), since a longer-but-milder
    dip is less worth calling out than a shorter-but-deeper one.

    :return: None if nothing qualifies, else {"start", "end" (hour indices,
        end exclusive), "mean_load", "n_hours"}.
    """
    is_low = load_factor < threshold
    if not is_low.any():
        return None

    # Contiguous below-threshold runs as (start, end) half-open hour ranges.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], is_low.astype(int), [0]))))
    runs = list(zip(edges[0::2], edges[1::2]))

    # Merge runs separated by a short above-threshold gap.
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap_hours:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    candidates = [(s, e) for s, e in merged if (e - s) >= min_hours]
    if not candidates:
        return None

    best_start, best_end = min(candidates, key=lambda se: load_factor[se[0]:se[1]].mean())
    return {
        "start": best_start,
        "end": best_end,
        "mean_load": float(load_factor[best_start:best_end].mean()),
        "n_hours": best_end - best_start,
    }


def _find_significant_load_drops(
    load_factor: np.ndarray, threshold: float = INFLOW_LOW_LOAD_THRESHOLD,
    min_hours: int = INFLOW_MIN_DROP_HOURS, merge_gap_hours: int = INFLOW_MERGE_GAP_HOURS,
) -> list[dict]:
    """
    Every sustained below-threshold stretch of an hourly load-factor series,
    with the same rules as _find_significant_load_drop (runs merged across
    short gaps, then runs shorter than min_hours discarded) -- but returning
    ALL of them, and treating the year as cyclic: a dip that runs over New
    Year (low at both the first and the last hours) is one drop, not two
    halves that may each fall under min_hours.

    :return: list of {"start", "end" (hour indices, end exclusive, end may
        exceed len(load_factor) for a drop wrapping into January),
        "mean_load", "n_hours"}, in chronological order of start.
    """
    n = len(load_factor)
    is_low = load_factor < threshold
    if not is_low.any():
        return []
    if is_low.all():
        return [{"start": 0, "end": n, "mean_load": float(load_factor.mean()), "n_hours": n}]
    # rotate so the series starts at an above-threshold hour: no drop is cut by the year boundary
    offset = int(np.flatnonzero(~is_low)[0])
    lf = np.roll(load_factor, -offset)
    low = lf < threshold
    edges = np.flatnonzero(np.diff(np.concatenate(([0], low.astype(int), [0]))))
    merged = []
    for start, end in zip(edges[0::2], edges[1::2]):
        if merged and start - merged[-1][1] <= merge_gap_hours:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    drops = []
    for start, end in merged:
        if end - start < min_hours:
            continue
        drops.append({"start": (start + offset) % n, "end": (start + offset) % n + (end - start),
                      "mean_load": float(lf[start:end].mean()), "n_hours": int(end - start)})
    return sorted(drops, key=lambda d: d["start"])


def _month_phrase(ts) -> str:
    """'early/mid/end <Mon>' for a timestamp (days 1-10 / 11-20 / 21-31)."""
    part = "early" if ts.day <= 10 else "mid" if ts.day <= 20 else "end"
    return f"{part} {ts.strftime('%b')}"


def plot_node_inflow(h5_path: Path, node_name: str = "Eni S.p.A Casalborsetti"):
    """
    CO2 arriving at a node via the network (network_inflow on the
    CO2captured carrier balance), as an hourly profile, with the built
    inbound pipeline capacity as a dashed reference line and EVERY sustained
    low-load period annotated (see _find_significant_load_drops), e.g.
    "~74% load for ~5 weeks / mid Jul – end Aug".

    Pipeline capacity is the sum of every built CO2_Pipeline arc's own size
    (t/h) feeding INTO this node (load_built_arcs' "to" column) -- trucks/rail
    are excluded since the ask is specifically about pipeline load factor.
    If no inbound pipeline arc is found, the reference line/annotations are
    skipped and only the raw inflow trace is drawn. No title (the caption
    covers it).
    """
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        inflow_clustered = f["operation"]["energy_balance"]["period1"][node_name][
            "CO2captured"
        ]["network_inflow"][()]

    inflow_full = inflow_clustered[seq - 1]
    n = len(inflow_full)
    hours = np.arange(n)

    built_arcs = load_built_arcs(h5_path)
    inbound_pipeline = built_arcs[
        (built_arcs["to"] == node_name) & (built_arcs["mode"] == "CO2_Pipeline")
    ]
    pipeline_capacity = float(inbound_pipeline["size"].sum()) if len(inbound_pipeline) else 0.0

    fig, ax = plt.subplots(figsize=fs.TS_SIZE_IN)
    ax.plot(hours, inflow_full, color=fs.TS_MAIN, linewidth=0.7, drawstyle="steps-post", zorder=3,
            label="CO$_2$ received")
    y_top = max(inflow_full.max(), pipeline_capacity) * 1.2
    ax.set_ylim(0, y_top)
    ax.set_xlim(0, hours[-1])

    drops = []
    if pipeline_capacity > 0:
        ax.axhline(pipeline_capacity, color=fs.TS_REFERENCE, linewidth=0.8, linestyle=(0, (4, 2)),
                   zorder=2, label="Built pipeline capacity")
        load_factor = inflow_full / pipeline_capacity
        drops = _find_significant_load_drops(load_factor)
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        if len(drops) == 1 and drops[0]["n_hours"] / n >= 0.85:
            # a window covering nearly the whole year is chronic under-loading, not a seasonal dip
            ax.text(0.5, 0.25, f"{drops[0]['mean_load']:.0%} average load across the year\n"
                               f"(chronically under {INFLOW_LOW_LOAD_THRESHOLD:.0%} of built capacity)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=fs.TS_ANNOT_SIZE,
                    color=INK_PRIMARY, zorder=6)
        else:
            # annotate only the two deepest dips (winter and summer troughs);
            # milder intermediate dips are left unmarked
            drops = sorted(drops, key=lambda d: d["mean_load"])[:INFLOW_N_ANNOTATED]
            # label rows: each box goes to the lowest row where it clears the
            # previous box in that row by BOX_MIN_DX (axes fraction)
            row_y, BOX_MIN_DX, row_xs = (0.17, 0.42, 0.67), 0.40, {}
            # process in left-to-right order of where the arrow points
            drops = sorted(drops, key=lambda d: (d["start"] + d["end"]) / 2 % n
                           if d["end"] - n < n - d["start"] else (d["end"] - n) / 2)
            for d in drops:
                start, end = d["start"], d["end"]
                # shaded band(s): a drop wrapping over New Year is drawn at both ends of the axis
                parts = [(start, min(end, n))] + ([(0, end - n)] if end > n else [])
                for a, b in parts:
                    ax.axvspan(a, b, color=fs.TS_BAND, linewidth=0, zorder=1)
                a, b = max(parts, key=lambda ab: ab[1] - ab[0])  # arrow to the longer visible part
                x_target = (a + b) / 2
                y_target = float(np.mean(inflow_full[int(a):int(b)]))
                weeks = d["n_hours"] / (24 * 7)
                t0, t1 = dates[start % n], dates[(end - 1) % n]
                label = (f"{d['mean_load']:.0%} load for ~{weeks:.0f} week{'s' if round(weeks) != 1 else ''}\n"
                         f"{_month_phrase(t0)} – {_month_phrase(t1)}")
                x_text = float(np.clip(x_target / n, 0.24, 0.76))
                row = next((r for r in range(len(row_y))
                            if all(abs(x_text - x) >= BOX_MIN_DX for x in row_xs.get(r, []))), len(row_y) - 1)
                row_xs.setdefault(row, []).append(x_text)
                ax.annotate(label, xy=(x_target, y_target), xycoords="data",
                            xytext=(x_text, row_y[row]), textcoords="axes fraction",
                            ha="center", va="center", fontsize=fs.TS_ANNOT_SIZE, color=INK_PRIMARY,
                            arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, lw=0.6, mutation_scale=6,
                                            shrinkA=0, shrinkB=2, connectionstyle="arc3,rad=0.15"),
                            zorder=6)

    ax.set_ylabel("CO$_2$ received (t/h)")
    style_timeseries_axes(ax)
    ax.legend(loc="upper right", ncol=2, frameon=False, handlelength=1.6, borderaxespad=0.3, fontsize=fs.TS_LEGEND_SIZE)
    fig.tight_layout(pad=0.3)
    out_file = out_path(f"inflow_{node_name.replace(' ', '_').replace('.', '')}")
    save_figure(fig, out_file)
    plt.close(fig)
    print(f"Inflow {node_name}: " + (", ".join(
        f"~{d['mean_load']:.0%} load for {d['n_hours'] / 24:.0f} days" for d in drops) or "no flagged drop"))


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
    # constant_injection scenarios only (None otherwise, see compute_cost_breakdown)
    if cost_breakdown.get("temporary_storage") is not None:
        stages["Temporary\nstorage"] = cost_breakdown["temporary_storage"]

    divisor = cost_breakdown["total_stored_t"] if per_tonne else 1e6
    unit = "€/tCO$_2$ stored" if per_tonne else "M€/y"

    fig, ax = plt.subplots(figsize=(max(9.5, 2.3 * len(stages) + 3.5), 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(stages))
    bottoms = np.zeros(len(stages))
    stage_totals = np.array([sum(v.values()) for v in stages.values()]) / divisor

    for component, color in COMPONENT_COLORS.items():
        heights = np.array([stages[s][component] for s in stages]) / divisor
        bars = ax.bar(x, heights, bottom=bottoms, color=color, width=0.55,
                      label=component, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        # direct-label segments that are large enough to read
        for i, h in enumerate(heights):
            if stage_totals[i] > 0 and h / stage_totals[i] > 0.06:
                ax.text(x[i], bottoms[i] + h / 2, f"{h:,.1f}" if per_tonne else f"{h:,.0f}",
                        ha="center", va="center", fontsize=9, color=text_color_on(color), weight="bold", zorder=4)
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
    save_figure(fig, out_path(f"cost_breakdown_{suffix}"))
    plt.close(fig)


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
    colors = [fs.KPI_INK] * len(present_families)
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
    # dedicated cost_breakdown_* figures for the component-level split)
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    cats = list(STAGE_LABELS)
    vals = [
        sum(cost_breakdown["capture"].values()),
        sum(cost_breakdown["transport"].values()),
        sum(cost_breakdown["storage"].values()),
    ]
    colors = [STAGE_COLORS_NEUTRAL[c] for c in STAGE_LABELS]
    # constant_injection scenarios only (None otherwise, see compute_cost_breakdown)
    if cost_breakdown.get("temporary_storage") is not None:
        cats.append("Temp.\nstorage")
        vals.append(sum(cost_breakdown["temporary_storage"].values()))
        colors.append(TEMPORARY_STORAGE_COLOR)
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
    colors = [fs.KPI_INK] * len(modes)
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
    save_figure(fig, out_path("summary_dashboard"))
    plt.close(fig)


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
    plot_cost_factor_input_map()
    plot_all_candidate_network_maps()
    plot_trunk_highlight(built_arcs, nodes_gdf, ccs_df, summary)
    plot_emitter_zoom(RESULTS_H5, node_name="SILLA 2", tech_name="WasteToEnergyEmitter_existing")
    plot_node_inflow(RESULTS_H5, node_name="Eni S.p.A Casalborsetti")
    plot_cost_breakdown(cost_breakdown, per_tonne=True)
    plot_cost_breakdown(cost_breakdown, per_tonne=False)
    plot_summary_dashboard(built_arcs, ccs_df, cost_breakdown)


if __name__ == "__main__":
    main()
