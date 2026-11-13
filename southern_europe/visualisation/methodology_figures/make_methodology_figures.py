"""
Methodology figures for the Italy CCS chain case study (main_italy.py).

All figures are input-data / method figures (no optimization results needed)
and are saved (.png + .pdf, white background, final print size) next to this
script:

  1. pipeline_size_class_ranges_linear  - power-law CAPEX/km fit and the
                                          small/medium/large size-class ranges
                                          (linear axes), redrawn from
                                          analysis/output/built_arcs_true_capex.csv
                                          with the fit/breakpoints of
                                          analysis/pipeline_size_class_ranges.py
  2. italy_emitters_by_sector           - every emitter (circle, sector colour),
                                          transport hubs and the storage site
  3. italy_cost_factor_<kt>kt            - integrated pipeline cost-factor grid
                                          for a pipeline sized at COST_FACTOR_KT_PER_YEAR,
                                          with the emitters by sector on top;
                                          *_colored: grid only, green
                                          (cmcrameri navia_r) colour scale
  4. italy_{truck,railway}_network,
     italy_pipeline_network_{small,medium,large,all}
                                        - candidate arcs per transport mode;
                                          "all" is the pipeline sheet of
                                          node_metrics_paper.xlsx without the
                                          per-size-class masks of
                                          pipeline_size_class_connections.xlsx

Colours, markers and line styles come from figure_style.py (shared with the
result figures); map drawing is reused from ccs_chain_plots.py, whose OUT_DIR
is redirected here.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
VIS_DIR = FIG_DIR.parent
SOUTHERN_EUROPE_DIR = VIS_DIR.parent
REPO_ROOT = SOUTHERN_EUROPE_DIR.parent
ANALYSIS_DIR = SOUTHERN_EUROPE_DIR / "analysis"

# ccs_chain_plots resolves its data paths relative to visualisation/
os.chdir(VIS_DIR)
sys.path.insert(0, str(VIS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import ccs_chain_plots as ccp  # noqa: E402
import figure_style as fs  # noqa: E402

ccp.OUT_DIR = FIG_DIR

# Pipeline size the cost-factor map is drawn for
COST_FACTOR_KT_PER_YEAR = 250


# ============================================================
# 1. Pipeline size-class interpolation (linear axes)
# ============================================================
def plot_size_class_interpolation():
    """Redraws analysis/pipeline_size_class_ranges.py's linear-scale figure
    from its saved per-arc table (same fit, breakpoints and class ranges),
    without re-running the Oeuvray model or touching the CO2_Pipeline_{size}
    .json files that script writes. Size classes are not sectors, so the
    bands are neutral greys (light -> dark with size), labelled directly."""
    import pipeline_size_class_ranges as pscr
    from pipeline_capex_per_arc_calculator import SIZE_CLASS_MASSFLOW_RANGES_KG_S as ranges

    df = pd.read_csv(ANALYSIS_DIR / "output" / "built_arcs_true_capex.csv")
    a, b, r2 = pscr.fit_power_law(df["flow_kg_s"].values, df["capex_per_km"].values)
    breakpoints = pscr.find_breakpoints(a, b, df["flow_kg_s"].min(), df["flow_kg_s"].max())

    fig, ax = plt.subplots(figsize=(fs.HALF_WIDTH_IN * 1.45, 3.0))
    band = {"small": "#f4f4f4", "medium": "#e9e9e9", "large": "#dddddd"}
    for name, (lo, hi) in ranges.items():
        ax.axvspan(lo, hi, color=band[name], linewidth=0, zorder=0)
    for bp in breakpoints:
        ax.axvline(bp, color=fs.INK_MUTED, linestyle=(0, (1, 1.5)), linewidth=0.7, zorder=1)

    grid = np.linspace(df["flow_kg_s"].min(), df["flow_kg_s"].max(), 200)
    ax.plot(grid, a * grid ** b / 1e6, color=fs.INK_PRIMARY, linestyle=(0, (4, 2)), linewidth=1.0, zorder=3,
            label=f"Power-law fit (R$^2$ = {r2:.2f})")
    ax.scatter(df["flow_kg_s"], df["capex_per_km"] / 1e6, s=10, facecolors="none",
               edgecolors=fs.INK_SECONDARY, linewidths=0.6, zorder=4,
               label="Arcs built in reference run (true capex)")

    ymin, ymax = 0, ax.get_ylim()[1]
    for name, (lo, hi) in ranges.items():
        # the 'small' band is narrower than its label -> vertical label inside it
        rot = 90 if name == "small" else 0
        ax.text((lo + hi) / 2, ymax * 0.97, name.capitalize(), ha="center", va="top", rotation=rot,
                fontsize=fs.TICK_SIZE, fontweight="bold", color=fs.INK_SECONDARY)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, ranges["large"][1])
    ax.set_xlabel("Pipeline mass flow (kg CO$_2$/s)")
    ax.set_ylabel("CAPEX per unit length (M€/km)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fs.save_figure(fig, FIG_DIR / "pipeline_size_class_ranges_linear")
    plt.close(fig)


# ============================================================
# 2. Emitters by sector
# ============================================================
def plot_emitters_by_sector(out_name: str = "italy_emitters_by_sector"):
    """Every emitter as a circle in its sector colour (the same colours as
    the result maps and MACCs), plus the transport hubs and the storage site."""
    ccp.plot_emitters_by_sector(out_name)
    # same map, circle area = annual CO2 emissions as used in main_italy.py
    ccp.plot_emitters_by_sector_sized(out_name + "_sized")


# ============================================================
# 3. Cost factor map at a given pipeline size
# ============================================================
def pipeline_dn_for_kt_per_year(kt_per_year: float) -> float:
    """Pipeline DN (mm) the Oeuvray cost model uses for this mass flow --
    the same conversion it applies before looking up the cost-factor
    coefficients, clamped to the cost_factor_table.xlsx range just as its
    _interpolate_cost_factors does."""
    from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import (
        CO2_Pipeline_CostModel,
    )
    with contextlib.redirect_stdout(io.StringIO()):  # emoji prints crash a cp1252 console
        model = CO2_Pipeline_CostModel("CO2_Pipeline")
        flow_kg_s = kt_per_year * 1e6 / (8760 * 3600)
        dn = model._get_pipeline_category_from_massflow(flow_kg_s)
    categories = model.cost_factor_table["pipeline_category"]
    dn_clamped = min(max(dn, categories.min()), categories.max())
    print(f"{kt_per_year} kt/y = {flow_kg_s:.2f} kg/s -> DN {dn:.0f} mm"
          + (f" (outside cost-factor table, using DN {dn_clamped:.0f})" if dn_clamped != dn else ""))
    return dn_clamped


def plot_cost_factor_map(kt_per_year: float = COST_FACTOR_KT_PER_YEAR):
    dn = pipeline_dn_for_kt_per_year(kt_per_year)
    ccp.plot_cost_factor_input_map(out_name=f"italy_cost_factor_{kt_per_year:g}kt", category=dn)
    # same grid, green colour scale (cmcrameri navia_r), no emitters
    ccp.plot_cost_factor_grid_only_map(out_name=f"italy_cost_factor_{kt_per_year:g}kt_colored", category=dn)


# ============================================================
# 4. Candidate networks per transport mode
# ============================================================
def plot_candidate_networks():
    for size_class in ("small", "medium", "large"):
        ccp.plot_candidate_network_map("CO2_Pipeline", size_class)
    ccp.plot_candidate_network_map("CO2_Pipeline", None, out_name="italy_pipeline_network_all")
    ccp.plot_candidate_network_map("CO2Truck")
    ccp.plot_candidate_network_map("CO2Railway")


if __name__ == "__main__":
    plot_size_class_interpolation()
    plot_emitters_by_sector()
    plot_cost_factor_map()
    plot_candidate_networks()
