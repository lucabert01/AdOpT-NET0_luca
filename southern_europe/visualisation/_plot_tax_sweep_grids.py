"""
Ad-hoc driver: paper figures for the "technology_selection" CO2-tax sweep in
main_italy.py's SCENARIOS -- cost-minimizing at carbon_tax = 150/200/250
EUR/t plus the emissions-minimizing run (scenario order and labels:
figure_style.SCENARIOS).

  network_map_tax_sweep   2x2 maps of the built network. Colour = sector,
                          shape = capture technology, marker area = the
                          emitter's total annual CO2 emissions, line style =
                          transport mode (neutral grey). Scales shared by all
                          four panels. Scenario (CO2 tax) and KPIs (% of
                          baseline emissions avoided, Mt/yr captured) as a
                          note in each map's upper-left corner; one shared
                          legend.
  macc_tax_sweep          2x2 MACCs, bars coloured by sector, CO2 tax as a
                          dashed line in each tax panel, shared axes.
  capture_by_technology_tax_sweep
                          captured CO2 per capture technology (rows) for each
                          scenario (panels), bars stacked by sector colour --
                          technology is a position, never a colour.

Each map/MACC panel is also saved on its own (..._tax_sweep_<slug>), and
the hourly CO2 received at Eni S.p.A Casalborsetti (just upstream of
storage) is plotted per scenario (inflow_Eni_SpA_Casalborsetti_<slug>).
Every figure is saved as PNG and PDF (figure_style.save_figure), drawn at
final print size (7.2 in wide).

Reuses ccs_chain_plots._draw_ccs_network_on_ax and
ccs_chain_emitter_cost_ranking._draw_macc_on_ax, the same drawing code as
the single-scenario figures, so the panels are consistent with those.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ccs_chain_plots as m
import ccs_chain_emitter_cost_ranking as cr
import figure_style as fs

OUT_DIR = Path("ccs_chain_results/scenario_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_NAME = "technology_selection"

# Fixed x-axis extent (Mt/yr) shared by every MACC panel, so bar widths are
# directly comparable across scenarios. Widened automatically if any panel's
# total capture exceeds it.
MACC_XMAX_MT = 15

# --- 2x2 map layout (inches) ---
MAP_LEFT_IN, MAP_RIGHT_IN, MAP_COL_GAP_IN = 0.36, 0.04, 0.10
TOP_IN = 0.04         # margin above the top row (scenario labels sit inside the maps)
ROW_GAP_IN = 0.06
XTICK_IN = 0.17       # longitude tick labels under the bottom row
LEGEND_IN = 1.0


def _load_panel_data():
    """Resolves each scenario to its h5 and pre-loads everything the figures
    need from it, so each h5 is only read once."""
    nodes_gdf = m.gpd.read_file(m.GIS_NODES)
    panels = []
    for letter, (slug, objective, carbon_tax, label) in zip(fs.PANEL_LETTERS, fs.SCENARIOS):
        h5_path = m.find_run_h5(SCENARIO_NAME, objective, carbon_tax=carbon_tax)
        built_arcs = m.attach_route_geometries(m.load_built_arcs(h5_path), nodes_gdf)
        ccs_df = m.load_ccs_status(h5_path)
        macc_df, storage_node = cr.build_emitter_cost_table(h5_path)
        summary = m.load_summary(h5_path)
        real_df = m.real_emitters_df(ccs_df)
        panels.append({
            "letter": letter, "label": label, "slug": slug, "carbon_tax": carbon_tax,
            "h5_path": h5_path, "built_arcs": built_arcs, "ccs_df": ccs_df, "real_df": real_df,
            "summary": summary, "macc_df": macc_df, "storage_node": storage_node,
            "captured_mt": real_df["captured_annual"].sum() / 1e6,
            "avoided": m.fraction_avoided(summary["emissions_pos"]),
        })
        print(f"Loaded {label!r} ({SCENARIO_NAME}, {objective}, carbon_tax={carbon_tax}) <- {h5_path}")
    return nodes_gdf, panels


def _panel_title(ax, panel, *, y=1.03):
    """'a  CO2 tax 150 €/t' above a panel: bold letter, plain scenario label."""
    ax.text(0.0, y, panel["letter"], transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fs.PANEL_TITLE_SIZE + 0.5, fontweight="bold", color=fs.INK_PRIMARY)
    # offset in points, so the gap after the letter is the same at any panel width
    ax.annotate(panel["label"], xy=(0.0, y), xycoords="axes fraction", xytext=(8.5, 0),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=fs.PANEL_TITLE_SIZE, color=fs.INK_PRIMARY)


# ============================================================
# Network maps
# ============================================================
def _map_grid_axes():
    """2x2 map axes placed in inches so every panel has the exact
    equal-aspect height of MAP_BOUNDS (no dead space), plus the legend band."""
    lon = m.MAP_BOUNDS["maxx"] - m.MAP_BOUNDS["minx"]
    lat = m.MAP_BOUNDS["maxy"] - m.MAP_BOUNDS["miny"]
    W = fs.FULL_WIDTH_IN
    pw = (W - MAP_LEFT_IN - MAP_RIGHT_IN - MAP_COL_GAP_IN) / 2
    ph = pw * lat / lon
    H = TOP_IN + ph + ROW_GAP_IN + ph + XTICK_IN + LEGEND_IN
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        top = TOP_IN + r * (ph + ROW_GAP_IN)
        for c in range(2):
            left = MAP_LEFT_IN + c * (pw + MAP_COL_GAP_IN)
            axes[r, c] = fig.add_axes([left / W, 1 - (top + ph) / H, pw / W, ph / H])
    legend_box = (MAP_LEFT_IN / W, 0.0, 1 - (MAP_LEFT_IN + MAP_RIGHT_IN) / W, (LEGEND_IN - 0.08) / H)
    return fig, axes, legend_box


def _map_note(ax, panel, *, size: float = fs.PANEL_TITLE_SIZE):
    """Note in the map's upper-left corner (outside Italy, plain white):
    bold panel letter + scenario (CO2 tax). The KPIs -- % of baseline
    emissions avoided (fraction_avoided) and Mt/yr captured -- go in a
    black-framed box in the upper-right corner."""
    backing = dict(facecolor=fs.SURFACE, edgecolor="none", pad=0.4, alpha=0.7)
    ax.annotate(panel["letter"], xy=(0, 1), xycoords="axes fraction", xytext=(3.5, -3),
                textcoords="offset points", ha="left", va="top", fontsize=size + 0.5,
                fontweight="bold", color=fs.INK_PRIMARY, zorder=30, bbox=backing)
    ax.annotate(panel["label"], xy=(0, 1), xycoords="axes fraction", xytext=(3.5 + size * 1.25, -3),
                textcoords="offset points", ha="left", va="top", fontsize=size,
                color=fs.INK_PRIMARY, zorder=30, bbox=backing)
    m.kpi_box(ax, panel["avoided"], panel["captured_mt"], fontsize=size - 1.0)


def plot_tax_sweep_maps(nodes_gdf, panels):
    italy = m.gpd.read_file(m.ITALY_SHP)
    # shared scales across all panels: line width (built capacity) and marker area (emissions)
    ref = m.reference_capacity(*[p["built_arcs"] for p in panels])
    e_ref = m.emission_reference(*[p["real_df"] for p in panels])

    fig, axes, (lx, ly, lw, lh) = _map_grid_axes()
    for (r, c), panel in zip([(0, 0), (0, 1), (1, 0), (1, 1)], panels):
        ax = axes[r, c]
        m.setup_base_map(ax, italy, lon_labels=(r == 1), lat_labels=(c == 0))
        m._draw_ccs_network_on_ax(ax, panel["built_arcs"], nodes_gdf, panel["ccs_df"], ref,
                                  emission_ref_t=e_ref, s_range=m.EMITTER_S_RANGE_PANEL)
        _map_note(ax, panel, size=fs.PANEL_TITLE_SIZE - 0.5)

    all_real = pd.concat([p["real_df"] for p in panels], ignore_index=True)
    modes = set().union(*[set(p["built_arcs"]["mode"]) for p in panels])
    legend_ax = fig.add_axes([lx, ly, lw, lh])
    legend_ax.axis("off")
    _sweep_legend(legend_ax, all_real, modes, e_ref)
    fs.save_figure(fig, OUT_DIR / "network_map_tax_sweep")
    plt.close(fig)

    # --- one figure per scenario (same scales as the grid) ---
    for panel in panels:
        fig, ax, lax = m.map_figure(legend_in=1.0, top_in=0.04)
        m.setup_base_map(ax, italy)
        m._draw_ccs_network_on_ax(ax, panel["built_arcs"], nodes_gdf, panel["ccs_df"], ref,
                                  emission_ref_t=e_ref, s_range=m.EMITTER_S_RANGE_FULL)
        _map_note(ax, panel, size=fs.PANEL_TITLE_SIZE + 1)
        m.add_map_legend(lax, panel["real_df"], modes=modes, emission_ref_t=e_ref,
                         s_range=m.EMITTER_S_RANGE_FULL)
        fs.save_figure(fig, OUT_DIR / f"network_map_tax_sweep_{panel['slug']}")
        plt.close(fig)


def _sweep_legend(lax, all_real, modes, e_ref):
    """Shared map legend: Sector (3x2) over Capture technology (3x2) on the
    left; Transport over Infrastructure in the middle; marker-size key
    (annual CO2 emissions, same area function as the panels) on the right."""
    sectors = m._present_sectors(all_real)
    fams = m._present_families(all_real)
    has_no = bool((~all_real["ccs_installed"]).any())
    size_keys = fs.size_handles(e_ref, lambda v: m.emitter_area(v, e_ref, m.EMITTER_S_RANGE_PANEL))
    blocks = [
        ("Sector (colour)", fs.sector_handles(sectors), 3, (0.0, 1.0), {}),
        ("Capture technology (shape)", fs.technology_handles(fams, include_no_capture=has_no), 3,
         (0.0, 0.47), {}),
        ("Transport", fs.mode_handles(modes), 1, (0.565, 1.0), {}),
        ("Infrastructure", fs.node_handles(), 1, (0.565, 0.47), {}),
        ("Annual CO$_2$ emissions (size)", size_keys, 1, (0.78, 1.0), {"labelspacing": 0.7}),
    ]
    for title, handles, ncol, (x, y), kw in blocks:
        leg = fs.legend_block(lax, handles, title, loc="upper left", bbox=(x, y), ncol=ncol,
                              bbox_transform=lax.transAxes, columnspacing=0.9, **kw)
        lax.add_artist(leg)


# ============================================================
# MACCs
# ============================================================
def _macc_limits(panels):
    max_capture_mt = max(p["macc_df"]["captured_annual_t"].sum() / 1e6 for p in panels)
    if max_capture_mt > MACC_XMAX_MT:
        print(f"Warning: max capture {max_capture_mt:.2f} Mt/yr exceeds MACC_XMAX_MT={MACC_XMAX_MT}; widening x-axis")
    x_max = max(MACC_XMAX_MT, max_capture_mt)
    # shared y: tallest bar or tax line + 10% headroom, rounded up to 50 €/t
    max_cost = max(max(p["macc_df"]["total_eur_per_t"].max(), p["carbon_tax"] or 0) for p in panels)
    y_max = np.ceil(max_cost * 1.1 / 50) * 50
    return x_max, y_max


def _draw_macc_panel(ax, panel, x_max, y_max):
    """One scenario's MACC on ax: sector-coloured bars, dashed CO2-tax line
    (direct-labelled) in tax panels. Returns the legend handles."""
    handles = cr._draw_macc_on_ax(ax, panel["macc_df"], annotate_fertilizers=False,
                                  carbon_tax=panel["carbon_tax"])
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 50))
    if panel["carbon_tax"] is not None:
        # left end: the cheapest bars sit far below the tax, so the label never
        # collides with a bar (the right end is where bars approach the tax)
        ax.text(x_max * 0.012, panel["carbon_tax"] + y_max * 0.015, f"CO$_2$ tax {panel['carbon_tax']} €/tCO$_2$",
                ha="left", va="bottom", fontsize=fs.TICK_SIZE, color=fs.INK_PRIMARY, zorder=6,
                bbox=dict(facecolor=fs.SURFACE, edgecolor="none", pad=0.6))
    return handles


def plot_tax_sweep_macc(panels):
    x_max, y_max = _macc_limits(panels)
    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL_WIDTH_IN, 4.1), sharex=True, sharey=True)
    for ax, panel in zip(axes.flat, panels):
        _draw_macc_panel(ax, panel, x_max, y_max)
        _panel_title(ax, panel)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    fig.supxlabel(fs.CAPTURED_LABEL, fontsize=fs.LABEL_SIZE, y=0.075)
    fig.supylabel(fs.COST_LABEL, fontsize=fs.LABEL_SIZE, x=0.012)

    all_macc = pd.concat([p["macc_df"] for p in panels], ignore_index=True)
    handles = cr.sector_patch_handles(all_macc) + [
        plt.Line2D([0], [0], color=fs.INK_PRIMARY, linestyle=(0, (4, 2)), linewidth=0.8, label="CO$_2$ tax")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=len(handles),
               frameon=False, handlelength=1.3, handletextpad=0.45, columnspacing=1.1)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.155, hspace=0.32, wspace=0.06)
    fs.save_figure(fig, OUT_DIR / "macc_tax_sweep")
    plt.close(fig)

    # --- one figure per scenario, same shared axis limits as the grid ---
    for panel in panels:
        fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH_IN, 2.9))
        handles = _draw_macc_panel(ax, panel, x_max, y_max)
        _panel_title(ax, panel)
        ax.set_xlabel(fs.CAPTURED_LABEL)
        ax.set_ylabel(fs.COST_LABEL)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  title="Sector", alignment="left")
        fig.tight_layout()
        fs.save_figure(fig, OUT_DIR / f"macc_tax_sweep_{panel['slug']}")
        plt.close(fig)


# ============================================================
# Captured CO2 by capture technology
# ============================================================
def capture_by_technology_table(panels) -> pd.DataFrame:
    """Mt/yr captured per (scenario, capture technology, sector), from every
    capture-classified technology row of load_ccs_status -- its own family
    and its own captured tonnes, so the totals reconcile exactly with the
    maps' real_emitters_df sums (see _real_emitter_rows' docstring)."""
    rows = []
    for p in panels:
        df = p["ccs_df"]
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({"scenario": p["slug"], "family": r["family"],
                         "sector": m._row_sector(r), "captured_mt": r["captured_annual"] / 1e6})
    out = pd.DataFrame(rows).groupby(["scenario", "family", "sector"], as_index=False)["captured_mt"].sum()
    for p in panels:  # sanity: same total as the maps
        tot = out.loc[out["scenario"] == p["slug"], "captured_mt"].sum()
        assert abs(tot - p["captured_mt"]) < 1e-6, (p["slug"], tot, p["captured_mt"])
    return out


def plot_capture_by_technology(panels):
    table = capture_by_technology_table(panels)
    table.to_csv(OUT_DIR / "capture_by_technology_tax_sweep.csv", index=False)
    print(f"Saved: {OUT_DIR / 'capture_by_technology_tax_sweep.csv'}")

    families = list(fs.FAMILY_ORDER)
    y = np.arange(len(families))[::-1]
    totals = table.groupby(["scenario", "family"])["captured_mt"].sum()
    x_max = np.ceil(totals.max() * 1.22 / 2) * 2

    fig, axes = plt.subplots(1, len(panels), figsize=(fs.FULL_WIDTH_IN, 2.05), sharey=True)
    for ax, panel in zip(axes, panels):
        sub = table[table["scenario"] == panel["slug"]]
        left = np.zeros(len(families))
        for sector in fs.SECTOR_ORDER:
            vals = np.array([sub.loc[(sub["family"] == f) & (sub["sector"] == sector), "captured_mt"].sum()
                             for f in families])
            if vals.sum() <= 0:
                continue
            ax.barh(y, vals, left=left, height=0.62, color=fs.SECTOR_COLORS[sector],
                    edgecolor=fs.SURFACE, linewidth=0.4, zorder=3)
            left += vals
        for yi, tot in zip(y, left):
            if tot > 0:
                ax.text(tot + x_max * 0.02, yi, f"{tot:.1f}" if tot >= 0.05 else "<0.1", va="center",
                        ha="left", fontsize=fs.TICK_SIZE, color=fs.INK_PRIMARY)
        ax.set_xlim(0, x_max)
        cr.style_value_axes(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
        _panel_title(ax, panel, y=1.04)

    # technology labels with the same shape glyph as the maps (shape only)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([fs.FAMILY_SHORT_LABELS[f] for f in families])
    axes[0].tick_params(axis="y", pad=11)
    for yi, f in zip(y, families):
        axes[0].scatter(-0.075, yi, marker=fs.FAMILY_MARKERS[f], s=16 * fs.FAMILY_MARKER_SCALE[f],
                        facecolors=fs.TECH_GLYPH_FACE, edgecolors=fs.MARK_EDGE, linewidths=0.6,
                        transform=axes[0].get_yaxis_transform(), clip_on=False, zorder=5)
    axes[0].set_ylim(-0.6, len(families) - 0.4)
    fig.supxlabel("CO$_2$ captured (Mt/y)", fontsize=fs.LABEL_SIZE, y=0.125)
    fig.legend(handles=fs.sector_handles(set(table["sector"]), kind="patch"), loc="lower center",
               bbox_to_anchor=(0.56, 0.0), ncol=6, frameon=False, handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.88, bottom=0.27, wspace=0.12)
    fs.save_figure(fig, OUT_DIR / "capture_by_technology_tax_sweep")
    plt.close(fig)


# ============================================================
# Hourly CO2 received upstream of storage, per scenario
# ============================================================
INFLOW_NODE = "Eni S.p.A Casalborsetti"


def plot_inflow_per_scenario(panels, node_name: str = INFLOW_NODE):
    """ccs_chain_plots.plot_node_inflow for each scenario, written to
    OUT_DIR as inflow_<node>_<slug>.{png,pdf}."""
    saved = (m.OUT_DIR, m.FIG_TAG)
    try:
        m.OUT_DIR = OUT_DIR
        for panel in panels:
            m.FIG_TAG = panel["slug"]
            m.plot_node_inflow(panel["h5_path"], node_name=node_name)
    finally:
        m.OUT_DIR, m.FIG_TAG = saved


def main():
    nodes_gdf, panels = _load_panel_data()
    plot_tax_sweep_maps(nodes_gdf, panels)
    plot_tax_sweep_macc(panels)
    plot_capture_by_technology(panels)
    plot_inflow_per_scenario(panels)


if __name__ == "__main__":
    main()
