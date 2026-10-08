"""
Ad-hoc driver: builds two paper-ready 2x2 panel figures for the
"technology_selection" CO2-tax sweep in main_italy.py's SCENARIOS --
cost-minimizing at carbon_tax = 150/200/250 EUR/t plus the emissions-
minimizing run -- one figure of the built network maps, one of the MACC
(marginal abatement cost curve) charts.

No panel title/figure title is added (the paper's own caption covers that);
each panel instead carries a small in-axes annotation ("CO2 tax = 150 EUR/t"
etc.) plus that scenario's total captured CO2. A single legend, built from
the union of what's present across all four scenarios, is shared by the
whole figure rather than repeated per panel.

Reuses ccs_chain_plots._draw_ccs_network_on_ax and
ccs_chain_emitter_cost_ranking._draw_macc_on_ax -- the exact same drawing
code the single-scenario ccs_chain_network_map.png / ccs_chain_emitter_macc.png
figures use -- so these panels are pixel-for-pixel consistent with those.

Output: ccs_chain_results/scenario_comparison/ccs_chain_{network_map,macc}_
tax_sweep.{png,pdf} -- PDF for the paper (vector), PNG for quick previewing.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import ccs_chain_plots as m
import ccs_chain_emitter_cost_ranking as cr

OUT_DIR = Path("ccs_chain_results/scenario_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_NAME = "technology_selection"

# (label, objective, carbon_tax) -- carbon_tax=None for the emissions-
# minimizing run, which doesn't vary carbon_tax (see find_run_h5's docstring).
PANELS = [
    ("CO$_2$ tax = 150 €/t", "costs", 150),
    ("CO$_2$ tax = 200 €/t", "costs", 200),
    ("CO$_2$ tax = 250 €/t", "costs", 250),
    ("Emissions-minimizing", "emissions_minC", None),
]

# Fixed x-axis extent (Mt/yr) shared by every MACC panel, so bar widths are
# directly comparable across scenarios. Widened automatically if any panel's
# total capture exceeds it.
MACC_XMAX_MT = 15


def _save_fig(fig, stem: str, facecolor: str):
    """Saves fig as both PNG (raster, dpi=300) and PDF (vector) under
    OUT_DIR/<stem>.{png,pdf} -- the PDF is what actually goes in the paper,
    the PNG is for quick previewing."""
    for ext in ("png", "pdf"):
        out_file = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=facecolor, pad_inches=0.2)
        print(f"Saved: {out_file}")


def _load_panel_data():
    """Resolves each PANELS entry to its h5 and pre-loads everything both
    figures need from it, so each h5 is only read once."""
    nodes_gdf = m.gpd.read_file(m.GIS_NODES)
    panels = []
    for label, objective, carbon_tax in PANELS:
        h5_path = m.find_run_h5(SCENARIO_NAME, objective, carbon_tax=carbon_tax)
        built_arcs = m.load_built_arcs(h5_path)
        built_arcs = m.attach_route_geometries(built_arcs, nodes_gdf)
        ccs_df = m.load_ccs_status(h5_path)
        macc_df, storage_node = cr.build_emitter_cost_table(h5_path)
        panels.append({
            "label": label,
            "h5_path": h5_path,
            "built_arcs": built_arcs,
            "ccs_df": ccs_df,
            "macc_df": macc_df,
            "storage_node": storage_node,
        })
        print(f"Loaded {label!r} ({SCENARIO_NAME}, {objective}, carbon_tax={carbon_tax}) <- {h5_path}")
    return nodes_gdf, panels


def plot_tax_sweep_maps(nodes_gdf, panels):
    italy = m.gpd.read_file(m.ITALY_SHP)

    # Height chosen to match MAP_BOUNDS's own aspect ratio (7.5deg lon x
    # 3.5deg lat per panel, aspect="equal" via setup_base_map) -- otherwise
    # each axes box is taller than its equal-aspect content needs, and
    # tight_layout can't reclaim that dead space since it's resolved at draw
    # time, after layout (leaves a large blank band between the two rows).
    fig, axes = plt.subplots(2, 2, figsize=(19, 11.2), gridspec_kw=dict(hspace=0.12, wspace=0.08))
    fig.patch.set_facecolor(m.SURFACE)

    real_dfs = []
    for ax, panel in zip(axes.flat, panels):
        ax.set_facecolor(m.SURFACE)
        m.setup_base_map(ax, italy, "")  # no per-panel title -- caption covers it
        real_df = m._draw_ccs_network_on_ax(ax, panel["built_arcs"], nodes_gdf, panel["ccs_df"])
        real_dfs.append(real_df)

        total_capture_mt = real_df["captured_annual"].sum() / 1e6
        ax.text(
            0.02, 0.98, f"{panel['label']}\n{total_capture_mt:.2f} Mt/yr CO$_2$ captured",
            transform=ax.transAxes, fontsize=12, color=m.INK_PRIMARY,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=m.GRIDLINE, alpha=0.92),
        )

    # Union across all four panels so the shared legend covers every family/
    # status that appears in ANY of them, not just the first panel's.
    all_ccs_df = pd.concat([p["ccs_df"] for p in panels], ignore_index=True)
    legend_handles = [
        Line2D([0], [0], color=m.MODE_COLORS["CO2_Pipeline"], lw=3, label="Pipeline"),
        Line2D([0], [0], color=m.MODE_COLORS["CO2Truck"], lw=3, label="Truck"),
        Line2D([0], [0], color=m.MODE_COLORS["CO2Railway"], lw=3, label="Railway"),
        *m._capture_legend_handles(all_ccs_df),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=m.TRANSPORT_COLOR, markeredgecolor="white",
               markersize=10, label="Transport hub", linestyle="None"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=m.STORAGE_COLOR, markeredgecolor="white",
               markersize=17, label="CO$_2$ storage", linestyle="None"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=True, fontsize=12, framealpha=0.95, edgecolor=m.GRIDLINE,
    )

    fig.tight_layout(rect=[0, 0.075, 1, 1])
    _save_fig(fig, "ccs_chain_network_map_tax_sweep", m.SURFACE)
    plt.close(fig)


def plot_tax_sweep_macc(panels):
    fig, axes = plt.subplots(2, 2, figsize=(19, 11), gridspec_kw=dict(hspace=0.22, wspace=0.1))
    fig.patch.set_facecolor(cr.SURFACE)

    max_capture_mt = max(p["macc_df"]["captured_annual_t"].sum() / 1e6 for p in panels)
    if max_capture_mt > MACC_XMAX_MT:
        print(f"Warning: max capture {max_capture_mt:.2f} Mt/yr exceeds MACC_XMAX_MT={MACC_XMAX_MT}; widening x-axis")
    x_max = max(MACC_XMAX_MT, max_capture_mt)

    # Shared y-axis too: tallest bar across all panels + 10% headroom, rounded
    # up to the next 25 €/t so the top tick is a clean number.
    max_cost = max(p["macc_df"]["total_eur_per_t"].max() for p in panels)
    y_max = np.ceil(max_cost * 1.1 / 25) * 25

    all_low_cf = False
    for ax, panel in zip(axes.flat, panels):
        ax.set_facecolor(cr.SURFACE)
        df = panel["macc_df"]
        # annotate_fertilizers=False: 4 cramped panels have no room for the
        # per-bar callout arrows without them overlapping -- sector color
        # (still shared, canonical across every panel) is enough here.
        _, low_cf_mask = cr._draw_macc_on_ax(ax, df, annotate_fertilizers=False)
        all_low_cf = all_low_cf or low_cf_mask.any()
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

        ax.set_xlabel("Cumulative CO$_2$ captured (Mt/yr)", fontsize=10.5)
        ax.set_ylabel("€/t CO$_2$", fontsize=10.5)

        total_capture_mt = df["captured_annual_t"].sum() / 1e6
        ax.text(
            0.02, 0.98, f"{panel['label']}\n{total_capture_mt:.2f} Mt/yr CO$_2$ captured",
            transform=ax.transAxes, fontsize=12, color=cr.INK_PRIMARY,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=cr.GRIDLINE, alpha=0.92),
        )

    # Shared legend built from the union of sectors present across all four
    # panels (sector_display_label already renders "Other" as "Industrial
    # cluster of Ravenna", same as every other plot -- see ccs_chain_plots.py).
    all_macc_df = pd.concat([p["macc_df"] for p in panels], ignore_index=True)
    seen_labels = set()
    legend_handles = []
    for s in cr.SECTOR_ORDER:
        label = cr.SECTOR_LEGEND_LABEL[s]
        if s not in set(all_macc_df["sector"]) or label in seen_labels:
            continue
        seen_labels.add(label)
        legend_handles.append(Patch(facecolor=cr.SECTOR_COLORS[s], edgecolor="white", label=label))
    if all_low_cf:
        legend_handles.append(Patch(
            facecolor="none", edgecolor=cr.INK_SECONDARY, hatch="////",
            label=f"Capacity factor < {cr.LOW_CAPACITY_FACTOR_THRESHOLD:.0%}",
        ))
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_handles), frameon=True, fontsize=11.5, framealpha=0.95,
        edgecolor=cr.GRIDLINE, title="Sector", title_fontsize=11.5,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_fig(fig, "ccs_chain_macc_tax_sweep", cr.SURFACE)
    plt.close(fig)


def main():
    nodes_gdf, panels = _load_panel_data()
    plot_tax_sweep_maps(nodes_gdf, panels)
    plot_tax_sweep_macc(panels)


if __name__ == "__main__":
    main()
