"""
Temporal analysis figures (main_italy.py runs at a CO2 tax of 200 EUR/t),
written to this folder as PNG + PDF:

  outflow_Eni_SpA_Casalborsetti_daily_tax200  daily-average CO2 sent out of Casalborsetti (network_outflow),
                                        technology_selection at tax 200
                                        (ccs_chain_plots.plot_node_inflow)
  emitter_zoom_SILLA_2_tax200           captured vs. emitted CO2 at SILLA 2,
                                        same run (ccs_chain_plots.plot_emitter_zoom)
  outflow_Casalborsetti_constant_injection_daily
                                        daily-average CO2 sent out of Casalborsetti (network_outflow):
                                        reference (technology_selection, tax 200)
                                        vs. constant injection with a minimum
                                        of 85 % (c_inj) and 95 % (c_inj_min95)
                                        of the injection capacity
  buffer_storage_size                   size of the temporary CO2 buffer storage
                                        (kt CO2) for Ref / 85 % / 95 %, with
                                        Northern Lights' buffer (8 kt) as a
                                        dashed reference line and, above each
                                        constant-injection bar, the increase in
                                        the cost of CO2 avoided vs. Ref (EUR/t)

Cost of CO2 avoided = total chain cost (capture + transport + storage +
temporary storage, ccs_chain_emitter_cost_ranking.build_emitter_cost_table)
/ CO2 avoided (baseline - emissions_pos), i.e. the avoided-weighted average
of the per-emitter cost per tonne avoided -- the same definition as
_plot_abatement_cost_vs_fraction.py.

Scenarios whose results are not available yet (e.g. still running on the
server) are skipped in the inflow figure and drawn as an empty "pending" bar.
Run from anywhere: python plot_temporal_analysis.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIS_DIR = HERE.parent
# ccs_chain_plots resolves its data/result paths relative to visualisation/
os.chdir(VIS_DIR)
sys.path.insert(0, str(VIS_DIR))

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ccs_chain_plots as m  # noqa: E402
import ccs_chain_emitter_cost_ranking as cr  # noqa: E402
import figure_style as fs  # noqa: E402

CARBON_TAX = 200
INFLOW_NODE = "Eni S.p.A Casalborsetti"
EMITTER_NODE = "SILLA 2"
NORTHERN_LIGHTS_BUFFER_KT = 8.0

# (key, scenario_name, legend/bar label, line style)
CASES = [
    ("ref", "technology_selection", "Ref", "-"),
    ("min85", "c_inj", "Min. 85 %", (0, (4, 2))),
    ("min95", "c_inj_min95", "Min. 95 %", (0, (1.2, 1.2))),
]
LINE_COLORS = {"ref": fs.TS_LIGHT, "min85": fs.TS_MAIN, "min95": fs.TS_REFERENCE}


def resolve_cases() -> dict:
    """key -> h5 path, or None when the run has no result yet."""
    out = {}
    for key, scenario, label, _ in CASES:
        try:
            out[key] = m.find_run_h5(scenario, "costs", carbon_tax=CARBON_TAX)
            print(f"{label:10s} <- {out[key]}")
        except FileNotFoundError:
            out[key] = None
            print(f"{label:10s} <- no result yet ({scenario}, tax {CARBON_TAX})")
    return out


def hourly_outflow(h5_path: Path, node_name: str = INFLOW_NODE) -> np.ndarray:
    """CO2 leaving node_name via the network (t/h, network_outflow of the
    CO2captured balance), full 8760-h year."""
    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        inflow = f["operation"]["energy_balance"]["period1"][node_name]["CO2captured"]["network_outflow"][()]
    return inflow[seq - 1]


def buffer_size_kt(h5_path: Path) -> float:
    """Total installed size of the temporary CO2 buffer storage (kt CO2)."""
    total = 0.0
    with h5py.File(h5_path, "r") as f:
        nodes = f["design"]["nodes"]["period1"]
        for node_name in nodes.keys():
            for tech in nodes[node_name].keys():
                if m.is_temporary_storage(tech):
                    total += float(np.ravel(nodes[node_name][tech]["size"][()])[0])
    return total / 1e3


def cost_per_t_avoided(h5_path: Path) -> float:
    """Total chain cost / CO2 avoided (EUR/t), see module docstring."""
    df, _ = cr.build_emitter_cost_table(h5_path)
    chain_cost = (df["total_eur_per_t"] * df["captured_annual_t"]).sum()
    avoided = m.load_baseline_emissions_t() - m.load_summary(h5_path)["emissions_pos"]
    return chain_cost / avoided


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def plot_reference_figures(ref_h5: Path):
    """Casalborsetti outflow and SILLA 2 zoom of the reference run, saved here."""
    saved = (m.OUT_DIR, m.FIG_TAG)
    try:
        m.OUT_DIR, m.FIG_TAG = HERE, f"tax{CARBON_TAX}"
        m.plot_node_inflow(ref_h5, node_name=INFLOW_NODE, direction="outflow", daily=True)
        m.plot_emitter_zoom(ref_h5, node_name=EMITTER_NODE)
    finally:
        m.OUT_DIR, m.FIG_TAG = saved


def plot_inflow_comparison(h5_paths: dict):
    """Hourly CO2 sent out of Casalborsetti: Ref vs. constant injection cases."""
    fig, ax = plt.subplots(figsize=fs.TS_SIZE_IN)
    y_max = 0.0
    for key, _, label, ls in CASES:
        if h5_paths[key] is None:
            continue
        inflow = m.daily_average(hourly_outflow(h5_paths[key]))
        hours = np.arange(len(inflow))
        ax.plot(hours, inflow, color=LINE_COLORS[key], linestyle=ls, linewidth=0.8 if key == "ref" else 0.7,
                drawstyle="steps-post", zorder=3 if key == "ref" else 4, label=label)
        y_max = max(y_max, inflow.max())
        ax.set_xlim(0, hours[-1])
    ax.set_ylim(0, y_max * 1.2)
    ax.set_ylabel("CO$_2$ injected (t/h)")
    m.style_timeseries_axes(ax)
    ax.legend(loc="upper right", ncol=3, frameon=False, handlelength=2.2, borderaxespad=0.3,
              fontsize=fs.TS_LEGEND_SIZE)
    fig.tight_layout(pad=0.3)
    fs.save_figure(fig, HERE / "outflow_Casalborsetti_constant_injection_daily.png")
    plt.close(fig)


def plot_buffer_storage(h5_paths: dict):
    """Buffer storage size per case, Northern Lights reference, and the
    increase in cost of CO2 avoided vs. Ref above each constant-injection bar."""
    labels = [label for _, _, label, _ in CASES]
    sizes = {k: buffer_size_kt(p) if p is not None else None for k, p in h5_paths.items()}
    costs = {k: cost_per_t_avoided(p) if p is not None else None for k, p in h5_paths.items()}
    for key, _, label, _ in CASES:
        if sizes[key] is not None:
            print(f"{label:10s} buffer = {sizes[key]:8.1f} kt   cost of CO2 avoided = {costs[key]:.2f} €/t")

    fig, ax = plt.subplots(figsize=fs.TS_SIZE_IN)
    x = np.arange(len(CASES))
    known = [v for v in sizes.values() if v is not None]
    y_top = max(known + [NORTHERN_LIGHTS_BUFFER_KT]) * 1.35
    for i, (key, _, _, _) in enumerate(CASES):
        if sizes[key] is None:
            ax.bar(x[i], y_top * 0.25, width=0.55, facecolor="none", edgecolor=fs.INK_MUTED,
                   linewidth=0.6, linestyle=(0, (2, 2)), zorder=3)
            ax.text(x[i], y_top * 0.17, "pending", ha="center", va="center",
                    fontsize=fs.TS_ANNOT_SIZE, color=fs.INK_MUTED)
            continue
        ax.bar(x[i], sizes[key], width=0.55, color=fs.TS_MAIN, zorder=3)
        ax.text(x[i], sizes[key] + y_top * 0.015, f"{sizes[key]:,.0f}", ha="center", va="bottom",
                fontsize=fs.TS_ANNOT_SIZE, color=fs.INK_PRIMARY)
        if key != "ref" and costs["ref"] is not None:
            delta = costs[key] - costs["ref"]
            base = sizes[key] + y_top * 0.09
            ax.annotate("", xy=(x[i], base + y_top * 0.12), xytext=(x[i], base),
                        arrowprops=dict(arrowstyle="-|>", color=fs.INK_SECONDARY, lw=0.8, mutation_scale=7),
                        zorder=5)
            ax.text(x[i] + 0.06, base + y_top * 0.06, f"{delta:+.1f} €/t", ha="left", va="center",
                    fontsize=fs.TS_ANNOT_SIZE, color=fs.INK_PRIMARY)

    ax.axhline(NORTHERN_LIGHTS_BUFFER_KT, color=fs.TS_REFERENCE, linewidth=0.8, linestyle=(0, (4, 2)), zorder=4)
    ax.text(-0.55, NORTHERN_LIGHTS_BUFFER_KT + y_top * 0.01, "Northern Lights (8 kt)", ha="left", va="bottom",
            fontsize=fs.TS_ANNOT_SIZE, color=fs.INK_PRIMARY)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(CASES) - 0.4)
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Buffer storage size (kt)", fontsize=fs.TS_LABEL_SIZE, color="black")
    ax.tick_params(axis="both", colors="black", labelsize=fs.TS_TICK_SIZE, width=0.8, length=3)
    cr.style_value_axes(ax)
    fig.tight_layout(pad=0.3)
    fs.save_figure(fig, HERE / "buffer_storage_size.png")
    plt.close(fig)


def main():
    h5_paths = resolve_cases()
    if h5_paths["ref"] is None:
        raise FileNotFoundError(f"No technology_selection result at tax {CARBON_TAX}")
    plot_reference_figures(h5_paths["ref"])
    plot_inflow_comparison(h5_paths)
    plot_buffer_storage(h5_paths)


if __name__ == "__main__":
    main()
