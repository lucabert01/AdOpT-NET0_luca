"""
Ad-hoc driver: one point per technology-selection scenario in main_italy.py's
SCENARIOS, with
  x = fraction of baseline emissions avoided (ccs_chain_plots.fraction_avoided,
      the same KPI as on the maps), and
  y = cost per tonne avoided: total annual CCS chain cost (sum over emitters
      of ccs_chain_emitter_cost_ranking's total_eur_per_t x captured tonnes,
      i.e. capture + transport + storage) divided by the annual tonnes avoided
      (baseline - emissions_pos, same baseline as fraction_avoided). Unlike
      the per-tonne-captured average, this charges technologies that add CO2
      (oxyfuel, CaL) for the extra CO2 they capture but that was never part of
      the baseline.

Scenarios without a usable result are skipped (printed). Output:
ccs_chain_results/scenario_comparison/abatement_cost_vs_fraction_avoided.{png,pdf}
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

# (scenario_name, objective, carbon_tax, label, marker, label offset in points)
SCENARIOS = [
    ("technology_selection", "costs", 175, "175 €/t", "o", (6, -9)),
    ("technology_selection", "costs", 200, "200 €/t", "o", (-6, -11)),
    ("technology_selection", "costs", 250, "250 €/t", "o", (5, -10)),
    ("technology_selection", "emissions_minC", None, "Emis.-min.", "D", (2, 6)),
    ("technology_selection_wasteCaL", "emissions_minC", None, "Emis.-min., CaL on WtE", "s", (0, -11)),
]


def collect() -> pd.DataFrame:
    rows = []
    for scenario_name, objective, carbon_tax, label, marker, offset in SCENARIOS:
        try:
            h5_path = m.find_run_h5(scenario_name, objective, carbon_tax=carbon_tax)
        except FileNotFoundError:
            print(f"SKIPPED (no result): {scenario_name} / {objective} / tax {carbon_tax}")
            continue
        df, _ = cr.build_emitter_cost_table(h5_path)
        summary = m.load_summary(h5_path)
        captured = df["captured_annual_t"].sum()
        chain_cost = (df["total_eur_per_t"] * df["captured_annual_t"]).sum()
        avoided = m.load_baseline_emissions_t() - summary["emissions_pos"]
        cost = chain_cost / avoided if avoided > 0 else np.nan
        rows.append({
            "scenario": scenario_name, "objective": objective, "carbon_tax": carbon_tax,
            "label": label, "marker": marker, "offset": offset, "h5": str(h5_path),
            "captured_Mt": captured / 1e6, "avoided_Mt": avoided / 1e6,
            "chain_cost_MEUR": chain_cost / 1e6,
            "fraction_avoided": m.fraction_avoided(summary["emissions_pos"]),
            "cost_per_t_avoided": cost,
        })
        print(f"{label:28s} avoided={rows[-1]['fraction_avoided']:.1%}  "
              f"cost={cost:.1f} €/t avoided  captured={captured / 1e6:.2f} Mt/y  ({h5_path.parent.name})")
    return pd.DataFrame(rows)


XLIM = (70, 95)    # % of baseline emissions avoided (tax 175 sits at ~76%)
YLIM = (100, 250)  # €/t avoided


def draw_abatement_on_ax(ax, res: pd.DataFrame):
    """Scenario points + labels, axis labels and fixed limits on ax (shared by
    the standalone figure and panel d of _plot_tax_sweep_grids' macc_tax_sweep)."""
    for _, r in res.iterrows():
        ax.scatter(r["fraction_avoided"] * 100, r["cost_per_t_avoided"], marker=r["marker"],
                   s=28, facecolor=fs.TS_MAIN, edgecolor=fs.MARK_EDGE, linewidth=0.5, zorder=3)
        ax.annotate(r["label"], (r["fraction_avoided"] * 100, r["cost_per_t_avoided"]),
                    xytext=r["offset"], textcoords="offset points", fontsize=fs.TICK_SIZE,
                    ha="right" if r["offset"][0] < 0 else "center" if r["offset"][0] == 0 else "left",
                    color=fs.INK_SECONDARY)
    ax.set_xlabel("Baseline CO$_2$ emissions avoided (%)")
    ax.set_ylabel("Cost of CO$_2$ avoided (€/tCO$_2$)")
    outside = res[~res["fraction_avoided"].mul(100).between(*XLIM) | ~res["cost_per_t_avoided"].between(*YLIM)]
    if len(outside):
        print(f"Warning: points outside the fixed axis limits: {list(outside['label'])}")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(np.arange(XLIM[0], XLIM[1] + 1, 5))
    ax.set_yticks(np.arange(YLIM[0], YLIM[1] + 1, 50))
    cr.style_value_axes(ax)


def plot(res: pd.DataFrame):
    fs.apply_rc()
    fig, ax = plt.subplots(figsize=(fs.HALF_WIDTH_IN, 2.6))
    draw_abatement_on_ax(ax, res)
    fs.save_figure(fig, OUT_DIR / "abatement_cost_vs_fraction_avoided.png")
    plt.close(fig)


if __name__ == "__main__":
    res = collect()
    res.drop(columns=["marker", "offset"]).to_csv(OUT_DIR / "abatement_cost_vs_fraction_avoided.csv", index=False)
    plot(res)
