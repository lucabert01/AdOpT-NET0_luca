"""
Ad-hoc driver: "inframarginal" cost of raising the CO2 tax, per tonne of
EXTRA CO2 avoided, for the technology-selection tax steps 175 -> 200 -> 250.

For a tax step T_prev -> T:
  extra_avoided_t  = emissions_pos(T_prev) - emissions_pos(T)
                     (= avoided(T) - avoided(T_prev), same KPI as the maps)
  residual_t       = CO2 still emitted, in the T run, by the emitters that
                     were ALREADY capturing at T_prev (they did not need the
                     higher tax to capture, but pay it on what they still emit)
  extra_tax_eur    = (T - T_prev) x residual_t
  cost_per_t_extra = extra_tax_eur / extra_avoided_t

The first tax (175) has no previous step, so its cost is 0 by definition.
residual_prev_t (the same emitters' residual in the T_prev run) and the
cost computed from it are written to the CSV as well, for comparison.

Output: ccs_chain_results/scenario_comparison/tax_increase_cost.{csv,png,pdf}
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
TAXES = [175, 200, 250]


def _load(carbon_tax):
    h5_path = m.find_run_h5(SCENARIO_NAME, "costs", carbon_tax=carbon_tax)
    macc_df, _ = cr.build_emitter_cost_table(h5_path)
    capturing = set(zip(macc_df.loc[macc_df["captured_annual_t"] > 0, "node"].str.strip(),
                        macc_df.loc[macc_df["captured_annual_t"] > 0, "sector"]))
    return {"tax": carbon_tax, "h5": h5_path, "capturing": capturing,
            "emitted": cr.emitted_by_emitter(h5_path),
            "emissions_pos": m.load_summary(h5_path)["emissions_pos"]}


def collect() -> pd.DataFrame:
    runs = [_load(t) for t in TAXES]
    rows = [{"carbon_tax": runs[0]["tax"], "prev_tax": np.nan, "extra_avoided_t": 0.0,
             "residual_t": 0.0, "residual_prev_t": 0.0, "extra_tax_MEUR": 0.0,
             "cost_per_t_extra_avoided": 0.0, "cost_per_t_extra_avoided_prev_residual": 0.0,
             "n_emitters_already_capturing": 0}]
    for prev, cur in zip(runs, runs[1:]):
        dtax = cur["tax"] - prev["tax"]
        extra_avoided = prev["emissions_pos"] - cur["emissions_pos"]
        residual = sum(cur["emitted"].get(k, 0.0) for k in prev["capturing"])
        residual_prev = sum(prev["emitted"].get(k, 0.0) for k in prev["capturing"])
        cost = dtax * residual / extra_avoided if extra_avoided > 0 else np.nan
        cost_prev = dtax * residual_prev / extra_avoided if extra_avoided > 0 else np.nan
        rows.append({"carbon_tax": cur["tax"], "prev_tax": prev["tax"], "extra_avoided_t": extra_avoided,
                     "residual_t": residual, "residual_prev_t": residual_prev,
                     "extra_tax_MEUR": dtax * residual / 1e6, "cost_per_t_extra_avoided": cost,
                     "cost_per_t_extra_avoided_prev_residual": cost_prev,
                     "n_emitters_already_capturing": len(prev["capturing"])})
    res = pd.DataFrame(rows)
    for _, r in res.iterrows():
        print(f"tax {r['carbon_tax']:>3}: extra avoided {r['extra_avoided_t'] / 1e6:6.3f} Mt/y, "
              f"residual of {int(r['n_emitters_already_capturing'])} already-capturing emitters "
              f"{r['residual_t'] / 1e6:6.3f} Mt/y -> {r['cost_per_t_extra_avoided']:7.1f} €/t extra avoided")
    return res


def draw_on_ax(ax, res: pd.DataFrame):
    x = np.arange(len(res))
    ax.bar(x, res["cost_per_t_extra_avoided"], width=0.55, color=fs.TS_MAIN,
           edgecolor=fs.SURFACE, linewidth=0.4, zorder=3)
    y_max = np.nanmax(res["cost_per_t_extra_avoided"]) or 1.0
    for xi, (_, r) in zip(x, res.iterrows()):
        txt = (f"{r['cost_per_t_extra_avoided']:.0f} €/t\n(+{r['extra_avoided_t'] / 1e6:.2f} Mt/y)"
               if r["extra_avoided_t"] > 0 else "0")
        ax.text(xi, r["cost_per_t_extra_avoided"] + y_max * 0.03, txt, ha="center", va="bottom",
                fontsize=fs.TICK_SIZE, color=fs.INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f} €/t" for t in res["carbon_tax"]])
    ax.set_xlabel("CO$_2$ tax")
    ax.set_ylabel("Extra tax on residual emissions\nper extra t avoided (€/tCO$_2$)")
    ax.set_ylim(0, y_max * 1.3)
    cr.style_value_axes(ax)


def plot(res: pd.DataFrame):
    fs.apply_rc()
    fig, ax = plt.subplots(figsize=(fs.HALF_WIDTH_IN, 2.6))
    draw_on_ax(ax, res)
    fig.tight_layout()
    fs.save_figure(fig, OUT_DIR / "tax_increase_cost")
    plt.close(fig)


if __name__ == "__main__":
    res = collect()
    res.to_csv(OUT_DIR / "tax_increase_cost.csv", index=False)
    print(f"Saved: {OUT_DIR / 'tax_increase_cost.csv'}")
    plot(res)
