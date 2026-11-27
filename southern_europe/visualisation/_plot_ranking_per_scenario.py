"""
Per-scenario cost ranking of EVERY emitter in the Italian case study,
including those the optimization left without capture.

For each scenario (figure_style.SCENARIOS plus the emissions-minimizing run
with CaL on every waste-to-energy plant):
  - captured emitters: one horizontal bar each, stacked capture / transport /
    storage, in € per tonne of CO2 AVOIDED -- each stage of
    ccs_chain_emitter_cost_ranking's per-tonne-captured split scaled by
    captured / avoided (add_avoided_columns), so the bar total equals the
    MACC bar height of the tax-sweep figures. Ranked cheapest at the top.
  - emitters without capture: listed below a separator, no bar, marked
    "no CCS", ordered by emissions.

Emitter names are "<sector> <location>" (e.g. "Cement Robilante",
"WtE Torino"), built from the paper_names sheet of node_metrics_paper.xlsx
(paper_name minus its facility-type suffix). The sector
is the bar colour (same colours as the maps/MACCs); the cost stage is its
opacity (full = capture, 60% = transport, 30% = storage).

Output: ranking_per_scenario/emitter_ranking_<slug>.{png,pdf,csv}
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ccs_chain_plots as m
import ccs_chain_emitter_cost_ranking as cr
import figure_style as fs

OUT_DIR = Path(__file__).resolve().parent / "ranking_per_scenario"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (slug, scenario_name, objective, carbon_tax, title)
RANKING_SCENARIOS = [(slug, "technology_selection", obj, tax, label) for slug, obj, tax, label in fs.SCENARIOS] + [
    ("emissions_minC_wasteCaL", "technology_selection_wasteCaL", "emissions_minC", None,
     "Emissions-minimizing, CaL on WtE"),
]

# facility-type suffixes stripped from paper_name to leave the location
_SUFFIX_RE = re.compile(r"\s+(WtE plant|cement plant|lime plant|refinery|fertiliser plant.*|industrial cluster)$")
# characters lost in node_metrics_paper.xlsx's encoding
_ENCODING_FIXES = {"Dolc�": "Dolcè", "Forl�": "Forlì"}
SECTOR_PREFIX = {"Cement": "Cement", "Waste": "WtE", "Refining": "Refinery", "Lime": "Lime",
                 "FertilizersCombustion": "Fertilizer", "FertilizersSMR": "Fertilizer",
                 "Other": "Industrial cluster"}
SECTOR_SUFFIX = {"FertilizersCombustion": " (non-SMR)", "FertilizersSMR": " (SMR)"}

# stage = opacity of the sector-coloured bar
STAGE_ALPHA = {"Capture": 1.0, "Transport": 0.6, "Storage": 0.3, "Temporary storage": 0.15}
STAGES = [("Capture", "capture_eur_per_t"), ("Transport", "transport_eur_per_t"),
          ("Storage", "storage_eur_per_t"), ("Temporary storage", "temporary_storage_eur_per_t")]


def emitter_display_names() -> dict:
    """{(node_name, sector): '<sector> <location>'} for every emitter."""
    names = pd.read_excel(m.NODE_METRICS_PAPER, sheet_name="paper_names")
    out = {}
    for node, sector, paper in names[["node_name", "node_type", "paper_name"]].itertuples(index=False):
        sector = sector.strip()
        if sector not in SECTOR_PREFIX:
            continue
        for bad, good in _ENCODING_FIXES.items():
            paper = paper.replace(bad, good)
        location = _SUFFIX_RE.sub("", paper.strip())
        out[(node.strip(), sector)] = f"{SECTOR_PREFIX[sector]} {location}{SECTOR_SUFFIX.get(sector, '')}"
    return out


def ranking_table(h5_path: Path, names: dict) -> pd.DataFrame:
    """One row per baseline emitter: captured ones with their €/t avoided
    split, uncaptured ones with NaN costs. Sorted: captured by total cost
    ascending, then uncaptured by baseline emissions descending."""
    df, _ = cr.build_emitter_cost_table(h5_path)
    df = cr.add_avoided_columns(df, h5_path)
    df["node"] = df["node"].str.strip()
    scale = df["captured_annual_t"] / df["avoided_annual_t"]
    for _, col in STAGES:
        if col in df.columns:
            df[col.replace("_per_t", "_per_t_avoided")] = df[col] * scale
    df["captured"] = True

    baseline = m.load_baseline_emissions_by_emitter()
    emitted = cr.emitted_by_emitter(h5_path)
    have = set(zip(df["node"], df["sector"]))
    rest = [{"node": n, "sector": s, "baseline_t": e, "emitted_t": emitted.get((n, s), e), "captured": False}
            for n, s, e in baseline[["node_name", "node_type", "annual_emission"]].itertuples(index=False)
            if (n, s) not in have]
    parts = [df.sort_values("total_eur_per_t_avoided")]
    if rest:
        parts.append(pd.DataFrame(rest).sort_values("baseline_t", ascending=False))
    out = pd.concat(parts, ignore_index=True)
    missing = [k for k in zip(out["node"], out["sector"]) if k not in names]
    if missing:
        raise KeyError(f"No paper name for emitter(s) {missing}")
    out.insert(0, "emitter", [names[k] for k in zip(out["node"], out["sector"])])
    return out


def plot_ranking(table: pd.DataFrame, title: str, path: Path, x_max: float):
    n = len(table)
    n_cap = int(table["captured"].sum())
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH_IN, 0.9 + 0.13 * n))
    y = np.arange(n)[::-1]          # first row at the top
    yc = y[:n_cap]
    cap = table.iloc[:n_cap]

    left = np.zeros(n_cap)
    used_stages = []
    for stage, col in STAGES:
        col = col.replace("_per_t", "_per_t_avoided")
        if col not in cap.columns:
            continue
        vals = cap[col].fillna(0).to_numpy()
        if not vals.any():
            continue
        # sector colour fill; the stage is the opacity
        ax.barh(yc, vals, left=left, height=0.7, color=[fs.sector_color(sc) for sc in cap["sector"]],
                alpha=STAGE_ALPHA[stage], linewidth=0, zorder=3)
        ax.barh(yc, vals, left=left, height=0.7, fill=False, edgecolor=fs.SURFACE, linewidth=0.4, zorder=4)
        left += vals
        used_stages.append(stage)
    for yi, tot in zip(yc, left):
        ax.text(min(tot, x_max) + x_max * 0.006, yi, f"{tot:,.0f}", va="center", ha="left",
                fontsize=fs.TICK_SIZE - 1, color=fs.INK_PRIMARY, clip_on=False)

    # uncaptured emitters: no bar, just a short note
    for yi in y[n_cap:]:
        ax.text(x_max * 0.006, yi, "no CCS", va="center", ha="left", fontsize=fs.TICK_SIZE - 1,
                color=fs.INK_SECONDARY, style="italic")
    if n_cap < n:
        ax.axhline(y[n_cap - 1] - 0.5 if n_cap else n - 0.5, color=fs.INK_SECONDARY, linewidth=0.5,
                   linestyle=(0, (2, 2)))

    ax.set_yticks(y)
    ax.set_yticklabels(table["emitter"], fontsize=fs.TICK_SIZE - 0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Cost of CO$_2$ avoided (€/tCO$_2$)")
    cr.style_value_axes(ax, grid_axis="x")

    sector_handles = cr.sector_patch_handles(table)
    stage_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fs.INK_PRIMARY, alpha=STAGE_ALPHA[s], linewidth=0,
                                   label=s) for s in used_stages]
    # in a band reserved below the axes, so it never covers a bar: two titled
    # blocks side by side (cost stage | sector), as in the tax-sweep map legend
    fig_h = fig.get_size_inches()[1]
    band_in = 0.2 + 0.16 * max(len(stage_handles), int(np.ceil(len(sector_handles) / 3)))
    fig.tight_layout(rect=(0, band_in / fig_h, 1, 1))
    x0, x1 = ax.get_position().x0, ax.get_position().x1
    blocks = [
        ("Cost stage (opacity)", stage_handles, 1, x0),
        ("Sector (colour)", sector_handles, 3, x0 + 0.3 * (x1 - x0)),
    ]
    for block_title, handles, ncol, x in blocks:
        fs.legend_block(fig, handles, block_title, loc="upper left", bbox=(x, (band_in - 0.05) / fig_h),
                        ncol=ncol, fontsize=fs.TICK_SIZE - 0.5, title_fontsize=fs.TICK_SIZE - 0.5,
                        handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    fs.save_figure(fig, path)
    plt.close(fig)


def main():
    fs.apply_rc()
    names = emitter_display_names()
    tables = {}
    for slug, scenario_name, objective, carbon_tax, title in RANKING_SCENARIOS:
        try:
            h5_path = m.find_run_h5(scenario_name, objective, carbon_tax=carbon_tax)
        except FileNotFoundError:
            print(f"SKIPPED (no result): {slug}")
            continue
        tables[slug] = (ranking_table(h5_path, names), title)
        print(f"Loaded {slug} <- {h5_path}")

    for slug, (table, title) in tables.items():
        # per-scenario x-axis, capped so a few extreme CaL-on-WtE bars (up to
        # ~1,750 €/t avoided) don't flatten everything else; their value labels
        # still show the true total
        totals = table.loc[table["captured"], "total_eur_per_t_avoided"]
        x_max = min(np.ceil(totals.max() * 1.12 / 50) * 50, 400)
        table.to_csv(OUT_DIR / f"emitter_ranking_{slug}.csv", index=False)
        plot_ranking(table, title, OUT_DIR / f"emitter_ranking_{slug}", x_max)


if __name__ == "__main__":
    main()
