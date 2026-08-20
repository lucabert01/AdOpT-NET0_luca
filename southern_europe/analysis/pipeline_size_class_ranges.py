"""
Derivation of the CO2 pipeline size-class mass-flow ranges
============================================================

Supplementary-methods script. Documents and reproduces how the mass-flow
evaluation ranges (kg/s) for the three CO2 pipeline network technologies -
CO2_Pipeline_small / _medium / _large (see italy_data/networks/) - were
chosen. Each class is calibrated as a straight-line CAPEX approximation
(gamma1 + gamma2 * size) fit to the Oeuvray et al. (2024) pipeline+
compression cost model evaluated over that class's fixed [min, max] kg/s
range (see pipeline_capex_per_arc_calculator.py, which actually computes the
gammas). This script only concerns itself with *where the three range
boundaries come from*.

Why three fixed classes instead of one range per arc
------------------------------------------------------
The earlier approach derived a *different* min/max mass-flow window per arc
from that arc's own upstream emissions, and fit gamma1/gamma2 to that
window. This produces a locally accurate straight-line approximation for
each arc individually, but each arc then effectively has its own bespoke
"technology" - which does not match the intent of representing pipeline
CAPEX with a small, discrete set of installable technologies (as is done for
e.g. truck/rail capacity tiers). The fix: pick a handful of shared,
economy-of-scale-informed size classes, each with one fixed evaluation
range applied uniformly to every arc, and let the optimizer pick which
class (or classes) to build on each arc.

Method
------
1. Take the arcs actually built in a reference optimization run
   (Results_CCSchainOptimization/20260711184007_emissions_minC-1) as a
   realistic SAMPLE of pipeline sizes that occur in this network. Network
   *topology* (which arcs carry large vs. small flows) is driven mostly by
   geography and the emitter layout, not by the exact cost coefficients, so
   this sample remains informative even though that particular run priced
   every arc with the same flat 5-10 kg/s default (a bug fixed after this
   run - see git history around TONNES_TO_KG / create_base_options).
2. For each built arc, recompute the TRUE capex at its actual built flow
   using the bug-fixed Oeuvray model (CO2_Pipeline_CostModel, single-point
   evaluation, i.e. massflow_min_kg_per_s == massflow_max_kg_per_s == that
   arc's flow) - this uses that arc's real length, terrain and geographic
   cost factors.
3. Normalize by pipeline length (capex / length_km) to isolate the
   size-driven component (diameter, wall thickness, compression) from pure
   route length, and fit a power law
       capex_per_km(flow) = a * flow_kg_s^b
   via log-log OLS. b < 1 quantifies economies of scale: every doubling of
   flow multiplies capex_per_km by only 2^b, i.e. a strongly concave curve -
   steep at small flow, flat at large flow.
4. Because that curve is concave, no single straight line fits it well
   everywhere. Evaluate the fitted power law on a fine, log-spaced grid and
   find the two breakpoints that minimize the *total* sum-of-squared error
   of a 3-segment piecewise-linear fit to that curve (brute-force search
   over all candidate breakpoint pairs - O(n^2) fits, trivial at this grid
   size). This concretely captures the paper's design goal: segments sized
   so each class's own linear gamma1+gamma2*size approximation is locally
   accurate, with more (narrower) resolution where the curve bends hardest.
   (Searching directly on the sparse, noisy real arc data instead of the
   smooth fitted curve was tried first and rejected - see the note in
   find_breakpoints() - it degenerates by isolating outliers into
   single-point segments.)
5. Pad the outer edges to the network's true floor (smallest single
   emitter's own flow) and ceiling (total network emissions), so that every
   physically possible arc flow - not just the ones built in the reference
   run - falls inside one of the three classes.

Outputs
-------
Run as a script (`python pipeline_size_class_ranges.py`) to reproduce:
  - analysis/output/pipeline_size_class_ranges.png   (figure for supplementary info)
  - analysis/output/built_arcs_true_capex.csv         (per-arc raw data table)
  - printed summary of the final (small, medium, large) kg/s and t/h ranges
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SOUTHERN_EUROPE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SOUTHERN_EUROPE_DIR.parent
CALC_DIR = SOUTHERN_EUROPE_DIR / "data_process" / "updated_network"

sys.path.append(str(REPO_ROOT))
sys.path.append(str(CALC_DIR))

from arc_specific_functions import (  # noqa: E402
    load_network_data,
    load_intersection_data,
    get_all_possible_arcs,
    get_pipeline_length,
    determine_arc_terrain,
    create_base_options,
    add_geographical_options,
    calculate_global_max_massflow,
    calculate_global_min_massflow,
    suppress_stdout,
    TONNES_TO_KG,
)
from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import (  # noqa: E402
    CO2_Pipeline_CostModel as EnhancedModel,
)

DATA_PATH = SOUTHERN_EUROPE_DIR / "italy_data"
REFERENCE_RUN = SOUTHERN_EUROPE_DIR / "Results_CCSchainOptimization" / "20260711184007_emissions_minC-1" / "optimization_results.h5"
OUTPUT_DIR = SCRIPT_DIR / "output"

KG_S_TO_T_H = 3600 / 1000


# ============================================================================
# 1. Extract the arcs actually built in the reference run
# ============================================================================
def load_built_pipeline_arcs(h5_path):
    """
    Returns a list of (from_node_name, to_node_name, size_t_h) for every
    CO2_Pipeline arc with a nonzero built size in the reference run.
    """
    built = []
    with h5py.File(h5_path, "r") as f:
        group = f["design/networks/period1/CO2_Pipeline"]
        for arc_key in group.keys():
            sub = group[arc_key]
            size_t_h = float(np.array(sub["size"]))
            if size_t_h <= 1e-6:
                continue
            from_name = sub["fromNode"][()]
            to_name = sub["toNode"][()]
            if isinstance(from_name, bytes):
                from_name = from_name.decode()
            if isinstance(to_name, bytes):
                to_name = to_name.decode()
            built.append((from_name, to_name, size_t_h))
    return built


# ============================================================================
# 2. Recompute the TRUE (bug-fixed) capex at each arc's actual built flow
# ============================================================================
def compute_true_capex_per_built_arc(built_arcs, data_dict):
    """
    For each (from_name, to_name, size_t_h), evaluates the Oeuvray cost
    model at a single point (massflow_min == massflow_max == that arc's
    actual flow), using that arc's real length/terrain/geographic factors.

    Returns a DataFrame with one row per arc: flow_kg_s, length_km, terrain,
    true_capex (EUR), and capex_per_km (EUR/km, the size-driven component
    with the pure length scaling divided out).
    """
    node_names = data_dict["network_nodes"]["node_name"]
    name_to_id = {}
    for nid, name in node_names.items():
        name_to_id.setdefault(name, nid)

    rows = []
    for from_name, to_name, size_t_h in built_arcs:
        from_id, to_id = name_to_id.get(from_name), name_to_id.get(to_name)
        if from_id is None or to_id is None:
            print(f"  skipping (name not found): {from_name} -> {to_name}")
            continue

        length_km = get_pipeline_length(from_id, to_id, data_dict["network_distance"])
        terrain = determine_arc_terrain(from_id, to_id, data_dict)
        flow_kg_s = size_t_h / KG_S_TO_T_H

        base_options = create_base_options(
            length_km, flow_kg_s, flow_kg_s,
            data_dict["avg_electricity_price_eur_mwh"], terrain=terrain, evaluation_points=1,
        )
        pipeline_name = f"{from_id}_{to_id}"
        intersection = data_dict["intersection_data"].get(
            pipeline_name, {"intersected_grids": [], "intersected_proportions": []}
        )
        options = add_geographical_options(
            base_options, data_dict["morpho_data"], data_dict["soil_data"], data_dict["anthro_data"],
            intersection["intersected_grids"], intersection["intersected_proportions"],
        )

        model = EnhancedModel("CO2_Pipeline")
        with suppress_stdout():
            result = model.calculate_indicators(options)
        true_capex = float(result["costs_detailed"]["updated_capex_total"].iloc[0])

        rows.append({
            "from": from_name, "to": to_name, "from_id": from_id, "to_id": to_id,
            "size_t_h": size_t_h, "flow_kg_s": flow_kg_s, "length_km": length_km,
            "terrain": terrain, "true_capex": true_capex,
            "capex_per_km": true_capex / length_km,
        })

    return pd.DataFrame(rows).sort_values("flow_kg_s").reset_index(drop=True)


# ============================================================================
# 3. Fit the economies-of-scale power law: capex_per_km = a * flow_kg_s^b
# ============================================================================
def fit_power_law(flow_kg_s, capex_per_km):
    log_x, log_y = np.log(flow_kg_s), np.log(capex_per_km)
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    (b, log_a), *_ = np.linalg.lstsq(A, log_y, rcond=None)
    a = np.exp(log_a)

    pred = A @ [b, log_a]
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return a, b, r2


# ============================================================================
# 4. Optimal 2-breakpoint piecewise-linear approximation of the fitted curve
# ============================================================================
def _segment_sse(x, y, lo, hi):
    """SSE of the best-fit OLS line through points [lo:hi)."""
    xs, ys = x[lo:hi], y[lo:hi]
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    return float(np.sum((ys - pred) ** 2))


def find_breakpoints(a, b, flow_min, flow_max, n_grid=400):
    """
    Finds the 2 breakpoints (3 segments) minimizing total piecewise-linear
    approximation SSE against the smooth fitted power-law curve, evaluated
    on a fine log-spaced grid.

    Note: running this same brute-force search directly on the sparse (~40
    point), noisy real per-arc data instead of the smooth fitted curve was
    tried first and produces degenerate results - with so few points and
    heavy right-skew, unconstrained SSE minimization just isolates the 1-2
    largest trunk-line outliers into their own segment (near-zero SSE by
    construction), rather than reflecting genuine curvature. Fitting the
    smooth power law first and searching on that removes this instability.
    """
    grid = np.geomspace(flow_min, flow_max, n_grid)
    curve = a * grid ** b

    best = None
    for i in range(5, n_grid - 10):
        for j in range(i + 5, n_grid - 5):
            total = (
                _segment_sse(grid, curve, 0, i)
                + _segment_sse(grid, curve, i, j)
                + _segment_sse(grid, curve, j, n_grid)
            )
            if best is None or total < best[0]:
                best = (total, i, j)

    _, i, j = best
    return grid[i], grid[j]


# ============================================================================
# 5. Figure for supplementary info
# ============================================================================
def make_figure(df, a, b, r2, breakpoints, final_ranges, output_path, log_scale=True):
    """
    log_scale=True gives the log-log view (clearest for showing the power-law
    economies-of-scale fit across 2 orders of magnitude of flow). log_scale=
    False gives a linear-axes version of the same data/fit/ranges - the
    'large' class dominates the x-range there and small/medium arcs bunch up
    near the origin, but it directly shows the actual EUR/km magnitudes
    without a reader needing to interpret log axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(df["flow_kg_s"], df["capex_per_km"], s=28, alpha=0.75, color="#2c3e50",
               label="Arcs built in reference run\n(true capex, bug-fixed model)")

    if log_scale:
        grid = np.geomspace(df["flow_kg_s"].min(), df["flow_kg_s"].max(), 200)
    else:
        grid = np.linspace(df["flow_kg_s"].min(), df["flow_kg_s"].max(), 200)
    ax.plot(grid, a * grid ** b, color="#7f8c8d", linestyle="--", linewidth=1.5,
            label=f"Fitted power law: {a:,.0f}$\\cdot$flow$^{{{b:.3f}}}$ (R$^2$={r2:.2f})")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()

    colors = {"small": "#e74c3c", "medium": "#f39c12", "large": "#27ae60"}
    for name, (lo, hi) in final_ranges.items():
        ax.axvspan(lo, hi, color=colors[name], alpha=0.12)
        mid = np.sqrt(max(lo, 1e-6) * hi) if log_scale else (lo + hi) / 2
        label_y = ymin * 1.5 if log_scale else ymin + 0.03 * (ymax - ymin)
        ax.annotate(name, xy=(mid, label_y), ha="center", fontsize=11, fontweight="bold",
                    color=colors[name])

    for bp in breakpoints:
        ax.axvline(bp, color="#34495e", linestyle=":", linewidth=1)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Pipeline mass flow (kg CO$_2$/s)")
    ax.set_ylabel("CAPEX per unit length (EUR/km)")
    ax.set_title("Deriving the three CO2 pipeline size-class ranges" + ("" if log_scale else " (linear scale)"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    print(f"Saved figure: {output_path} (+ .pdf)")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("PIPELINE SIZE-CLASS RANGE DERIVATION")
    print("=" * 80)

    print(f"\nLoading network data and reference run: {REFERENCE_RUN}")
    data_dict = load_network_data(str(DATA_PATH))
    possible_arcs = get_all_possible_arcs(data_dict["network_pipeline"])
    pipeline_names = [f"{f}_{t}" for f, t in possible_arcs]
    data_dict["intersection_data"] = load_intersection_data(
        DATA_PATH / "geographical_feature" / "route_grid_intersections.xlsx", pipeline_names
    )

    built_arcs = load_built_pipeline_arcs(REFERENCE_RUN)
    print(f"Found {len(built_arcs)} built pipeline arcs in the reference run")

    print("\nRecomputing TRUE capex at each arc's actual built flow (bug-fixed Oeuvray model)...")
    df = compute_true_capex_per_built_arc(built_arcs, data_dict)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "built_arcs_true_capex.csv", index=False)
    print(f"Saved per-arc table: {OUTPUT_DIR / 'built_arcs_true_capex.csv'} ({len(df)} arcs)")

    a, b, r2 = fit_power_law(df["flow_kg_s"].values, df["capex_per_km"].values)
    print(f"\nFitted power law: capex_per_km = {a:,.1f} * flow_kg_s^{b:.4f}   (log-log R2={r2:.3f})")
    print(f"  -> economies of scale: doubling flow multiplies capex/km by only {2**b:.2f}x")

    bp1, bp2 = find_breakpoints(a, b, df["flow_kg_s"].min(), df["flow_kg_s"].max())
    print(f"\nOptimal breakpoints (piecewise-linear fit to the smooth curve): {bp1:.2f}, {bp2:.2f} kg/s")

    global_min_kg_s = calculate_global_min_massflow(data_dict["network_emission_flux"])
    global_max_kg_s = calculate_global_max_massflow(data_dict["network_emission_flux"])

    # Pad the outer edges so every physically possible arc flow falls inside
    # one of the three classes: floor rounded DOWN to the network's true
    # minimum (smallest single emitter), ceiling rounded UP to the nearest
    # 10 kg/s above the network's true maximum (total network emissions) -
    # not to the maximum itself, so a future run with slightly different
    # emissions data doesn't immediately exceed the calibrated range.
    floor_kg_s = np.floor(global_min_kg_s * 10) / 10
    ceiling_kg_s = np.ceil(global_max_kg_s / 10) * 10
    final_ranges = {
        "small": (floor_kg_s, round(bp1, 0)),
        "medium": (round(bp1, 0), round(bp2, 0)),
        "large": (round(bp2, 0), ceiling_kg_s),
    }

    print("\n" + "=" * 80)
    print("FINAL SIZE-CLASS RANGES")
    print("=" * 80)
    for name, (lo, hi) in final_ranges.items():
        print(f"  {name:8s}  {lo:7.1f} - {hi:7.1f} kg/s   |   {lo*KG_S_TO_T_H:8.1f} - {hi*KG_S_TO_T_H:8.1f} t/h")

    make_figure(df, a, b, r2, [bp1, bp2], final_ranges, OUTPUT_DIR / "pipeline_size_class_ranges.png", log_scale=True)
    make_figure(df, a, b, r2, [bp1, bp2], final_ranges, OUTPUT_DIR / "pipeline_size_class_ranges_linear.png", log_scale=False)

    print("\nThese ranges are hardcoded (not re-read from this script) into:")
    print("  - data_process/updated_network/pipeline_capex_per_arc_calculator.py :: SIZE_CLASS_MASSFLOW_RANGES_KG_S")
    print("  - italy_data/networks/CO2_Pipeline_{small,medium,large}.json :: size_min / size_max (t/h)")
    print("  - main_italy.py :: pipeline_size_class_max_capacity_t_h (t/h)")


if __name__ == "__main__":
    main()
