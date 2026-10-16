"""
Find the variables / coefficients behind Gurobi's "large bounds" warning
=========================================================================

The tax200 technology_selection run reports

    Matrix range     [1e-03, 1e+08]
    Bounds range     [1e+00, 5e+10]
    Warning: Model contains large bounds

and its best bound stalls at ~2.3% gap. Loose variable bounds and big-M
coefficients weaken the LP relaxation, so this script builds the exact
model main_italy.py would solve (same data pipeline, same objective, same
cement exclusivity constraint, same scaling), hands it to Gurobi WITHOUT
optimizing, and reports which Pyomo components the extreme values come from.

How it works: ModelHub._call_solver is replaced by a function that loads the
model into a gurobi_persistent instance, analyses the gurobipy model, and
stops the run -- so nothing is solved and no result folder is created.
Note that run_scenario still rewrites Italy_CaseStudy/<scenario_name>
exactly as a normal main_italy.py run does.

Outputs (in analysis/output/large_bounds_<case>/):
    large_bounds_by_group.csv      variable groups with |bound| >= --bound-threshold
    large_bounds_variables.csv     every such variable (name, lb, ub, type)
    extreme_coefficients.csv       constraint x variable groups with
                                   |coef| >= --coef-high or <= --coef-low
    large_rhs.csv                  constraint groups with |rhs| >= --rhs-threshold
    model.mps.bz2                  (only with --write-mps) for grbtune / testing
                                   other Gurobi versions on the same model

A "group" is the Pyomo component name with indices stripped, except the
technology / network / compressor block index (e.g.
periods.node_blocks.tech_blocks_active[MEA_large].var_size), so each row
points at a specific technology's variable.

Usage (from the repository root or from southern_europe/):
    python southern_europe/analysis/find_large_bounds.py --scenario technology_selection --tax 200
    python southern_europe/analysis/find_large_bounds.py --tax 200 --write-mps
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

SOUTHERN_EUROPE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = SOUTHERN_EUROPE_DIR / "analysis" / "output"

# Block segments whose index is kept in the group name, so groups separate by
# technology / network type instead of lumping every technology together.
KEEP_INDEX_BLOCKS = (
    "tech_blocks_active",
    "tech_blocks_new",
    "tech_blocks_existing",
    "network_block",
    "compressor_blocks_active",
)
_SEGMENT_INDEX = re.compile(r"([A-Za-z0-9_]+)\[([^\]]*)\]")


def group_name(name: str) -> str:
    """Strips Pyomo indices from a component name, keeping block indices in KEEP_INDEX_BLOCKS."""
    def _repl(match):
        segment, index = match.group(1), match.group(2)
        if segment in KEEP_INDEX_BLOCKS:
            return f"{segment}[{index}]"
        return segment

    return _SEGMENT_INDEX.sub(_repl, name)


class _ModelExported(Exception):
    """Raised by the patched _call_solver to stop run_scenario before solving."""


def analyse_model(model, out_dir: Path, args) -> None:
    """
    Loads a Pyomo model into Gurobi (no solve) and writes the bound / coefficient reports.

    :param model: the Pyomo ConcreteModel that would be passed to the solver
    :param Path out_dir: folder for the csv reports
    :param args: parsed command-line arguments (thresholds, write_mps)
    """
    print("Loading model into gurobi_persistent (no optimization)...", flush=True)
    opt = pyo.SolverFactory("gurobi_persistent")
    opt.set_instance(model, symbolic_solver_labels=False)
    gm = opt._solver_model
    gm.update()
    print(f"Gurobi model: {gm.NumConstrs} rows, {gm.NumVars} columns, {gm.NumNZs} nonzeros")
    gm.printStats()

    # Pyomo names indexed by Gurobi column / row position
    var_names = np.empty(gm.NumVars, dtype=object)
    for pyo_var, grb_var in opt._pyomo_var_to_solver_var_map.items():
        var_names[grb_var.index] = pyo_var.name
    con_names = np.empty(gm.NumConstrs, dtype=object)
    for pyo_con, grb_con in opt._pyomo_con_to_solver_con_map.items():
        if hasattr(grb_con, "index"):  # skip SOS constraints
            con_names[grb_con.index] = pyo_con.name
    var_groups = np.array([group_name(n) for n in var_names], dtype=object)
    con_groups = np.array([group_name(n) for n in con_names], dtype=object)

    lb = np.array(gm.getAttr("LB", gm.getVars()))
    ub = np.array(gm.getAttr("UB", gm.getVars()))
    vtype = np.array(gm.getAttr("VType", gm.getVars()))
    inf = 1e20  # Gurobi's GRB.INFINITY

    # ---- Large variable bounds ----
    lb_big = (np.abs(lb) >= args.bound_threshold) & (np.abs(lb) < inf)
    ub_big = (np.abs(ub) >= args.bound_threshold) & (np.abs(ub) < inf)
    flagged = lb_big | ub_big
    df_vars = pd.DataFrame({
        "group": var_groups[flagged],
        "name": var_names[flagged],
        "lb": lb[flagged],
        "ub": ub[flagged],
        "vtype": vtype[flagged],
    })
    df_vars["max_abs_bound"] = np.maximum(
        np.where(np.abs(df_vars["lb"]) < inf, np.abs(df_vars["lb"]), 0),
        np.where(np.abs(df_vars["ub"]) < inf, np.abs(df_vars["ub"]), 0),
    )
    df_vars = df_vars.sort_values("max_abs_bound", ascending=False)
    df_vars.to_csv(out_dir / "large_bounds_variables.csv", index=False)

    df_bound_groups = (
        df_vars.groupby("group")
        .agg(n_vars=("name", "size"),
             max_abs_bound=("max_abs_bound", "max"),
             min_abs_bound=("max_abs_bound", "min"),
             vtypes=("vtype", lambda s: "".join(sorted(set(s)))),
             example=("name", "first"))
        .sort_values("max_abs_bound", ascending=False)
    )
    df_bound_groups.to_csv(out_dir / "large_bounds_by_group.csv")

    # ---- Extreme matrix coefficients (big-M and badly scaled rows) ----
    A = gm.getA().tocoo()
    abs_val = np.abs(A.data)
    extreme = (abs_val >= args.coef_high) | ((abs_val <= args.coef_low) & (abs_val > 0))
    df_coef = pd.DataFrame({
        "constraint_group": con_groups[A.row[extreme]],
        "variable_group": var_groups[A.col[extreme]],
        "variable_vtype": vtype[A.col[extreme]],
        "abs_coef": abs_val[extreme],
        "example_constraint": con_names[A.row[extreme]],
    })
    df_coef["kind"] = np.where(df_coef["abs_coef"] >= args.coef_high, "high", "low")
    df_coef_groups = (
        df_coef.groupby(["kind", "constraint_group", "variable_group", "variable_vtype"])
        .agg(n_nonzeros=("abs_coef", "size"),
             max_abs_coef=("abs_coef", "max"),
             min_abs_coef=("abs_coef", "min"),
             example_constraint=("example_constraint", "first"))
        .reset_index()
        .sort_values(["kind", "max_abs_coef"], ascending=[True, False])
    )
    df_coef_groups.to_csv(out_dir / "extreme_coefficients.csv", index=False)

    # ---- Large right-hand sides ----
    rhs = np.array(gm.getAttr("RHS", gm.getConstrs()))
    rhs_big = np.abs(rhs) >= args.rhs_threshold
    df_rhs = (
        pd.DataFrame({"group": con_groups[rhs_big], "abs_rhs": np.abs(rhs[rhs_big]),
                      "example": con_names[rhs_big]})
        .groupby("group")
        .agg(n_rows=("abs_rhs", "size"), max_abs_rhs=("abs_rhs", "max"),
             example=("example", "first"))
        .sort_values("max_abs_rhs", ascending=False)
    )
    df_rhs.to_csv(out_dir / "large_rhs.csv")

    # ---- Console summary ----
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"\n=== Variable groups with |bound| >= {args.bound_threshold:.0e} "
          f"({int(flagged.sum())} variables) ===")
    print(df_bound_groups.drop(columns="example").head(40).to_string())
    print(f"\n=== Coefficients >= {args.coef_high:.0e} (big-M candidates; vtype B = binary) ===")
    print(df_coef_groups[df_coef_groups["kind"] == "high"]
          .drop(columns=["kind", "example_constraint"]).head(30).to_string(index=False))
    print(f"\n=== Coefficients <= {args.coef_low:.0e} ===")
    print(df_coef_groups[df_coef_groups["kind"] == "low"]
          .drop(columns=["kind", "example_constraint"]).head(30).to_string(index=False))
    print(f"\n=== Constraint groups with |rhs| >= {args.rhs_threshold:.0e} ===")
    print(df_rhs.drop(columns="example").head(20).to_string())

    if args.write_mps:
        mps_path = out_dir / "model.mps.bz2"
        print(f"\nWriting {mps_path} ...", flush=True)
        gm.write(str(mps_path))

    print(f"\nReports written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="technology_selection",
                        help="scenario name in main_italy.SCENARIOS")
    parser.add_argument("--objective", default="costs")
    parser.add_argument("--tax", type=int, default=200, help="carbon_tax of the scenario entry")
    parser.add_argument("--bound-threshold", type=float, default=1e6)
    parser.add_argument("--coef-high", type=float, default=1e5)
    parser.add_argument("--coef-low", type=float, default=1e-2)
    parser.add_argument("--rhs-threshold", type=float, default=1e6)
    parser.add_argument("--write-mps", action="store_true",
                        help="also write the model as .mps.bz2 (for grbtune / Gurobi version tests)")
    args = parser.parse_args()

    # main_italy.py uses paths relative to southern_europe/ and imports data_process from there
    os.chdir(SOUTHERN_EUROPE_DIR)
    sys.path.insert(0, str(SOUTHERN_EUROPE_DIR))
    import adopt_net0 as adopt
    import main_italy

    matches = [s for s in main_italy.SCENARIOS
               if s["name"] == args.scenario
               and s.get("objective", "costs") == args.objective
               and (s.get("carbon_tax") or main_italy.default_carbon_tax) == args.tax]
    if not matches:
        raise ValueError(f"No scenario {args.scenario} / {args.objective} / tax {args.tax} "
                         f"in main_italy.SCENARIOS")
    scenario = matches[0]

    case = f"tax{args.tax}_{args.objective}_{args.scenario}"
    out_dir = OUTPUT_DIR / f"large_bounds_{case}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _export_instead_of_solve(self):
        config = self.data.model_config
        if config["scaling"]["scaling_on"]["value"] == 1:
            self.scale_model()
            model = self.model["scaled"]
        else:
            model = self.model[self.info_solving_algorithms["aggregation_model"]]
        analyse_model(model, out_dir, args)
        raise _ModelExported

    adopt.ModelHub._call_solver = _export_instead_of_solve

    try:
        main_italy.run_scenario(
            scenario["name"],
            scenario["tech_for_cement"],
            scenario["tech_for_waste"],
            objective=scenario.get("objective", "costs"),
            carbon_tax=scenario.get("carbon_tax"),
            flatten_profiles=scenario.get("flatten_profiles", False),
            nr_dd_days=scenario.get("nr_dd_days"),
            injection_rate_tolerance=scenario.get("injection_rate_tolerance"),
            temporary_storage_nodes=scenario.get("temporary_storage_nodes"),
        )
    except _ModelExported:
        pass


if __name__ == "__main__":
    main()
