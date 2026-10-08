"""
Ad-hoc diagnostic (not part of the regular figure pipeline): for every node
where main_italy.py's technology-selection design offers more than one
candidate capture technology for the SAME real-world plant (per
node_metrics_paper.xlsx -- see ccs_chain_plots._load_node_sectors), reports
each candidate's actual annualized PRODUCT output (clinker for Cement,
processed waste for Waste -- the technology's own main_output_carrier, not
CO2), so it's possible to judge whether the split is a meaningful
double-installation of the same job or a negligible numerical residual.
"""
import h5py
import numpy as np
import ccs_chain_plots as m
from pathlib import Path

OUTPUT_KEY_BY_SECTOR = {
    "Cement": "clinker_output",
    "Waste": "waste_output",
}


def analyze(h5_path, label):
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    ccs_df = m.load_ccs_status(Path(h5_path))
    node_sectors = m._load_node_sectors()

    with h5py.File(h5_path, "r") as f:
        seq = f["k_means_specs"]["period1"]["sequence"][()]
        op = f["operation"]["technology_operation"]["period1"]

        for node, group in ccs_df.groupby("node"):
            if len(group) <= 1:
                continue
            sectors = node_sectors.get(node)
            if sectors and len(sectors) > 1:
                tagged = group.copy()
                tagged["_sector"] = tagged["tech"].map(m._tech_sector)
                subgroups = [g for _, g in tagged.groupby("_sector") if len(g) > 1]
            else:
                subgroups = [group]

            for sub in subgroups:
                sector = m._tech_sector(sub.iloc[0]["tech"])
                out_key = OUTPUT_KEY_BY_SECTOR.get(sector)
                print(f"\n{node}  [{sector}]")
                rows_out = []
                total = 0.0
                for _, r in sub.iterrows():
                    tech = r["tech"]
                    if out_key and out_key in op[node][tech]:
                        series = op[node][tech][out_key][()]
                        annual = float(series[seq - 1].sum())
                        peak = float(series.max())
                    else:
                        annual, peak = float("nan"), float("nan")
                    rows_out.append((tech, r["family"], annual, peak, r["captured_annual"], r["total_emissions"]))
                    if not np.isnan(annual):
                        total += annual
                for tech, fam, annual, peak, cap, tot_em in rows_out:
                    share = (annual / total * 100) if total > 0 and not np.isnan(annual) else float("nan")
                    print(f"   {tech:35s} [{fam:16s}] output={annual:>12,.0f} t/yr  ({share:5.1f}%)  peak={peak:6.1f} t/h  |  CO2 captured={cap:>10,.0f} t/yr  total_em={tot_em:>10,.0f} t/yr")
                if out_key:
                    print(f"   -> combined product output: {total:,.0f} t/yr")


analyze(
    "../Results_CCSchainOptimization/technology_selection/20260914153327_emissions_minC_technology_selection-1/optimization_results.h5",
    "TECHNOLOGY_SELECTION (latest rerun)",
)
analyze(
    "../Results_CCSchainOptimization/technology_selection_wasteCaL/20260913002012_emissions_minC_technology_selection_wasteCaL-1/optimization_results.h5",
    "TECHNOLOGY_SELECTION_WASTECAL (older run)",
)
