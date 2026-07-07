"""
chain_results.py

Generates publication-quality static figures for the CCS chain optimization results:
  1. A geographic map of emitters (colored by sector, sized by CCS captured capacity,
     ringed if CCS is installed) with the built CO2 transport network overlaid
     (colored/width-scaled by mode: pipeline / rail / truck).
  2. Summary bar charts: installed CCS capacity by emitter, and total CO2 transported
     by mode.

Run from: southern_europe/visualisation/
Outputs saved to: southern_europe/visualisation/figures/
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# CONFIG — relative paths (script assumed to run from southern_europe/visualisation/)
# ----------------------------------------------------------------------------
H5_PATH = "../Results_CCSchainOptimization/202607061310065/optimization_results.h5"
NODE_METRICS_PATH = "../italy_data/geographical_feature/node_metrics.xlsx"
OUTPUT_DIR = "figures"
PERIOD = "period1"

# Sector -> color (colorblind-friendly, Okabe-Ito inspired)
SECTOR_COLORS = {
    "Cement": "#D55E00",
    "WasteToEnergy": "#0072B2",
    "Refinery": "#009E73",
    "Unspecified": "#999999",
    "Storage": "#000000",
    "None": "#CCCCCC",  # transport-only nodes with no technology installed
}

# Network mode -> color / label
NETWORK_STYLE = {
    "CO2_Pipeline": {"color": "#CC79A7", "label": "Pipeline"},
    "CO2Railway": {"color": "#E69F00", "label": "Railway"},
    "CO2Truck": {"color": "#56B4E9", "label": "Truck"},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. LOAD NODE COORDINATES
# ----------------------------------------------------------------------------
def load_node_coords(path):
    df = pd.read_excel(path, sheet_name="nodes")
    for col in ["latitude", "longitude"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    df = df.set_index("node_name")
    return df


# ----------------------------------------------------------------------------
# 2. LOAD EMITTER / CCS DESIGN DATA
# ----------------------------------------------------------------------------
def infer_sector(tech_name):
    if "Cement" in tech_name:
        return "Cement"
    if "WasteToEnergy" in tech_name:
        return "WasteToEnergy"
    if "Refinery" in tech_name:
        return "Refinery"
    if "PermanentStorage" in tech_name:
        return "Storage"
    if "Unspecified" in tech_name:
        return "Unspecified"
    return "Unspecified"


def load_emitter_design(h5file):
    nodes_grp = h5file[f"design/nodes/{PERIOD}"]
    records = []
    for node_name in nodes_grp.keys():
        node_grp = nodes_grp[node_name]
        tech_keys = list(node_grp.keys())

        if len(tech_keys) == 0:
            records.append({
                "node_name": node_name,
                "sector": "None",
                "technology": None,
                "size_ccs": 0.0,
                "has_ccs": False,
                "size": 0.0,
            })
            continue

        for tech_name in tech_keys:
            tech_grp = node_grp[tech_name]
            sector = infer_sector(tech_name)

            size = float(tech_grp["size"][()]) if "size" in tech_grp else 0.0

            if "size_ccs" in tech_grp:
                size_ccs = float(tech_grp["size_ccs"][()])
            else:
                size_ccs = 0.0

            records.append({
                "node_name": node_name,
                "sector": sector,
                "technology": tech_name,
                "size_ccs": size_ccs,
                "has_ccs": size_ccs > 0,
                "size": size,
            })

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# 3. LOAD TRANSPORT NETWORK DESIGN DATA
# ----------------------------------------------------------------------------
def load_network_design(h5file, node_names):
    networks_grp = h5file[f"design/networks/{PERIOD}"]
    records = []

    node_name_set = set(node_names)
    sorted_names = sorted(node_name_set, key=len, reverse=True)

    def split_edge_key(edge_key):
        for name in sorted_names:
            if edge_key.startswith(name):
                remainder = edge_key[len(name):]
                if remainder in node_name_set:
                    return name, remainder
        return None, None

    for net_type in networks_grp.keys():
        net_grp = networks_grp[net_type]
        for edge_key in net_grp.keys():
            edge_grp = net_grp[edge_key]

            if "fromNode" in edge_grp and "toNode" in edge_grp:
                from_node = edge_grp["fromNode"][()]
                to_node = edge_grp["toNode"][()]
                if isinstance(from_node, bytes):
                    from_node = from_node.decode()
                if isinstance(to_node, bytes):
                    to_node = to_node.decode()
            else:
                from_node, to_node = split_edge_key(edge_key)

            size = float(edge_grp["size"][()]) if "size" in edge_grp else 0.0
            total_flow = float(edge_grp["total_flow"][()]) if "total_flow" in edge_grp else 0.0

            if size > 0 and from_node is not None and to_node is not None:
                records.append({
                    "network_type": net_type,
                    "from_node": from_node,
                    "to_node": to_node,
                    "size": size,
                    "total_flow": total_flow,
                })

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# 4. FIGURE 1 — GEOGRAPHIC MAP
# ----------------------------------------------------------------------------
def plot_map(coords, emitters, network, output_path):
    use_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception:
        use_cartopy = False

    fig = plt.figure(figsize=(10, 11))

    if use_cartopy:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#F5F5F0", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#E6F2FA", zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#888888", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555", zorder=1)
        transform = ccrs.PlateCarree()
    else:
        ax = plt.axes()
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        transform = None

    lon_min, lon_max = coords["longitude"].min(), coords["longitude"].max()
    lat_min, lat_max = coords["latitude"].min(), coords["latitude"].max()
    pad_lon = (lon_max - lon_min) * 0.08
    pad_lat = (lat_max - lat_min) * 0.08
    extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]

    if use_cartopy:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    max_flow = network["total_flow"].max() if len(network) else 1.0
    for _, row in network.iterrows():
        style = NETWORK_STYLE.get(row["network_type"], {"color": "gray", "label": row["network_type"]})
        try:
            lon1, lat1 = coords.loc[row["from_node"], ["longitude", "latitude"]]
            lon2, lat2 = coords.loc[row["to_node"], ["longitude", "latitude"]]
        except KeyError:
            continue

        lw = 0.8 + 3.5 * (row["total_flow"] / max_flow if max_flow > 0 else 0)
        kwargs = dict(color=style["color"], linewidth=lw, alpha=0.75, zorder=2, solid_capstyle="round")
        if use_cartopy:
            ax.plot([lon1, lon2], [lat1, lat2], transform=transform, **kwargs)
        else:
            ax.plot([lon1, lon2], [lat1, lat2], **kwargs)

    max_ccs = emitters["size_ccs"].max() if emitters["size_ccs"].max() > 0 else 1.0
    for _, row in emitters.iterrows():
        node = row["node_name"]
        if node not in coords.index:
            continue
        lon, lat = coords.loc[node, ["longitude", "latitude"]]
        color = SECTOR_COLORS.get(row["sector"], "#999999")

        base_size = 60
        size = base_size + 340 * (row["size_ccs"] / max_ccs if row["size_ccs"] > 0 else 0)

        plot_kwargs = dict(
            s=size, facecolor=color, edgecolor="black", linewidth=0.6,
            zorder=3, alpha=0.9,
        )
        if use_cartopy:
            ax.scatter(lon, lat, transform=transform, **plot_kwargs)
        else:
            ax.scatter(lon, lat, **plot_kwargs)

        if row["has_ccs"]:
            ring_kwargs = dict(
                s=size * 1.9, facecolor="none", edgecolor="#111111",
                linewidth=1.6, zorder=4,
            )
            if use_cartopy:
                ax.scatter(lon, lat, transform=transform, **ring_kwargs)
            else:
                ax.scatter(lon, lat, **ring_kwargs)

    sector_handles = [
        mpatches.Patch(facecolor=color, edgecolor="black", label=sector)
        for sector, color in SECTOR_COLORS.items()
        if sector in emitters["sector"].unique() and sector != "None"
    ]
    ccs_handle = Line2D(
        [], [], marker="o", markerfacecolor="none", markeredgecolor="#111111",
        markeredgewidth=1.6, linestyle="None", markersize=10, label="CCS installed"
    )
    network_handles = [
        Line2D([], [], color=style["color"], linewidth=2.5, label=style["label"])
        for style in NETWORK_STYLE.values()
    ]

    legend1 = ax.legend(handles=sector_handles + [ccs_handle], loc="lower left",
                         title="Emitter sector", fontsize=9, title_fontsize=10, framealpha=0.92)
    ax.add_artist(legend1)
    ax.legend(handles=network_handles, loc="lower right",
              title="Transport mode", fontsize=9, title_fontsize=10, framealpha=0.92)

    ax.set_title("CO$_2$ Capture and Transport Network — Northern Italy", fontsize=13, weight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=350, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}.png / .pdf  (cartopy basemap: {use_cartopy})")


# ----------------------------------------------------------------------------
# 5. FIGURE 2 — SUMMARY BAR CHARTS
# ----------------------------------------------------------------------------
def plot_summary(emitters, network, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ccs_df = emitters[emitters["size_ccs"] > 0].sort_values("size_ccs", ascending=True)
    colors = [SECTOR_COLORS.get(s, "#999999") for s in ccs_df["sector"]]
    axes[0].barh(ccs_df["node_name"], ccs_df["size_ccs"], color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("Installed CCS capacity")
    axes[0].set_title("(a) Installed CCS capacity by emitter", fontsize=11, weight="bold")
    axes[0].tick_params(axis="y", labelsize=7)

    sector_legend = [mpatches.Patch(facecolor=c, edgecolor="black", label=s)
                      for s, c in SECTOR_COLORS.items() if s in ccs_df["sector"].unique()]
    axes[0].legend(handles=sector_legend, fontsize=8, loc="lower right")

    mode_totals = network.groupby("network_type")["total_flow"].sum()
    mode_labels = [NETWORK_STYLE.get(m, {"label": m})["label"] for m in mode_totals.index]
    mode_colors = [NETWORK_STYLE.get(m, {"color": "gray"})["color"] for m in mode_totals.index]
    axes[1].bar(mode_labels, mode_totals.values, color=mode_colors, edgecolor="black", linewidth=0.6)
    axes[1].set_ylabel("Total CO$_2$ transported (annual)")
    axes[1].set_title("(b) Total CO$_2$ transported by mode", fontsize=11, weight="bold")
    for i, v in enumerate(mode_totals.values):
        axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=350, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}.png / .pdf")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    coords = load_node_coords(NODE_METRICS_PATH)

    with h5py.File(H5_PATH, "r") as f:
        emitters = load_emitter_design(f)
        network = load_network_design(f, coords.index.tolist())

    n_ccs = emitters[emitters["has_ccs"]]["node_name"].nunique()
    print(f"Loaded {emitters['node_name'].nunique()} nodes, {n_ccs} with CCS installed, "
          f"{len(network)} active transport links.")

    plot_map(coords, emitters, network, os.path.join(OUTPUT_DIR, "ccs_chain_map"))
    plot_summary(emitters, network, os.path.join(OUTPUT_DIR, "ccs_chain_summary"))


if __name__ == "__main__":
    main()