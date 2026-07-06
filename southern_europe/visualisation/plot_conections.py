"""
generate_north_network_plots.py
================================
Extends generate_network_plots.py to render the node connections
(pipeline / truck / railway) restricted to nodes located in
Northern Italy, drawn on top of the Italy country boundary — using
the same "Northern Italy" latitude threshold/bounding-box logic as
the cost-factor mapping script.

Produces one PNG per transport mode, plus a nodes-only overview:
    exports/network_pipeline_NORTH.png
    exports/network_truck_NORTH.png
    exports/network_railway_NORTH.png
    exports/network_nodes_only_NORTH.png

Run directly:
    python generate_north_network_plots.py

Requires:
    pip install pandas geopandas openpyxl matplotlib
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ==========================================
# 1. PATHS
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = SCRIPT_DIR.parent / "italy_data" / "geographical_feature" / "node_metrics_150.xlsx"
ITALY_SHP_PATH = SCRIPT_DIR.parent / "italy_data" / "raw_data" / "gis_data" / "italy_WGS1984.shp"
EXPORT_DIR = SCRIPT_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Could not locate Excel file at: {EXCEL_PATH}")
if not ITALY_SHP_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Could not locate Italy shapefile at: {ITALY_SHP_PATH}")

MATRIX_MODES = ["pipeline", "truck", "railway"]

# Nodes are still selected using the same threshold as the cost-factor
# mapping script, so the two analyses stay consistent about which nodes
# count as "Northern".
NORTH_LAT_THRESHOLD = 44

# The map view itself is fixed to this latitude band (tighter zoom than
# the raw node bounding box), longitude is still padded around the data.
NORTH_YLIM = (43, 48)
BBOX_PAD = 0.3  # degrees of padding around the northern nodes' longitude extent


# ==========================================
# 2. DATA LOADING
# ==========================================
def load_network_data(matrix_type: str):
    df_nodes = pd.read_excel(EXCEL_PATH, sheet_name="nodes")
    df_matrix = pd.read_excel(EXCEL_PATH, sheet_name=matrix_type, index_col=0)
    df_matrix.columns = df_matrix.columns.astype(int)
    return df_nodes, df_matrix


def load_italy_boundary() -> gpd.GeoDataFrame:
    return gpd.read_file(ITALY_SHP_PATH)


# ==========================================
# 3. NODE TYPE CLASSIFICATION
# ==========================================
# Waste, Cement, Refining, and any other unrecognised node_type are all
# grouped together and labelled "Emitter". Transport and Storage keep
# their own category.
EMITTER_SUBTYPES = {"Waste", "Cement", "Refining"}
KNOWN_TYPES = {"Transport", "Storage"}

CATEGORY_STYLES = {
    "Emitter":   {"color": "#c0392b", "marker": "o"},  # red circle
    "Transport": {"color": "#2c3e50", "marker": "s"},  # dark square
    "Storage":   {"color": "#8e44ad", "marker": "D"},  # purple diamond
}
CATEGORY_ORDER = ["Emitter", "Transport", "Storage"]

LINE_COLORS = {
    "pipeline": "#2ecc71",
    "truck":    "#e67e22",
    "railway":  "#3498db",
}


def classify_node_category(node_type) -> str:
    """Waste / Cement / Refining / missing / anything unrecognised -> Emitter.
    Transport and Storage keep their own label."""
    if pd.isna(node_type):
        return "Emitter"
    nt = str(node_type).strip()
    if nt in KNOWN_TYPES:
        return nt
    return "Emitter"


def annotate_node_categories(nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    if "node_type" in nodes.columns:
        nodes["category"] = nodes["node_type"].apply(classify_node_category)
    else:
        nodes["node_type"] = "N/A"
        nodes["category"] = "Emitter"
    return nodes


# ==========================================
# 4. NORTHERN ITALY FILTERING
# ==========================================
def filter_nodes_north(nodes: pd.DataFrame, threshold: float = NORTH_LAT_THRESHOLD) -> pd.DataFrame:
    """Keep only nodes whose latitude places them in Northern Italy."""
    return nodes[nodes["latitude"] > threshold].copy()


def filter_matrix_to_nodes(matrix: pd.DataFrame, node_ids) -> pd.DataFrame:
    """Restrict a connectivity matrix to rows/columns whose id is in node_ids
    (so only connections between two Northern nodes survive)."""
    node_ids = set(node_ids)
    keep_rows = [i for i in matrix.index if i in node_ids]
    keep_cols = [c for c in matrix.columns if c in node_ids]
    return matrix.loc[keep_rows, keep_cols]


# ==========================================
# 5. PLOT BUILDER
# ==========================================
def build_north_network_plot(nodes_north: pd.DataFrame, matrix_north: pd.DataFrame,
                              matrix_name: str, italy_boundary: gpd.GeoDataFrame):
    nodes_north = annotate_node_categories(nodes_north)
    line_color = LINE_COLORS.get(matrix_name, "#7f8c8d")

    fig, ax = plt.subplots(figsize=(10, 10))

    # ── Italy country outline for geographic context ────────────────────
    italy_boundary.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=0)

    # ── Draw connections as arrows, only between Northern nodes ─────────
    n_connections = 0
    for idx in matrix_north.index:
        for col in matrix_north.columns:
            if idx == col:
                continue
            val = matrix_north.loc[idx, col]
            if pd.notna(val) and val > 0:
                node_a = nodes_north[nodes_north["node_id"] == idx]
                node_b = nodes_north[nodes_north["node_id"] == col]
                if node_a.empty or node_b.empty:
                    continue

                x_a, y_a = node_a.iloc[0]["longitude"], node_a.iloc[0]["latitude"]
                x_b, y_b = node_b.iloc[0]["longitude"], node_b.iloc[0]["latitude"]

                arrow = FancyArrowPatch(
                    (x_a, y_a), (x_b, y_b),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    shrinkA=8, shrinkB=8,
                    color=line_color,
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )
                ax.add_patch(arrow)
                n_connections += 1

    # ── Draw nodes, coloured by category ────────────────────────────────
    for category in CATEGORY_ORDER:
        group = nodes_north[nodes_north["category"] == category]
        if group.empty:
            continue
        style = CATEGORY_STYLES[category]
        ax.scatter(
            group["longitude"], group["latitude"],
            s=110,
            c=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=1.2,
            label=category,
            zorder=2,
        )

    # ── Node ID labels ───────────────────────────────────────────────────
    for row in nodes_north.itertuples():
        ax.annotate(
            str(row.node_id),
            (row.longitude, row.latitude),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            zorder=3,
        )

    # ── Zoom: fixed latitude band, longitude padded around the data ─────
    if not nodes_north.empty:
        minx, maxx = nodes_north["longitude"].min(), nodes_north["longitude"].max()
        ax.set_xlim(minx - BBOX_PAD, maxx + BBOX_PAD)
    ax.set_ylim(*NORTH_YLIM)

    ax.set_title(f"{matrix_name.upper()} Network — Northern Italy ({n_connections} connections)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Node Type", loc="upper right", frameon=True)
    fig.tight_layout()
    return fig, ax


def build_north_nodes_only_plot(nodes_north: pd.DataFrame, italy_boundary: gpd.GeoDataFrame):
    """Same styling as build_north_network_plot, but with no connection arrows —
    just the nodes, so the site locations can be viewed on their own."""
    nodes_north = annotate_node_categories(nodes_north)

    fig, ax = plt.subplots(figsize=(10, 10))

    # ── Italy country outline for geographic context ────────────────────
    italy_boundary.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=0)

    # ── Draw nodes, coloured by category ────────────────────────────────
    for category in CATEGORY_ORDER:
        group = nodes_north[nodes_north["category"] == category]
        if group.empty:
            continue
        style = CATEGORY_STYLES[category]
        ax.scatter(
            group["longitude"], group["latitude"],
            s=110,
            c=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=1.2,
            label=category,
            zorder=2,
        )

    # ── Node ID labels ───────────────────────────────────────────────────
    for row in nodes_north.itertuples():
        ax.annotate(
            str(row.node_id),
            (row.longitude, row.latitude),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            zorder=3,
        )

    # ── Zoom: fixed latitude band, longitude padded around the data ─────
    if not nodes_north.empty:
        minx, maxx = nodes_north["longitude"].min(), nodes_north["longitude"].max()
        ax.set_xlim(minx - BBOX_PAD, maxx + BBOX_PAD)
    ax.set_ylim(*NORTH_YLIM)

    ax.set_title("Nodes — Northern Italy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Node Type", loc="upper right", frameon=True)
    fig.tight_layout()
    return fig, ax


# ==========================================
# 6. EXPORT
# ==========================================
def export_north_network_png(matrix_name: str, italy_boundary: gpd.GeoDataFrame, dpi: int = 300) -> Path:
    """Build and save a single mode's Northern-Italy network map as a PNG."""
    nodes, matrix = load_network_data(matrix_name)
    nodes_north = filter_nodes_north(nodes)
    matrix_north = filter_matrix_to_nodes(matrix, nodes_north["node_id"])

    fig, _ = build_north_network_plot(nodes_north, matrix_north, matrix_name, italy_boundary)
    out_path = EXPORT_DIR / f"network_{matrix_name}_NORTH.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_north_nodes_only_png(italy_boundary: gpd.GeoDataFrame, dpi: int = 300) -> Path:
    """Build and save a Northern-Italy map showing only the nodes (no connections)."""
    # Node list is identical across sheets, so any mode's "nodes" sheet works.
    nodes, _ = load_network_data(MATRIX_MODES[0])
    nodes_north = filter_nodes_north(nodes)

    fig, _ = build_north_nodes_only_plot(nodes_north, italy_boundary)
    out_path = EXPORT_DIR / "network_nodes_only_NORTH.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_all_north_network_pngs() -> list[Path]:
    italy_boundary = load_italy_boundary()
    paths = [export_north_network_png(mode, italy_boundary) for mode in MATRIX_MODES]
    paths.append(export_north_nodes_only_png(italy_boundary))
    return paths


# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Reading data from: {EXCEL_PATH}")
    saved_paths = export_all_north_network_pngs()
    print("Saved Northern Italy network PNG plots:")
    for p in saved_paths:
        print(f"  - {p}")