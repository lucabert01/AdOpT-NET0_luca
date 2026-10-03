import h5py
from pathlib import Path
from adopt_net0.result_management.read_results import (
    print_h5_tree,
    extract_datasets_from_h5group,
)
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.ticker import FuncFormatter
import warnings


def save_figure_for_paper(fig, filename, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(folder / f"{filename}.pdf", bbox_inches='tight')
    fig.savefig(folder / f"{filename}.jpg", dpi=300, bbox_inches='tight')


def setup_matplotlib_for_paper(column="single"):
    text_width_pt = 469.75539
    inches_per_pt = 1 / 72.27

    widths = {"double": text_width_pt, "single": (text_width_pt - 10) / 2}
    fig_width_in = widths[column] * inches_per_pt

    # Use a slightly taller aspect ratio for 'single' to give legends room
    aspect_ratio = 0.618 if column == "double" else 0.618
    fig_height_in = fig_width_in * aspect_ratio

    fs = 9 if column == "double" else 8  # Don't go below 8 for legibility

    plt.rcParams.update({
        "figure.figsize": (fig_width_in, fig_height_in),
        "figure.dpi": 300,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", # Matches math symbols to the text
        "font.size": fs,
        "axes.labelsize": fs-1,
        "axes.titlesize": fs + 1,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 3,
        "legend.title_fontsize": fs,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.labelspacing": 0.3,  # CRITICAL: Shrinks vertical gap between entries
        "legend.edgecolor": "0.8",  # Light border is less "heavy"
        "legend.handletextpad": 0.2,  # Tighten space
        "legend.columnspacing": 0.8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "figure.constrained_layout.use": True,  # CRITICAL: Auto-adjusts for legends
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return fig_width_in, fig_height_in

def print_h5_structure(file_path, indent=0):
    """
        Print the structure of the h5file for results

        Parameters:
                Directory of the h5file
        """
    with h5py.File(file_path, "r") as hdf_file:
        for key in hdf_file.keys():
            item = hdf_file[key]
            if isinstance(item, h5py.Group):
                print("  " * indent + f"[Group] {key}")
                print_h5_structure(item, indent + 1)
            elif isinstance(item, h5py.Dataset):
                print("  " * indent + f"[Dataset] {key}, shape={item.shape}, dtype={item.dtype}")
            else:
                print("  " * indent + f"[Unknown] {key}")

# ------------------------------------------------------------
# TECHNOLOGY SELECTION PLOTS
# ------------------------------------------------------------
# Colors of the installed technology: grey if nothing is installed, the other ones roughly follow batlow
TECH_COLORS = {
    "none": "#D3D3D3",
    "MEA": "#2F6B9E",
    "CaL": "#D887B5",
    "Oxyfuel": "#9D892B",
    "Oxyfuel + PCC": "#D887B5",
}

# Colormaps (light to dark) of the variables shown next to the technology selection
SEQUENTIAL_CMAPS = {
    "fraction_avoided": LinearSegmentedColormap.from_list("fraction_avoided", ["#FBEEDD", "#C27A2C"]),
    "size": LinearSegmentedColormap.from_list("size", ["#E4EEEF", "#1C5A62"]),
    "load_factor": LinearSegmentedColormap.from_list("load_factor", ["#EEF1E2", "#687B3E"]),
}


def _text_color(facecolor):
    """Black or white text, depending on the luminance of the cell"""
    r, g, b = to_rgb(facecolor)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4 else "black"


def _draw_cells(ax, facecolors, texts):
    """
    Draws a grid of colored cells with a text in each cell

    :param ax: matplotlib axes
    :param pd.DataFrame facecolors: color of each cell
    :param pd.DataFrame texts: text of each cell
    """
    n_rows, n_cols = facecolors.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1,
                    facecolor=facecolors.iloc[i, j],
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
            ax.text(
                j + 0.5, i + 0.5, texts.iloc[i, j],
                ha="center", va="center",
                color=_text_color(facecolors.iloc[i, j]),
                fontsize=rcParams["axes.labelsize"] - 2,
                fontweight="bold",
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([x + 0.5 for x in range(n_cols)])
    ax.set_yticks([y + 0.5 for y in range(n_rows)])
    ax.set_xticklabels(facecolors.columns)
    ax.set_yticklabels(facecolors.index)
    ax.invert_yaxis()


def draw_tech_selection(fig, ax, type_matrix, cost_matrix, types,
                        label="Technology installed"):
    """
    Draws the grid of the installed technology, with the cost of CO2 avoided in each cell

    The legend is a colorbar with one color per technology, so that the axes have the same size as the ones of
    draw_heatmap.

    :param fig: matplotlib figure
    :param ax: matplotlib axes
    :param pd.DataFrame type_matrix: technology installed in each cell
    :param pd.DataFrame cost_matrix: cost of CO2 avoided in each cell
    :param list types: technologies in the legend (keys of TECH_COLORS)
    :param str label: label of the colorbar
    """
    facecolors = type_matrix.apply(lambda col: col.map(TECH_COLORS))
    texts = pd.DataFrame(
        [[f"{cost_matrix.iloc[i, j]:.1f} €/t" if type_matrix.iloc[i, j] != "none" else "-"
          for j in range(type_matrix.shape[1])] for i in range(type_matrix.shape[0])],
        index=type_matrix.index, columns=type_matrix.columns,
    )
    _draw_cells(ax, facecolors, texts)

    cmap = ListedColormap([TECH_COLORS[t] for t in types])
    norm = BoundaryNorm(range(len(types) + 1), cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=label, ticks=[i + 0.5 for i in range(len(types))], drawedges=True)
    cbar.ax.set_yticklabels(types)
    cbar.ax.tick_params(length=0)
    return cbar


def draw_heatmap(fig, ax, df, label, cmap, is_pct=False, zero_color=TECH_COLORS["none"]):
    """
    Draws a heatmap with the value in each cell. Cells equal to 0 (nothing installed) have the zero_color

    :param fig: matplotlib figure
    :param ax: matplotlib axes
    :param pd.DataFrame df: value of each cell
    :param str label: label of the colorbar
    :param cmap: colormap (name or matplotlib colormap)
    :param bool is_pct: if True, values are shown as percentages
    :param zero_color: color of the cells equal to 0
    """
    # colors are based on the values rounded as they are shown in the cells, so that equal numbers have equal colors
    data = df.round(3 if is_pct else 1).to_numpy()
    nonzero = data[data != 0]
    vmin = np.nanmin(nonzero) if len(nonzero) > 0 else 0
    vmax = np.nanmax(nonzero) if len(nonzero) > 0 else 1
    single_value = vmin if vmin == vmax else None
    if single_value is not None:
        # all the cells have the same value: use the middle of the colormap
        vmin, vmax = 0.99 * vmin, 1.01 * vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    facecolors = df.round(3 if is_pct else 1).apply(
        lambda col: col.map(lambda val: zero_color if val == 0 else to_hex(cmap_obj(norm(val))))
    )
    texts = df.apply(lambda col: col.map(lambda val: "-" if val == 0 else f"{val:.1%}" if is_pct else f"{val:.1f}"))
    _draw_cells(ax, facecolors, texts)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=label)
    if single_value is not None:
        cbar.set_ticks([single_value])
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}" if is_pct else f"{x:.1f}"))
    elif is_pct:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    return cbar
