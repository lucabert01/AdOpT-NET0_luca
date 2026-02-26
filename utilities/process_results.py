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
import warnings


def save_figure_for_paper(fig, filename, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(folder / f"{filename}.pdf", bbox_inches='tight')
    fig.savefig(folder / f"{filename}.jpg", dpi=300, bbox_inches='tight')


def setup_matplotlib_for_paper(column="double"):
    text_width_pt = 469.75539
    inches_per_pt = 1 / 72.27

    widths = {"double": text_width_pt, "single": (text_width_pt - 10) / 2}
    fig_width_in = widths[column] * inches_per_pt

    # Use a slightly taller aspect ratio for 'single' to give legends room
    aspect_ratio = 1 if column == "double" else 0.618
    fig_height_in = fig_width_in * aspect_ratio

    fs = 9 if column == "double" else 8  # Don't go below 8 for legibility

    plt.rcParams.update({
        "figure.figsize": (fig_width_in, fig_height_in),
        "figure.dpi": 300,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs + 1,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "legend.title_fontsize": fs,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",  # Light border is less "heavy"
        "legend.handletextpad": 0.4,  # Tighten space
        "legend.columnspacing": 1.0,
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