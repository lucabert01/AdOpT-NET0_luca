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

def setup_matplotlib_for_paper(column="single"):
    """
    Configure matplotlib for journal-quality figures.

    Parameters
    ----------
    column : {"single", "double"}
        Single- or double-column figure.
    """

    # Journal-standard column widths (pt)
    column_width_pt = {
        "single": 432,
        "double": 648,
    }

    if column not in column_width_pt:
        raise ValueError("column must be 'single' or 'double'")

    aspect_ratio = 2 / 3
    inches_per_pt = 1 / 72.27
    fig_width_in = column_width_pt[column] * inches_per_pt
    fig_height_in = fig_width_in * aspect_ratio

    # Font sizes tuned per column width
    if column == "single":
        font = dict(
            base=8,
            label=8,
            tick=7,
            legend=7,
            title=8,
        )
    else:  # double
        font = dict(
            base=9,
            label=9,
            tick=8,
            legend=8,
            title=9,
        )

    rcParams.update({
        # Figure
        'figure.figsize': (fig_width_in, fig_height_in),
        'figure.dpi': 300,

        # Fonts
        'font.family': 'Arial',
        'font.size': font["base"],
        'axes.labelsize': font["label"],
        'axes.titlesize': font["title"],
        'xtick.labelsize': font["tick"],
        'ytick.labelsize': font["tick"],
        'legend.fontsize': font["legend"],

        # Line & axes aesthetics
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,

        # Mathtext safety (CO$_2$, etc.)
        'mathtext.default': 'regular',
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