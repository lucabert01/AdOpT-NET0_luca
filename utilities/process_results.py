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


def save_figure_for_paper(fig, filename, file_path_results):
    """
    Save a matplotlib figure with settings similar to the provided MATLAB function.

    Parameters:
        fig : matplotlib.figure.Figure
            The figure handle to save.
        filename : str
            The base filename for saving the figure.
        file_path_results : str or Path
            The directory where the figure should be saved.
    """
    from matplotlib import rcParams

    # Convert to Path object if needed
    file_path_results = Path(file_path_results)
    file_path_results.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Set figure size (width x height in inches)
    width_in, height_in = 432 / 72, 288 / 72  # Convert from points (1 pt = 1/72 inch)
    fig.set_size_inches(width_in, height_in)


    # # Save in PDF and JPG formats
    fig.savefig(file_path_results / f"{filename}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(file_path_results / f"{filename}.jpg", format='jpeg', dpi=300, bbox_inches='tight')



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