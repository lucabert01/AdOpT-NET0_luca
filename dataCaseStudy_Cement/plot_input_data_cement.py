from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
from matplotlib import rcParams
import warnings

from utilities.process_results import save_figure_for_paper, print_h5_structure


colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']


possible_plants = ["Vernasca", "Robilante", "Monselice", "Fanna"]
plant_analyzed = "Vernasca"
path_processed_data = Path("./dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
av_el_price = electricity_price_data["el_price_itNord"].mean()
electricity_price_norm = electricity_price_data["el_price_itNord"]/av_el_price

clinker_data = pd.read_excel(path_processed_data, sheet_name="clinker_production")
clinker_demand = clinker_data[f"clinker_{plant_analyzed}"]

# --- Time series: Electricity price ---
plt.figure()
plt.plot(electricity_price_norm.index,
         electricity_price_norm,
         color=batlow_colors[0])
plt.title("Normalized Electricity Price – IT Nord")
plt.xlabel("Time")
plt.ylabel("Normalized price")
plt.tight_layout()
plt.show()


# --- Time series: Clinker demand ---
plt.figure()
plt.plot(clinker_demand.index,
         clinker_demand,
         color=batlow_colors[2])
plt.title(f"Clinker Demand – {plant_analyzed}")
plt.xlabel("Time")
plt.ylabel("Clinker demand")
plt.tight_layout()
plt.show()


# --- ECDF: Electricity price ---
x_el = np.sort(electricity_price_norm.values)
y_el = np.arange(1, len(x_el) + 1) / len(x_el)

plt.figure()
plt.plot(x_el, y_el, color=batlow_colors[4])
plt.title("ECDF – Normalized Electricity Price")
plt.xlabel("Normalized price")
plt.ylabel("Cumulative probability")
plt.tight_layout()
plt.show()


# --- ECDF: Clinker demand ---
x_cl = np.sort(clinker_demand.values)
y_cl = np.arange(1, len(x_cl) + 1) / len(x_cl)

plt.figure()
plt.plot(x_cl, y_cl, color=batlow_colors[5])
plt.title(f"ECDF – Clinker Demand ({plant_analyzed})")
plt.xlabel("Clinker demand")
plt.ylabel("Cumulative probability")
plt.tight_layout()
plt.show()