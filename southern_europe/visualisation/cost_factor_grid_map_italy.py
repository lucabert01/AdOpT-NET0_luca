import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmcrameri.cm as cmc
import matplotlib as mpl
from pathlib import Path
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


# ——— Choose which pipeline size to map ———
# Must match (or fall between) the pipeline_category values in cost_factor_table.xlsx
PIPELINE_CATEGORY = 300  # <-- change this to whichever size you want to plot


# Import data
path_data_case_study = Path("../italy_data")

path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_grids = path_data_case_study / "geographical_feature"

# Path to the cost factor coefficient table, relative to this script's location
# (southern_europe/visualisation/ -> adopt_net0/database/data/networks/enhanced_co2_transport_cost_model/)
path_cost_factor_table = Path(
    "../../adopt_net0/database/data/networks/enhanced_co2_transport_cost_model/cost_factor_table.xlsx"
)

italy = gpd.read_file(path_files_gis / "italy_WGS1984.shp")  # italy boundary
fishnet = gpd.read_file(path_files_gis / "fishnet_italy_5km.shp").reset_index().rename(
    columns={"index": "GRID_OID"})
soil_data = pd.read_csv(path_files_grids / "soil_type_grids_italy.csv")
anthro_data = pd.read_csv(path_files_grids / "anthropisation_grids_italy.csv")
morpho_data = pd.read_csv(path_files_grids / "morphological_feature_grids_italy.csv")
cost_factor_table = pd.read_excel(path_cost_factor_table)


def get_cost_coefficients(table: pd.DataFrame, category: float) -> dict:
    """
    Look up the k-coefficients for a given pipeline_category.

    If `category` matches a row exactly, that row is returned as-is.
    Otherwise the coefficients are linearly interpolated between the
    two nearest pipeline_category values in the table (so you're not
    restricted to only the discrete sizes listed in the sheet).
    """
    table = table.sort_values("pipeline_category").reset_index(drop=True)

    exact_match = table.loc[table["pipeline_category"] == category]
    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    if category < table["pipeline_category"].min() or category > table["pipeline_category"].max():
        raise ValueError(
            f"PIPELINE_CATEGORY={category} is outside the range covered by the table "
            f"({table['pipeline_category'].min()}–{table['pipeline_category'].max()}). "
            f"Extrapolation is not supported — pick a value within range."
        )

    coeffs = {"pipeline_category": category}
    for col in table.columns:
        if col == "pipeline_category":
            continue
        coeffs[col] = float(np.interp(category, table["pipeline_category"], table[col]))
    return coeffs


coeffs = get_cost_coefficients(cost_factor_table, PIPELINE_CATEGORY)
print(f"Using cost factor coefficients for pipeline_category={PIPELINE_CATEGORY}:")
print(coeffs)

# Merge all attributes into fishnet
fishnet = (fishnet
           .merge(soil_data, on="GRID_OID")
           .merge(anthro_data, on="GRID_OID")
           .merge(morpho_data, on="GRID_OID"))

# Calculate factors using the coefficients looked up above
fishnet['SOIL_FACTOR'] = (coeffs['k_soil_non_rock'] * fishnet['NON_ROCK_S']
                           + coeffs['k_soil_rock'] * fishnet['ROCK_S'])
fishnet['ANTHRO_FACTOR'] = (coeffs['k_anthro_non_anthropised'] * fishnet['NON_ANTHROPISED_A']
                             + coeffs['k_anthro_anthropised'] * fishnet['ANTHROPISED_A'])
fishnet['MORPH_FACTOR'] = (coeffs['k_morpho_plain'] * fishnet['PLAIN_M']
                            + coeffs['k_morpho_hill'] * fishnet['HILL_M']
                            + coeffs['k_morpho_mountain'] * fishnet['MOUNTAIN_M'])
fishnet['COST_FACTOR'] = fishnet[['SOIL_FACTOR', 'ANTHRO_FACTOR', 'MORPH_FACTOR']].sum(axis=1)

# Clip to Italy boundary
fishnet_clipped = gpd.clip(fishnet, italy)

# ——— Define northern Italy bounding box ———
NORTH_LAT_THRESHOLD = 44

northern_subset = fishnet_clipped[fishnet_clipped.geometry.centroid.y > NORTH_LAT_THRESHOLD]
minx, miny, maxx, maxy = northern_subset.total_bounds

pad = 0.3  # degrees of padding around the bounding box
xlim = (minx - pad, maxx + pad)
ylim = (miny - pad, maxy + pad)

# ——— Plot 1: four‐panel row for the individual factors and integrated cost factor ———
fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=False)
plt.subplots_adjust(wspace=0.05, right=0.85)
fig.suptitle(f"Pipeline category: {PIPELINE_CATEGORY}", fontsize=14, y=1.02)

min_val = min(northern_subset['MORPH_FACTOR'].min(),
              northern_subset['SOIL_FACTOR'].min(),
              northern_subset['ANTHRO_FACTOR'].min(),
              northern_subset['COST_FACTOR'].min())
max_val = max(northern_subset['MORPH_FACTOR'].max(),
              northern_subset['SOIL_FACTOR'].max(),
              northern_subset['ANTHRO_FACTOR'].max(),
              northern_subset['COST_FACTOR'].max())

norm = Normalize(vmin=min_val, vmax=max_val)
cmap = cmc.navia_r

panel_info = [
    ('MORPH_FACTOR', 'Geomorphological feature'),
    ('SOIL_FACTOR', 'Soil type'),
    ('ANTHRO_FACTOR', 'Anthropization'),
    ('COST_FACTOR', 'Integrated cost factor')
]

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for i, (col, title) in enumerate(panel_info):
    ax = axes[i]
    italy.boundary.plot(ax=ax, color='black', linewidth=0.8)
    fishnet_clipped.plot(column=col, ax=ax, cmap=cmap, norm=norm, legend=False)
    fishnet_clipped.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, y=-0.1, fontsize=12)
    ax.set_axis_off()

cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Factor Value', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.savefig(f'italy_incremental_cost_factors_with_integrated_NORTH_cat{PIPELINE_CATEGORY}.png',
            dpi=600, bbox_inches='tight')
plt.show()

# ——— Plot 2: single map for the total cost factor (kept for comparison/standalone use) ———
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
italy.boundary.plot(ax=ax, color='black', linewidth=1)

fishnet_clipped.plot(column='COST_FACTOR', ax=ax, cmap=cmc.navia_r, legend=False)
fishnet_clipped.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()
ax.set_title(f"Cost factor — pipeline category {PIPELINE_CATEGORY}", fontsize=12)

cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
sm2 = plt.cm.ScalarMappable(cmap=cmc.navia_r, norm=plt.Normalize(
    northern_subset['COST_FACTOR'].min(), northern_subset['COST_FACTOR'].max()))
cbar = fig.colorbar(sm2, cax=cbar_ax)
cbar.set_label('Cost Factor Value', fontsize=12)

plt.subplots_adjust(right=0.8)

plt.savefig(f'italy_cost_factor_NORTH_cat{PIPELINE_CATEGORY}.png', dpi=600, bbox_inches='tight')
plt.show()