import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmcrameri.cm as cmc
import matplotlib as mpl
from pathlib import Path
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from shapely.geometry import Point, LineString


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
nodes_selected = gpd.read_file(path_files_gis / "all_nodes_italy.shp")
routes_pipeline = gpd.read_file(path_files_gis / "routes_distances_pipelines.shp")
network_pipeline = pd.read_excel(path_files_grids / "node_metrics.xlsx", index_col=0, sheet_name='pipeline')
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


nodes_selected = nodes_selected.to_crs(italy.crs)
routes_pipeline = routes_pipeline.to_crs(italy.crs)

pipeline_color = cmc.navia(0.15)  # matches route_colors['pipeline'] in routes_connection.py


def get_route_directionality_fixed(routes_gdf, network_matrix, route_type):
    """Determine flow direction for each route, matching routes_connection.py exactly."""
    route_directions = {}

    for idx, route in routes_gdf.iterrows():
        try:
            from_node_id = None
            to_node_id = None

            if 'Node' in route.index and pd.notna(route['Node']):
                node_str = str(route['Node']).strip()
                separators = [',', '-', ';', '|', ' ']
                for sep in separators:
                    if sep in node_str:
                        node_parts = node_str.split(sep)
                        if len(node_parts) >= 2:
                            try:
                                from_node_id = int(node_parts[0].strip())
                                to_node_id = int(node_parts[1].strip())
                                break
                            except ValueError:
                                continue
                        break

            line = route.geometry
            start_point = Point(line.coords[0])
            end_point = Point(line.coords[-1])

            start_distances = nodes_selected.geometry.distance(start_point)
            end_distances = nodes_selected.geometry.distance(end_point)

            closest_to_start_idx = start_distances.idxmin()
            closest_to_end_idx = end_distances.idxmin()

            geometry_start_node = closest_to_start_idx + 1
            geometry_end_node = closest_to_end_idx + 1

            if from_node_id is None or to_node_id is None:
                from_node_id = geometry_start_node
                to_node_id = geometry_end_node

            forward_connection = False
            backward_connection = False

            if from_node_id is not None and to_node_id is not None:
                try:
                    if from_node_id in network_matrix.index and to_node_id in network_matrix.columns:
                        forward_connection = network_matrix.loc[from_node_id, to_node_id] > 0
                    if to_node_id in network_matrix.index and from_node_id in network_matrix.columns:
                        backward_connection = network_matrix.loc[to_node_id, from_node_id] > 0
                except Exception:
                    pass

            if forward_connection and backward_connection:
                direction = 'bidirectional'
                inlet_position = 'both_ends'
            elif forward_connection:
                direction = 'forward'
                inlet_position = 'start' if geometry_start_node == from_node_id else 'end'
            elif backward_connection:
                direction = 'backward'
                inlet_position = 'start' if geometry_start_node == to_node_id else 'end'
            else:
                direction = 'none'
                inlet_position = 'start'

            route_directions[idx] = {'direction': direction, 'inlet_position': inlet_position}

        except Exception as e:
            print(f"Error processing {route_type} route {idx}: {e}")
            route_directions[idx] = {'direction': 'error', 'inlet_position': 'start'}

    return route_directions


def _point_and_tangent_at_fraction(coords, fraction):
    """Return (point, (dx, dy)) at a given fraction of the route's arc length."""
    cum = [0.0]
    total = 0.0
    for i in range(1, len(coords)):
        d = ((coords[i][0] - coords[i - 1][0]) ** 2 + (coords[i][1] - coords[i - 1][1]) ** 2) ** 0.5
        total += d
        cum.append(total)

    if total == 0:
        return coords[0], (1.0, 0.0)

    target = fraction * total

    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg_start = coords[i - 1]
            seg_end = coords[i]
            seg_len = cum[i] - cum[i - 1]
            seg_frac = (target - cum[i - 1]) / seg_len if seg_len > 0 else 0.0
            point = (
                seg_start[0] + seg_frac * (seg_end[0] - seg_start[0]),
                seg_start[1] + seg_frac * (seg_end[1] - seg_start[1])
            )
            dx = seg_end[0] - seg_start[0]
            dy = seg_end[1] - seg_start[1]
            return point, (dx, dy)

    dx = coords[-1][0] - coords[-2][0]
    dy = coords[-1][1] - coords[-2][1]
    return coords[-1], (dx, dy)


def draw_direction_arrow(ax, coords, color, reverse=False, fraction=0.5,
                          arrow_length=0.09, linewidth=1.5, zorder=10):
    """Draw a single small arrow along the route indicating flow direction."""
    point, (dx, dy) = _point_and_tangent_at_fraction(coords, fraction)
    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm == 0:
        return
    ux, uy = dx / norm, dy / norm
    if reverse:
        ux, uy = -ux, -uy

    start = (point[0] - ux * arrow_length / 2, point[1] - uy * arrow_length / 2)
    end = (point[0] + ux * arrow_length / 2, point[1] + uy * arrow_length / 2)

    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='-|>', color=color, lw=linewidth, mutation_scale=14),
        zorder=zorder
    )


def plot_route_simple_with_arrow(ax, route, color, direction_info, linewidth=1.2, alpha=1.0):
    """Plot a thin, uniform-opacity route line with a directional arrow."""
    line = route.geometry
    coords = list(line.coords)
    coords = [(x, y) for x, y, *_ in coords]  # normalize to 2D

    if len(coords) < 2:
        return

    gpd.GeoSeries([LineString(coords)]).plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=5)

    direction = direction_info.get('direction', 'unknown')
    inlet_position = direction_info.get('inlet_position', 'start')

    if direction == 'bidirectional':
        draw_direction_arrow(ax, coords, color, reverse=True, fraction=0.4, linewidth=linewidth)
        draw_direction_arrow(ax, coords, color, reverse=False, fraction=0.6, linewidth=linewidth)
    else:
        reverse = (inlet_position == 'end')
        draw_direction_arrow(ax, coords, color, reverse=reverse, fraction=0.5, linewidth=linewidth)


pipeline_directions = get_route_directionality_fixed(routes_pipeline, network_pipeline, 'Pipeline')

# ——— Node styling, matching routes_connection.py exactly ———
CATEGORY_STYLES = {
    'Emitter':   {'marker': 'o', 'color': 'red'},
    'Transport': {'marker': 's', 'color': 'purple'},
    'Storage':   {'marker': 's', 'color': 'black'},
}
NON_EMITTER_TYPES = {'Transport', 'Storage'}


def get_node_category(node_type):
    if node_type in NON_EMITTER_TYPES:
        return node_type
    return 'Emitter'


def draw_nodes_by_type(ax, nodes_gdf, markersize=80, zorder=20, edgecolor='white', linewidth=1.5, legend=True):
    """Draw nodes with consistent marker+color per category (Emitter/Transport only here)."""
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['_category'] = nodes_gdf['node_type'].apply(
        lambda t: get_node_category(t) if pd.notna(t) else None
    )
    nodes_gdf = nodes_gdf[nodes_gdf['_category'].isin(['Emitter', 'Transport'])]

    present_categories = [c for c in nodes_gdf['_category'].dropna().unique()]
    ordered_categories = [c for c in ['Emitter', 'Transport'] if c in present_categories]

    for category in ordered_categories:
        style = CATEGORY_STYLES[category]
        subset = nodes_gdf[nodes_gdf['_category'] == category]
        subset.plot(ax=ax, marker=style['marker'], color=style['color'], markersize=markersize,
                    edgecolors=edgecolor, linewidth=linewidth, zorder=zorder, label=category)

    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=len(ordered_categories),
                  fontsize=12, frameon=True, fancybox=True, shadow=True, markerscale=1.3,
                  handletextpad=0.6, columnspacing=1.2)


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

# ——— Plot 3: total cost factor with pipeline connections + Transport/Emitter nodes overlaid ———
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
italy.boundary.plot(ax=ax, color='black', linewidth=1)

fishnet_clipped.plot(column='COST_FACTOR', ax=ax, cmap=cmc.navia_r, legend=False)
fishnet_clipped.boundary.plot(ax=ax, color='gray', linewidth=0.3, alpha=0.5)

for idx, route in routes_pipeline.iterrows():
    plot_route_simple_with_arrow(ax, route, pipeline_color, pipeline_directions[idx], linewidth=1.2)

draw_nodes_by_type(ax, nodes_selected, markersize=60, legend=False)

route_legend = [plt.Line2D([0], [0], color=pipeline_color, lw=2, label='Pipeline')]
node_legend = [
    plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'],
               markersize=9, markeredgecolor='white', markeredgewidth=1.2, label=category, linestyle='None')
    for category, style in CATEGORY_STYLES.items() if category in ['Emitter', 'Transport']
]
ax.legend(handles=route_legend + node_legend, loc='upper center', bbox_to_anchor=(0.5, -0.06),
          ncol=3, fontsize=12, frameon=True, fancybox=True, shadow=True, markerscale=1.3,
          handletextpad=0.6, columnspacing=1.2)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_axis_off()
ax.set_title(f"Cost factor with pipeline network — pipeline category {PIPELINE_CATEGORY}", fontsize=12)

cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
sm3 = plt.cm.ScalarMappable(cmap=cmc.navia_r, norm=plt.Normalize(
    northern_subset['COST_FACTOR'].min(), northern_subset['COST_FACTOR'].max()))
cbar = fig.colorbar(sm3, cax=cbar_ax)
cbar.set_label('Cost Factor Value', fontsize=12)

plt.subplots_adjust(right=0.8)

plt.savefig(f'italy_cost_factor_with_nodes_NORTH_cat{PIPELINE_CATEGORY}.png', dpi=600, bbox_inches='tight')
plt.show()