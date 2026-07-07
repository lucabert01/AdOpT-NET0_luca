import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

# Import cmcrameri for navia colormap
try:
    import cmcrameri.cm as cmc

    navia_available = True
    print("CMC colormaps loaded successfully!")
except ImportError:
    print("Warning: cmcrameri not available. Install with: pip install cmcrameri")
    print("Falling back to matplotlib's viridis colormap")
    navia_available = False

warnings.filterwarnings('ignore')

# ============================================================
# Load data
# ============================================================
path_data_case_study = Path("../italy_data")
path_files_gis = path_data_case_study / "raw_data/gis_data"
path_files_node_flux = path_data_case_study / "geographical_feature"

italy = gpd.read_file(path_files_gis / "italy_WGS1984.shp")
nodes_selected = gpd.read_file(path_files_gis / "all_nodes_italy.shp")
routes_pipeline = gpd.read_file(path_files_gis / "routes_distances_pipelines.shp")
routes_railway = gpd.read_file(path_files_gis / "truck_italy_150.shp")  # TODO: point to actual railway shapefile
routes_truck = gpd.read_file(path_files_gis / "truck_italy_150.shp")

# Load network matrices for transport directions
network_pipeline = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='pipeline')
network_truck = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='truck')
network_railway = pd.read_excel(path_files_node_flux / "node_metrics.xlsx", index_col=0, sheet_name='railway')

# --- Reproject route layers to match italy/nodes CRS ---
target_crs = italy.crs
routes_pipeline = routes_pipeline.to_crs(target_crs)
routes_railway = routes_railway.to_crs(target_crs)
routes_truck = routes_truck.to_crs(target_crs)

print("Data loaded successfully!")
print(f"Italy boundary: {italy.shape[0]} features")
print(f"Selected nodes: {nodes_selected.shape[0]} nodes")
print(f"Pipeline routes: {routes_pipeline.shape[0]} routes")
print(f"Railway routes: {routes_railway.shape[0]} routes")
print(f"Truck routes: {routes_truck.shape[0]} routes")

print("\nNode type distribution (from shapefile):")
print(nodes_selected['node_type'].value_counts(dropna=False))
print("Unique node_type values:", nodes_selected['node_type'].unique())

# ============================================================
# Consistent node styling, bucketed by category (marker, color)
# ============================================================
# node_type can hold several emitter subtypes (e.g. 'Cement', 'Waste refining',
# 'Other', etc.) alongside the two infrastructure types 'Transport' and
# 'Storage'. Anything that is NOT explicitly 'Transport' or 'Storage' is
# treated as an emitter and gets the same red-circle style, regardless of
# its specific subtype.
CATEGORY_STYLES = {
    'Emitter':   {'marker': 'o', 'color': 'red'},
    'Transport': {'marker': 's', 'color': 'purple'},
    'Storage':   {'marker': 's', 'color': 'black'},
}

# Node types that map to their own dedicated category (exact match, case-sensitive).
# Everything else falls into 'Emitter'.
NON_EMITTER_TYPES = {'Transport', 'Storage'}


def get_node_category(node_type):
    if node_type in NON_EMITTER_TYPES:
        return node_type
    return 'Emitter'


def get_node_style(node_type):
    category = get_node_category(node_type)
    return CATEGORY_STYLES[category]


def draw_nodes_by_type(ax, nodes_gdf, markersize=80, zorder=20, edgecolor='white', linewidth=1.5, legend=True):
    """Draw nodes with consistent marker+color per category, used across all plots."""
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['_category'] = nodes_gdf['node_type'].apply(
        lambda t: get_node_category(t) if pd.notna(t) else None
    )

    present_categories = [c for c in nodes_gdf['_category'].dropna().unique()]
    # Keep a consistent legend order: Emitter, Transport, Storage
    ordered_categories = [c for c in ['Emitter', 'Transport', 'Storage'] if c in present_categories]

    for category in ordered_categories:
        style = CATEGORY_STYLES[category]
        subset = nodes_gdf[nodes_gdf['_category'] == category]
        subset.plot(ax=ax, marker=style['marker'], color=style['color'], markersize=markersize,
                    edgecolors=edgecolor, linewidth=linewidth, zorder=zorder, label=category)

    missing = nodes_gdf[nodes_gdf['_category'].isna()]
    if len(missing) > 0:
        missing.plot(ax=ax, marker='D', color='gray',
                     markersize=markersize, edgecolors=edgecolor, linewidth=linewidth, zorder=zorder, label='Unknown')

    if legend:
        n_items = len(ordered_categories) + (1 if len(missing) > 0 else 0)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=max(n_items, 1),
                  fontsize=12, frameon=True, fancybox=True, shadow=True, markerscale=1.3,
                  handletextpad=0.6, columnspacing=1.2)


# ============================================================
# Route colors
# ============================================================
def setup_navia_colors():
    """Setup specific colors from CMC navia colormap for each route type"""
    if navia_available:
        navia_cmap = cmc.navia
        route_colors = {
            'pipeline': navia_cmap(0.15),
            'truck': navia_cmap(0.5),
            'railway': navia_cmap(0.85)
        }
        colormap = navia_cmap
    else:
        viridis_cmap = plt.cm.viridis
        route_colors = {
            'pipeline': viridis_cmap(0.15),
            'truck': viridis_cmap(0.5),
            'railway': viridis_cmap(0.85)
        }
        colormap = viridis_cmap
    return route_colors, colormap


route_colors, colormap = setup_navia_colors()

# ============================================================
# Map bounds (shared across all plots)
# ============================================================
map_bounds = {
    'minx': 6.5, 'maxx': 14.0, 'miny': 43.5, 'maxy': 47.0
}


# ============================================================
# Direction analysis (needed to know which end of each route
# is the flow origin, so we can draw the arrow correctly)
# ============================================================
def get_route_directionality_fixed(routes_gdf, network_matrix, route_type):
    route_directions = {}
    debug_info = []

    print(f"\nAnalyzing {route_type} routes...")

    for idx, route in routes_gdf.iterrows():
        try:
            from_node_id = None
            to_node_id = None
            method_used = "unknown"

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
                                method_used = f"node_column_{sep}_separated"
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
                method_used = "geometry_fallback"

            forward_connection = False
            backward_connection = False
            forward_value = 0
            backward_value = 0

            if from_node_id is not None and to_node_id is not None:
                try:
                    if from_node_id in network_matrix.index and to_node_id in network_matrix.columns:
                        forward_value = network_matrix.loc[from_node_id, to_node_id]
                        forward_connection = forward_value > 0

                    if to_node_id in network_matrix.index and from_node_id in network_matrix.columns:
                        backward_value = network_matrix.loc[to_node_id, from_node_id]
                        backward_connection = backward_value > 0
                except Exception as e:
                    print(f"    Error checking network matrix: {e}")

            if forward_connection and backward_connection:
                direction = 'bidirectional'
                inlet_position = 'both_ends'
                flow_origin_node = None
            elif forward_connection:
                direction = 'forward'
                flow_origin_node = from_node_id
                if geometry_start_node == from_node_id:
                    inlet_position = 'start'
                elif geometry_end_node == from_node_id:
                    inlet_position = 'end'
                else:
                    inlet_position = 'start'
            elif backward_connection:
                direction = 'backward'
                flow_origin_node = to_node_id
                if geometry_start_node == to_node_id:
                    inlet_position = 'start'
                elif geometry_end_node == to_node_id:
                    inlet_position = 'end'
                else:
                    inlet_position = 'end'
            else:
                direction = 'none'
                inlet_position = 'start'
                flow_origin_node = None

            route_directions[idx] = {
                'direction': direction,
                'inlet_position': inlet_position,
                'from_node': from_node_id,
                'to_node': to_node_id,
                'geometry_start_node': geometry_start_node,
                'geometry_end_node': geometry_end_node,
                'flow_origin_node': flow_origin_node,
                'method': method_used,
                'forward_value': forward_value,
                'backward_value': backward_value
            }

            debug_info.append(route_directions[idx])

        except Exception as e:
            print(f"Error processing {route_type} route {idx}: {e}")
            route_directions[idx] = {
                'direction': 'error',
                'inlet_position': 'start',
                'from_node': None,
                'to_node': None,
                'geometry_start_node': None,
                'geometry_end_node': None,
                'flow_origin_node': None,
                'method': 'error',
                'forward_value': 0,
                'backward_value': 0
            }

    direction_counts = {}
    for info in debug_info:
        d = info['direction']
        direction_counts[d] = direction_counts.get(d, 0) + 1
    print(f"{route_type} direction analysis:")
    for direction, count in direction_counts.items():
        print(f"  {direction}: {count} routes")

    return route_directions


# ============================================================
# Simple thin-line plotting with a directional arrow
# ============================================================
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


def plot_simple_route(ax, route, color, linewidth=1.2, alpha=1.0):
    """Route line only, no arrow (used in the overview plot)."""
    line = route.geometry
    gpd.GeoSeries([line]).plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=5)


def setup_base_map(ax, title):
    italy.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.7)
    italy.plot(ax=ax, color='lightgray', alpha=0.2)
    ax.set_xlim(map_bounds['minx'], map_bounds['maxx'])
    ax.set_ylim(map_bounds['miny'], map_bounds['maxy'])
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')


# ============================================================
# Run direction analysis
# ============================================================
print("\nAnalyzing route directions...")
pipeline_directions = get_route_directionality_fixed(routes_pipeline, network_pipeline, 'Pipeline')
truck_directions = get_route_directionality_fixed(routes_truck, network_truck, 'Truck')
railway_directions = get_route_directionality_fixed(routes_railway, network_railway, 'Railway')

# ============================================================
# PLOT 1 — Overview (all modes, simple lines, no arrows)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 10))
setup_base_map(ax1, 'Full overview')

for idx, route in routes_pipeline.iterrows():
    plot_simple_route(ax1, route, route_colors['pipeline'], linewidth=1.2, alpha=1.0)
for idx, route in routes_truck.iterrows():
    plot_simple_route(ax1, route, route_colors['truck'], linewidth=1.0, alpha=1.0)
for idx, route in routes_railway.iterrows():
    plot_simple_route(ax1, route, route_colors['railway'], linewidth=1.2, alpha=1.0)

draw_nodes_by_type(ax1, nodes_selected, markersize=70, legend=False)

route_legend = [
    plt.Line2D([0], [0], color=route_colors['pipeline'], lw=2, label='Pipeline'),
    plt.Line2D([0], [0], color=route_colors['truck'], lw=2, label='Truck'),
    plt.Line2D([0], [0], color=route_colors['railway'], lw=2, label='Railway'),
]
node_legend = [
    plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'],
               markersize=9, markeredgecolor='white', markeredgewidth=1.2, label=category, linestyle='None')
    for category, style in CATEGORY_STYLES.items()
]
ax1.legend(handles=route_legend + node_legend, loc='upper center', bbox_to_anchor=(0.5, -0.10),
           ncol=6, fontsize=12, frameon=True, fancybox=True, shadow=True, markerscale=1.3,
           handletextpad=0.6, columnspacing=1.2)

fig1.tight_layout(rect=[0, 0.05, 1, 1])
fig1.savefig("italy_overview.png", dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("Saved: italy_overview.png")

# ============================================================
# PLOT 2 — Pipeline network (standalone)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(9, 9))
setup_base_map(ax2, f'Pipeline network, ({len(routes_pipeline)} arcs)')
for idx, route in routes_pipeline.iterrows():
    plot_route_simple_with_arrow(ax2, route, route_colors['pipeline'], pipeline_directions[idx], linewidth=1.2)
draw_nodes_by_type(ax2, nodes_selected, markersize=60)
fig2.tight_layout(rect=[0, 0.05, 1, 1])
fig2.savefig("italy_pipeline_network.png", dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("Saved: italy_pipeline_network.png")

# ============================================================
# PLOT 3 — Truck network (standalone)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(9, 9))
setup_base_map(ax3, f'Truck network, ({len(routes_truck)} arcs)')
for idx, route in routes_truck.iterrows():
    plot_route_simple_with_arrow(ax3, route, route_colors['truck'], truck_directions[idx], linewidth=1.2)
draw_nodes_by_type(ax3, nodes_selected, markersize=60)
fig3.tight_layout(rect=[0, 0.05, 1, 1])
fig3.savefig("italy_truck_network.png", dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("Saved: italy_truck_network.png")

# ============================================================
# PLOT 4 — Railway network (standalone)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(9, 9))
setup_base_map(ax4, f'Railway network, ({len(routes_railway)} arcs)')
for idx, route in routes_railway.iterrows():
    plot_route_simple_with_arrow(ax4, route, route_colors['railway'], railway_directions[idx], linewidth=1.2)
draw_nodes_by_type(ax4, nodes_selected, markersize=60)
fig4.tight_layout(rect=[0, 0.05, 1, 1])
fig4.savefig("italy_railway_network.png", dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("Saved: italy_railway_network.png")

# ============================================================
# PLOT 5 — All nodes only, no connections, styled by node_type
# ============================================================
fig5, ax5 = plt.subplots(figsize=(9, 9))
setup_base_map(ax5, 'Emitters and transport nodes')
draw_nodes_by_type(ax5, nodes_selected, markersize=80)
fig5.tight_layout(rect=[0, 0.05, 1, 1])
fig5.savefig("italy_nodes_by_type.png", dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("Saved: italy_nodes_by_type.png")

# ============================================================
# PLOT 6 — Non-Transport nodes only, no connections, styled by node_type
# ============================================================
fig6, ax6 = plt.subplots(figsize=(9, 9))
setup_base_map(ax6, "Emitters")
non_transport = nodes_selected[nodes_selected['node_type'].apply(get_node_category) != 'Transport']
draw_nodes_by_type(ax6, non_transport, markersize=80)
fig6.tight_layout(rect=[0, 0.05, 1, 1])
fig6.savefig("italy_nodes_non_transport.png", dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("Saved: italy_nodes_non_transport.png")

plt.show()