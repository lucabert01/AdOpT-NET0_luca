#!/usr/bin/env python3
"""
Visualisation of Simulated Network with Mass Balance Verification
with emission data extracted from H5 file for each node.
"""

import re
import h5py
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

try:
    import cmcrameri.cm as cmc
    _HAS_CRAMERI = True
except ImportError:
    _HAS_CRAMERI = False

# ============================================================================
# Configuration
# ============================================================================
HOURS_PER_YEAR = 8760          # for unit conversion: simulation hours → t/yr
FLOW_THRESHOLD = 1e-6          # minimum flow to consider a connection active (t)
ABATEMENT_FULL_THRESHOLD = 0.5  # ≥85 % capture → "full capture"
ABATEMENT_PARTIAL_THRESHOLD = 0.01  # ≥1 % capture → "partial capture"
OFFSET_DISTANCE = 0.015        # degree offset for bidirectional lane separation

# ============================================================================
# Setup paths
# ============================================================================
print("=" * 60)
print("CO2 Transport Network Analysis")
print("=" * 60)

script_dir = Path(__file__).parent
path_data_case_study = script_dir.parent / "italy_data"
path_files_gis = path_data_case_study / "raw_data" / "gis_data"
results_data_path = script_dir.parent / "Results_CCSchainOptimization"
simulation_round_data_path = results_data_path / "20251113121620-1"
h5_file_path = simulation_round_data_path / "optimization_results.h5"
nodes_shp_path = path_files_gis / "all_nodes_italy.shp"
italy_shp_path = path_files_gis / "italy_WGS1984.shp"

# ============================================================================
# STEP 1: Load nodes from shapefile
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 1: Loading nodes...")
print(f"{'=' * 60}")

nodes_gdf = gpd.read_file(nodes_shp_path)
nodes_gdf["Name"] = nodes_gdf["node_name"]

unique_locations = nodes_gdf.groupby("geometry").size()
unique_nodes = nodes_gdf["Name"].nunique()

coord_dict: dict[str, tuple[float, float]] = {
    row["Name"]: (row.geometry.x, row.geometry.y)
    for _, row in nodes_gdf.iterrows()
}

print(f"✅ Loaded {len(nodes_gdf)} node records "
      f"({unique_nodes} unique names, {len(unique_locations)} unique locations)")
print(f"   Node types: {nodes_gdf['node_type'].value_counts().to_dict()}")

# ============================================================================
# STEP 2: Extract emission data from H5 for each node
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 2: Extracting emission data...")
print(f"{'=' * 60}")

# node_emissions[node_name] accumulates across ALL technologies at that node.
node_emissions: dict[str, dict] = {}

with h5py.File(h5_file_path, "r") as f:
    period = list(f["operation/technology_operation"].keys())[0]
    tech_op_group = f[f"operation/technology_operation/{period}"]

    for node_name in tech_op_group.keys():
        node_group = tech_op_group[node_name]

        # Accumulate across all technologies at this node (FIX #3)
        node_total_captured = 0.0
        node_total_emissions_pos = 0.0
        node_technologies = []

        for tech_name in node_group.keys():
            tech_group = node_group[tech_name]

            if "CO2captured_var_output_ccs" not in tech_group:
                continue

            co2_captured_data = tech_group["CO2captured_var_output_ccs"][:]
            co2_captured_annual = float(np.sum(co2_captured_data))

            emissions_pos_annual = 0.0
            if "emissions_pos" in tech_group:
                emissions_pos_data = tech_group["emissions_pos"][:]
                emissions_pos_annual = float(np.sum(emissions_pos_data))

            node_total_captured += co2_captured_annual
            node_total_emissions_pos += emissions_pos_annual
            node_technologies.append(tech_name)

        if node_technologies:
            node_emissions[node_name] = {
                "co2_captured": node_total_captured,
                "emissions_pos": node_total_emissions_pos,
                "technologies": node_technologies,
            }
            print(f"  {node_name}")
            print(f"    Technologies: {', '.join(node_technologies)}")
            print(f"    CO2 Captured (annualised): {node_total_captured:,.2f} t/yr")
            print(f"    Emissions Positive (annualised): {node_total_emissions_pos:,.2f} t/yr")

# ============================================================================
# STEP 3: Calculate abatement rates
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 3: Calculating abatement rates...")
print(f"{'=' * 60}")

node_categories: dict[str, str] = {}

for node_name, emission_data in node_emissions.items():
    co2_captured = emission_data["co2_captured"]
    emissions_pos = emission_data["emissions_pos"]
    total_co2 = co2_captured + emissions_pos

    if total_co2 > 0:
        abatement_rate = co2_captured / total_co2
        if abatement_rate >= ABATEMENT_FULL_THRESHOLD:
            category = "emitter_full"
        elif abatement_rate >= ABATEMENT_PARTIAL_THRESHOLD:
            category = "emitter_partial"
        else:
            category = "emitter_none"
        print(f"  {node_name}: {abatement_rate * 100:.1f}% abatement → {category}")
    else:
        category = "emitter_none"
        print(f"  {node_name}: 0% CO2 produced → emitter_none")

    node_categories[node_name] = category

# Add storage nodes from design data
with h5py.File(h5_file_path, "r") as f:
    design_key = "design/nodes/period1"
    if design_key in f:
        for node_name in f[design_key].keys():
            if node_name not in node_categories:
                node_group = f[f"{design_key}/{node_name}"]
                for tech_name in node_group.keys():
                    if "Storage" in tech_name:
                        node_categories[node_name] = "storage"
                        print(f"  {node_name}: storage site")
                        break

# ============================================================================
# STEP 4: Extract active flows
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 4: Extracting flows...")
print(f"{'=' * 60}")

active_connections: list[dict] = []

with h5py.File(h5_file_path, "r") as f:
    period = list(f["operation/networks"].keys())[0]
    network_types = list(f[f"operation/networks/{period}"].keys())

    for network_type in network_types:
        network_group = f[f"operation/networks/{period}/{network_type}"]
        type_active = 0

        for connection in network_group.keys():
            if "flow" not in network_group[connection]:
                continue
            flow_data = network_group[connection]["flow"][:]
            total_flow = float(np.sum(flow_data))

            if total_flow > FLOW_THRESHOLD:
                active_connections.append({
                    "network_type": network_type,
                    "connection": connection,
                    "total_flow": total_flow,
                })
                type_active += 1

        print(f"  {network_type}: {type_active} active connections")

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================
print(f"\n{'=' * 60}")
print("SUMMARY STATISTICS")
print(f"{'=' * 60}")

emitters_full    = [n for n, c in node_categories.items() if c == "emitter_full"]
emitters_partial = [n for n, c in node_categories.items() if c == "emitter_partial"]
emitters_none    = [n for n, c in node_categories.items() if c == "emitter_none"]
storage_sites    = [n for n, c in node_categories.items() if c == "storage"]

print(f"\n📊 NETWORK COMPOSITION:")
print(f"  CO2 Emitters:")
print(f"    • Full Capture (≥{ABATEMENT_FULL_THRESHOLD*100:.0f}%): {len(emitters_full)}")
print(f"    • Partial Capture ({ABATEMENT_PARTIAL_THRESHOLD*100:.0f}–{ABATEMENT_FULL_THRESHOLD*100:.0f}%): {len(emitters_partial)}")
print(f"    • No Capture: {len(emitters_none)}")
print(f"  Storage Sites: {len(storage_sites)}")
print(f"  Active Transport Connections: {len(active_connections)}")

total_captured = sum(d["co2_captured"] for d in node_emissions.values())
total_vented   = sum(d["emissions_pos"] for d in node_emissions.values())
overall_abatement = (total_captured / (total_captured + total_vented)
                     if (total_captured + total_vented) > 0 else 0.0)

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"  Total CO2 Captured:  {total_captured:,.2f} t/yr")
print(f"  Total CO2 Vented:    {total_vented:,.2f} t/yr")
print(f"  Network Abatement Rate: {overall_abatement * 100:.1f}%")

# ============================================================================
# STEP 6: Parse connections — robust exact-delimiter matching (FIX #2)
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 6: Parsing connections with exact-delimiter matching...")
print(f"{'=' * 60}")

nodes_set = set(nodes_gdf["Name"].tolist())

def parse_connection(conn_name: str, known_nodes: set[str]) -> tuple[str | None, str | None]:
    """
    Extract (from_node, to_node) from a connection string.

    Tries three strategies in order:
      1. Split on common delimiters ('__', '_to_', '-').
         Both halves must be exact node names.
      2. Scan the sorted-by-length node list: find the longest node name
         that is a prefix of conn_name, then check if the remainder
         (after stripping delimiters) is also an exact node name.
         This avoids the partial-match false-positives of the original code.
      3. Return (None, None) if nothing matches.
    """
    # Strategy 1: explicit delimiter split
    for sep in ("__", "_to_", "-"):
        parts = conn_name.split(sep, 1)
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a in known_nodes and b in known_nodes and a != b:
                return a, b

    # Strategy 2: longest-prefix scan (prevents 'Milano' matching 'Milano_Nord')
    sorted_nodes = sorted(known_nodes, key=len, reverse=True)
    for n1 in sorted_nodes:
        if conn_name.startswith(n1):
            remainder = conn_name[len(n1):].lstrip("_- ")
            for n2 in sorted_nodes:
                if n2 == n1:
                    continue
                # n2 must match the full remainder (or its stripped version)
                if remainder == n2 or remainder.startswith(n2 + "_") or remainder.startswith(n2 + "-"):
                    return n1, n2

    return None, None


# Build registry: {sorted_pair: [conn_info, ...]}
connection_registry: dict[tuple[str, str], list[dict]] = {}

n_unparsed = 0
for conn in active_connections:
    from_node, to_node = parse_connection(conn["connection"], nodes_set)

    if from_node and to_node:
        pair = tuple(sorted([from_node, to_node]))
        connection_registry.setdefault(pair, []).append({
            "from": from_node,
            "to": to_node,
            "data": conn,
        })
    else:
        n_unparsed += 1
        print(f"  ⚠️  Could not parse connection: '{conn['connection']}'")

if n_unparsed:
    print(f"  ⚠️  {n_unparsed} connection(s) could not be matched to node pairs.")

bidirectional_pairs = {
    pair for pair, conns in connection_registry.items() if len(conns) > 1
}

if bidirectional_pairs:
    print(f"\n⚠️  WARNING: {len(bidirectional_pairs)} bidirectional flow pair(s) detected!")
    print("   Possible causes: reversible infrastructure, seasonal flow reversal,")
    print("   or a mass-balance constraint issue in the optimisation model.")
    for pair in list(bidirectional_pairs)[:3]:
        for flow in connection_registry[pair]:
            print(f"     • {flow['from']} → {flow['to']}: "
                  f"{flow['data']['total_flow']:,.0f} t/yr "
                  f"via {flow['data']['network_type']}")

# ============================================================================
# STEP 7: Visualisation
# ============================================================================
print(f"\n{'=' * 60}")
print("STEP 7: Creating visualisation...")
print(f"{'=' * 60}")

# --- Map node categories back to GeoDataFrame ---
active_nodes_in_flow: set[str] = set()
for conns in connection_registry.values():
    for ci in conns:
        active_nodes_in_flow.add(ci["from"])
        active_nodes_in_flow.add(ci["to"])


def assign_category(row) -> str:
    name = row["Name"]
    if name in node_categories:
        return node_categories[name]
    return "hub_active" if name in active_nodes_in_flow else "hub_inactive"


nodes_gdf["category"] = nodes_gdf.apply(assign_category, axis=1)
nodes_gdf["annual_captured"] = nodes_gdf["Name"].map(
    lambda x: node_emissions.get(x, {}).get("co2_captured", 0.0)
)

# --- Colors ---
if _HAS_CRAMERI:
    navia = cmc.navia
    network_colors = {
        "CO2_Pipeline": navia(0.15),
        "CO2Railway":   navia(0.85),
        "CO2Truck":     navia(0.50),
    }
else:
    network_colors = {
        "CO2_Pipeline": "#1f77b4",
        "CO2Railway":   "#d62728",
        "CO2Truck":     "#7f7f7f",
    }

node_colors = {
    "emitter_full":    "#88c879",
    "emitter_partial": "#e2e9ba",
    "emitter_none":    "#D32F2F",
    "hub_active":      "#044977",
    "hub_inactive":    "#BDBDBD",
    "storage":         "#041a39",
    "unknown":         "#CCCCCC",
}

# --- Figure ---
fig, ax = plt.subplots(figsize=(24, 12))

if italy_shp_path.exists():
    italy_boundary = gpd.read_file(italy_shp_path)
    if nodes_gdf.crs != italy_boundary.crs:
        italy_boundary = italy_boundary.to_crs(nodes_gdf.crs)
    italy_boundary.plot(ax=ax, color="#eeeeee", edgecolor="#bcbcbc",
                        linewidth=1, zorder=1)
    print("✅ Base map loaded")

ax.set_xlim(7.2, 14.0)
ax.set_ylim(43.9, 46.5)

max_flow = max((c["data"]["total_flow"]
                for conns in connection_registry.values()
                for c in conns), default=1.0)

plotted_types: set[str] = set()

for pair, conns in connection_registry.items():
    is_bidir = pair in bidirectional_pairs

    for idx, conn_info in enumerate(conns):
        from_node = conn_info["from"]
        to_node   = conn_info["to"]
        conn      = conn_info["data"]
        net_type  = conn["network_type"]

        p1 = np.array(coord_dict[from_node])
        p2 = np.array(coord_dict[to_node])

        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue

        if is_bidir:
            perp = np.array([-direction[1], direction[0]]) / length
            sign = 1 if idx == 0 else -1
            offset_vec = perp * OFFSET_DISTANCE * sign
            p1_plot = p1 + offset_vec
            p2_plot = p2 + offset_vec
        else:
            p1_plot, p2_plot = p1, p2

        flow_ratio = conn["total_flow"] / max_flow
        lw = 2 + 8 * flow_ratio
        line_style = "-" if "Pipeline" in net_type else "--" if "Railway" in net_type else ":"
        label = net_type if net_type not in plotted_types else ""

        ax.plot(
            [p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]],
            color=network_colors.get(net_type, "gray"),
            linewidth=lw, linestyle=line_style, alpha=0.6,
            zorder=3, label=label,
        )

        # Direction arrow at 55% of route
        arrow_pos = 0.55
        arrow_base = p1_plot + arrow_pos * (p2_plot - p1_plot)
        arrow_tip  = p1_plot + (arrow_pos + 0.05) * (p2_plot - p1_plot)
        base_arrow_width = max(2.5, lw * 0.8)
        mutation_scale   = max(20, 15 * (lw / 10))

        # White outline for thin flows to improve visibility
        if flow_ratio < 0.3:
            ax.annotate(
                "", xy=arrow_tip, xytext=arrow_base,
                arrowprops=dict(
                    arrowstyle="-|>", lw=base_arrow_width + 1.5,
                    color="white", alpha=0.8,
                    mutation_scale=mutation_scale + 2,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3.5,
            )

        ax.annotate(
            "", xy=arrow_tip, xytext=arrow_base,
            arrowprops=dict(
                arrowstyle="-|>", lw=base_arrow_width,
                color=network_colors.get(net_type, "gray"), alpha=0.95,
                mutation_scale=mutation_scale,
                shrinkA=0, shrinkB=0,
            ),
            zorder=4,
        )

        plotted_types.add(net_type)

print(f"📊 Plotted {sum(len(c) for c in connection_registry.values())} connections")

# ============================================================================
# --- Plot nodes (Continuous Size Scaling) ---
# ============================================================================
print("📌 Plotting nodes with continuous size scaling...")

# Define bounds for your visual markers (in points squared)
MIN_MARKER_SIZE = 80
MAX_MARKER_SIZE = 1200
STORAGE_SITE_SIZE = 600  # Dedicated size for storage sites since they don't "capture"

# Find the maximum captured value to scale against
max_captured_value = nodes_gdf["annual_captured"].max()

for _, row in nodes_gdf.iterrows():
    cat = row["category"]
    val = row["annual_captured"]

    # Calculate continuous size based on captured CO2
    if cat == "storage":
        size = STORAGE_SITE_SIZE
        marker = "s"  # Square for storage
    else:
        marker = "o"  # Circle for emitters/hubs
        if max_captured_value > 0 and val > 0:
            scale_ratio = val / max_captured_value
            size = MIN_MARKER_SIZE + (MAX_MARKER_SIZE - MIN_MARKER_SIZE) * scale_ratio
        else:
            size = MIN_MARKER_SIZE

    color = node_colors.get(cat, node_colors["unknown"])

    ax.scatter(
        row.geometry.x, row.geometry.y,
        s=size, marker=marker,
        facecolor=color, edgecolor="black",
        linewidth=1.5 if cat == "storage" else 1.2,
        zorder=5,
    )
# --- Legends ---
legend_elements = [
    Patch(facecolor=node_colors["emitter_full"],    edgecolor="k",
          label=f"Full Capture (≥{ABATEMENT_FULL_THRESHOLD*100:.0f}%)"),
    Patch(facecolor=node_colors["emitter_partial"], edgecolor="k",
          label="Partial Capture"),
    Patch(facecolor=node_colors["emitter_none"],    edgecolor="k",
          label="No Capture"),
    Patch(facecolor=node_colors["hub_active"],      edgecolor="k",
          label="Active Hub"),
    Patch(facecolor=node_colors["storage"],         edgecolor="k",
          label="Storage Site"),
]
leg1 = ax.legend(handles=legend_elements, loc="lower right",
                 title="Infrastructure Status", frameon=True, fontsize=11)
ax.add_artist(leg1)

if plotted_types:
    leg2 = ax.legend(loc="upper right", title="Transport Mode", fontsize=11)
    ax.add_artist(leg2)

ax.set_title(
    "Northern Italy CO2 Transport Network\n"
    "objective = emissions_minC & MAX storage capacity",
    fontsize=16, pad=20, weight="bold",
)
ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Latitude (°N)", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
ax.set_aspect("equal")

plt.tight_layout()
out_path = "co2_network_final_map.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"🎨 Map saved as: {out_path}")
plt.show()

# ============================================================================
# Final Summary & Mass Balance
# ============================================================================
print(f"\n{'=' * 60}")
print("✅ VISUALISATION COMPLETE")
print(f"{'=' * 60}")
print(f"📊 Network Summary:")
print(f"   Total Nodes: {len(nodes_gdf)}")
print(f"   Parsed Connections: {sum(len(c) for c in connection_registry.values())}")
print(f"   Unparsed Connections: {n_unparsed}")
print(f"   Bidirectional Routes: {len(bidirectional_pairs)}")

print(f"\n   Node Categories:")
for cat_key, label in [
    ("emitter_full",    "Full Capture Sites"),
    ("emitter_partial", "Partial Capture Sites"),
    ("emitter_none",    "No Capture Sites"),
    ("storage",         "Storage Sites"),
    ("hub_active",      "Active Hubs"),
    ("hub_inactive",    "Inactive Hubs"),
]:
    count = (nodes_gdf["category"] == cat_key).sum()
    print(f"     • {label}: {count}")

# Mass balance uses only parsed connections for consistency (FIX #1)
total_transported = sum(
    ci["data"]["total_flow"]
    for conns in connection_registry.values()
    for ci in conns
)
total_captured_nodes = nodes_gdf["annual_captured"].sum()

print(f"\n   Mass Balance Check (parsed connections only):")
print(f"     • Total CO2 Captured (nodes):     {total_captured_nodes:>15,.0f} t/yr")
print(f"     • Total CO2 Transported (pipes):  {total_transported:>15,.0f} t/yr")

if total_captured_nodes > 0:
    balance_ratio = (total_transported / total_captured_nodes) * 100
    print(f"     • Transport / Capture Ratio:      {balance_ratio:>14.1f}%")
    if balance_ratio > 110:
        print(f"     ⚠️  WARNING: transport exceeds capture by "
              f"{balance_ratio - 100:.1f}% — check mass-balance constraints")
    elif n_unparsed > 0:
        print(f"     ℹ️  Note: {n_unparsed} unparsed connection(s) excluded; "
              f"ratio may be understated")