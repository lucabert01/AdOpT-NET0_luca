"""
Adds new pipeline arcs to the Italy case-study input data, without rerunning the
full GIS / gamma pipeline for every existing arc.

For every arc in NEW_PIPELINE_ARCS this script
1. writes the arc's length into the 'pipeline' sheet of node_metrics_paper.xlsx.
   The route is a straight line between the two nodes, and its length is the
   WGS84 geodesic length, the same as the existing routes in
   raw_data/gis_data/routes_distances_pipeline.shp.
2. intersects that straight line with the 5 km fishnet and writes the
   "{from_id}_{to_id}" sheet of route_grid_intersections.xlsx, in the same
   format as geo_data_processing/identify_intersection.py.
3. computes gamma1/gamma2 for the small, medium and large pipeline size classes
   with the same cost model and mass-flow ranges as
   pipeline_capex_per_arc_calculator.py, and patches them into
   capex_defined_per_arc_{size}.xlsx and gamma_defined_per_arc_pipeline_{size}.xlsx.

The arcs are directed, like the other arcs in the 'pipeline' sheet (towards the
storage site). The script can be run again: it overwrites the same cells and
sheets. Run it again after identify_intersection.py or
pipeline_capex_per_arc_calculator.py, because those rewrite the files from
routes_distances_pipeline.shp, which does not contain these arcs.

Run from this folder: python add_pipeline_arcs.py
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import openpyxl
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo_data_processing/')))

from arc_specific_functions import (
    load_network_data,
    load_intersection_data,
    calculate_arc_gammas,
    determine_arc_terrain,
    patch_gamma_matrix_file,
)
from defined_functions import analyze_route_grid_intersections
from pipeline_capex_per_arc_calculator import SIZE_CLASS_MASSFLOW_RANGES_KG_S

# (from node_name, to node_name), as in the 'nodes' sheet of node_metrics_paper.xlsx
NEW_PIPELINE_ARCS = [
    ("Ferrara", "Eni S.p.A Casalborsetti"),
    ("Italcementi di Rezzato", "Ferrara"),
    ("MONSELICE cement plant", "Ferrara"),
    # in the 'pipeline' sheet already (73.92 km), but not in
    # routes_distances_pipeline.shp, so it had no route grid cells
    ("Segrate", "TERMOVALORIZZATORE DI BRESCIA"),
]
# Minimum outer diameter (inch) of the arcs above, converted to a capacity in t/h
# by gas_pipeline_capacity_kg_s (printed only; the values used are in main_italy.py)
MINIMUM_DIAMETER_INCH = {
    ("Ferrara", "Eni S.p.A Casalborsetti"): 36,
    ("Italcementi di Rezzato", "Ferrara"): 26,
}

PATH_DATA = Path(__file__).resolve().parent.parent.parent / "italy_data"
PATH_NODE_METRICS = PATH_DATA / "geographical_feature" / "node_metrics_paper.xlsx"
PATH_INTERSECTIONS = PATH_DATA / "geographical_feature" / "route_grid_intersections.xlsx"
PATH_FISHNET = PATH_DATA / "raw_data" / "gis_data" / "fishnet_italy_5km.shp"
PATH_CAPEX = PATH_DATA / "network_capex_metrics"


def gas_pipeline_capacity_kg_s(od_inch, length_km, terrain="Onshore"):
    """
    Largest CO2 mass flow (kg/s) a pipe of outer diameter od_inch can carry over
    length_km under the assumptions of the gas-phase node-pressure mode of the
    Oeuvray model (CO2Chain_Oeuvray._calculate_gas_node_pressure_configuration):
    every segment starts at NODE_PRESSURE_BAR (40 bar), ends at no less than the
    lowest suction pressure (15 bar), the velocity at the segment end stays below
    vRange_max (20 m/s), and booster stations are at least 40 km apart (as many
    segments as that allows). The wall thickness is the one of the lowest steel
    grade (X42) at the 40 bar design pressure, which gives the smallest inner
    diameter and so a conservative capacity.

    This converts the minimum pipe diameters in main_italy.py's
    MINIMUM_PIPELINE_CAPACITY_T_H into t/h (x 3.6).
    """
    import math
    from adopt_net0.database.components.networks.utilities.co2_pipelines_oeuvray import (
        CO2Chain_Oeuvray,
    )
    from arc_specific_functions import NODE_PRESSURE_BAR

    p_in_mpa, p_min_mpa, min_station_spacing_km = NODE_PRESSURE_BAR / 10, 1.5, 40
    model = CO2Chain_Oeuvray()
    model.length_km = length_km
    model.terrain = terrain
    model.timeframe = "mid-term"  # steel grades available, as in create_base_options
    model._preprocess_data()
    model.phase = "gas"
    od_m = od_inch * 0.0254
    id_m = min(od_m - 2 * model._calculate_pipe_thickness(od_m, p_in_mpa, grade.S_MPa)
               for _, grade in model.steel_data.iterrows())
    v_max = model.terrain_specific_data["gas"][terrain]["vRange_max"]
    segment_m = length_km * 1000 / max(1, math.floor(length_km / min_station_spacing_km))

    def feasible(m_kg_s):
        model.m_kg_per_s = m_kg_s
        p_end_mpa = model._calculate_gas_segment_outlet_pressure(p_in_mpa, id_m, segment_m)
        if p_end_mpa < p_min_mpa:
            return False
        rho = model._gas_property(p_end_mpa, "Density (kg/m3)")
        return 4 * m_kg_s / (math.pi * id_m ** 2 * rho) <= v_max

    low, high = 0.0, 2000.0
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if feasible(mid) else (low, mid)
    return low


def node_ids_and_coordinates():
    nodes = pd.read_excel(PATH_NODE_METRICS, index_col=0, sheet_name="nodes")
    nodes = nodes[~nodes.index.duplicated(keep="first")]
    nodes["node_name"] = nodes["node_name"].str.strip()
    return {row.node_name: (node_id, row.longitude, row.latitude)
            for node_id, row in nodes.iterrows()}


def write_pipeline_distances(arcs):
    """Writes {(from_id, to_id): length_km} into the 'pipeline' sheet, keeping all other cells."""
    wb = openpyxl.load_workbook(PATH_NODE_METRICS)
    ws = wb["pipeline"]
    col_of = {cell.value: cell.column for cell in ws[1] if isinstance(cell.value, int)}
    row_of = {ws.cell(row, 1).value: row for row in range(2, ws.max_row + 1)
              if isinstance(ws.cell(row, 1).value, int)}
    for (from_id, to_id), length_km in arcs.items():
        ws.cell(row_of[from_id], col_of[to_id]).value = length_km
    wb.save(PATH_NODE_METRICS)


def write_intersections(routes):
    """
    Writes one "{from_id}_{to_id}" sheet per route into route_grid_intersections.xlsx,
    replacing a sheet of that name if it exists.

    :param dict routes: {(from_id, to_id): (route name, LineString in EPSG:4326)}
    """
    fishnet = gpd.read_file(PATH_FISHNET).to_crs(epsg=4326)
    fishnet["grid_id"] = fishnet.index + 1  # GRID_OID of the raster tables, see identify_intersection.py
    route_gdf = gpd.GeoDataFrame(
        {"Name": [name for name, _ in routes.values()]},
        geometry=[line for _, line in routes.values()], crs="EPSG:4326",
    )
    results = analyze_route_grid_intersections(route_gdf, fishnet)

    with pd.ExcelWriter(PATH_INTERSECTIONS, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        for (from_id, to_id), (name, _) in routes.items():
            metadata_row = pd.DataFrame({"grid_id": [f"Route: {name}"],
                                         "intersection_length": [""], "proportion": [""]})
            sheet = pd.concat([metadata_row, results[name]], ignore_index=True)
            sheet.to_excel(writer, sheet_name=f"{from_id}_{to_id}", index=False)
            print(f"  {from_id}_{to_id}: {len(results[name])} grid cells")


def main():
    nodes = node_ids_and_coordinates()
    geod = Geod(ellps="WGS84")

    routes, lengths = {}, {}
    for from_name, to_name in NEW_PIPELINE_ARCS:
        from_id, from_lon, from_lat = nodes[from_name]
        to_id, to_lon, to_lat = nodes[to_name]
        line = LineString([(from_lon, from_lat), (to_lon, to_lat)])
        lengths[(from_id, to_id)] = round(geod.geometry_length(line) / 1000, 6)
        routes[(from_id, to_id)] = (f"{from_name} - {to_name}", line)
        print(f"{from_name} ({from_id}) -> {to_name} ({to_id}): {lengths[(from_id, to_id)]:.3f} km")
        if (from_name, to_name) in MINIMUM_DIAMETER_INCH:
            od_inch = MINIMUM_DIAMETER_INCH[(from_name, to_name)]
            capacity_kg_s = gas_pipeline_capacity_kg_s(od_inch, lengths[(from_id, to_id)])
            print(f"  capacity of a {od_inch} inch pipe: {capacity_kg_s:.1f} kg/s = "
                  f"{capacity_kg_s * 3.6:.1f} t/h (MINIMUM_PIPELINE_CAPACITY_T_H in main_italy.py)")

    print("\nWriting pipeline distances to node_metrics_paper.xlsx")
    write_pipeline_distances(lengths)

    print("\nIntersecting routes with the 5 km fishnet")
    write_intersections(routes)

    print("\nComputing gammas per pipeline size class")
    data_dict = load_network_data(PATH_DATA)
    data_dict["intersection_data"] = load_intersection_data(
        PATH_INTERSECTIONS, [f"{f}_{t}" for f, t in routes]
    )
    for size_class, (min_kg_s, max_kg_s) in SIZE_CLASS_MASSFLOW_RANGES_KG_S.items():
        for from_id, to_id in routes:
            gamma1, gamma2 = calculate_arc_gammas(
                from_id, to_id, data_dict,
                terrain=determine_arc_terrain(from_id, to_id, data_dict),
                massflow_min_kg_s_override=min_kg_s,
                massflow_max_kg_s_override=max_kg_s,
            )
            if gamma1 == 0 and gamma2 == 0:
                raise RuntimeError(f"Gamma calculation failed for {from_id} -> {to_id} ({size_class})")
            for file_name in [f"capex_defined_per_arc_{size_class}.xlsx",
                              f"gamma_defined_per_arc_pipeline_{size_class}.xlsx"]:
                patch_gamma_matrix_file(PATH_CAPEX / file_name, from_id, to_id, gamma1, gamma2)
            print(f"  {size_class:6s} {from_id} -> {to_id}: gamma1 = {gamma1:,.0f} EUR, "
                  f"gamma2 = {gamma2:,.1f} EUR/(t/h)")


if __name__ == "__main__":
    main()
