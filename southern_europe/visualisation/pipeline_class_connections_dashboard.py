"""
Pipeline Size-Class Connections Dashboard

The three CO2 pipeline size classes (CO2_Pipeline_small/medium/large - see
main_italy.py's pipeline_size_class_max_capacity_t_h and
data_process/updated_network/pipeline_capex_per_arc_calculator.py::
SIZE_CLASS_MASSFLOW_RANGES_KG_S) all reuse the exact same physical pipeline
connectivity by default: a 'large' pipeline (calibrated for ~133-470 kg/s)
is currently just as buildable on an arc that only ever carries a small
emitter's flow as a 'small' one is. This dashboard lets you remove specific
arcs from an individual size class instead of every class sharing the same
flat connectivity.

Workflow:
    1. Pick a size class in the sidebar (or just click cells in the table -
       see below).
    2. Click a "Small"/"Medium"/"Large" cell in the arc table to toggle that
       arc on/off for that specific class - every cell saves immediately.
       Alternatively, pick a class with the radio buttons and click an arc
       directly on the map to toggle it for the selected class.
    3. main_italy.py picks up your edits automatically on the next run (via
       defined_functions.load_pipeline_class_connection_matrix reading
       pipeline_size_class_connections.xlsx) - no code changes needed.

An arc not touched here stays enabled for every class (identical to the
flat, pre-size-class behaviour), so this dashboard only ever narrows
connectivity, never adds arcs beyond what node_metrics_150.xlsx's 'pipeline'
sheet already allows.

Run with: python pipeline_class_connections_dashboard.py, then open
http://127.0.0.1:8052
"""

import math
from pathlib import Path

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ==========================================
# 1. PATH SETUP
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "italy_data"
CAPEX_METRICS_DIR = DATA_PATH / "network_capex_metrics"

# Must match main_italy.py's node_metrics_suffix (default 150) - this is the
# exact 'pipeline' connectivity/distance matrix main_italy.py reads into
# network_pipeline before applying the per-class overrides this dashboard
# curates.
NODE_METRICS_PATH = DATA_PATH / "geographical_feature" / "node_metrics_150.xlsx"
OVERRIDES_PATH = CAPEX_METRICS_DIR / "pipeline_size_class_connections.xlsx"

if not NODE_METRICS_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Could not locate {NODE_METRICS_PATH}")

SIZE_CLASSES = ["small", "medium", "large"]

# Reference only, kept in sync with pipeline_capex_per_arc_calculator.py::
# SIZE_CLASS_MASSFLOW_RANGES_KG_S and main_italy.py::
# pipeline_size_class_max_capacity_t_h - shown in the sidebar/table to help
# judge which arcs make sense for which class.
SIZE_CLASS_RANGES_KG_S = {"small": (3.1, 29.0), "medium": (29.0, 133.0), "large": (133.0, 470.0)}
SIZE_CLASS_MAX_CAPACITY_T_H = {"small": 104.4, "medium": 478.8, "large": 1692.0}

SECONDS_PER_YEAR = 365.25 * 24 * 3600
TONNES_TO_KG = 1000  # 'annual_flux' is in tonnes CO2/year despite the column name

# ==========================================
# 2. LOAD NETWORK DATA ONCE AT STARTUP
# ==========================================
print("=" * 80)
print("Loading network data for the pipeline class connections dashboard...")
print("=" * 80)

_nodes_raw = pd.read_excel(NODE_METRICS_PATH, sheet_name="nodes", index_col=0)
BASE_PIPELINE = pd.read_excel(NODE_METRICS_PATH, sheet_name="pipeline", index_col=0)
BASE_PIPELINE.columns = BASE_PIPELINE.columns.astype(int)

# A node_id can be shared by several co-located facility rows (e.g. one node
# has both a Waste and a Cement row) - collapse those for display and sum
# their emissions, same convention as massflow_editor_dashboard.py.
_nodes_raw["node_type"] = _nodes_raw["node_type"].astype(str).str.strip()
NODES_RAW = _nodes_raw.groupby(_nodes_raw.index).agg(
    node_name=("node_name", "first"),
    longitude=("longitude", "first"),
    latitude=("latitude", "first"),
    node_type=("node_type", lambda s: "+".join(sorted(set(s)))),
    annual_flux=("annual_flux", "sum"),
)
NODES_RAW.index.name = "node_id"
NODES_RAW["emission_kg_s"] = NODES_RAW["annual_flux"].fillna(0.0) * TONNES_TO_KG / SECONDS_PER_YEAR
NODE_NAME = NODES_RAW["node_name"].to_dict()

ALL_NODE_IDS = sorted(set(BASE_PIPELINE.index) | set(BASE_PIPELINE.columns))

# All directed arcs that physically exist (value > 0). The underlying matrix
# is NOT symmetric - e.g. 1->3 can be connected while 3->1 is not - so each
# direction is tracked and toggled independently.
POSSIBLE_ARCS = [
    (f, t) for f in BASE_PIPELINE.index for t in BASE_PIPELINE.columns
    if pd.notna(BASE_PIPELINE.loc[f, t]) and BASE_PIPELINE.loc[f, t] > 0
]

print(f"Loaded {len(NODES_RAW)} nodes, {len(POSSIBLE_ARCS)} directed pipeline arcs.")
print("=" * 80)


# ==========================================
# 3. OVERRIDE FILE HELPERS
# ==========================================
def load_class_mask(size_class):
    """1 = enabled, 0 = disabled, for every (from,to) node_id pair. Defaults
    to all-enabled if the override file/sheet doesn't exist yet - matches
    main_italy.py / load_pipeline_class_connection_matrix's default."""
    default = pd.DataFrame(1, index=ALL_NODE_IDS, columns=ALL_NODE_IDS)
    if not OVERRIDES_PATH.exists():
        return default
    try:
        xl = pd.ExcelFile(OVERRIDES_PATH)
        if size_class not in xl.sheet_names:
            return default
        df = pd.read_excel(OVERRIDES_PATH, sheet_name=size_class, index_col=0)
        df.index = df.index.astype(int)
        df.columns = df.columns.astype(int)
    except Exception as e:
        print(f"Warning: could not load '{size_class}' override sheet: {e}")
        return default
    return df.reindex(index=ALL_NODE_IDS, columns=ALL_NODE_IDS, fill_value=1).fillna(1).astype(int)


def save_class_mask(size_class, mask_df):
    """Upsert one class's mask sheet into pipeline_size_class_connections.xlsx,
    leaving every other class's sheet untouched."""
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_sheets = {}
    if OVERRIDES_PATH.exists():
        all_sheets = pd.read_excel(OVERRIDES_PATH, sheet_name=None, index_col=0)
    all_sheets[size_class] = mask_df
    with pd.ExcelWriter(OVERRIDES_PATH, engine="openpyxl") as writer:
        for name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=name, index=True)


def toggle_arc(size_class, from_node, to_node):
    mask_df = load_class_mask(size_class)
    current = int(mask_df.loc[from_node, to_node])
    new_val = 0 if current == 1 else 1
    mask_df.loc[from_node, to_node] = new_val
    save_class_mask(size_class, mask_df)
    return new_val


def set_all_arcs(size_class, value):
    mask_df = load_class_mask(size_class)
    for f, t in POSSIBLE_ARCS:
        mask_df.loc[f, t] = value
    save_class_mask(size_class, mask_df)


def build_arcs_table_data():
    masks = {sc: load_class_mask(sc) for sc in SIZE_CLASSES}
    rows = []
    for f, t in POSSIBLE_ARCS:
        row = {
            "from_name": NODE_NAME.get(f, f"Node {f}"),
            "from_node": f,
            "to_name": NODE_NAME.get(t, f"Node {t}"),
            "to_node": t,
            "distance_km": round(float(BASE_PIPELINE.loc[f, t]), 2),
            "from_flow_kg_s": round(float(NODES_RAW.loc[f, "emission_kg_s"]), 3) if f in NODES_RAW.index else None,
        }
        for sc in SIZE_CLASSES:
            row[sc] = "✅" if bool(masks[sc].loc[f, t]) else "❌"
        rows.append(row)
    return rows


# ==========================================
# 4. MAP RENDERING
# ==========================================
def _bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _arrow_tip(lat1, lon1, lat2, lon2, offset_frac=0.2):
    tip_lat = lat2 + offset_frac * (lat1 - lat2)
    tip_lon = lon2 + offset_frac * (lon1 - lon2)
    return tip_lat, tip_lon


ENABLED_COLOR = "rgba(39, 174, 96, 0.85)"
DISABLED_COLOR = "rgba(192, 57, 43, 0.55)"
ENABLED_SOLID = "#27ae60"
DISABLED_SOLID = "#c0392b"


def generate_map_figure(size_class):
    mask_df = load_class_mask(size_class)
    fig = go.Figure()

    arrow_lons, arrow_lats, arrow_angles, arrow_hovers, arrow_colors, arrow_customdata = [], [], [], [], [], []
    hit_lons, hit_lats, hit_hovers, hit_customdata = [], [], [], []

    for f, t in POSSIBLE_ARCS:
        if f not in NODES_RAW.index or t not in NODES_RAW.index:
            continue
        node_a, node_b = NODES_RAW.loc[f], NODES_RAW.loc[t]
        enabled = bool(mask_df.loc[f, t])
        color = ENABLED_COLOR if enabled else DISABLED_COLOR
        solid = ENABLED_SOLID if enabled else DISABLED_SOLID

        hover_txt = (
            f"<b>{node_a.node_name}</b> (#{f}) &rarr; <b>{node_b.node_name}</b> (#{t})<br>"
            f"Distance: {BASE_PIPELINE.loc[f, t]:.2f} km<br>"
            f"'{size_class}' status: {'ENABLED' if enabled else 'DISABLED'} (click to toggle)"
        )

        fig.add_trace(go.Scattergeo(
            lon=[node_a.longitude, node_b.longitude],
            lat=[node_a.latitude, node_b.latitude],
            mode="lines",
            line=dict(width=2.5, color=color),
            hoverinfo="skip",
        ))

        hit_lons += [node_a.longitude, node_b.longitude, None]
        hit_lats += [node_a.latitude, node_b.latitude, None]
        hit_hovers += [hover_txt, hover_txt, hover_txt]
        hit_customdata += [[int(f), int(t)], [int(f), int(t)], [int(f), int(t)]]

        tip_lat, tip_lon = _arrow_tip(node_a.latitude, node_a.longitude, node_b.latitude, node_b.longitude)
        bearing = _bearing(node_a.latitude, node_a.longitude, node_b.latitude, node_b.longitude)
        arrow_lons.append(tip_lon)
        arrow_lats.append(tip_lat)
        arrow_angles.append(bearing)
        arrow_hovers.append(hover_txt)
        arrow_colors.append(solid)
        arrow_customdata.append([int(f), int(t)])

    if hit_lons:
        fig.add_trace(go.Scattergeo(
            lon=hit_lons, lat=hit_lats, mode="lines",
            line=dict(width=14, color="rgba(0,0,0,0.001)"),
            hoverinfo="text", hovertext=hit_hovers, customdata=hit_customdata,
            name="Arc (click target)",
        ))

    if arrow_lons:
        fig.add_trace(go.Scattergeo(
            lon=arrow_lons, lat=arrow_lats, mode="markers",
            marker=dict(symbol="arrow", size=11, color=arrow_colors, angle=arrow_angles, line=dict(width=0)),
            hoverinfo="text", hovertext=arrow_hovers, customdata=arrow_customdata, name="Direction",
        ))

    node_hover = [
        f"{row.node_name} (#{nid})<br>Type: {row.node_type}<br>Flow: {row.emission_kg_s:.3f} kg/s"
        for nid, row in NODES_RAW.iterrows()
    ]
    fig.add_trace(go.Scattergeo(
        lon=NODES_RAW["longitude"], lat=NODES_RAW["latitude"],
        mode="markers+text",
        text=NODES_RAW.index.astype(str),
        textposition="top right",
        marker=dict(size=10, color="#2c3e50", line=dict(width=1.2, color="#ffffff")),
        hoverinfo="text", hovertext=node_hover, name="Nodes",
    ))

    fig.update_layout(
        geo=dict(scope="europe", center=dict(lon=12.6, lat=44.8), projection_scale=6.5,
                 showland=True, landcolor="#f9f9f9", countrycolor="#bdc3c7"),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        uirevision="pipeline-class-map",
        hovermode="closest",
        clickmode="event",
    )
    return fig


# ==========================================
# 5. APP LAYOUT
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Pipeline Size-Class Connections"

CLASS_OPTIONS = [
    {
        "label": f" {sc.capitalize()}  ({SIZE_CLASS_RANGES_KG_S[sc][0]:.1f}-{SIZE_CLASS_RANGES_KG_S[sc][1]:.1f} kg/s, "
                  f"≤{SIZE_CLASS_MAX_CAPACITY_T_H[sc]:,.0f} t/h)",
        "value": sc,
    }
    for sc in SIZE_CLASSES
]

LEGEND = dbc.Row([
    dbc.Col(html.Span("─ ", style={"color": ENABLED_SOLID, "fontWeight": "bold"}), width="auto"),
    dbc.Col("Enabled for selected class", width="auto", className="me-3"),
    dbc.Col(html.Span("─ ", style={"color": DISABLED_SOLID, "fontWeight": "bold"}), width="auto"),
    dbc.Col("Disabled for selected class", width="auto"),
], className="small text-muted mb-2")

app.layout = dbc.Container([
    dcc.Store(id="data-version", data=0),

    dbc.Row([dbc.Col(html.H3("CO2 Pipeline Size-Class Connections", className="text-center my-3"), width=12)]),
    dbc.Row([dbc.Col(LEGEND, width=12)]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="network-map", style={"height": "70vh"}),
        ], width=7),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Size class (for map click + bulk actions)", className="card-title"),
                    dcc.RadioItems(
                        id="class-selector",
                        options=CLASS_OPTIONS,
                        value="small",
                        labelStyle={"display": "block", "margin-bottom": "6px"},
                        className="mb-2",
                    ),
                    html.Div(id="class-status-text", className="small text-muted mb-3"),

                    dbc.Row([
                        dbc.Col(dbc.Button("Enable all for this class", id="enable-all-btn",
                                            color="success", outline=True, size="sm", className="w-100"), width=6),
                        dbc.Col(dbc.Button("Disable all for this class", id="disable-all-btn",
                                            color="danger", outline=True, size="sm", className="w-100"), width=6),
                    ], className="mb-2"),
                    html.Div(id="bulk-status-text", className="small mt-1"),

                    html.Hr(),
                    html.P(
                        "Tip: click a ✅/❌ cell in the Small/Medium/Large columns below to toggle "
                        "that arc for that specific class directly, regardless of which class is selected "
                        "above. Click an arc on the map to toggle it for the selected class.",
                        className="small text-muted",
                    ),
                ])
            ], style={"height": "70vh", "overflowY": "auto"}),
        ], width=5),
    ]),

    html.Hr(),
    html.H5("All pipeline arcs"),
    dash_table.DataTable(
        id="arcs-table",
        columns=[
            {"name": "From", "id": "from_name"}, {"name": "id", "id": "from_node"},
            {"name": "To", "id": "to_name"}, {"name": "id", "id": "to_node"},
            {"name": "Distance (km)", "id": "distance_km"},
            {"name": "From-node flow (kg/s)", "id": "from_flow_kg_s"},
            {"name": "Small", "id": "small"},
            {"name": "Medium", "id": "medium"},
            {"name": "Large", "id": "large"},
        ],
        data=build_arcs_table_data(),
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 12, "padding": "4px", "textAlign": "center"},
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"} for c in ["from_name", "to_name"]
        ],
        style_data_conditional=[
            {"if": {"column_id": sc}, "cursor": "pointer"} for sc in SIZE_CLASSES
        ],
        sort_action="native",
        filter_action="native",
        page_size=15,
    ),
], fluid=True)


# ==========================================
# 6. CALLBACKS
# ==========================================
@app.callback(Output("network-map", "figure"),
              Input("class-selector", "value"), Input("data-version", "data"))
def refresh_map(size_class, _version):
    return generate_map_figure(size_class)


@app.callback(Output("arcs-table", "data"), Input("data-version", "data"))
def refresh_table(_version):
    return build_arcs_table_data()


@app.callback(Output("class-status-text", "children"), Input("class-selector", "value"), Input("data-version", "data"))
def show_class_status(size_class, _version):
    mask_df = load_class_mask(size_class)
    enabled = sum(1 for f, t in POSSIBLE_ARCS if bool(mask_df.loc[f, t]))
    total = len(POSSIBLE_ARCS)
    return f"'{size_class}': {enabled}/{total} arcs enabled."


@app.callback(
    Output("data-version", "data", allow_duplicate=True),
    Input("network-map", "clickData"),
    State("class-selector", "value"),
    State("data-version", "data"),
    prevent_initial_call=True,
)
def toggle_from_map(click_data, size_class, version):
    if not click_data:
        return dash.no_update
    point = click_data["points"][0]
    if "customdata" not in point or not isinstance(point["customdata"], list):
        return dash.no_update
    from_node, to_node = point["customdata"]
    toggle_arc(size_class, int(from_node), int(to_node))
    return (version or 0) + 1


@app.callback(
    Output("data-version", "data", allow_duplicate=True),
    Input("arcs-table", "active_cell"),
    State("arcs-table", "data"),
    State("data-version", "data"),
    prevent_initial_call=True,
)
def toggle_from_table(active_cell, table_data, version):
    if not active_cell or active_cell["column_id"] not in SIZE_CLASSES:
        return dash.no_update
    row = table_data[active_cell["row"]]
    size_class = active_cell["column_id"]
    toggle_arc(size_class, int(row["from_node"]), int(row["to_node"]))
    return (version or 0) + 1


@app.callback(
    Output("bulk-status-text", "children"),
    Output("data-version", "data", allow_duplicate=True),
    Input("enable-all-btn", "n_clicks"),
    Input("disable-all-btn", "n_clicks"),
    State("class-selector", "value"),
    State("data-version", "data"),
    prevent_initial_call=True,
)
def bulk_toggle(_enable_clicks, _disable_clicks, size_class, version):
    triggered = dash.ctx.triggered_id
    if triggered == "enable-all-btn":
        set_all_arcs(size_class, 1)
        msg = f"✅ All arcs enabled for '{size_class}'."
    elif triggered == "disable-all-btn":
        set_all_arcs(size_class, 0)
        msg = f"\U0001F5D1️ All arcs disabled for '{size_class}'."
    else:
        return dash.no_update, dash.no_update
    return msg, (version or 0) + 1


if __name__ == "__main__":
    app.run(debug=True, port=8052)
