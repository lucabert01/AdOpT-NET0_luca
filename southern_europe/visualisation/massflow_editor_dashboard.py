"""
Mass-Flow Range Editor Dashboard

Lets you manually curate, per pipeline arc, which emitters are considered
"behind" that arc and what mass-flow range (min/max, kg/s) the Oeuvray cost
model should be evaluated over - instead of the automatic min/max derivation
in arc_specific_functions.calculate_arc_gammas (own-emission-or-global-min
for the floor, network-wide total for the ceiling).

Workflow per arc:
    1. Click an arc on the map (or a row in the table below).
    2. Tick the emitter(s) you consider upstream of / feeding into this arc.
    3. Either use the auto-suggested range (single emitter: +/-50%; multiple:
       smallest x0.5 for the floor, sum for the ceiling) or type your own.
    4. "Save range" persists the arc's range + reasoning to
       massflow_overrides_per_arc.xlsx (does not touch the gammas yet).
    5. "Recompute gamma" re-runs the real Oeuvray cost model for just this
       arc with the saved range and patches the resulting gamma1/gamma2
       directly into capex_defined_per_arc.xlsx and (if present)
       gamma_defined_per_arc_pipeline.xlsx - the file main_italy.py actually
       reads - leaving every other arc untouched.

Run with: python massflow_editor_dashboard.py, then open http://127.0.0.1:8051
"""

import sys
import os
import math
import datetime
from pathlib import Path

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ==========================================
# 1. PATH SETUP
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent          # .../AdOpT-NET0_luca  (for `adopt_net0`)
CALC_DIR = SCRIPT_DIR.parent / "data_process" / "updated_network"

sys.path.append(str(REPO_ROOT))
sys.path.append(str(CALC_DIR))

from arc_specific_functions import (  # noqa: E402
    load_network_data,
    load_intersection_data,
    get_all_possible_arcs,
    get_node_emission,
    get_pipeline_length,
    calculate_arc_gammas,
    determine_arc_terrain,
    load_massflow_overrides_df,
    save_massflow_override,
    delete_massflow_override,
    patch_gamma_matrix_file,
    suppress_stdout,
    TONNES_TO_KG,
)

DATA_PATH = SCRIPT_DIR.parent / "italy_data"
CAPEX_METRICS_DIR = DATA_PATH / "network_capex_metrics"
OVERRIDES_PATH = CAPEX_METRICS_DIR / "massflow_overrides_per_arc.xlsx"
CAPEX_FILE = CAPEX_METRICS_DIR / "capex_defined_per_arc.xlsx"
GAMMA_PIPELINE_FILE = CAPEX_METRICS_DIR / "gamma_defined_per_arc_pipeline.xlsx"

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# ==========================================
# 2. LOAD NETWORK DATA ONCE AT STARTUP
# ==========================================
print("=" * 80)
print("Loading network data for the mass-flow editor dashboard...")
print("=" * 80)

DATA_DICT = load_network_data(str(DATA_PATH))

POSSIBLE_ARCS = get_all_possible_arcs(DATA_DICT["network_pipeline"])
_pipeline_names = [f"{f}_{t}" for f, t in POSSIBLE_ARCS]
_intersection_file = DATA_PATH / "geographical_feature" / "route_grid_intersections.xlsx"
DATA_DICT["intersection_data"] = load_intersection_data(_intersection_file, _pipeline_names)

ALL_NODES = sorted(set(DATA_DICT["network_pipeline"].index) | set(DATA_DICT["network_pipeline"].columns))

# --- Deduplicated node display table -----------------------------------
# A node_id can be shared by several co-located facility rows (e.g. node 20
# "Piacenza" has both a Waste and a Cement row) - collapse those for display
# and sum their emissions via get_node_emission (already handles this).
_nodes_raw = DATA_DICT["network_nodes"]


def _agg_types(s):
    uniq = sorted(set(str(v) for v in s.dropna()))
    return "+".join(uniq) if uniq else "Unknown"


NODES_DF = _nodes_raw.groupby(_nodes_raw.index).agg(
    node_name=("node_name", "first"),
    longitude=("longitude", "first"),
    latitude=("latitude", "first"),
    node_type=("node_type", _agg_types),
)
NODES_DF.index.name = "node_id"
NODES_DF["is_emitter"] = ~NODES_DF["node_type"].isin(["Transport", "Storage"])
# get_node_emission() actually returns TONNES/year despite the "_kg_" naming
# convention used elsewhere in this codebase - emission_profile_emitters.xlsx
# is in t CO2/h (see arc_specific_functions.TONNES_TO_KG for the full story).
NODES_DF["emission_tonnes_year"] = [
    get_node_emission(nid, DATA_DICT["network_emission_flux"]) for nid in NODES_DF.index
]
NODES_DF["emission_kg_s"] = NODES_DF["emission_tonnes_year"] * TONNES_TO_KG / SECONDS_PER_YEAR

NODE_NAME = NODES_DF["node_name"].to_dict()

EMITTERS_DF = NODES_DF[NODES_DF["is_emitter"] & (NODES_DF["emission_kg_s"] > 0)].sort_values("node_name")
EMITTER_OPTIONS = [
    {
        "label": f"{row.node_name} (#{nid}) - {row.emission_kg_s:.3f} kg/s [{row.node_type}]",
        "value": int(nid),
    }
    for nid, row in EMITTERS_DF.iterrows()
]

# --- Precompute static per-arc info (length, terrain) -------------------
ARC_INFO = {}
for f, t in POSSIBLE_ARCS:
    ARC_INFO[(f, t)] = {
        "length_km": get_pipeline_length(f, t, DATA_DICT["network_distance"]),
        "terrain": determine_arc_terrain(f, t, DATA_DICT),
    }

print(f"Loaded {len(NODES_DF)} nodes ({len(EMITTERS_DF)} emitters), {len(POSSIBLE_ARCS)} arcs.")
print("=" * 80)


# ==========================================
# 3. HELPERS
# ==========================================
def compute_suggested_range(selected_emitter_ids):
    """User's heuristic: single emitter -> +/-50%; multiple -> smallest*0.5
    for the floor, sum for the ceiling. Returns (min_kg_s, max_kg_s) or
    (None, None) if nothing is selected."""
    values = [
        NODES_DF.loc[nid, "emission_kg_s"]
        for nid in selected_emitter_ids
        if nid in NODES_DF.index
    ]
    if not values:
        return None, None
    if len(values) == 1:
        v = values[0]
        return round(v * 0.5, 4), round(v * 1.5, 4)
    return round(min(values) * 0.5, 4), round(sum(values), 4)


def kg_s_to_t_h(v):
    return None if v is None else v * 3600 / 1000


def read_gamma_cell(path, from_node, to_node):
    """Best-effort read of the current gamma1/gamma2 for one arc from an
    existing gamma matrix workbook. Returns (gamma1, gamma2) or (None, None)."""
    if not Path(path).exists():
        return None, None
    try:
        sheets = pd.read_excel(path, sheet_name=["gamma1", "gamma2"], index_col=0)
        g1_sheet = sheets["gamma1"]
        g2_sheet = sheets["gamma2"]
        g1_sheet.columns = g1_sheet.columns.astype(int)
        g2_sheet.columns = g2_sheet.columns.astype(int)
        if from_node in g1_sheet.index and to_node in g1_sheet.columns:
            return float(g1_sheet.loc[from_node, to_node]), float(g2_sheet.loc[from_node, to_node])
    except Exception as e:
        print(f"Warning: could not read gamma cell for {from_node}->{to_node}: {e}")
    return None, None


def arc_status(overrides_df, from_node, to_node):
    """'default' | 'saved' | 'computed'"""
    if overrides_df.empty:
        return "default"
    row = overrides_df[(overrides_df["from_node"] == from_node) & (overrides_df["to_node"] == to_node)]
    if row.empty:
        return "default"
    if pd.notna(row.iloc[0].get("computed_gamma1")):
        return "computed"
    return "saved"


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


STATUS_COLOR = {
    "default": "rgba(149, 165, 166, 0.55)",
    "saved": "rgba(230, 126, 34, 0.9)",
    "computed": "rgba(39, 174, 96, 0.9)",
}
STATUS_SOLID = {"default": "#95a5a6", "saved": "#e67e22", "computed": "#27ae60"}


def generate_map_figure(selected_arc=None):
    overrides_df = load_massflow_overrides_df(OVERRIDES_PATH)
    fig = go.Figure()

    arrow_lons, arrow_lats, arrow_angles, arrow_hovers, arrow_colors, arrow_customdata = [], [], [], [], [], []
    hit_lons, hit_lats, hit_hovers, hit_customdata = [], [], [], []

    for f, t in POSSIBLE_ARCS:
        if f not in NODES_DF.index or t not in NODES_DF.index:
            continue
        node_a, node_b = NODES_DF.loc[f], NODES_DF.loc[t]
        status = arc_status(overrides_df, f, t)
        is_selected = selected_arc is not None and selected_arc == (f, t)
        color = "#2c3e50" if is_selected else STATUS_COLOR[status]
        width = 4.5 if is_selected else 2.2

        info = ARC_INFO.get((f, t), {})
        hover_txt = (
            f"<b>{node_a.node_name}</b> (#{f}) &rarr; <b>{node_b.node_name}</b> (#{t})<br>"
            f"Length: {info.get('length_km')} km | Terrain: {info.get('terrain')}<br>"
            f"Status: {status}"
        )

        # Thin line: purely visual, no hover/click of its own (a 2-3px wide
        # line is a very small, fiddly hit target on a geo map).
        fig.add_trace(go.Scattergeo(
            lon=[node_a.longitude, node_b.longitude],
            lat=[node_a.latitude, node_b.latitude],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
        ))

        # Fat, near-invisible line drawn on top of it: this is the actual
        # click/hover target, much easier to hit with the cursor. Collected
        # into a single batched trace below (one trace per arc would be
        # visually identical but far slower to add/patch).
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
        arrow_colors.append("#2c3e50" if is_selected else STATUS_SOLID[status])
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

    node_colors = NODES_DF["is_emitter"].map({True: "#e74c3c", False: "#8e44ad"})
    node_symbols = NODES_DF["is_emitter"].map({True: "circle", False: "square"})
    node_sizes = NODES_DF["emission_kg_s"].apply(lambda v: max(8, min(28, 8 + (v ** 0.5) * 10)))
    node_hover = [
        f"{row.node_name} (#{nid})<br>Type: {row.node_type}<br>Flow: {row.emission_kg_s:.3f} kg/s"
        for nid, row in NODES_DF.iterrows()
    ]

    fig.add_trace(go.Scattergeo(
        lon=NODES_DF["longitude"], lat=NODES_DF["latitude"],
        mode="markers+text",
        text=NODES_DF.index.astype(str),
        textposition="top right",
        marker=dict(size=node_sizes, color=node_colors, symbol=node_symbols,
                    line=dict(width=1.2, color="#ffffff")),
        hoverinfo="text", hovertext=node_hover, name="Nodes",
    ))

    fig.update_layout(
        geo=dict(scope="europe", center=dict(lon=12.6, lat=44.8), projection_scale=6.5,
                 showland=True, landcolor="#f9f9f9", countrycolor="#bdc3c7"),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        # Every click/save/recompute rebuilds this figure from scratch; without
        # a fixed uirevision Plotly treats each rebuild as a brand new figure
        # and resets pan/zoom, which is what made arc selection feel jumpy.
        uirevision="massflow-map",
        hovermode="closest",
        clickmode="event",
    )
    return fig


# ==========================================
# 5. APP LAYOUT
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Mass-Flow Range Editor"

LEGEND = dbc.Row([
    dbc.Col(html.Span("● ", style={"color": "#e74c3c"}), width="auto"),
    dbc.Col("Emitter", width="auto", className="me-3"),
    dbc.Col(html.Span("■ ", style={"color": "#8e44ad"}), width="auto"),
    dbc.Col("Transport/Storage", width="auto", className="me-3"),
    dbc.Col(html.Span("— ", style={"color": "#95a5a6"}), width="auto"),
    dbc.Col("Default range", width="auto", className="me-3"),
    dbc.Col(html.Span("— ", style={"color": "#e67e22"}), width="auto"),
    dbc.Col("Saved (not recomputed)", width="auto", className="me-3"),
    dbc.Col(html.Span("— ", style={"color": "#27ae60"}), width="auto"),
    dbc.Col("Recomputed", width="auto"),
], className="small text-muted mb-2")

app.layout = dbc.Container([
    dcc.Store(id="selected-arc-store", data=None),
    dcc.Store(id="data-version", data=0),

    dbc.Row([dbc.Col(html.H3("CO2 Pipeline Mass-Flow Range Editor", className="text-center my-3"), width=12)]),
    dbc.Row([dbc.Col(LEGEND, width=12)]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="network-map", style={"height": "78vh"}),
        ], width=7),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Selected arc", className="card-title"),
                    html.Div(id="arc-info-text", className="mb-1 fw-bold text-primary"),
                    html.Div(id="from-node-info-text", className="mb-2 text-muted small"),

                    html.Hr(),
                    html.H6("Contributing emitters (tick what's 'behind' this arc)"),
                    html.Div(
                        dcc.Checklist(
                            id="emitter-checklist",
                            options=EMITTER_OPTIONS,
                            value=[],
                            labelStyle={"display": "block"},
                            inputStyle={"margin-right": "6px"},
                        ),
                        style={"maxHeight": "22vh", "overflowY": "auto",
                               "border": "1px solid #dee2e6", "borderRadius": "6px", "padding": "8px"},
                    ),
                    html.Div(id="suggested-range-text", className="small text-muted mt-2"),
                    dbc.Button("Use suggestion", id="use-suggestion-btn", size="sm",
                               color="secondary", outline=True, className="mt-1"),

                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Min (kg/s)"),
                            dbc.Input(id="min-input", type="number", step=0.001),
                        ], width=6),
                        dbc.Col([
                            html.Label("Max (kg/s)"),
                            dbc.Input(id="max-input", type="number", step=0.001),
                        ], width=6),
                    ]),
                    html.Div(id="t-per-h-text", className="small text-muted mt-1"),

                    html.Div([
                        html.Label("Note", className="mt-2"),
                        dbc.Textarea(id="note-input", placeholder="Why these emitters / this range?", rows=2),
                    ]),

                    dbc.Row([
                        dbc.Col(dbc.Button("Save range", id="save-btn", color="primary",
                                            className="w-100 mt-2"), width=6),
                        dbc.Col(dbc.Button("Delete override", id="delete-btn", color="outline-danger",
                                            className="w-100 mt-2"), width=6),
                    ]),
                    html.Div(id="save-status-text", className="small mt-1"),

                    dbc.Button("Recompute gamma for this arc", id="recompute-btn", color="success",
                               className="w-100 mt-3"),
                    dcc.Loading(html.Div(id="recompute-status-text", className="small mt-1"), type="dot"),
                ])
            ], style={"height": "78vh", "overflowY": "auto"}),
        ], width=5),
    ]),

    html.Hr(),
    html.H5("Saved overrides"),
    dash_table.DataTable(
        id="overrides-table",
        columns=[
            {"name": "From", "id": "from_name"}, {"name": "from_node", "id": "from_node"},
            {"name": "To", "id": "to_name"}, {"name": "to_node", "id": "to_node"},
            {"name": "Min (kg/s)", "id": "massflow_min_kg_s"},
            {"name": "Max (kg/s)", "id": "massflow_max_kg_s"},
            {"name": "Contributing emitters", "id": "contributing_emitters_names"},
            {"name": "gamma1", "id": "computed_gamma1"},
            {"name": "gamma2", "id": "computed_gamma2"},
            {"name": "Updated", "id": "last_updated"},
            {"name": "Note", "id": "note"},
        ],
        row_selectable="single",
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 12, "padding": "4px"},
        page_size=10,
    ),
], fluid=True)


# ==========================================
# 6. CALLBACKS
# ==========================================
@app.callback(Output("network-map", "figure"), Input("data-version", "data"),
              Input("selected-arc-store", "data"))
def refresh_map(_version, selected_arc):
    sel = tuple(selected_arc) if selected_arc else None
    return generate_map_figure(sel)


@app.callback(Output("overrides-table", "data"), Input("data-version", "data"))
def refresh_table(_version):
    df = load_massflow_overrides_df(OVERRIDES_PATH)
    return df.to_dict("records")


@app.callback(Output("selected-arc-store", "data"), Input("network-map", "clickData"),
              prevent_initial_call=True)
def select_from_map(click_data):
    if not click_data:
        return dash.no_update
    point = click_data["points"][0]
    if "customdata" in point and isinstance(point["customdata"], list):
        return list(point["customdata"])
    return dash.no_update


@app.callback(Output("selected-arc-store", "data", allow_duplicate=True),
              Input("overrides-table", "selected_rows"),
              State("overrides-table", "data"), prevent_initial_call=True)
def select_from_table(selected_rows, table_data):
    if not selected_rows or not table_data:
        return dash.no_update
    row = table_data[selected_rows[0]]
    return [int(row["from_node"]), int(row["to_node"])]


@app.callback(
    Output("arc-info-text", "children"),
    Output("from-node-info-text", "children"),
    Output("emitter-checklist", "value"),
    Output("min-input", "value"),
    Output("max-input", "value"),
    Output("note-input", "value"),
    Input("selected-arc-store", "data"),
)
def populate_editor(selected_arc):
    if not selected_arc:
        return "No arc selected - click one on the map or a row below.", "", [], None, None, ""

    from_node, to_node = int(selected_arc[0]), int(selected_arc[1])
    info = ARC_INFO.get((from_node, to_node), {})
    from_name = NODE_NAME.get(from_node, f"Node {from_node}")
    to_name = NODE_NAME.get(to_node, f"Node {to_node}")
    arc_text = f"{from_name} (#{from_node}) → {to_name} (#{to_node})  |  {info.get('length_km')} km, {info.get('terrain')}"

    if from_node in NODES_DF.index and NODES_DF.loc[from_node, "is_emitter"]:
        own_flow = NODES_DF.loc[from_node, "emission_kg_s"]
        from_info = f"From-node own flow: {own_flow:.3f} kg/s ({kg_s_to_t_h(own_flow):.3f} t/h)"
    else:
        from_info = "From-node is Transport/Storage - no direct emission of its own."

    overrides_df = load_massflow_overrides_df(OVERRIDES_PATH)
    row = overrides_df[(overrides_df["from_node"] == from_node) & (overrides_df["to_node"] == to_node)]

    if not row.empty:
        r = row.iloc[0]
        checked = [int(x) for x in str(r["contributing_emitters"]).split(",") if x.strip().isdigit()]
        min_val = float(r["massflow_min_kg_s"]) if pd.notna(r["massflow_min_kg_s"]) else None
        max_val = float(r["massflow_max_kg_s"]) if pd.notna(r["massflow_max_kg_s"]) else None
        note_val = r["note"] if pd.notna(r["note"]) else ""
        if pd.notna(r.get("computed_gamma1")):
            arc_text += f"  |  γ1={r['computed_gamma1']:,.0f} EUR, γ2={r['computed_gamma2']:,.1f} EUR/(t/h)"
    else:
        checked = [from_node] if from_node in EMITTERS_DF.index else []
        min_val, max_val = compute_suggested_range(checked)
        note_val = ""

    return arc_text, from_info, checked, min_val, max_val, note_val


@app.callback(Output("suggested-range-text", "children"), Input("emitter-checklist", "value"))
def show_suggestion(selected_emitters):
    min_v, max_v = compute_suggested_range(selected_emitters or [])
    if min_v is None:
        return "Tick at least one emitter to get a suggested range."
    n = len(selected_emitters)
    rule = "±50% of the single emitter" if n == 1 else "smallest×0.5 .. sum of selected"
    return f"Suggested ({rule}): {min_v:.3f} - {max_v:.3f} kg/s ({kg_s_to_t_h(min_v):.3f} - {kg_s_to_t_h(max_v):.3f} t/h)"


@app.callback(
    Output("min-input", "value", allow_duplicate=True),
    Output("max-input", "value", allow_duplicate=True),
    Input("use-suggestion-btn", "n_clicks"),
    State("emitter-checklist", "value"),
    prevent_initial_call=True,
)
def apply_suggestion(_n, selected_emitters):
    min_v, max_v = compute_suggested_range(selected_emitters or [])
    return min_v, max_v


@app.callback(Output("t-per-h-text", "children"), Input("min-input", "value"), Input("max-input", "value"))
def show_t_per_h(min_v, max_v):
    if min_v is None or max_v is None:
        return ""
    return f"= {kg_s_to_t_h(min_v):.3f} - {kg_s_to_t_h(max_v):.3f} t/h"


@app.callback(
    Output("save-status-text", "children"),
    Output("data-version", "data", allow_duplicate=True),
    Input("save-btn", "n_clicks"),
    State("selected-arc-store", "data"), State("min-input", "value"), State("max-input", "value"),
    State("emitter-checklist", "value"), State("note-input", "value"), State("data-version", "data"),
    prevent_initial_call=True,
)
def save_range(_n, selected_arc, min_v, max_v, checked, note, version):
    if not selected_arc:
        return "⚠️ Select an arc first.", dash.no_update
    if min_v is None or max_v is None:
        return "⚠️ Enter both a min and a max mass flow.", dash.no_update
    if min_v > max_v:
        return "⚠️ Min cannot be greater than max.", dash.no_update

    from_node, to_node = int(selected_arc[0]), int(selected_arc[1])
    names = [NODE_NAME.get(n, str(n)) for n in (checked or [])]
    save_massflow_override(
        OVERRIDES_PATH, from_node, to_node,
        NODE_NAME.get(from_node, str(from_node)), NODE_NAME.get(to_node, str(to_node)),
        min_v, max_v, contributing_emitters=checked or [], contributing_emitters_names=names,
        note=note or "",
    )
    return f"✅ Saved range for {from_node} → {to_node}.", (version or 0) + 1


@app.callback(
    Output("save-status-text", "children", allow_duplicate=True),
    Output("data-version", "data", allow_duplicate=True),
    Input("delete-btn", "n_clicks"), State("selected-arc-store", "data"), State("data-version", "data"),
    prevent_initial_call=True,
)
def delete_override(_n, selected_arc, version):
    if not selected_arc:
        return "⚠️ Select an arc first.", dash.no_update
    from_node, to_node = int(selected_arc[0]), int(selected_arc[1])
    delete_massflow_override(OVERRIDES_PATH, from_node, to_node)
    return f"🗑️ Override removed for {from_node} → {to_node} (reverts to automatic range).", (version or 0) + 1


@app.callback(
    Output("recompute-status-text", "children"),
    Output("data-version", "data", allow_duplicate=True),
    Input("recompute-btn", "n_clicks"),
    State("selected-arc-store", "data"), State("min-input", "value"), State("max-input", "value"),
    State("emitter-checklist", "value"), State("note-input", "value"), State("data-version", "data"),
    prevent_initial_call=True,
)
def recompute_gamma(_n, selected_arc, min_v, max_v, checked, note, version):
    if not selected_arc:
        return "⚠️ Select an arc first.", dash.no_update
    if min_v is None or max_v is None:
        return "⚠️ Enter both a min and a max mass flow before recomputing.", dash.no_update
    if min_v > max_v:
        return "⚠️ Min cannot be greater than max.", dash.no_update
    if not CAPEX_FILE.exists():
        return (f"⚠️ {CAPEX_FILE.name} does not exist yet - run pipeline_capex_per_arc_calculator.py "
                f"once first to establish baseline gammas for all arcs, then use this button to "
                f"touch up individual arcs. (Recomputing into a missing file would leave every "
                f"other arc at 0.)"), dash.no_update

    from_node, to_node = int(selected_arc[0]), int(selected_arc[1])
    terrain = ARC_INFO.get((from_node, to_node), {}).get("terrain", "Onshore")

    try:
        with suppress_stdout():
            gamma1, gamma2 = calculate_arc_gammas(
                from_node, to_node, DATA_DICT, terrain=terrain,
                massflow_min_kg_s_override=min_v, massflow_max_kg_s_override=max_v,
            )
    except Exception as e:
        return f"❌ Cost model failed: {e}", dash.no_update

    if gamma1 == 0 and gamma2 == 0:
        return "❌ Cost model returned 0/0 (failed) - check terminal log for details.", dash.no_update

    patch_gamma_matrix_file(CAPEX_FILE, from_node, to_node, gamma1, gamma2, all_nodes=ALL_NODES)
    patched_live = False
    if GAMMA_PIPELINE_FILE.exists():
        patch_gamma_matrix_file(GAMMA_PIPELINE_FILE, from_node, to_node, gamma1, gamma2, all_nodes=ALL_NODES)
        patched_live = True

    names = [NODE_NAME.get(n, str(n)) for n in (checked or [])]
    save_massflow_override(
        OVERRIDES_PATH, from_node, to_node,
        NODE_NAME.get(from_node, str(from_node)), NODE_NAME.get(to_node, str(to_node)),
        min_v, max_v, contributing_emitters=checked or [], contributing_emitters_names=names,
        note=note or "", computed_gamma1=gamma1, computed_gamma2=gamma2,
    )

    msg = f"✅ γ1={gamma1:,.0f} EUR, γ2={gamma2:,.1f} EUR/(t/h). Patched capex_defined_per_arc.xlsx"
    msg += " and gamma_defined_per_arc_pipeline.xlsx." if patched_live else " (gamma_defined_per_arc_pipeline.xlsx not found - run main_italy's copy step, or rename this file, to pick it up)."
    return msg, (version or 0) + 1


if __name__ == "__main__":
    app.run(debug=True, port=8051)
