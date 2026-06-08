import math
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. DYNAMIC PATH HANDLING (GIT-SAFE)
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = SCRIPT_DIR.parent / "italy_data" / "geographical_feature" / "node_metrics.xlsx"

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Could not locate Excel file at: {EXCEL_PATH}")


# ==========================================
# 2. DATA LOADING FUNCTION
# ==========================================
def load_network_data(matrix_type="pipeline"):
    df_nodes = pd.read_excel(EXCEL_PATH, sheet_name="nodes")
    df_matrix = pd.read_excel(EXCEL_PATH, sheet_name=matrix_type, index_col=0)
    df_matrix.columns = df_matrix.columns.astype(int)
    return df_nodes, df_matrix


# ==========================================
# 3. DASH WEB APP CONFIGURATION
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("AdOpT-NET0 Infrastructure Network Modifier",
                        className="text-center my-3 text-dark"), width=12)
    ]),
    dbc.Row([
        # Interactive Map Display Panel
        dbc.Col([
            dcc.Graph(id='network-map', style={'height': '80vh'})
        ], width=8),

        # Sidebar Layer Controls
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # ── Section A: Transport Mode ──────────────────────────
                    html.H5("1. Select Transport Mode", className="card-title text-secondary"),
                    dcc.RadioItems(
                        id='matrix-selector',
                        options=[
                            {'label': ' 🟢 Pipeline Network', 'value': 'pipeline'},
                            {'label': ' 🚚 Truck Network',    'value': 'truck'},
                            {'label': ' 🚂 Railway Network',  'value': 'railway'}
                        ],
                        value='pipeline',
                        labelStyle={'display': 'block', 'margin-bottom': '10px'},
                        className="mb-4 fw-bold"
                    ),
                    html.Hr(),

                    # ── Section B: Modify Existing Connection ──────────────
                    html.H5("2. Modify Existing Connection", className="card-title text-secondary"),
                    html.Small("Click a connection line on the map to select it.",
                               className="text-muted d-block mb-2"),
                    html.Div([
                        html.Label("Selected Element:"),
                        html.P(id='selected-nodes-text', children="None selected",
                               className="fw-bold text-primary")
                    ]),
                    html.Div([
                        html.Label("Current Matrix Weight:"),
                        html.P(id='current-value-text', children="-", className="fw-bold")
                    ]),
                    html.Div([
                        html.Label("New Value:"),
                        dbc.Input(id='new-matrix-value', type='number',
                                  placeholder='Enter updated weight...'),
                    ], className="mb-3"),
                    dbc.Button("Apply & Overwrite Excel", id='update-btn',
                               color="danger", className="w-100 fw-bold", n_clicks=0),
                    html.Div(id='status-message', className="text-center fw-bold mt-2"),

                    html.Hr(),

                    # ── Section C: Add New Connection ─────────────────────
                    html.H5("3. Add New Connection", className="card-title text-secondary"),
                    html.Small("Enter the two node IDs and a weight to create a new edge.",
                               className="text-muted d-block mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("From Node ID:"),
                            dbc.Input(id='new-node-a', type='number',
                                      placeholder='e.g. 1', min=0, step=1),
                        ], width=6),
                        dbc.Col([
                            html.Label("To Node ID:"),
                            dbc.Input(id='new-node-b', type='number',
                                      placeholder='e.g. 5', min=0, step=1),
                        ], width=6),
                    ], className="mb-2"),
                    html.Div([
                        html.Label("Connection Weight:"),
                        dbc.Input(id='new-edge-value', type='number',
                                  placeholder='Enter weight value...'),
                    ], className="mb-3"),
                    dbc.Button("➕ Add Connection", id='add-edge-btn',
                               color="success", className="w-100 fw-bold", n_clicks=0),
                    html.Div(id='add-edge-status', className="text-center fw-bold mt-2"),
                ])
            ], style={'height': '80vh', 'overflowY': 'auto'})
        ], width=4)
    ])
], fluid=True)


# ==========================================
# 4. MAP LAYOUT RENDERING ENGINE
# ==========================================
def _bearing(lat1, lon1, lat2, lon2):
    """Compass bearing in degrees from point A → point B."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _arrow_tip(lat1, lon1, lat2, lon2, offset_frac=0.18):
    """
    Return a point slightly back from the destination along the line,
    so the arrowhead sits ON the line rather than overlapping the node marker.
    offset_frac = fraction of the total segment length to pull back.
    """
    tip_lat = lat2 + offset_frac * (lat1 - lat2)
    tip_lon = lon2 + offset_frac * (lon1 - lon2)
    return tip_lat, tip_lon


def generate_map_figure(nodes, matrix, matrix_name):
    fig = go.Figure()

    color_map = {
        'pipeline': 'rgba(46, 204, 113, 0.9)',
        'truck':    'rgba(230, 126, 34, 0.9)',
        'railway':  'rgba(52, 152, 219, 0.9)'
    }
    # Solid colour for arrowheads (no alpha channel so marker colour matches)
    solid_color_map = {
        'pipeline': '#27ae60',
        'truck':    '#e67e22',
        'railway':  '#2980b9'
    }
    line_color  = color_map.get(matrix_name, 'rgba(149, 165, 166, 0.9)')
    arrow_color = solid_color_map.get(matrix_name, '#95a5a6')

    # Collect arrowhead positions for a single batched marker trace
    arrow_lons, arrow_lats, arrow_angles, arrow_hovers = [], [], [], []

    for idx in matrix.index:
        for col in matrix.columns:
            val = matrix.loc[idx, col]
            if pd.notna(val) and val > 0:
                node_a = nodes[nodes['node_id'] == idx]
                node_b = nodes[nodes['node_id'] == col]

                if not node_a.empty and not node_b.empty:
                    lon_a, lat_a = node_a.iloc[0]['longitude'], node_a.iloc[0]['latitude']
                    lon_b, lat_b = node_b.iloc[0]['longitude'], node_b.iloc[0]['latitude']
                    name_a = node_a.iloc[0]['node_name']
                    name_b = node_b.iloc[0]['node_name']

                    hover_txt = (
                        f"<b>From:</b> {name_a} (Node {idx})<br>"
                        f"<b>To:</b> {name_b} (Node {col})<br>"
                        f"<b>Mode:</b> {matrix_name.upper()}<br>"
                        f"<b>Weight:</b> {val}"
                    )

                    # ── Line ──────────────────────────────────────────────
                    fig.add_trace(go.Scattergeo(
                        lon=[lon_a, lon_b],
                        lat=[lat_a, lat_b],
                        mode='lines',
                        line=dict(width=2.5, color=line_color),
                        hoverinfo='text',
                        hovertext=hover_txt,
                        customdata=[[int(idx), int(col)]],
                    ))

                    # ── Arrowhead data (batched below) ────────────────────
                    tip_lat, tip_lon = _arrow_tip(lat_a, lon_a, lat_b, lon_b)
                    bearing = _bearing(lat_a, lon_a, lat_b, lon_b)
                    arrow_lons.append(tip_lon)
                    arrow_lats.append(tip_lat)
                    arrow_angles.append(bearing)
                    arrow_hovers.append(hover_txt)

    # ── Batch arrowhead markers (one trace, much faster than per-edge traces) ──
    if arrow_lons:
        fig.add_trace(go.Scattergeo(
            lon=arrow_lons,
            lat=arrow_lats,
            mode='markers',
            marker=dict(
                symbol='arrow',          # Plotly built-in arrow symbol
                size=10,
                color=arrow_color,
                angle=arrow_angles,      # each marker rotated to its bearing
                line=dict(width=0),
            ),
            hoverinfo='text',
            hovertext=arrow_hovers,
            name='Direction',
        ))

    # ── Node markers ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scattergeo(
        lon=nodes['longitude'],
        lat=nodes['latitude'],
        mode='markers+text',
        text=nodes['node_id'],
        textposition="top right",
        marker=dict(size=11, color='#2c3e50', line=dict(width=1.5, color='#ffffff')),
        hoverinfo='text',
        hovertext=nodes['node_name'],
        name='Infrastructure Nodes'
    ))

    fig.update_layout(
        geo=dict(
            scope='europe',
            center=dict(lon=12.6, lat=44.8),
            projection_scale=6.5,
            showland=True,
            landcolor="#f9f9f9",
            countrycolor="#bdc3c7"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    return fig


# ==========================================
# 5. RE-RENDER MAP ON USER CHANGE
# ==========================================
@app.callback(
    Output('network-map', 'figure'),
    Input('matrix-selector', 'value'),
    Input('status-message', 'children'),
    Input('add-edge-status', 'children'),   # also refresh after adding a new edge
)
def update_map_view(selected_matrix, _modify_msg, _add_msg):
    nodes, matrix = load_network_data(selected_matrix)
    return generate_map_figure(nodes, matrix, selected_matrix)


# ==========================================
# 6. HANDLE MAP CLICKS TO EXTRACT VALUES
# ==========================================
@app.callback(
    Output('selected-nodes-text', 'children'),
    Output('current-value-text', 'children'),
    Input('network-map', 'clickData'),
    State('matrix-selector', 'value'),
    prevent_initial_call=True
)
def handle_map_click(clickData, selected_matrix):
    if not clickData:
        return "None selected", "-"

    point_data = clickData['points'][0]

    if 'customdata' in point_data and isinstance(point_data['customdata'], list):
        node_a_id, node_b_id = point_data['customdata']
        _, matrix = load_network_data(selected_matrix)
        current_val = matrix.loc[node_a_id, node_b_id]
        return f"Node {node_a_id} ↔ Node {node_b_id}", f"{current_val}"

    return "Click cleanly onto a connection segment line.", "-"


# ==========================================
# 7. MODIFY EXISTING CONNECTION IN EXCEL
# ==========================================
@app.callback(
    Output('status-message', 'children'),
    Input('update-btn', 'n_clicks'),
    State('matrix-selector', 'value'),
    State('selected-nodes-text', 'children'),
    State('new-matrix-value', 'value'),
    prevent_initial_call=True
)
def write_to_excel(n_clicks, selected_matrix, selected_text, new_value):
    if n_clicks == 0 or "↔" not in selected_text or new_value is None:
        return ""

    try:
        parts = selected_text.replace("Node ", "").split(" ↔ ")
        node_a_id, node_b_id = int(parts[0]), int(parts[1])
        return _write_edge(selected_matrix, node_a_id, node_b_id, float(new_value),
                           action="Updated")
    except Exception as e:
        return f"❌ Save Failed: {str(e)}"


# ==========================================
# 8. ADD A NEW CONNECTION TO EXCEL
# ==========================================
@app.callback(
    Output('add-edge-status', 'children'),
    Input('add-edge-btn', 'n_clicks'),
    State('matrix-selector', 'value'),
    State('new-node-a', 'value'),
    State('new-node-b', 'value'),
    State('new-edge-value', 'value'),
    prevent_initial_call=True
)
def add_new_connection(n_clicks, selected_matrix, node_a, node_b, weight):
    if n_clicks == 0:
        return ""

    # Input validation
    if node_a is None or node_b is None:
        return "⚠️ Please enter both node IDs."
    if weight is None:
        return "⚠️ Please enter a connection weight."
    if int(node_a) == int(node_b):
        return "⚠️ Source and destination nodes must be different."

    node_a_id, node_b_id = int(node_a), int(node_b)

    try:
        nodes, matrix = load_network_data(selected_matrix)

        # Verify both nodes exist in the nodes table
        valid_ids = set(nodes['node_id'].tolist())
        missing = [n for n in [node_a_id, node_b_id] if n not in valid_ids]
        if missing:
            return f"⚠️ Node ID(s) not found in nodes sheet: {missing}"

        # Check the connection doesn't already exist (non-zero)
        if node_a_id in matrix.index and node_b_id in matrix.columns:
            existing = matrix.loc[node_a_id, node_b_id]
            if pd.notna(existing) and existing > 0:
                return (f"⚠️ Connection [{node_a_id} ↔ {node_b_id}] already exists "
                        f"(weight={existing}). Use Section 2 to modify it.")

        return _write_edge(selected_matrix, node_a_id, node_b_id, float(weight),
                           action="Added")
    except Exception as e:
        return f"❌ Add Failed: {str(e)}"


# ==========================================
# 9. SHARED EXCEL WRITE HELPER
# ==========================================
def _write_edge(selected_matrix: str, node_a_id: int, node_b_id: int,
                value: float, action: str = "Updated") -> str:
    """Write (or overwrite) a symmetric edge in the selected matrix sheet."""
    all_sheets = ['nodes', 'pipeline', 'truck', 'railway']

    with pd.ExcelWriter(EXCEL_PATH, mode='a', engine='openpyxl',
                        if_sheet_exists='replace') as writer:
        for sheet in all_sheets:
            if sheet == selected_matrix:
                matrix_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, index_col=0)
                matrix_df.columns = matrix_df.columns.astype(int)

                matrix_df.loc[node_a_id, node_b_id] = value
                matrix_df.loc[node_b_id, node_a_id] = value

                matrix_df.to_excel(writer, sheet_name=sheet, index=True)
            else:
                temp_df = pd.read_excel(
                    EXCEL_PATH, sheet_name=sheet,
                    index_col=0 if sheet != 'nodes' else None
                )
                temp_df.to_excel(writer, sheet_name=sheet,
                                 index=(sheet != 'nodes'))

    return (f"✅ {action}! {selected_matrix.upper()} matrix: "
            f"[{node_a_id} ↔ {node_b_id}] = {value}")


# ==========================================
# 10. LAUNCH APP SERVER
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)