import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. DYNAMIC PATH HANDLING (GIT-SAFE)
# ==========================================
# Resolves paths relative to this script: .../southern_europe/visualisation/map_connections.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Move up one level to 'southern_europe' and dive into 'italy_data'
EXCEL_PATH = SCRIPT_DIR.parent / "italy_data" / "geographical_feature" / "node_metrics.xlsx"

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Could not locate Excel file at: {EXCEL_PATH}")


# ==========================================
# 2. DATA LOADING FUNCTION
# ==========================================
def load_network_data(matrix_type="pipeline"):
    """
    Loads node features and dynamically targets the chosen logistics matrix tab.
    Expected sheet names: 'nodes', 'pipeline', 'truck', 'railway'
    """
    # Load geographic nodes info (lowercased sheet name match)
    df_nodes = pd.read_excel(EXCEL_PATH, sheet_name="nodes")

    # Target the requested matrix sheet
    df_matrix = pd.read_excel(EXCEL_PATH, sheet_name=matrix_type, index_col=0)
    df_matrix.columns = df_matrix.columns.astype(int)

    return df_nodes, df_matrix


# ==========================================
# 3. DASH WEB APP CONFIGURATION
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("AdOpT-NET0 Infrastructure Network Modifier", className="text-center my-3 text-dark"), width=12)
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
                    # Section A: Infrastructure Mode Selector
                    html.H5("1. Select Transport Mode", className="card-title text-secondary"),
                    dcc.RadioItems(
                        id='matrix-selector',
                        options=[
                            {'label': ' 🟢 Pipeline Network', 'value': 'pipeline'},
                            {'label': ' 🚚 Truck Network', 'value': 'truck'},
                            {'label': ' 🚂 Railway Network', 'value': 'railway'}
                        ],
                        value='pipeline',  # Default view layer
                        labelStyle={'display': 'block', 'margin-bottom': '10px'},
                        className="mb-4 fw-bold"
                    ),
                    html.Hr(),

                    # Section B: Data Modifier Form
                    html.H5("2. Modify Matrix Values", className="card-title text-secondary"),
                    html.Div([
                        html.Label("Target Element:"),
                        html.P(id='selected-nodes-text', children="None selected", className="fw-bold text-primary")
                    ]),
                    html.Div([
                        html.Label("Existing Matrix Weight:"),
                        html.P(id='current-value-text', children="-", className="fw-bold")
                    ]),
                    html.Div([
                        html.Label("Input New Value:"),
                        dbc.Input(id='new-matrix-value', type='number', placeholder='Type numerical updates here...'),
                    ], className="mb-3"),

                    dbc.Button("Apply & Overwrite Excel", id='update-btn', color="danger", className="w-100 fw-bold",
                               n_clicks=0),
                    html.Br(), html.Br(),
                    html.Div(id='status-message', className="text-center fw-bold")
                ])
            ], style={'height': '80vh', 'overflowY': 'auto'})
        ], width=4)
    ])
], fluid=True)


# ==========================================
# 4. MAP LAYOUT RENDERING ENGINE
# ==========================================
def generate_map_figure(nodes, matrix, matrix_name):
    fig = go.Figure()

    # Dynamic line colors based on transport mode selection
    color_map = {
        'pipeline': 'rgba(46, 204, 113, 0.7)',  # Emerald Green
        'truck': 'rgba(230, 126, 34, 0.7)',  # Orange
        'railway': 'rgba(52, 152, 219, 0.7)'  # Blue
    }
    line_color = color_map.get(matrix_name, 'rgba(149, 165, 166, 0.7)')

    # --- Draw Active Connections ---
    for idx in matrix.index:
        for col in matrix.columns:
            val = matrix.loc[idx, col]
            if pd.notna(val) and val > 0:  # Valid connection
                node_a = nodes[nodes['node_id'] == idx]
                node_b = nodes[nodes['node_id'] == col]

                if not node_a.empty and not node_b.empty:
                    lon_a, lat_a = node_a.iloc[0]['longitude'], node_a.iloc[0]['latitude']
                    lon_b, lat_b = node_b.iloc[0]['longitude'], node_b.iloc[0]['latitude']

                    fig.add_trace(go.Scattergeo(
                        lon=[lon_a, lon_b],
                        lat=[lat_a, lat_b],
                        mode='lines',
                        line=dict(width=2.5, color=line_color),
                        hoverinfo='text',
                        hovertext=f"<b>Route:</b> {node_a.iloc[0]['node_name']} ↔ {node_b.iloc[0]['node_name']}<br><b>Mode:</b> {matrix_name.upper()}<br><b>Value:</b> {val}",
                        customdata=[[int(idx), int(col)]],
                    ))

    # --- Draw Node Markers ---
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

    # Center focus maps view specifically over Italy region coordinates
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
# 5. RE-RENDER MAP COMPONENT ON USER CHANGE
# ==========================================
@app.callback(
    Output('network-map', 'figure'),
    Input('matrix-selector', 'value'),
    Input('status-message', 'children')  # Refreshes live when changes are written to Excel
)
def update_map_view(selected_matrix, _):
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

    # Catching edge parameters from clicked lines
    if 'customdata' in point_data and isinstance(point_data['customdata'], list):
        node_a_id, node_b_id = point_data['customdata']
        _, matrix = load_network_data(selected_matrix)
        current_val = matrix.loc[node_a_id, node_b_id]

        return f"Node {node_a_id} ↔ Node {node_b_id}", f"{current_val}"

    return "Click cleanly onto a connection segment line.", "-"


# ==========================================
# 7. COMMIT MODIFICATIONS STRAIGHT TO EXCEL
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
        # Extract ID pairs
        parts = selected_text.replace("Node ", "").split(" ↔ ")
        node_a_id, node_b_id = int(parts[0]), int(parts[1])

        # Read the current workbook mapping state completely
        with pd.ExcelWriter(EXCEL_PATH, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            # Preserve sheets not currently being edited
            all_sheets = ['nodes', 'pipeline', 'truck', 'railway']

            for sheet in all_sheets:
                if sheet == selected_matrix:
                    # Update matrix symmetrically for undirected graph
                    matrix_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, index_col=0)
                    matrix_df.columns = matrix_df.columns.astype(int)

                    matrix_df.loc[node_a_id, node_b_id] = float(new_value)
                    matrix_df.loc[node_b_id, node_a_id] = float(new_value)

                    matrix_df.to_excel(writer, sheet_name=sheet, index=True)
                else:
                    # Keep existing tabs perfectly preserved
                    temp_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, index_col=0 if sheet != 'nodes' else None)
                    temp_df.to_excel(writer, sheet_name=sheet, index=(sheet != 'nodes'))

        return f"✅ Saved! {selected_matrix.upper()} matrix altered: [{node_a_id} ↔ {node_b_id}] = {new_value}"
    except Exception as e:
        return f"❌ Save Failed: {str(e)}"


# ==========================================
# 8. LAUNCH APPS SERVER
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)