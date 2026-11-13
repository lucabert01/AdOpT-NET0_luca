"""
Shared visual language for every CCS-chain figure (methodology_figures/,
ccs_chain_results/). Import this module instead of defining colours,
markers or line styles locally.

One variable = one visual channel, in every figure:
  - COLOUR      = emitter sector (maps, MACCs, bar charts). Nothing else may
                  use a sector colour.
  - SHAPE       = capture technology (result maps only). On input-data maps
                  every emitter is a circle.
  - LINE STYLE  = transport mode, in neutral dark grey (solid pipeline,
                  dashed truck, dotted rail); LINE WIDTH is proportional to
                  the built capacity (t/h), on one scale shared by all modes.
  - Emitters without capture: small, hollow, grey.
  - Transport hubs / storage: neutral (white square, black star).

Sector palette: batlow hues (navy, ochre, teal, khaki, mauve, rust; see
cmcrameri's batlow) re-stepped so that every sector sits on its own
lightness level. Checked (OKLab dE x100, Machado-2009 CVD simulation, all
15 pairs): normal vision >= 17.4, protanopia >= 14.7, deuteranopia >= 15.0,
tritanopia >= 9.7 (previous palette: 14.5 / 11.7 / 14.4 / 11.5, with
Cement-Waste at 19; it is 40 now). Lime and Fertilizers are light (WCAG
contrast < 3 on white), so every sector mark carries a thin dark outline.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------------------------------------------
# Surfaces and ink
# ------------------------------------------------------------------
SURFACE = "#ffffff"        # figure, axes and sea: plain white everywhere
LAND = "#ececec"           # Italy fill
COASTLINE = "#9a9a9a"      # Italy boundary
GRIDLINE = "#d9d9d9"       # map graticule / chart grid
SPINE = "#bdbdbd"
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#8a8a8a"
MARK_EDGE = "#262626"      # outline of every sector-coloured mark
NO_CAPTURE_GREY = "#a6a6a6"
KPI_INK = "#4d4d4d"        # neutral fill for non-sector magnitudes (KPI bars)
# Neutral ramp (dark -> light) for categories that are NOT sectors (cost
# components, chain stages, captured vs. emitted): colour stays reserved for
# sector, so these are told apart by lightness and direct labels instead.
NEUTRAL_RAMP = ["#262626", "#555555", "#808080", "#a8a8a8", "#cfcfcf"]
HIGHLIGHT_INK = "#000000"  # emphasis (e.g. trunk line) -- by weight, not hue
# Hourly time-series figures (emitter zoom, node inflow): one shared palette
# so they read as a pair; deliberately not sector colours.
TS_MAIN = "#3a4b5c"        # main series (CO2 captured / CO2 received)
TS_LIGHT = "#b9c2cb"       # secondary series (CO2 emitted)
TS_BAND = "#e6e9ed"        # flagged low-load periods
TS_REFERENCE = "#1a1a1a"   # reference lines (built pipeline capacity), dashed
# Same footprint for every time-series figure: single-column width
# (3.5 in = 89 mm) with a conventional ~4:3 aspect (2.6 in = 66 mm high).
TS_SIZE_IN = (3.5, 2.6)
# Fonts for the time-series figures, at print size: journals ask for >= 7 pt
# after scaling, so nothing here goes below 7.
TS_LABEL_SIZE = 8
TS_TICK_SIZE = 7
TS_LEGEND_SIZE = 7.5
TS_ANNOT_SIZE = 7

# ------------------------------------------------------------------
# Sector = colour
# ------------------------------------------------------------------
SECTOR_ORDER = ["Cement", "Waste", "Refining", "Lime", "Fertilizers", "Other"]
SECTOR_COLORS = {
    "Cement": "#062d59",       # navy   (batlow 0.0)
    "Waste": "#b28737",        # ochre  (batlow ~0.6)
    "Refining": "#3b7c8b",     # teal   (batlow ~0.2, lighter)
    "Lime": "#c5d07b",         # khaki  (batlow ~0.5, lighter)
    "Fertilizers": "#c59bc0",  # mauve  (batlow pink end, muted)
    "Other": "#843825",        # rust   (batlow salmon, darkened)
}
SECTOR_LABELS = {
    "Cement": "Cement",
    "Waste": "Waste-to-energy",
    "Refining": "Refining",
    "Lime": "Lime",
    "Fertilizers": "Fertilizers",
    "Other": "Ravenna industrial cluster",
}
# node_metrics_paper.xlsx node types / technology sectors -> colour group
_SECTOR_ALIASES = {
    "FertilizersCombustion": "Fertilizers",
    "FertilizersSMR": "Fertilizers",
    "Unspecified": "Other",
}


def sector_group(sector) -> str:
    """Colour group of a sector / node type (fertilizer sub-sectors collapse
    to 'Fertilizers'; anything unknown is 'Other')."""
    if not isinstance(sector, str):
        return "Other"
    sector = _SECTOR_ALIASES.get(sector, sector)
    return sector if sector in SECTOR_COLORS else "Other"


def sector_color(sector) -> str:
    return SECTOR_COLORS[sector_group(sector)]


def sector_label(sector) -> str:
    return SECTOR_LABELS[sector_group(sector)]


# ------------------------------------------------------------------
# Capture technology = shape (result maps only)
# ------------------------------------------------------------------
FAMILY_ORDER = ["mea_retrofit", "oxyfuel", "oxyfuel_hybrid", "calcium_looping", "direct_capture"]
FAMILY_SHORT_LABELS = {
    "mea_retrofit": "MEA",
    "oxyfuel": "Oxyfuel",
    "oxyfuel_hybrid": "Oxyfuel + MEA",
    "calcium_looping": "Calcium looping",
    "direct_capture": "Pure process CO$_2$",
}
FAMILY_MARKERS = {
    "mea_retrofit": "o",
    "oxyfuel": "v",
    "oxyfuel_hybrid": "^",
    "calcium_looping": "D",
    "direct_capture": "P",
}
# scatter `s` is a bounding-box area: non-circular glyphs need a larger `s`
# to carry the same visual weight as a circle
FAMILY_MARKER_SCALE = {
    "mea_retrofit": 1.0, "oxyfuel": 1.35, "oxyfuel_hybrid": 1.35,
    "calcium_looping": 0.85, "direct_capture": 1.25,
}
TECH_GLYPH_FACE = "#ffffff"  # technology legend glyphs: shape only, no colour

# ------------------------------------------------------------------
# Transport mode = line style (neutral)
# ------------------------------------------------------------------
MODE_ORDER = ["CO2_Pipeline", "CO2Truck", "CO2Railway"]
MODE_LABELS = {"CO2_Pipeline": "Pipeline", "CO2Truck": "Truck", "CO2Railway": "Rail"}
MODE_COLOR = "#333333"
MODE_LINESTYLES = {
    "CO2_Pipeline": "solid",
    "CO2Truck": (0, (3.0, 1.6)),
    "CO2Railway": (0, (0.9, 1.4)),
}
MODE_COLORS = {m: MODE_COLOR for m in MODE_ORDER}  # kept for callers that look a mode up
# line width (pt) = LW_MIN + LW_SPAN * size / reference capacity (t/h), one
# scale for every mode so widths are comparable across modes and panels
LW_MIN, LW_SPAN = 0.45, 3.0

# ------------------------------------------------------------------
# Non-emitter nodes (neutral)
# ------------------------------------------------------------------
HUB_STYLE = dict(marker="s", facecolor="#ffffff", edgecolor="#333333", linewidth=0.9)
STORAGE_STYLE = dict(marker="*", facecolor="#1a1a1a", edgecolor="#ffffff", linewidth=0.5)

# ------------------------------------------------------------------
# Scenarios (technology_selection CO2-tax sweep in main_italy.py)
# ------------------------------------------------------------------
# (slug, objective, carbon_tax, panel title)
SCENARIOS = [
    ("tax150", "costs", 150, "CO$_2$ tax 150 €/tCO$_2$"),
    ("tax200", "costs", 200, "CO$_2$ tax 200 €/tCO$_2$"),
    ("tax250", "costs", 250, "CO$_2$ tax 250 €/tCO$_2$"),
    ("emissions_minC", "emissions_minC", None, "Emissions-minimizing"),
]
PANEL_LETTERS = "abcdefgh"
SCENARIO_SHORT_LABELS = {"tax150": "150 €/tCO$_2$", "tax200": "200 €/tCO$_2$", "tax250": "250 €/tCO$_2$",
                         "emissions_minC": "Emis.-min."}

# ------------------------------------------------------------------
# Typography / sizes (figures are drawn at final print size)
# ------------------------------------------------------------------
FULL_WIDTH_IN = 7.2   # two-column journal width
HALF_WIDTH_IN = 3.5
FONT_SIZE = 7.5
LABEL_SIZE = 7.5
TICK_SIZE = 6.5
LEGEND_SIZE = 6.8
PANEL_TITLE_SIZE = 8

# label text
COST_LABEL = "Cost of carbon avoided (€/tCO$_2$)"
CAPTURED_LABEL = "Cumulative CO$_2$ captured (Mt/y)"


def apply_rc():
    """Matplotlib defaults shared by every figure."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_PRIMARY,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": False,
        "legend.title_fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,   # editable text in the PDF
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })


def save_figure(fig, path, *, dpi: int = 400, pad_inches: float = 0.03):
    """Saves `path` as BOTH .png and .pdf on a white background, whatever
    extension `path` carries."""
    path = Path(path)
    for ext in (".png", ".pdf"):
        out = path.with_suffix(ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, facecolor=SURFACE)
        print(f"Saved: {out}")


# ------------------------------------------------------------------
# Legend handles
# ------------------------------------------------------------------
def sector_handles(groups=None, *, kind: str = "marker", markersize: float = 5.5) -> list:
    """Colour key (all sectors, or only `groups`, in SECTOR_ORDER).
    kind='marker' -> circles (maps); kind='patch' -> squares (bar charts)."""
    present = {sector_group(g) for g in groups} if groups is not None else set(SECTOR_ORDER)
    out = []
    for g in SECTOR_ORDER:
        if g not in present:
            continue
        if kind == "patch":
            out.append(Patch(facecolor=SECTOR_COLORS[g], edgecolor=MARK_EDGE, linewidth=0.4,
                             label=SECTOR_LABELS[g]))
        else:
            out.append(Line2D([0], [0], marker="o", linestyle="None", markersize=markersize,
                              markerfacecolor=SECTOR_COLORS[g], markeredgecolor=MARK_EDGE,
                              markeredgewidth=0.5, label=SECTOR_LABELS[g]))
    return out


def technology_handles(families=None, *, include_no_capture: bool = True, markersize: float = 5.5) -> list:
    """Shape key: white glyphs with dark outline (shape only -- colour is sector)."""
    present = set(families) if families is not None else set(FAMILY_ORDER)
    out = [
        Line2D([0], [0], marker=FAMILY_MARKERS[f], linestyle="None",
               markersize=markersize * FAMILY_MARKER_SCALE[f] ** 0.5,
               markerfacecolor=TECH_GLYPH_FACE, markeredgecolor=MARK_EDGE, markeredgewidth=0.7,
               label=FAMILY_SHORT_LABELS[f])
        for f in FAMILY_ORDER if f in present
    ]
    if include_no_capture:
        out.append(no_capture_handle(markersize))
    return out


def no_capture_handle(markersize: float = 5.5) -> Line2D:
    return Line2D([0], [0], marker="o", linestyle="None", markersize=markersize * 0.6,
                  markerfacecolor="none", markeredgecolor=NO_CAPTURE_GREY, markeredgewidth=0.8,
                  label="No capture")


def mode_handles(modes=None, *, lw: float = 1.4) -> list:
    present = set(modes) if modes is not None else set(MODE_ORDER)
    return [Line2D([0], [0], color=MODE_COLOR, lw=lw, linestyle=MODE_LINESTYLES[m], label=MODE_LABELS[m])
            for m in MODE_ORDER if m in present]


def width_handles(ref_capacity_t_h: float, values=(100, 500, 1500)) -> list:
    return [Line2D([0], [0], color=MODE_COLOR, solid_capstyle="butt",
                   lw=LW_MIN + LW_SPAN * v / ref_capacity_t_h, label=f"{v:,} t/h")
            for v in values if v <= ref_capacity_t_h * 1.05]


def transport_handles(modes, ref_capacity_t_h: float) -> list:
    """Mode line styles (column 1) and capacity widths (column 2) for a
    2-column legend: both columns padded to the same length with blanks,
    because matplotlib fills legend columns top-to-bottom."""
    left, right = mode_handles(modes), width_handles(ref_capacity_t_h)
    n = max(len(left), len(right))
    blank = lambda: Line2D([], [], linestyle="None", label=" ")  # noqa: E731
    return left + [blank() for _ in range(n - len(left))] + right + [blank() for _ in range(n - len(right))]


def size_handles(ref_t: float, area_fn, values_t=(1e5, 5e5, 1.5e6)) -> list:
    """Marker-size key (annual CO2 emissions): hollow neutral circles whose
    area is area_fn(value) -- the same function that sizes the map markers
    (scatter `s` is an area in pt^2, Line2D markersize a diameter in pt)."""
    return [Line2D([0], [0], marker="o", linestyle="None", markersize=area_fn(v) ** 0.5,
                   markerfacecolor="none", markeredgecolor=INK_SECONDARY, markeredgewidth=0.6,
                   label=f"{v / 1e6:g} Mt/y")
            for v in values_t if v <= ref_t * 1.02]


def node_handles(markersize: float = 5.5) -> list:
    return [
        Line2D([0], [0], linestyle="None", marker=HUB_STYLE["marker"], markersize=markersize * 0.85,
               markerfacecolor=HUB_STYLE["facecolor"], markeredgecolor=HUB_STYLE["edgecolor"],
               markeredgewidth=HUB_STYLE["linewidth"], label="Transport hub"),
        Line2D([0], [0], linestyle="None", marker=STORAGE_STYLE["marker"], markersize=markersize * 1.6,
               markerfacecolor=STORAGE_STYLE["facecolor"], markeredgecolor=STORAGE_STYLE["edgecolor"],
               markeredgewidth=STORAGE_STYLE["linewidth"], label="CO$_2$ storage"),
    ]


def legend_block(target, handles, title, *, loc, bbox, ncol=1, **kw):
    """One titled, frameless legend column; left-aligned title. Returns it
    (re-add with add_artist when several share an Axes)."""
    leg = target.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=bbox, ncol=ncol,
                        frameon=False, handlelength=kw.pop("handlelength", 1.6),
                        handletextpad=kw.pop("handletextpad", 0.5),
                        columnspacing=kw.pop("columnspacing", 1.0),
                        labelspacing=kw.pop("labelspacing", 0.35),
                        borderaxespad=0.0, borderpad=0.0, **kw)
    leg._legend_box.align = "left"
    leg.get_title().set_fontweight("bold")
    return leg
