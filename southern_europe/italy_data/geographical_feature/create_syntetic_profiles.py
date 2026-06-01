"""
Generate synthetic hourly emission profiles (8760 hours) for all emitters
that don't already have real-world data in emission_profile_emitters.xlsx.

Sectors handled:
  - Cement   → flat, 4-week summer shutdown + 2 random 1-week stops
  - Waste    → full capacity ±10% noise, 2 random 3-week half-capacity stops (not winter)
  - Refining → flat, 1 random 1-week stop
  - Other    → flat, 1 random 1-week stop
  - Transport / Storage → skipped

All profiles are scaled so that sum(hourly_profile) == annual_flux (tonnes/year).
annual_flux is read from the "annual_flux" column in node_metrics.xlsx (values in kg).

Output: sheet "synthetic_data" written to emission_profile_emitters.xlsx
Column naming convention: "[Sector - node_name]"
"""

from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE          = Path(r"C:\Users\0954659\PycharmProjects\AdOpT-NET0_luca\southern_europe\italy_data\geographical_feature")
NODES_FILE    = BASE / "node_metrics.xlsx"
EMITTERS_FILE = BASE / "emission_profile_emitters.xlsx"

HOURS = 8760
RNG   = np.random.default_rng(42)          # fixed seed → reproducible results

SKIP_SECTORS = {"Transport", "Storage"}

# ---------------------------------------------------------------------------
# Calendar windows (hour indices, non-leap year)
# ---------------------------------------------------------------------------
JAN_FEB_START,  JAN_FEB_END   =   0 * 24,  59 * 24   # 1 Jan – 28 Feb
APR_MAY_START,  APR_MAY_END   =  90 * 24, 151 * 24   # 1 Apr – 31 May
SUMMER_START,   SUMMER_END    = 196 * 24, 228 * 24   # 15 Jul – 15 Aug  (4 weeks = 672 h)
NO_WINTER_START, NO_WINTER_END =  59 * 24, 334 * 24  # 1 Mar – 30 Nov  (allowed for Waste stops)

WEEK  = 7  * 24   # 168 h
THREE_WEEKS = 3 * WEEK   # 504 h


def _rand_start(window_start: int, window_end: int, stop_len: int) -> int:
    """Uniform random start hour for a stop of `stop_len` hours inside the window."""
    latest = window_end - stop_len
    if latest <= window_start:
        return window_start
    return int(RNG.integers(window_start, latest))


# ---------------------------------------------------------------------------
# Normalized profile generators  (output in [0, 1])
# Scaling to physical units (tonnes/h) is done separately via annual_flux.
# ---------------------------------------------------------------------------

def _norm_cement() -> np.ndarray:
    """
    Flat 1.0 when running.
    Downtime:
      - fixed 4-week summer shutdown (15 Jul – 15 Aug)
      - 1 random 1-week stop in Jan–Feb
      - 1 random 1-week stop in Apr–May
    """
    profile = np.ones(HOURS)
    profile[SUMMER_START:SUMMER_END] = 0.0
    s1 = _rand_start(JAN_FEB_START, JAN_FEB_END, WEEK)
    profile[s1:s1 + WEEK] = 0.0
    s2 = _rand_start(APR_MAY_START, APR_MAY_END, WEEK)
    profile[s2:s2 + WEEK] = 0.0
    return profile


def _norm_waste() -> np.ndarray:
    """
    Full capacity (1.0 ± 10% uniform noise) when both lines running.
    2 random 3-week stops, not in winter (Mar–Nov only):
      - during a stop: 0.5 flat (one line down, no noise)
    Stops are guaranteed non-overlapping.
    """
    profile = RNG.uniform(0.9, 1.1, HOURS)      # full capacity with ±10% noise

    starts: list[int] = []
    for _ in range(2):
        for _ in range(10_000):                  # retry until non-overlapping
            s = _rand_start(NO_WINTER_START, NO_WINTER_END, THREE_WEEKS)
            if all(abs(s - s2) >= THREE_WEEKS for s2 in starts):
                starts.append(s)
                break

    for s in starts:
        profile[s:s + THREE_WEEKS] = 0.5        # half capacity, no noise

    return profile


def _norm_flat_one_stop() -> np.ndarray:
    """
    Flat 1.0 when running, one random 1-week stop anywhere in the year.
    Used for Refining and Other.
    """
    profile = np.ones(HOURS)
    s = _rand_start(0, HOURS, WEEK)
    profile[s:s + WEEK] = 0.0
    return profile


NORM_GENERATORS = {
    "Cement":   _norm_cement,
    "Waste":    _norm_waste,
    "Refining": _norm_flat_one_stop,
    "Other":    _norm_flat_one_stop,
}

# ---------------------------------------------------------------------------
# Scaling: normalized profile → tonnes/hour so that annual sum = annual_flux
# ---------------------------------------------------------------------------

def scale_profile(normalized: np.ndarray, annual_flux_kg: float) -> np.ndarray:
    """
    Scale normalized profile (dimensionless) to tonnes/hour.
    sum(result) == annual_flux_kg / 1000  [tonnes/year]
    """
    annual_flux_t = annual_flux_kg / 1000.0
    total = normalized.sum()
    if total == 0:
        return np.zeros(HOURS)
    return normalized * (annual_flux_t / total)


# ---------------------------------------------------------------------------
# Plotting Function with Subplots Per Emitter
# ---------------------------------------------------------------------------
def plot_profiles(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    """Generates three figures (Cement, Waste, Refining/Other), with a separate subplot for each emitter."""

    groups = {
        "Cement": ["Cement"],
        "Waste": ["Waste"],
        "Refining & Others": ["Refining", "Other"]
    }

    hours_axis = np.arange(HOURS)

    for plot_title, sectors in groups.items():
        # Identify all columns belonging to this group
        real_cols = [c for c in real_df.columns if any(c.startswith(f"{s} -") for s in sectors)]
        synth_cols = [c for c in synth_df.columns if any(c.startswith(f"{s} -") for s in sectors)]

        all_emitters = [(c, "REAL") for c in real_cols] + [(c, "SYNTHETIC") for c in synth_cols]
        num_emitters = len(all_emitters)

        if num_emitters == 0:
            continue

        # Dynamically compute grid rows and columns (max 3 columns wide)
        cols_grid = min(3, num_emitters)
        rows_grid = int(np.ceil(num_emitters / cols_grid))

        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(5 * cols_grid, 3.5 * rows_grid), squeeze=False)
        fig.suptitle(f"Hourly Emission Profiles: {plot_title}", fontsize=16, fontweight='bold', y=0.98)

        for idx, (col_name, data_source) in enumerate(all_emitters):
            r, c = divmod(idx, cols_grid)
            ax = axes[r, c]

            if data_source == "REAL":
                ax.scatter(hours_axis, real_df[col_name], s=0.5, alpha=0.6, color="crimson")
                ax.set_title(f"{col_name}\n[REAL DATA]", color="crimson", fontsize=10, fontweight="bold")
                # Light highlight for real data backgrounds
                ax.set_facecolor('#fff5f5')
            else:
                ax.scatter(hours_axis, synth_df[col_name], s=0.3, alpha=0.5, color="#1f77b4")
                ax.set_title(f"{col_name}\n[SYNTHETIC]", color="#1f77b4", fontsize=10)

            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=8)

            # Label only the edge plots to avoid cluttering
            if r == rows_grid - 1:
                ax.set_xlabel("Hour", fontsize=9)
            if c == 0:
                ax.set_ylabel("tonnes/h", fontsize=9)

        # Hide any unused subplot tiles in the grid matrix
        for idx in range(num_emitters, rows_grid * cols_grid):
            r, c = divmod(idx, cols_grid)
            fig.delaxes(axes[r, c])

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------------
    # Main
    # ---------------------------------------------------------------------------


def main() -> None:
    # 1. Load node list
    nodes_df = pd.read_excel(NODES_FILE, sheet_name="nodes")
    nodes_df = nodes_df[["node_id", "node_name", "node_type", "annual_flux"]].dropna(
        subset=["node_name", "node_type"]
    )
    nodes_df["node_name"] = nodes_df["node_name"].str.strip()
    nodes_df["node_type"] = nodes_df["node_type"].str.strip()
    nodes_df["annual_flux"] = pd.to_numeric(nodes_df["annual_flux"], errors="coerce").fillna(0.0)

    # 2. Load existing real-world profiles
    wb = load_workbook(EMITTERS_FILE)
    real_sheet = "raw_data"
    real_df = pd.read_excel(
        EMITTERS_FILE,
        sheet_name=real_sheet if real_sheet in wb.sheetnames else 0,
    )
    existing_cols = set(real_df.columns)

    # 3. Build synthetic profiles
    synthetic_data: dict[str, np.ndarray] = {}

    for _, row in nodes_df.iterrows():
        node_name = row["node_name"]
        sector = row["node_type"]
        annual_flux = float(row["annual_flux"])

        if sector in SKIP_SECTORS:
            continue

        col_header = f"{sector} - {node_name}"

        if col_header in existing_cols:
            continue

        if sector not in NORM_GENERATORS:
            print(f"[WARN] Unknown node_type '{sector}' for '{node_name}' — using Other generator")
            sector = "Other"
            col_header = f"{sector} - {node_name}"

        if col_header not in synthetic_data:
            normalized = NORM_GENERATORS[sector]()
            synthetic_data[col_header] = scale_profile(normalized, annual_flux)

    synth_df = pd.DataFrame(synthetic_data)

    # 4. Write synthetic_data sheet
    if not synthetic_data:
        print("No synthetic profiles needed — all nodes already have real-world data.")
    else:
        print(f"Writing {len(synth_df.columns)} synthetic profiles ({HOURS} hours each) …")
        with pd.ExcelWriter(EMITTERS_FILE, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            synth_df.to_excel(writer, sheet_name="synthetic_data", index=False)
        print(f"Done. 'synthetic_data' sheet written to:\n  {EMITTERS_FILE}")

    # 5. Generate and display subplot matrices
    print("\nGenerating subplot profiles...")
    plot_profiles(real_df, synth_df)

    # 6. Coverage summary
    print("\n=== Coverage summary ===")
    records = []
    for _, row in nodes_df.iterrows():
        node_name = row["node_name"]
        sector = row["node_type"]
        if sector in SKIP_SECTORS:
            continue
        col_header = f"{sector} - {node_name}"
        source = "real_data" if col_header in existing_cols else "synthetic_data"
        records.append((node_name, sector, round(row["annual_flux"] / 1000, 1), source))

    coverage = pd.DataFrame(records, columns=["node_name", "sector", "annual_flux_t", "source"])
    print(coverage.to_string(index=False))
    print(f"\nAll {len(coverage)} relevant nodes covered ✓")


if __name__ == "__main__":
    main()