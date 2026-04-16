from scipy.interpolate import RegularGridInterpolator
import numpy as np
from pathlib import Path
import pandas as pd


data_path = Path("../adopt_net0/database/templates/technology_data/Industrial/WasteCaL_data/wasteCaL_sheet.xlsx")

capex_data = pd.read_excel(
            data_path, sheet_name="capex_eur", index_col=0
        )
# 1. Define your axes (coordinates from your simulations)
concentrations = np.array(capex_data.columns, dtype=float)
sizes = np.array(capex_data.index, dtype=float)

z_values = capex_data.values

results = []

# Iterate through the grid intervals
# sizes.shape[0]-1 because we are looking at the 'boxes' between points
for i in range(len(sizes) - 1):
    for j in range(len(concentrations) - 1):
        # Define the bounds of this specific "patch"
        s0, s1 = sizes[i], sizes[i + 1]
        c0, c1 = concentrations[j], concentrations[j + 1]

        # Get values at the corners (z00 is the anchor)
        z00 = z_values[i, j]  # (Size_low, Conc_low)
        z10 = z_values[i + 1, j]  # (Size_high, Conc_low)
        z01 = z_values[i, j + 1]  # (Size_low, Conc_high)

        # Calculate local slopes (gradients) for this specific patch
        m_size = (z10 - z00) / (s1 - s0)
        m_conc = (z01 - z00) / (c1 - c0)

        # Calculate the intercept for this patch
        # Intercept = z - m1*s - m2*c
        intercept = z00 - (m_size * s0) - (m_conc * c0)

        results.append({
            'size_range': f"{s0}-{s1}",
            'conc_range': f"{c0}-{c1}",
            'slope_size': m_size,
            'slope_conc': m_conc,
            'intercept': intercept
        })

# Convert to a DataFrame for easy export or lookup
piecewise_df = pd.DataFrame(results)
print(piecewise_df.head())


# Test a point
# 1. Define the point
test_size = 3350
test_conc = 0.085

# 2. Find the correct row in the piecewise DataFrame
# We look for the row where test_size and test_conc fall within the ranges
# This logic assumes your ranges are formatted as "low-high" strings from your code
def is_in_range(val, range_str):
    low, high = map(float, range_str.split('-'))
    return low <= val <= high

# Find the specific parameters
mask = piecewise_df.apply(lambda row:
    is_in_range(test_size, row['size_range']) and
    is_in_range(test_conc, row['conc_range']), axis=1)

patch_params = piecewise_df[mask].iloc[0]

# 3. Apply the linear equation: z = m1*x + m2*y + b
m_s = patch_params['slope_size']
m_c = patch_params['slope_conc']
b = patch_params['intercept']

final_capex = (m_s * test_size) + (m_c * test_conc) + b

# 4. Output results
print("\n--- Piecewise Calculation Results ---")
print(f"Point:        Size {test_size}, Concentration {test_conc}")
print(f"Selected Row: Size Range {patch_params['size_range']} | Conc Range {patch_params['conc_range']}")
print(f"Equation:     ({m_s:.4f} * {test_size}) + ({m_c:.4f} * {test_conc}) + {b:.4f}")
print("-" * 40)
print(f"RESULTING CAPEX: {final_capex:,.2f} EUR")