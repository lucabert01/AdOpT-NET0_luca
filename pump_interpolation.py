import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from pathlib import Path
import json

# Given constants
rho_co2 = 505           # kg/m3
nu = 1/rho_co2           # Specific volume
p_pump_in = 100    # Inlet pressure in bar (constant)
eta_pump = 0.75    # Pump efficiency (assumed constant)
# Function to calculate the power consumption W_pump
def compute_W_pump(m_dot, p_pump_out):
    return m_dot * nu * (p_pump_out - p_pump_in) / eta_pump *0.1/3.6 # power in MW

# Define the ranges for p_pump_out (delta_p) and m_dot (flowrate)
range_delta_p = [101, 130] # in bar
range_flowrate = [0, 10] # in t/h

# Create linearly spaced values within the given ranges
p_pump_out_range = np.linspace(range_delta_p[0], range_delta_p[1], 100)  # Pump exit pressure  bar
m_dot_range = np.linspace(range_flowrate[0], range_flowrate[1], 100)     # Mass flow rate  t/h

# Create a meshgrid of values for p_pump_out and m_dot
p_pump_out_grid, m_dot_grid = np.meshgrid(p_pump_out_range, m_dot_range)

# Calculate the pump power for each pair of (m_dot, p_pump_out)
W_pump_values = compute_W_pump(m_dot_grid, p_pump_out_grid)

# fit W_pump with a plane W = a * p_pump_out + b * m_dot
def plane_model(X, a, b):
    p_pump_out, m_dot = X
    return a * m_dot + b * p_pump_out - b * p_pump_out_range[0]

# Flatten the grids for curve fitting
p_pump_out_flat = p_pump_out_grid.flatten()
m_dot_flat = m_dot_grid.flatten()
W_pump_flat = W_pump_values.flatten()

# Perform curve fitting to get the coefficients a and b
params, _ = curve_fit(plane_model, (p_pump_out_flat, m_dot_flat), W_pump_flat)

# Extract the fitted parameters
a, b = params
print(f"Fitted parameters: a = {a:.5f}, b = {b:.5f}")

# Visualization of the fitted plane and the original data
batlow_cmap = cmc.batlow

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the original W_pump data points
ax.scatter(p_pump_out_grid, m_dot_grid, W_pump_values, color=batlow_cmap(0.1), label='Original Data')

# Generate the interpolated (fitted) plane
W_fitted = plane_model((p_pump_out_grid, m_dot_grid), a, b)
ax.plot_surface(p_pump_out_grid, m_dot_grid, W_fitted, color=batlow_cmap(0.8), alpha=0.5, label='Fitted Plane')

# Labels and plot settings
ax.set_xlabel('Pump exit pressure [bar]]')
ax.set_ylabel('Mass flow rate [t/h]')
ax.set_zlabel('Power consumption [MW]]')
plt.legend()
plt.show()



# save results of interpolation
data_path = Path(__file__).parent/ "adopt_net0/data/technology_data/Sink/SalineAquifer_data"
data_path.mkdir(parents=True, exist_ok=True)

params_file = data_path / "pump_params.json"
params_dict = {"a": a, "b": b}
with open(params_file, "w") as f:
    json.dump(params_dict, f)