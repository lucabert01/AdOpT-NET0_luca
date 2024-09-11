import pandas as pd
import numpy as np

# Case studies parameters
time_sim = 10 # years simulated


# Timeseries for cement and w2e
file_path = 'C:/Users/0954659/OneDrive - Universiteit Utrecht/Documents/PhD/5 - Research/HERCCULES WP6/Emissions database/DS.06.01_v2/DS.06.01_emission profiles_v2.xlsx'
cement = {'Cement - FANNA', 'Cement-MONSELICE', 'Cement-ROBILANTE'}
w2e = {'Waste - SILLA 2'}

daily_emissions_cement = np.zeros((1, 365))
daily_emissions_w2e = np.zeros((1, 365))
i = 0
# Loop through each sheet and import the data, then extract the third column
for plant in cement:
    i= i+1
    # Load the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=plant)
    emissions = df.iloc[1:, 2].to_numpy()
    daily_emissions_cement = daily_emissions_cement + emissions# in t/h

emissions_tot_cement = daily_emissions_cement.sum()/10**6


# Silla 2
daily_emissions_silla = np.zeros((1, 365))
df = pd.read_excel(file_path, sheet_name='Waste - SILLA 2')
emissions = df.iloc[1:, 2].to_numpy()
emissions_silla = np.append(emissions,emissions[-1])# in t/h (adding an extra timestep missing in the excel)

for i in range(0,365):
    daily_emissions_silla[0,i] = np.sum(emissions_silla[i*24:(i+1)*24])

emissions_tot_silla = daily_emissions_silla.sum()/10**6

# Waste to energy Como
daily_emissions_como = np.zeros((1, 365))
df = pd.read_excel(file_path, sheet_name='Waste - TERMOVALORIZZATORE COMO')
emissions = df.iloc[1:, 2].to_numpy()
emissions_como = emissions# in t/h

for i in range(0,365):
    daily_emissions_como[0,i] = np.sum(emissions_como[i*48:(i+1)*48])

emissions_tot_como = daily_emissions_como.sum()/10**6

daily_emissions_w2e = daily_emissions_como + daily_emissions_silla



fullprofile_emissions_cement = np.tile(daily_emissions_cement, time_sim)
fullprofile_emissions_w2e = np.tile(daily_emissions_w2e, time_sim)



# CO2 price based on IEA 2022 APS https://2024.entsos-tyndp-scenarios.eu/download/

co2_price_2030 = 113.4
co2_price_2040 = 147
co2_price_2050 = 168

range_co2_price_3040 = np.linspace(co2_price_2030,co2_price_2040, num=365*10)
range_co2_price_4050 = np.linspace(co2_price_2040,co2_price_2050, num=365*10)
range_co2_prices_fulll = np.append(range_co2_price_3040, range_co2_price_4050)

fullprofile_co2_price = range_co2_prices_fulll[0:time_sim*365]

