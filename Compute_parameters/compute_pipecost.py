from adopt_net0.database.components.networks import CO2_Pipeline_CostModel

options = {
    "length_km": 100,
    "discount_rate": 0.1,
    "financial_year_out": 2024,
    "currency_out": "EUR",
    "terrain": "Onshore"
}

model = CO2_Pipeline_CostModel("CO2_Pipeline")
model.calculate_indicators(options)


gamma1 = model.financial_indicators["gamma1"]  # EUR
gamma2 = model.financial_indicators["gamma2"]  # EUR per t/h



energy_cons = model.technical_indicators["energyconsumption"]


a=1
