"""
CO2 container-based (isotainer) transport costs for truck and train, as a
function of distance and discount rate.

Source: Oeuvray et al. 2024, "Multi-criteria assessment of inland and
offshore carbon dioxide transport options", J. Clean. Prod. 443, 140781,
and its Supplementary Information (SI).

IMPORTANT — data provenance
----------------------------
The paper states explicitly (SI, p.2): "The data about transport that has
been gathered directly from industry is confidential." This means:

  - PUBLIC (used as-is below): isotainer capacity and specs (Table A.1),
    discount rate, truck driver salary, HGVT rates, operating hours,
    loading times, train speed (Tables A.2-A.4, B.2), and the FITTED
    aggregate cost curve (Table C.1) UC = alpha1 + alpha2/d [EUR/(t*km)].
  - CONFIDENTIAL / NOT DISCLOSED: isotainer purchase price, tractor/trailer
    price, fuel consumption, maintenance EUR/km, transshipment/weighing/
    rail-linehaul fees. These are marked "ASSUMED" below with placeholder
    values, calibrated so the mechanistic model matches published_fit() to
    within a few percent (see the calibration comment on each PARAMS dict).
    Replace them with your own quotes for a precise breakdown.

Two ways to get a cost:
  1. published_fit(distance_km, option) -> exact total levelised cost
     [EUR/t], backed by real industrial data (R^2 = 0.996-1.0). No
     capex/opex split, and implicitly assumes the fleet is exactly sized
     to whatever tonnage you end up transporting in a year (no idle
     capacity).
  2. truck_costs_per_capacity(...) / train_costs_per_capacity(...) -> a
     mechanistic breakdown, normalised per unit of guaranteed transport
     CAPACITY (t/h) rather than per tonne transported:
       - capex_eur_per_tph_y, fixed_opex_eur_per_tph_y [EUR/(t/h)/y]:
         the annualised cost of owning enough trucks/isotainers (or train
         isotainers) to be able to move CO2 at a given rate, indefinitely,
         regardless of how much you actually end up shipping.
       - variable_opex_eur_per_t [EUR/t] (truck only; folded into
         fixed_opex for train, see below): the extra cost incurred only
         when a shipment actually runs (fuel, maintenance, driver hours),
         genuinely proportional to tonnes moved, independent of fleet size.
     For train, everything the rail operator charges per shipment
     (transshipment, weighing, the rail linehaul rate) is folded into
     fixed_opex_eur_per_tph_y instead of being split out, since in
     practice this is normally bought as one bundled per-isotainer rail
     service rather than operated and fuelled directly like a truck.
"""

import math

# ---------------------------------------------------------------------------
# 1. Published aggregate fit (Table C.1) — exact, backed by real quotes
# ---------------------------------------------------------------------------

# UC [EUR / (t . km)] = alpha1 + alpha2 / d[km]  ->  LC[EUR/t] = alpha1*d + alpha2
PUBLISHED_FIT_PARAMS = {
    "container_truck": {"alpha1": 0.15, "alpha2": 5.58},
    "container_train": {"alpha1": 0.07, "alpha2": 28.9},
}


def published_fit(distance_km, option="container_truck"):
    """Levelised cost per tonne CO2 [EUR/t] from the paper's fitted curve.

    Independent of mass flow: the paper shows container-based options (and
    dedicated truck/train) have unitary costs that depend only on distance,
    since capacity per carrier is small and doesn't benefit from scale.
    """
    p = PUBLISHED_FIT_PARAMS[option]
    unitary_cost = p["alpha1"] + p["alpha2"] / distance_km  # EUR/(t.km)
    levelised_cost = unitary_cost * distance_km  # EUR/t
    return {"unitary_cost_eur_per_t_km": unitary_cost, "levelised_cost_eur_per_t": levelised_cost}


# ---------------------------------------------------------------------------
# 2. Mechanistic, capacity-based breakdown
# ---------------------------------------------------------------------------

def capital_recovery_factor(r, lifetime_y):
    return r / (1 - (1 + r) ** (-lifetime_y))


TRUCK_PARAMS = {
    # --- PUBLIC (Oeuvray et al. 2024, Tables A.1, A.2, B.2) ---
    "isotainer_capacity_t": 20.0,          # Table A.1
    "remaining_onboard_frac": 0.04,        # Table A.1, note 2
    "supplementary_isotainer_frac": 0.20,  # Table A.2, epsilon_st
    "t_op_h_per_y": 8500.0,                # Table A.1, container truck
    "load_unload_h": 1.4 + 1.4,            # Table A.4, container truck
    "border_h": 0.0,                       # Table A.4 (set >0 if crossing a border, e.g. 0.5)
    "driver_salary_eur_per_h": 21.9,       # Table A.2

    # --- ASSUMED / CONFIDENTIAL (industry data not disclosed) ---
    # Calibrated by least-squares fit against published_fit("container_truck")
    # over d = 50-2000 km at 1 Mt/y (mass-flow-independent regime), subject to
    # staying within a plausible EU heavy-goods-vehicle cost range. Achieves
    # < 3% gap to the published curve across that range.
    "avg_speed_kmh": 45.6,                 # paper uses Google Maps travel time; this is the calibrated average
    "isotainer_cost_eur": 60000.0,
    "isotainer_lifetime_y": 15.0,
    "isotainer_choice": "buy",             # "buy" or "rent"
    "isotainer_rent_eur_per_y": 8000.0,    # used only if isotainer_choice == "rent"
    "tractor_cost_eur": 240000.0,
    "tractor_lifetime_y": 6.0,
    "trailer_cost_eur": 80000.0,
    "trailer_lifetime_y": 12.0,
    "fuel_l_per_km": 0.34,
    "fuel_price_eur_per_l": 1.60,
    "maintenance_eur_per_km": 0.146,
    "hgvt_eur_per_km": 0.0,                # e.g. CH 40t+20t combined ~1.37 CHF/km, Table A.3
    "hgvt_eur_per_y": 0.0,                 # e.g. Eurovignette 1250 EUR/y, Table A.3
    "insurance_eur_per_y": 12000.0,
    "vehicle_tax_eur_per_y": 6000.0,
    "admin_eur_per_y": 8000.0,
    "tires_eur_per_y": 10000.0,
    "infrastructure_eur_per_y": 4000.0,
}

TRAIN_PARAMS = {
    # --- PUBLIC ---
    "isotainer_capacity_t": 20.0,
    "remaining_onboard_frac": 0.04,
    "supplementary_isotainer_frac": 0.20,
    "t_op_h_per_y": 8760.0,                # Table A.1, container train
    "load_unload_h": 8.0,                  # Table A.4, container train (RNE)

    # --- ASSUMED / CONFIDENTIAL ---
    # Calibrated the same way as TRUCK_PARAMS; achieves < 0.2% gap to the
    # published curve over d = 50-2000 km at 1 Mt/y.
    "avg_speed_kmh": 15.9,                 # Table A.4 footnote 13 gives ~18 km/h; calibrated value is close
    "isotainer_cost_eur": 30300.0,
    "isotainer_lifetime_y": 15.0,
    "isotainer_choice": "buy",
    "isotainer_rent_eur_per_y": 8000.0,
    "transshipment_eur_per_isotainer": 223.0,   # per loading/unloading event, Eq. (26)
    "weighing_eur_per_isotainer": 30.0,         # Eq. (27)
    "transport_base_eur_per_isotainer": 74.0,   # distance-independent part of rail linehaul, Eq. (29)
    "transport_eur_per_isotainer_km": 1.283,    # distance-dependent part of rail linehaul, Eq. (29)
}


def truck_costs_per_capacity(distance_km, discount_rate=0.08, params=None):
    """Capacity-based capex/fixed-opex, plus a usage-based variable rate, for
    container-based truck transport.

    distance_km: one-way distance [km]
    discount_rate: r used in the capital recovery factor

    Returns:
      capex_eur_per_tph_y, fixed_opex_eur_per_tph_y [EUR/(t/h)/y]:
        annualised cost of owning enough tractors/trailers/isotainers to
        sustain a guaranteed transport rate of 1 t/h indefinitely (derived
        continuously via Little's law: a truck "in flight" for
        roundtrip_h hours carries mc_co2 tonnes, so 1/mc_co2 * roundtrip_h
        trucks are needed per unit of t/h capacity). Independent of how
        much you actually ship in a given year - this is a pure capacity
        cost, like paying for a car + insurance regardless of mileage.
      variable_opex_eur_per_t [EUR/t]: fuel + maintenance + driver time +
        per-km HGVT for one round trip, divided by the tonnes carried in
        it - the genuinely usage-based part, only incurred per shipment
        actually run (like a car's fuel).
    """
    p = dict(TRUCK_PARAMS)
    if params:
        p.update(params)

    mc_co2 = p["isotainer_capacity_t"] * (1 - p["remaining_onboard_frac"])
    roundtrip_h = p["load_unload_h"] + 2 * distance_km / p["avg_speed_kmh"] + 2 * p["border_h"]

    # Shipments/year needed to average 1 t/h over the year - independent of
    # trip duration, since a shipment always carries mc_co2 tonnes.
    n_shipment_per_tph_y = 8760.0 / mc_co2
    # Concurrent fleet ("carriers") needed to run that many shipments given
    # each ties one up for roundtrip_h hours, within t_op_h_per_y of
    # available operating time per carrier per year (Little's law).
    n_carrier_per_tph = n_shipment_per_tph_y * roundtrip_h / p["t_op_h_per_y"]
    n_isotainer_per_tph = n_carrier_per_tph * (1 + p["supplementary_isotainer_frac"])

    r = discount_rate
    a_tractor = capital_recovery_factor(r, p["tractor_lifetime_y"])
    a_trailer = capital_recovery_factor(r, p["trailer_lifetime_y"])
    a_iso = capital_recovery_factor(r, p["isotainer_lifetime_y"])

    if p["isotainer_choice"] == "buy":
        capex_eur_per_tph_y = n_carrier_per_tph * (p["tractor_cost_eur"] * a_tractor + p["trailer_cost_eur"] * a_trailer) \
            + n_isotainer_per_tph * p["isotainer_cost_eur"] * a_iso
        iso_fixed_opex_per_tph_y = 0.0
    else:
        capex_eur_per_tph_y = n_carrier_per_tph * (p["tractor_cost_eur"] * a_tractor + p["trailer_cost_eur"] * a_trailer)
        iso_fixed_opex_per_tph_y = n_isotainer_per_tph * p["isotainer_rent_eur_per_y"]

    fixed_opex_eur_per_tph_y = n_carrier_per_tph * (
        p["insurance_eur_per_y"] + p["vehicle_tax_eur_per_y"] + p["admin_eur_per_y"]
        + p["tires_eur_per_y"] + p["infrastructure_eur_per_y"] + p["hgvt_eur_per_y"]
    ) + iso_fixed_opex_per_tph_y

    beta1_eur_per_km = p["fuel_l_per_km"] * p["fuel_price_eur_per_l"] + p["maintenance_eur_per_km"]
    variable_opex_eur_per_t = (
        beta1_eur_per_km * 2 * distance_km
        + roundtrip_h * p["driver_salary_eur_per_h"]
        + 2 * distance_km * p["hgvt_eur_per_km"]
    ) / mc_co2

    return {
        "capex_eur_per_tph_y": capex_eur_per_tph_y,
        "fixed_opex_eur_per_tph_y": fixed_opex_eur_per_tph_y,
        "variable_opex_eur_per_t": variable_opex_eur_per_t,
    }


def train_costs_per_capacity(distance_km, discount_rate=0.08, params=None):
    """Capacity-based capex/fixed-opex for container-based train transport.

    Same capacity logic as truck_costs_per_capacity(), but all per-shipment
    rail-service fees (transshipment, weighing, linehaul rate) are folded
    into fixed_opex_eur_per_tph_y rather than split into a separate
    variable rate - train capacity here is normally bought as a single
    bundled per-isotainer rail service, not operated/fuelled directly.

    Returns capex_eur_per_tph_y, fixed_opex_eur_per_tph_y [EUR/(t/h)/y].
    """
    p = dict(TRAIN_PARAMS)
    if params:
        p.update(params)

    mc_co2 = p["isotainer_capacity_t"] * (1 - p["remaining_onboard_frac"])
    roundtrip_h = p["load_unload_h"] + 2 * distance_km / p["avg_speed_kmh"]

    # Shipments/year needed to average 1 t/h over the year - independent of
    # trip duration (see truck_costs_per_capacity for the same derivation).
    n_shipment_per_tph_y = 8760.0 / mc_co2
    # Concurrent isotainer fleet needed to run that many shipments given
    # each ties one up for roundtrip_h hours, within t_op_h_per_y available.
    n_carrier_per_tph = n_shipment_per_tph_y * roundtrip_h / p["t_op_h_per_y"]
    n_isotainer_per_tph = n_carrier_per_tph * (1 + p["supplementary_isotainer_frac"])

    a_iso = capital_recovery_factor(discount_rate, p["isotainer_lifetime_y"])

    if p["isotainer_choice"] == "buy":
        capex_eur_per_tph_y = n_isotainer_per_tph * p["isotainer_cost_eur"] * a_iso
        iso_fixed_opex_per_tph_y = 0.0
    else:
        capex_eur_per_tph_y = 0.0
        iso_fixed_opex_per_tph_y = n_isotainer_per_tph * p["isotainer_rent_eur_per_y"]

    # Per-shipment rail-service fees scale with SHIPMENTS/year, not with the
    # concurrent fleet - a transshipment fee is charged once per isotainer
    # loaded/unloaded, regardless of how long that isotainer's trip takes.
    rail_service_eur_per_tph_y = n_shipment_per_tph_y * (
        2 * p["transshipment_eur_per_isotainer"] + p["weighing_eur_per_isotainer"]
        + p["transport_base_eur_per_isotainer"] + p["transport_eur_per_isotainer_km"] * distance_km
    )
    fixed_opex_eur_per_tph_y = rail_service_eur_per_tph_y + iso_fixed_opex_per_tph_y

    return {
        "capex_eur_per_tph_y": capex_eur_per_tph_y,
        "fixed_opex_eur_per_tph_y": fixed_opex_eur_per_tph_y,
    }


# ---------------------------------------------------------------------------
# 3. Demo: print a table and (if matplotlib available) a sanity-check plot
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    distances = [50, 100, 200, 500, 1000, 1500, 2000]
    discount_rate = 0.08

    print(f"Discount rate: {discount_rate:.0%}\n")
    header = (
        f"{'d [km]':>7} | {'Truck capex':>12} {'fix.opex':>10} {'[EUR/(t/h)/y]':>14} "
        f"{'variable':>10} {'[EUR/t]':>9} | "
        f"{'Train capex':>12} {'fix.opex':>10} {'[EUR/(t/h)/y]':>14}"
    )
    print(header)
    for d in distances:
        t = truck_costs_per_capacity(d, discount_rate)
        tr = train_costs_per_capacity(d, discount_rate)
        print(
            f"{d:7d} | {t['capex_eur_per_tph_y']:12,.0f} {t['fixed_opex_eur_per_tph_y']:10,.0f} "
            f"{'':>14} {t['variable_opex_eur_per_t']:10.2f} {'':>9} | "
            f"{tr['capex_eur_per_tph_y']:12,.0f} {tr['fixed_opex_eur_per_tph_y']:10,.0f}"
        )

    # -------------------------------------------------------------
    # Sanity check against published_fit(): if a fleet sized for
    # exactly 1 t/h ran continuously at that rate (100% utilisation,
    # i.e. mass_t_per_y = 1 t/h * 8760 h/y), the capacity-based capex
    # + fixed opex, spread over that mass, plus the variable rate,
    # should reproduce the same levelised cost as the original
    # mass-flow-based calibration (since it's the same underlying
    # TRUCK_PARAMS/TRAIN_PARAMS, just formulated continuously instead
    # of with ceil()). This is a consistency check, not a new fit.
    # -------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        d_arr = np.linspace(20, 2000, 200)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        capex_t, fixed_t, var_t = [], [], []
        capex_tr, fixed_tr = [], []
        lc_check_t, lc_check_tr, pub_t, pub_tr = [], [], [], []
        for d in d_arr:
            rt = truck_costs_per_capacity(d, discount_rate)
            rtr = train_costs_per_capacity(d, discount_rate)
            capex_t.append(rt["capex_eur_per_tph_y"])
            fixed_t.append(rt["fixed_opex_eur_per_tph_y"])
            var_t.append(rt["variable_opex_eur_per_t"])
            capex_tr.append(rtr["capex_eur_per_tph_y"])
            fixed_tr.append(rtr["fixed_opex_eur_per_tph_y"])

            # implied LC at 100% utilisation (mass_t_per_y = 1 t/h * 8760 h/y)
            mass_at_full_util = 1.0 * 8760
            lc_check_t.append((rt["capex_eur_per_tph_y"] + rt["fixed_opex_eur_per_tph_y"]) / 8760
                               + rt["variable_opex_eur_per_t"])
            lc_check_tr.append((rtr["capex_eur_per_tph_y"] + rtr["fixed_opex_eur_per_tph_y"]) / 8760)
            pub_t.append(published_fit(d, "container_truck")["levelised_cost_eur_per_t"])
            pub_tr.append(published_fit(d, "container_train")["levelised_cost_eur_per_t"])

        axes[0].plot(d_arr, capex_t, label="Truck capex")
        axes[0].plot(d_arr, fixed_t, label="Truck fixed opex")
        axes[0].plot(d_arr, capex_tr, "--", label="Train capex")
        axes[0].plot(d_arr, fixed_tr, "--", label="Train fixed opex")
        axes[0].set_xlabel("Distance [km]")
        axes[0].set_ylabel("EUR / (t/h) / y")
        axes[0].set_title("Capacity-based capex & fixed opex")
        axes[0].legend(fontsize=8)

        axes[1].plot(d_arr, var_t, color="C0")
        axes[1].set_xlabel("Distance [km]")
        axes[1].set_ylabel("EUR/t")
        axes[1].set_title("Truck variable opex rate\n(train's equivalent is folded into fixed opex)")

        axes[2].plot(d_arr, lc_check_t, color="C0", label="Truck: implied LC @ 100% utilisation")
        axes[2].plot(d_arr, pub_t, color="C0", linestyle="--", label="Truck: published fit")
        axes[2].plot(d_arr, lc_check_tr, color="C1", label="Train: implied LC @ 100% utilisation")
        axes[2].plot(d_arr, pub_tr, color="C1", linestyle="--", label="Train: published fit")
        axes[2].set_xlabel("Distance [km]")
        axes[2].set_ylabel("EUR/t")
        axes[2].set_title("Consistency check vs. published_fit()")
        axes[2].legend(fontsize=7)

        fig.suptitle(f"Capacity-based cost breakdown, discount rate = {discount_rate:.0%}")
        fig.tight_layout()
        plt.savefig("co2_container_transport_costs.png", dpi=150)
        print("\nPlot saved to co2_container_transport_costs.png")
    except ImportError:
        print("\n(matplotlib not installed - skipping plot)")
