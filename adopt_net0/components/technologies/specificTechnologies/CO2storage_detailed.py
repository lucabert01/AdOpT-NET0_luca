import copy
from warnings import warn
import scipy.io as sci
from pathlib import Path
import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit
import h5py
from ..technology import Technology, set_capex_model


from ..technology import Technology
from ....components.utilities import (
    annualize,
    set_discount_rate,
    get_attribute_from_dict,
    link_full_resolution_to_clustered,
)

class CO2storageDetailed(Technology):
    """
    This model resembles a permanent storage technology (sink). It takes energy and a main carrier (e.g. CO2, H2 etc)
    as inputs, and it has no output.


    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    **Constraint declarations:**

    - Maximal injection rate:

      .. math::
        Input_{t} \leq Inj_rate

    - Size constraint:

      .. math::
        E_{t} \leq Size

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} + Input_{t}

    - If an energy consumption for the injection is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

    """

    def __init__(self, tec_data):
        """
        Constructor

        :param dict tec_data: technology data
        """

        super().__init__(tec_data)

        self.component_options.emissions_based_on = "input"
        self.component_options.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]
        self.component_options.extra_equipment = tec_data["Extra_equipment"]

    def fit_technology_performance(self, climate_data, location):
        """
        Fits conversion technology type CO2storageDetailed and returns fitted parameters as a dict

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(CO2storageDetailed, self).fit_technology_performance(climate_data, location)

        time_steps = len(climate_data)

        if (
                "energy_consumption"
                in self.input_parameters.performance_data["performance"]
        ):
            self.processed_coeff.time_independent["energy_consumption"] = (
                self.input_parameters.performance_data["performance"][
                    "energy_consumption"
                ]
            )

        # Load the .mat file for ROM model of saline aquifer
        general_performance_data_path = Path(__file__).parent.parent.parent.parent
        aquifer_performance_data_path = (
            general_performance_data_path
            / "data/technology_data/Sink/SalineAquifer_data/matrices_for_ROM.mat"
        )

        self.processed_coeff.time_independent["matrices_data"] = sci.loadmat(aquifer_performance_data_path)
        phi = self.processed_coeff.time_independent['matrices_data']['phi']
        WI = self.processed_coeff.time_independent['matrices_data']['WI']
        mobApprox = self.processed_coeff.time_independent['matrices_data']['mobApprox']
        epsilon = self.processed_coeff.time_independent['matrices_data']['epsilon_mat']
        convert2bar = 10 ** 5
        cell_topwell = int(self.processed_coeff.time_independent['matrices_data']['cellTopWell'][0])
        lp = int(self.processed_coeff.time_independent['matrices_data']['lp']) # a mode is the equivalent of a grid block, but in the reduced space. So there is a pressure for every mode etc
        u = self.processed_coeff.time_independent['matrices_data']['u']
        g = 9.81
        delta_h = 1000
        self.processed_coeff.time_independent["rho_co2_surface"] = 505.71
        # TODO: check if using rho_co2_surface is fine (and not an average value)
        hydrostatic_pressure = g * delta_h * self.processed_coeff.time_independent["rho_co2_surface"] / convert2bar
        bhp_t0 = sum(phi[cell_topwell - 1, k] * epsilon[0, k]
                                               for k in range(0, lp)) + u[0,1]/(WI*mobApprox)/convert2bar
        whp_t0 = bhp_t0 - hydrostatic_pressure
        self.processed_coeff.time_independent["hydrostatic_pressure"] = hydrostatic_pressure
        # Perform pump interpolation
        # TODO: calculate value for p_loss in the offshore pipeline based on an assumed flowrate
        offshore_transport = {}
        offshore_transport["p_pump_in"] = 90 # Inlet pressure in bar (constant)
        offshore_transport["p_loss_offshorepipeline"] = 10 # Inlet pressure in bar (constant)
        p_pump_in = offshore_transport["p_pump_in"]
        p_loss_offshorepipeline = offshore_transport["p_loss_offshorepipeline"]
        nu = 1/self.processed_coeff.time_independent["rho_co2_surface"]
        eta_pump = 0.75
        pout_min = whp_t0[0,0] + p_loss_offshorepipeline
        range_delta_p = [pout_min, 125]  # in bar
        range_flowrate = [0, 3000]  # in t/day

        def compute_W_pump(m_dot, p_pump_out):
            return m_dot * nu * (p_pump_out - p_pump_in) / eta_pump * 0.1 / 3.6  # power in MWh/day

        # Create linearly spaced values within the given ranges
        p_pump_out_range = np.linspace(range_delta_p[0], range_delta_p[1], 100)  # Pump exit pressure  bar
        m_dot_range = np.linspace(range_flowrate[0], range_flowrate[1], 100)  # Mass flow rate  t/h

        # Create a meshgrid of values for p_pump_out and m_dot
        p_pump_out_grid, m_dot_grid = np.meshgrid(p_pump_out_range, m_dot_range)

        # Calculate the pump power for each pair of (m_dot, p_pump_out)
        W_pump_values = compute_W_pump(m_dot_grid, p_pump_out_grid)

        # fit W_pump with a plane W = a * m_dot + b * p_pump_out + constant
        def plane_model(X, a, b):
            p_pump_out, m_dot = X
            return a * m_dot + b * p_pump_out - b * p_pump_out_range[0]

        # Flatten the grids for curve fitting
        p_pump_out_flat = p_pump_out_grid.flatten()
        m_dot_flat = m_dot_grid.flatten()
        W_pump_flat = W_pump_values.flatten()

        # Perform curve fitting to get the coefficients a and b
        params, _ = curve_fit(plane_model, (p_pump_out_flat, m_dot_flat), W_pump_flat)


        self.processed_coeff.time_independent["pump_interp"] = params
        self.processed_coeff.time_independent["offshore_transport"] = offshore_transport

        # Get the values to save for result plotting and processing
        a, b = params
        p_pump_out_min = p_pump_out_range[0]
        data_to_save = {
            "a": a,
            "b": b,
            "p_pump_out_min": p_pump_out_min,
            "p_loss_offshore": offshore_transport["p_loss_offshorepipeline"],
            "nu": nu,
            "eta": eta_pump,
            "p_pump_in": offshore_transport["p_pump_in"]
        }
        file_path = Path(__file__).parent.parent.parent.parent.parent/"data_simulations_co2stor_linear/pump_coefficients.json"
        with open(file_path, "w") as file:
            json.dump(data_to_save, file, indent=4)
    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(CO2storageDetailed, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Main carrier (carrier to be stored)
        self.main_car = self.component_options.main_input_carrier

        # Input Bounds
        for car in self.component_options.input_carrier:
            if car == self.main_car:
                self.bounds["input"][car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps))
                     * self.input_parameters.performance_data["injection_rate_max"])
                )
            else:
                if "energy_consumption" in self.input_parameters.performance_data["performance"]:
                    energy_consumption = self.input_parameters.performance_data[
                        "performance"
                    ]["energy_consumption"]

                    self.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps)) * energy_consumption["in"][car],
                        )
                    )
                else:
                    size_pump_max = self.component_options.extra_equipment["size_pump_max"]
                    self.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps)) * size_pump_max,
                        )
                    )

    def construct_tech_model(
        self, b_tec: pyo.Block, data: dict, set_t: pyo.Set, set_t_clustered: pyo.Set
    ) -> pyo.Block:
        """
        Adds constraints to technology blocks for tec_type SINK, resembling a permanent storage technology

        :param b_tec:
        :param energyhub:
        :return: b_tec
        """

        super(CO2storageDetailed, self).construct_tech_model(b_tec, data, set_t, set_t_clustered)

        # DATA OF TECHNOLOGY
        config = data["config"]
        coeff_ti = self.processed_coeff.time_independent


        # Additional decision variables
        b_tec.var_storage_level = pyo.Var(
            set_t,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        # Size constraint
        def init_size_constraint(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = pyo.Constraint(set_t, rule=init_size_constraint)

        # Constraint storage level
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.modelled_with_full_res
        ):

            def init_storage_level(const, t):
                if t == 1:
                    return (
                        b_tec.var_storage_level[t]
                        == self.input[self.sequence[t - 1], self.main_car]
                    )
                else:
                    return (
                        b_tec.var_storage_level[t]
                        == b_tec.var_storage_level[t - 1]
                        + self.input[self.sequence[t - 1], self.main_car]
                    )

        else:

            def init_storage_level(const, t):
                if t == 1:
                    return b_tec.var_storage_level[t] == self.input[t, self.main_car]
                else:
                    return (
                        b_tec.var_storage_level[t]
                        == b_tec.var_storage_level[t - 1] + self.input[t, self.main_car]
                    )

            b_tec.const_storage_level = pyo.Constraint(set_t, rule=init_storage_level)


        # Create sets for allowing the reduced order model (ROM) to be run for a reduced number of timesteps (periods)
        length_t_red = self.input_parameters.performance_data['time_step_length']
        num_reduced_period = int(np.ceil(len(self.set_t_full) / length_t_red))
        b_tec.set_t_reduced = pyo.Set(initialize=range(1, num_reduced_period + 1))
        # TODO: check value of rho_co2_surface
        rho_co2_surface = coeff_ti["rho_co2_surface"] # [kg/m3] density of CO2 at surface conditions
        convert_inj_rate = 1/(rho_co2_surface*3.6) # converts t/h to m3/s, which is the unit required in the TPWL-POD model

        def init_reduced_set_t(set, t_red):
            return [x + (t_red-1)*length_t_red for x in list(range(1, length_t_red+1))]
        b_tec.set_t_for_reduced_period = pyo.Set(b_tec.set_t_reduced, initialize=init_reduced_set_t)

        b_tec.var_average_inj_rate = pyo.Var(b_tec.set_t_reduced, domain=pyo.NonNegativeReals,
                               bounds=[0, self.input_parameters.performance_data["injection_rate_max"]])

        # Var_average_inj_rate in m3/s
        def init_average_inj_rate(const, t_red):
            if t_red * length_t_red <= max(self.set_t_full):
                return b_tec.var_average_inj_rate[t_red] == sum(self.input[t, self.main_car]
                                                                for t in list(range((t_red -1) * length_t_red +1,
                                                                                    t_red * length_t_red+1)))/length_t_red*convert_inj_rate/24
            else:
                leftover_t_step = max(self.set_t_full) - (t_red-1) * length_t_red
                return b_tec.var_average_inj_rate[t_red] == sum(self.input[t, self.main_car]
                                                                for t in list(range((t_red-1) * length_t_red,
                                                                                    (t_red-1) * length_t_red +leftover_t_step+1)))/leftover_t_step*convert_inj_rate/24

        b_tec.const_average_inj = pyo.Constraint(b_tec.set_t_reduced, rule = init_average_inj_rate)

        # Setting up the ROM for the evolution of bottom-hole pressure
        ltot = int(coeff_ti['matrices_data']['ltot']) # this is actually the number of eigenvectors retrieved from the POD (ltot=lp+ls)
        lp = int(coeff_ti['matrices_data']['lp']) # a mode is the equivalent of a grid block, but in the reduced space. So there is a pressure for every mode etc
        cell_topwell = int(coeff_ti['matrices_data']['cellTopWell'][0])
        scale_down = 1
        epsilon = coeff_ti['matrices_data']['epsilon_mat']
        u = coeff_ti['matrices_data']['u']
        weight_distance_cwi = coeff_ti['matrices_data']['weight']
        invJred = coeff_ti['matrices_data']['invJred_mat']
        Ared = coeff_ti['matrices_data']['Ared_mat']
        Bred = coeff_ti['matrices_data']['Bred_mat']
        phi = coeff_ti['matrices_data']['phi']
        WI = coeff_ti['matrices_data']['WI']
        mobApprox = coeff_ti['matrices_data']['mobApprox']
        convert2bar = 10**5

        b_tec.set_modes = pyo.Set(initialize=range(1, lp +1)) # also refers to the eigenvectors retrieved and not to grid blocks
        # TODO: fix bounds of all variables
        b_tec.var_states = pyo.Var(b_tec.set_t_reduced, b_tec.set_modes, within= pyo.Reals,
                                   bounds=(epsilon.min()*3, epsilon.max()*3))
        b_tec.var_bhp = pyo.Var(b_tec.set_t_reduced, within=pyo.Reals, bounds=(100, 250))



        def init_states_time0(const, mode):
             return b_tec.var_states[1, mode] == epsilon[0, mode-1]

        b_tec.const_states_time0 = pyo.Constraint(b_tec.set_modes, rule=init_states_time0)

        # Calculate distance between states and training run
        search_range = 1
        s_search_indices = range(-search_range, search_range + 1)
        s_abs_index = [0, 1]
        # TODO: add proper bounds to the distance variables
        b_tec.var_distance = pyo.Var(b_tec.set_t_reduced, s_search_indices,
                                     within=pyo.Reals, bounds=(0, 1000))
        b_tec.var_d_min = pyo.Var(b_tec.set_t_reduced,
                                  within=pyo.Reals, bounds=(0, 100))
        # Distance cumulative injection
        b_tec.var_d_cuminj = pyo.Var(b_tec.set_t_reduced, s_search_indices,
                                     within=pyo.Reals, bounds=(0, 1000))
        # auxiliary for abs value
        b_tec.var_d_cuminj_auxpos = pyo.Var(b_tec.set_t_reduced, s_search_indices,
                                            within=pyo.Reals, bounds=(0, 1000))
        b_tec.var_d_cuminj_auxneg = pyo.Var(b_tec.set_t_reduced, s_search_indices,
                                            within=pyo.Reals, bounds=(0, 1000))

        # Absolute value for cumulative injection distance disjunction
        # note: we use the inj_rate instead of inj_rate*length_timestep because we use constant timsteps, so they cancel out in the equation

        self.big_m_transformation_required = 1

        def init_abs_dis_cuminj(dis, t_red, t_search, ind):
            def init_dist_pos_cuminj(const):
                if (t_red + t_search >= 1) and (t_red + t_search <= max(b_tec.set_t_reduced)):
                    return (
                            b_tec.var_d_cuminj_auxpos[t_red, t_search]
                            == (sum(b_tec.var_average_inj_rate[k] for k in range(1, t_red))
                                - sum(u[0, k - 1] for k in range(1, t_red + t_search))) / (
                            (sum(u[0, k - 1] for k in range(1, t_red + t_search)) + 0.001)
                            ) * ind
                    )
                else:
                    return pyo.Constraint.Skip

            dis.const_dist_cuminj_pos = pyo.Constraint(rule=init_dist_pos_cuminj)

            def init_dist_neg_cuminj(const):
                if (t_red + t_search >= 1) and (t_red + t_search <= max(b_tec.set_t_reduced)):
                    return (
                            b_tec.var_d_cuminj_auxneg[t_red, t_search]
                            == (-1) * (sum(b_tec.var_average_inj_rate[k] for k in range(1, t_red))
                                       - sum(u[0, k - 1] for k in range(1, t_red + t_search))) /
                            ((sum(u[0, k - 1] for k in range(1, t_red + t_search)) + 0.001))
                            * (1 - ind)
                    )
                else:
                    return pyo.Constraint.Skip

            dis.const_dist_cuminj_neg = pyo.Constraint(rule=init_dist_neg_cuminj)

        b_tec.dis_abs_dist_cuminj = gdp.Disjunct(
            b_tec.set_t_reduced, s_search_indices, s_abs_index, rule=init_abs_dis_cuminj
        )

        # Bind disjuncts for absolute value of cumulative injection distance
        def bind_disjunctions_abs_dist_cuminj(dis, t_red, t_search):
            return [b_tec.dis_abs_dist_cuminj[t_red, t_search, i] for i in s_abs_index]

        b_tec.disjunction_abs_dist_cuminj = gdp.Disjunction(
            b_tec.set_t_reduced, s_search_indices, rule=bind_disjunctions_abs_dist_cuminj
        )

        # Complete absolute value with d_cuminj = d_cuminj_pos + d_cuminj_neg
        def init_const_dcuminj_aux(const, t_red, t_search):
            return (b_tec.var_d_cuminj[t_red, t_search] == b_tec.var_d_cuminj_auxpos[t_red, t_search]
                    + b_tec.var_d_cuminj_auxneg[t_red, t_search])

        b_tec.const_abs_dcuminj_aux = pyo.Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_const_dcuminj_aux)

        def init_d_tot(const, t_red, t_search):
            return b_tec.var_distance[t_red, t_search] == b_tec.var_d_cuminj[t_red, t_search]
        b_tec.const_d_tot = pyo.Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_d_tot)


        # Find minimum distance per timestep
        def init_upper_bound_dmin(const, t_red, t_search):
            if (t_red + t_search >= 1) and (t_red + t_search <= max(b_tec.set_t_reduced)):
                return b_tec.var_d_min[t_red] <= b_tec.var_distance[t_red, t_search]
            else:
                return pyo.Constraint.Skip

        b_tec.const_upper_bound_dmin = pyo.Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_upper_bound_dmin)



        # disjunction
        def init_min_dist(dis, t_red, t_search):
            def init_lower_bound_dmin(const):
                if (t_red + t_search >= 1) and (t_red + t_search <= max(b_tec.set_t_reduced)):
                    return (
                        b_tec.var_d_min[t_red]
                        >= b_tec.var_distance[t_red, t_search]
                    )
                else:
                    # this makes sure that unwanted t_search are not picked (first
                    # and last time interval)
                    return b_tec.var_d_min[t_red] >= sum(b_tec.var_distance[t_red,
                    t_search] for t_search in s_search_indices)

            dis.const_lower_bound = pyo.Constraint(rule=init_lower_bound_dmin)


            #TPWL equation (note that t_red+t_search is the equivalent of i+1 in the paper)
            def init_states_calc(const, mode):
                if t_red > 1:
                    return  (b_tec.var_states[t_red, mode] == epsilon[t_red + t_search -1, mode-1]
                            - sum(invJred[t_red + t_search -1, mode-1, j-1]* (
                            sum(Ared[t_red + t_search -1, j-1, k-1] * (b_tec.var_states[t_red-1, k] -
                                                                          epsilon[t_red + t_search -2, k-1]) for k in b_tec.set_modes)
                            +
                                Bred[t_red + t_search - 1, j - 1] * (b_tec.var_average_inj_rate[t_red] -
                                                                               u[0,t_red + t_search -1])
                                )
                            for j in b_tec.set_modes))
                else:
                    return pyo.Constraint.Skip

            dis.const_states_calc = pyo.Constraint(b_tec.set_modes, rule=init_states_calc)

        b_tec.dis_min_distance = gdp.Disjunct(
            b_tec.set_t_reduced, s_search_indices, rule=init_min_dist
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t_red):
            return [b_tec.dis_min_distance[t_red, i] for i in s_search_indices]
        b_tec.disjunction_min_distance = gdp.Disjunction(b_tec.set_t_reduced, rule=bind_disjunctions)

        # rewrite the states in the new base (only done for the cell of interest - top well)
        def init_retrieve_bhp(const, t_red):
            return b_tec.var_bhp[t_red] == sum(phi[cell_topwell - 1, k - 1] * b_tec.var_states[t_red, k]
                                               for k in b_tec.set_modes) + b_tec.var_average_inj_rate[t_red]/(WI*mobApprox)/convert2bar

        b_tec.const_retrieve_bhp = pyo.Constraint(b_tec.set_t_reduced, rule=init_retrieve_bhp)

        # limit bhp to prevent fracture
        b_tec.para_p_max = 180

        def init_p_max(const, t_red):
            return b_tec.var_bhp[t_red] <= b_tec.para_p_max
        b_tec.const_limit_bhp = pyo.Constraint(b_tec.set_t_reduced, rule=init_p_max)

        b_tec.var_pwellhead = pyo.Var(b_tec.set_t_reduced, within=pyo.NonNegativeReals)
        # TODO: check if using rho_co2_surface is fine (and not an average value)
        hydrostatic_pressure = self.processed_coeff.time_independent["hydrostatic_pressure"]
        def init_pwellhad(const, t_red):
            return b_tec.var_pwellhead[t_red] == b_tec.var_bhp[t_red] - hydrostatic_pressure
        b_tec.const_pwellhead = pyo.Constraint(b_tec.set_t_reduced, rule=init_pwellhad)







        # Electricity consumption pump
        def init_const_size_pump(const, t):
            return self.input[t,'electricity'] <= b_tec.var_size_pump
        b_tec.const_size_pump = pyo.Constraint(set_t, rule=init_const_size_pump)

        a_pump, b_pump = coeff_ti["pump_interp"]
        p_loss_offshorepipeline = coeff_ti["offshore_transport"]["p_loss_offshorepipeline"]
        # Mulitplied times 0.99 to make sure that el consumption is not negative when the mass flow rate is 0
        def init_pump(const,t):
            t_red = int((t-1)/length_t_red) +1
            return (self.input[t,'electricity'] == a_pump * self.input[t, self.main_car] + b_pump *
                    (b_tec.var_pwellhead[t_red] + p_loss_offshorepipeline)
                    - b_pump * ((b_tec.var_pwellhead[1]+p_loss_offshorepipeline)*0.99))

        b_tec.const_pump = pyo.Constraint(set_t, rule=init_pump)


        # # Electricity consumption for compression (to be used if Co2 is transported in gas phase
        # b_tec.var_pratio = pyo.Var(b_tec.set_t_reduced, within=pyo.NonNegativeReals, bounds=[0, 100])
        # def init_pressure_ratio(const, t):
        #     return b_tec.var_pratio[t] == 5
        #
        # b_tec.const_pratio1 = pyo.Constraint(expr=b_tec.var_pratio[1] == 5)
        # b_tec.const_pratio2 = pyo.Constraint(expr=b_tec.var_pratio[2] == 15)
        # nr_segments =2
        # b_tec.set_pieces = pyo.RangeSet(1, nr_segments)
        # eta = [0.2, 0.8]
        # pratio_range = [0, 10, 20]
        #
        # self.big_m_transformation_required = 1
        # def init_input_output(dis, t_red, ind):
        #     def init_output(const, t):
        #         if t <= max(self.set_t_full):
        #             return (
        #                 self.input[t, "electricity"]
        #                 == eta[ind - 1] * self.input[t, self.main_car]
        #         )
        #         else:
        #             return pyo.Constraint.Skip
        #
        #     dis.const_output = pyo.Constraint(b_tec.set_t_for_reduced_period[t_red], rule=init_output)
        #
        #     # Lower bound on the energy input (eq. 5)
        #     def init_input_low_bound(const):
        #         return (
        #             pratio_range[ind - 1]
        #             <= b_tec.var_pratio[t_red]
        #         )
        #
        #     dis.const_input_on1 = pyo.Constraint(rule=init_input_low_bound)
        #
        #     # Upper bound on the energy input (eq. 5)
        #     def init_input_up_bound(const):
        #         return (
        #             b_tec.var_pratio[t_red]
        #             <= pratio_range[ind]
        #         )
        #
        #     dis.const_input_on2 = pyo.Constraint(rule=init_input_up_bound)
        #
        # b_tec.dis_input_output = gdp.Disjunct(
        #     b_tec.set_t_reduced, b_tec.set_pieces, rule=init_input_output
        # )
        #
        # # Bind disjuncts
        # def bind_disjunctions(dis, t_red):
        #     return [b_tec.dis_input_output[t_red, i] for i in b_tec.set_pieces]
        # b_tec.disjunction_input_output = gdp.Disjunction(b_tec.set_t_reduced, rule=bind_disjunctions)


        # Add offshore pipeline

        def init_const_size_offshore_pipeline(const, t):
            return self.input[t, 'CO2captured'] <= b_tec.var_size_offshore_pipeline

        b_tec.const_size_offshore_pipeline = pyo.Constraint(set_t, rule=init_const_size_offshore_pipeline)


        return b_tec

    def _define_capex_parameters(self, b_tec, data: dict):
        """
        Defines the capex parameters for a flexible storage technology of capex model 4 (for all other capex models,
        the parameters are defined in the Technology class).

        For capex model 4:
        - para_unit_capex_charging_cap: the capex per unit of charging capacity (i.e., per MW charging).
        - para_unit_capex_charging_cap_annual: annualized capex per unit of charging capacity.
        - para_unit_capex_discharging_cap: the capex per unit of discharging capacity (i.e., per MW discharging).
        - para_unit_capex_discharging_cap_annual: annualized capex per unit of discharging capacity.
        - para_unit_capex: capex per unit of energy storage capacity (i.e., per MWh storage).
        - para_unit_capex_annual: annualized capex per unit of storage capacity.

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        super(CO2storageDetailed, self)._define_capex_parameters(b_tec, data)

        config = data["config"]
        economics = self.economics
        extra_equipment = self.component_options.extra_equipment
        b_tec.para_size_pump_max = self.component_options.extra_equipment["size_pump_max"]
        b_tec.para_size_offshore_pipeline_max = self.component_options.extra_equipment["size_offshore_pipeline_max"]

        b_tec.var_size_offshore_pipeline = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, b_tec.para_size_offshore_pipeline_max))
        b_tec.var_size_pump = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,b_tec.para_size_pump_max))

        # CAPEX PARAMETERS
        capex_model = set_capex_model(config, economics)

        if capex_model == 4:
            discount_rate = set_discount_rate(config, economics)
            fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
            annualization_factor = annualize(
                discount_rate, economics.lifetime, fraction_of_year_modelled
            )

            # additional parameters
            b_tec.para_unit_capex_pump = pyo.Param(
                domain=pyo.Reals,
                initialize=extra_equipment["pump_CAPEX"],
                mutable=True,
            )
            # unit cost in EUR/t/day (capacity)
            b_tec.para_unit_capex_offshore_pipeline = pyo.Param(
                domain=pyo.Reals,
                initialize=extra_equipment["offshore_pipeline_CAPEX"]* extra_equipment['offshore_pipeline_length'],
                mutable=True,
            )


            # annualized parameters
            b_tec.para_unit_capex_pump_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=(annualization_factor * b_tec.para_unit_capex_pump),
                mutable=True,
            )
            b_tec.para_unit_capex_offshore_pipeline_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=(
                    annualization_factor * b_tec.para_unit_capex_offshore_pipeline
                ),
                mutable=True,
            )

        return b_tec

    def _define_capex_variables(self, b_tec, data: dict):
        """
        Defines variables related to storage technology capex for a flexible storage technology (with capex model 4).

        - var_capex_charging_cap: the capex of the charging capacity (MW) installed.
        - var_capex_discharging_cap: the capex of the discharging capacity (MW) installed.
        - var_capex_energy_cap: the capex of the energy storage capacity (MWh) installed.

        For a non-flexible storage technology (with fixed charging and discharging capacities), the capex
        variables are taken from the parent class (Technology).

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """

        super(CO2storageDetailed, self)._define_capex_variables(b_tec, data)

        config = data["config"]
        economics = self.economics
        capex_model = set_capex_model(config, economics)

        if capex_model == 4:

            # BOUNDS
            max_capex_pump = (
                b_tec.para_unit_capex_pump_annual
                * b_tec.para_size_pump_max
            )
            max_capex_offshore_pipeline = (
                b_tec.para_unit_capex_offshore_pipeline_annual
                * b_tec.para_size_offshore_pipeline_max
            )

            # VARIABLES
            b_tec.var_capex_pump = pyo.Var(
                domain=pyo.NonNegativeReals, bounds=(0, max_capex_pump)
            )
            b_tec.var_capex_offshore_pipeline = pyo.Var(
                domain=pyo.NonNegativeReals, bounds=(0, max_capex_offshore_pipeline)
            )


        return b_tec

    def _define_capex_constraints(self, b_tec, data: dict):
        """
        Defines constraints related to capex for CO2 storage capex model 4, for which the capex
        comprises the capex for pump and the offshore pipeline.

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """

        super(CO2storageDetailed, self)._define_capex_constraints(b_tec, data)

        config = data["config"]
        economics = self.economics
        capex_model = set_capex_model(config, economics)

        if capex_model == 4:

            b_tec.const_capex_pump = pyo.Constraint(
                expr=b_tec.var_size_pump
                * b_tec.para_unit_capex_pump_annual
                == b_tec.var_capex_pump
            )
            b_tec.const_capex_offshore_pipeline = pyo.Constraint(
                expr=b_tec.var_size_offshore_pipeline
                * b_tec.para_unit_capex_offshore_pipeline_annual
                == b_tec.var_capex_offshore_pipeline
            )

            b_tec.const_capex_aux = pyo.Constraint(
                expr=b_tec.var_capex_pump
                + b_tec.var_capex_offshore_pipeline
                == b_tec.var_capex_aux
            )

        return b_tec


    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies operations after optimization

        :param Block b_tec: technology model block
        :param h5py.Group h5_group: technology model block
        """
        super(CO2storageDetailed, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "storage_level",
            data=[model_block.var_storage_level[t].value for t in self.set_t_full],
        )

        h5_group.create_dataset(
            "bhp",
            data=[model_block.var_bhp[t_red].value for t_red in model_block.set_t_reduced for t
         in model_block.set_t_for_reduced_period[t_red]]
        )

        h5_group.create_dataset(
            "average_inj_rate",
            data=[model_block.var_average_inj_rate[t_red].value for t_red in model_block.set_t_reduced for t
         in model_block.set_t_for_reduced_period[t_red]]
        )

        h5_group.create_dataset(
            "whp",
            data=[model_block.var_pwellhead[t_red].value for t_red in model_block.set_t_reduced for t
         in model_block.set_t_for_reduced_period[t_red]]
        )

    def write_results_tec_design(self, h5_group: h5py.Group, model_block: pyo.Block):
        """
        Function to report results of technologies design after optimization

        :param  h5py.Group h5_group: h5 file structure
        :param Block b_tec: technology model block
        """
        super(CO2storageDetailed, self).write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset(
            "size_pump", data=[model_block.var_size_pump.value]
        )
