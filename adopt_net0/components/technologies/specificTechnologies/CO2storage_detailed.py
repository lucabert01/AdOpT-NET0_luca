import copy
from warnings import warn
import scipy.io as sci
from pathlib import Path
import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np
import pandas as pd

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

        # Load the .mat file
        self.processed_coeff.time_independent["matrices_data"] = sci.loadmat("C:/Users/0954659\OneDrive - Universiteit Utrecht/Documents/AdOpT-NET0_luca/adopt_net0/data/technology_data/Sink/SalineAquifer_data/matrices_for_ROM.mat")


        # Adding the necessary Jacobians
        length_t_red = self.input_parameters.performance_data['time_step_length']
        num_reduced_period = int(np.ceil(time_steps / length_t_red))
        nb = int(self.processed_coeff.time_independent["matrices_data"]["ltot"])*2
        self.processed_coeff.time_independent["jacobian"] = np.random.rand(num_reduced_period, nb, nb)
        self.processed_coeff.time_independent["jacobian"][1:2, :, :] = 0
        self.processed_coeff.time_independent["jacobian_injection"] = np.random.rand(num_reduced_period, nb)

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
        length_t_red = self.input_parameters.performance_data["performance"]['time_step_length']
        num_reduced_period = int(np.ceil(len(self.set_t) / length_t_red))
        b_tec.set_t_reduced = pyo.Set(initialize=range(1, num_reduced_period + 1))
        def init_reduced_set_t(set, t_red):
            return [x + (t_red-1)*length_t_red for x in list(range(1, length_t_red+1))]
        b_tec.set_t_for_reduced_period = pyo.Set(b_tec.set_t_reduced, initialize=init_reduced_set_t)

        # TODO: convert max injection_rate_max to m3/s
        b_tec.var_average_inj_rate = pyo.Var(b_tec.set_t_reduced, domain=pyo.NonNegativeReals,
                               bounds=[0, self.performance_data["injection_rate_max"]])

        # TODO: convert max pyo.Var_average_inj_rate to m3/s
        def init_average_inj_rate(const, t_red):
            if t_red * length_t_red <= max(self.set_t):
                return b_tec.var_average_inj_rate[t_red] == sum(self.input[t, self.main_car]
                                                                for t in list(range(1, t_red * length_t_red+1)))/length_t_red
            else:
                leftover_t_step = max(self.set_t) - (t_red-1) * length_t_red
                return b_tec.var_average_inj_rate[t_red] == sum(self.input[t, self.main_car]
                                                                for t in list(range(1, leftover_t_step+1)))/leftover_t_step

        b_tec.const_average_inj = pyo.Constraint(b_tec.set_t_reduced, rule = init_average_inj_rate)

        # Setting up the ROM for the evolution of bottom-hole pressure
        nb = int(coeff_ti['matrices_data']['ltot']) # this is actually the number of eigenvectors retrieved from the POD
        b_tec.set_grid_blocks = pyo.Set(initialize=range(1, nb +1)) # also refers to the eigenvectors retrieved and not to grid blocks
        # TODO: fix bounds var_states
        # TODO: rescale var_states (only pressure)
        b_tec.var_states = pyo.Var(b_tec.set_t_reduced, b_tec.set_grid_blocks, within= pyo.Reals,
                                   bounds=(-100000000000, 10000000000))
        b_tec.var_bhp = pyo.Var(b_tec.set_t_reduced, within=pyo.Reals)
        cell_topwell = 2
        jac = coeff_ti['matrices_data']['jacobian']
        jac_inj = coeff_ti['matrices_data']['jacobian_injection']

        epsilon = coeff_ti['matrices_data']['epsilon_mat']
        u = coeff_ti['matrices_data']['u']
        weight = coeff_ti['matrices_data']['weight']
        invJred = coeff_ti['matrices_data']['invJred_mat']
        Ared = coeff_ti['matrices_data']['Ared_mat']
        Bred = coeff_ti['matrices_data']['Bred_mat']
        phi = coeff_ti['matrices_data']['phi']

        # Approximate bhp by identifying the top well cell in the states
        def init_approx_bhp(const, t_red):
            return b_tec.var_bhp[t_red] == b_tec.var_states[t_red, cell_topwell]
        b_tec.const_approx_bhp = pyo.Constraint(b_tec.set_t_reduced, rule=init_approx_bhp)

        # Calculate distance between states and training run
        search_range = 1
        s_search_indices = range(-search_range, search_range + 1)
        # TODO: add proper bounds to the distance variables
        b_tec.var_distance = pyo.Var(b_tec.set_t_reduced, s_search_indices,
                                     within=pyo.Reals, bounds=(-1000000000, 10000000000))
        b_tec.var_d_min = pyo.Var(b_tec.set_t_reduced,
                                  within=pyo.Reals, bounds=(-10000000000, 100000000000))
        # TODO: add distance calculations
        def init_distance_calc(const, t_red, t_search):
            if t_red + t_search >= 1 and t_red + t_search <= max(b_tec.set_t_reduced):
                return b_tec.var_distance[t_red, t_search] == t_red + t_search
            else:
                return b_tec.var_distance[t_red, t_search] == 44444
        b_tec.const_distance_calc = pyo.Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_distance_calc)

        # Find minimum distance per timestep
        def init_upper_bound_dmin(const, t_red, t_search):
            return b_tec.var_d_min[t_red] <= b_tec.var_distance[t_red, t_search]
        b_tec.const_upper_bound_dmin = pyo.Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_upper_bound_dmin)


        self.big_m_transformation_required = 1

        def init_min_dist(dis, t_red, t_search):
            def init_lower_bound_dmin(const):
                return (
                    b_tec.var_d_min[t_red]
                    >= b_tec.var_distance[t_red, t_search]
                )

            dis.const_lower_bound = pyo.Constraint(rule=init_lower_bound_dmin)

            #TPWL equation (note that (t_red + t_search) is the equivalent of i+1 in the paper)
            def init_states_calc(const, cell):
                if t_red ==1:
                    return b_tec.var_states[t_red, cell] == epsilon[1, cell-1]
                elif t_red + t_search >= 1 and t_red + t_search <= max(b_tec.set_t_reduced) and t_red > 1:
                        return  (b_tec.var_states[t_red, cell] == epsilon[t_red + t_search -1, cell-1] -
                                sum(invJred[t_red + t_search -1, cell-1, j-1]*
                                sum(Ared[t_red + t_search -1, cell-1, k-1] * (b_tec.var_states[t_red-1, cell] -
                                                                              epsilon[t_red + t_search -2, cell-1]) +
                                    Bred[t_red + t_search - 1, k - 1] * (b_tec.var_average_inj_rate[t_red] -
                                                                                   u[0,t_red + t_search-1])
                                    for k in b_tec.set_grid_blocks)
                                for j in b_tec.set_grid_blocks))
                else:
                    return pyo.Constraint.Skip

            dis.const_states_calc = pyo.Constraint( b_tec.set_grid_blocks, rule=init_states_calc)

        b_tec.dis_min_distance = gdp.Disjunct(
            b_tec.set_t_reduced, s_search_indices, rule=init_min_dist
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t_red):
            return [b_tec.dis_min_distance[t_red, i] for i in s_search_indices]
        b_tec.disjunction_min_distance = gdp.Disjunction(b_tec.set_t_reduced, rule=bind_disjunctions)


        # Electricity consumption for compression
        b_tec.var_pwellhead = pyo.Var(b_tec.set_t_reduced, within=pyo.NonNegativeReals)
        b_tec.var_pratio = pyo.Var(b_tec.set_t_reduced, within=pyo.NonNegativeReals, bounds=[0, 100])
        # TODO add constraint on relationship bhp and wellhead pressure

        def init_pressure_ratio(const, t):
            return b_tec.var_pratio[t] == 5

        b_tec.const_pratio1 = pyo.Constraint(expr=b_tec.var_pratio[1] == 5)
        b_tec.const_pratio2 = pyo.Constraint(expr=b_tec.var_pratio[2] == 15)
        nr_segments =2
        b_tec.set_pieces = pyo.RangeSet(1, nr_segments)
        eta = [0.2, 0.8]
        pratio_range = [0, 10, 20]


        self.big_m_transformation_required = 1
        def init_input_output(dis, t_red, ind):
            def init_output(const, t):
                if t <= max(self.set_t):
                    return (
                        self.input[t, "electricity"]
                        == eta[ind - 1] * self.input[t, self.main_car]
                )
                else:
                    return pyo.Constraint.Skip

            dis.const_output = pyo.Constraint(b_tec.set_t_for_reduced_period[t_red], rule=init_output)

            # Lower bound on the energy input (eq. 5)
            def init_input_low_bound(const):
                return (
                    pratio_range[ind - 1]
                    <= b_tec.var_pratio[t_red]
                )

            dis.const_input_on1 = pyo.Constraint(rule=init_input_low_bound)

            # Upper bound on the energy input (eq. 5)
            def init_input_up_bound(const):
                return (
                    b_tec.var_pratio[t_red]
                    <= pratio_range[ind]
                )

            dis.const_input_on2 = pyo.Constraint(rule=init_input_up_bound)

        b_tec.dis_input_output = pyo.Disjunct(
            b_tec.set_t_reduced, b_tec.set_pieces, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t_red):
            return [b_tec.dis_input_output[t_red, i] for i in b_tec.set_pieces]
        b_tec.disjunction_input_output = pyo.Disjunction(b_tec.set_t_reduced, rule=bind_disjunctions)




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


    #def convert2matrix(self, ):