from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np
import scipy.io as sci

from src.components.technologies.utilities import FittedPerformance
from src.components.technologies.technology import Technology


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
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

    def fit_technology_performance(self, climate_data, location):
        """
        Fits conversion technology type SINK and returns fitted parameters as a dict

        :param node_data: contains data on demand, climate data, etc.
        """

        time_steps = len(climate_data)

        # Main carrier (carrier to be stored)
        self.main_car = self.performance_data["main_input_carrier"]

        # Input Bounds
        for car in self.performance_data["input_carrier"]:
            if car == self.performance_data["main_input_carrier"]:
                self.fitted_performance.bounds["input"][car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
            else:
                if "energy_consumption" in self.performance_data["performance"]:
                    energy_consumption = self.performance_data["performance"][
                        "energy_consumption"
                    ]
                    self.fitted_performance.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps)) * energy_consumption["in"][car],
                        )
                    )

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 0

        # Adding the necessary Jacobians
        length_t_red = self.performance_data['time_step_length']
        num_reduced_period = int(np.ceil(time_steps / length_t_red))
        nb = self.performance_data['num_grid_blocks']*2
        self.performance_data['time_step_length']
        self.fitted_performance.jacobian = np.random.rand(num_reduced_period, nb, nb)
        self.fitted_performance.jacobian[1:2, :, :] = 0
        self.fitted_performance.jacobian_injection = np.random.rand(num_reduced_period, nb)


    def construct_tech_model(
        self, b_tec: Block, data: dict, set_t: Set, set_t_clustered: Set
    ) -> Block:
        """
        Adds constraints to technology blocks for tec_type SINK, resembling a permanent storage technology

        :param b_tec:
        :param energyhub:
        :return: b_tec
        """

        super(CO2storageDetailed, self).construct_tech_model(b_tec, data, set_t, set_t_clustered)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        config = data["config"]


        # Additional decision variables
        b_tec.var_storage_level = Var(
            set_t,
            domain=NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        # Size constraint
        def init_size_constraint(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

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

            b_tec.const_storage_level = Constraint(set_t, rule=init_storage_level)

        # Maximal injection rate
        def init_maximal_injection(const, t):
            return (
                self.input[t, self.main_car]
                <= self.performance_data["injection_rate_max"]
            )

        b_tec.const_max_injection = Constraint(self.set_t, rule=init_maximal_injection)

        # Create sets for allowing the reduced order model (ROM) to be run for a reduced number of timesteps (periods)
        length_t_red = self.performance_data['time_step_length']
        num_reduced_period = int(np.ceil(len(self.set_t) / length_t_red))
        b_tec.set_t_reduced = Set(initialize=range(1, num_reduced_period + 1))
        def init_reduced_set_t(set, t_red):
            return [x + (t_red-1)*length_t_red for x in list(range(1, length_t_red+1))]
        b_tec.set_t_for_reduced_period = Set(b_tec.set_t_reduced, initialize=init_reduced_set_t)

        # TODO: convert max injection_rate_max to m3/s
        b_tec.var_average_inj_rate = Var(b_tec.set_t_reduced, domain=NonNegativeReals,
                               bounds=[0, self.performance_data["injection_rate_max"]])

        # TODO: convert max var_average_inj_rate to m3/s
        def init_average_inj_rate(const, t_red):
            if t_red * length_t_red <= max(self.set_t):
                return b_tec.var_average_inj_rate[t_red] == sum(self.input[t, self.main_car]
                                                                for t in list(range(1, t_red * length_t_red+1)))/length_t_red
            else:
                leftover_t_step = max(self.set_t) - (t_red-1) * length_t_red
                return b_tec.var_average_inj_rate[t_red] == sum(self.input[t, self.main_car]
                                                                for t in list(range(1, leftover_t_step+1)))/leftover_t_step

        b_tec.const_average_inj = Constraint(b_tec.set_t_reduced, rule = init_average_inj_rate)

        # Setting up the ROM for the evolution of bottom-hole pressure
        nb = self.performance_data['num_grid_blocks']
        b_tec.set_grid_blocks = Set(initialize=range(1, 2*nb +1))
        # TODO: fix bounds var_states
        # TODO: rescale var_states (only pressure)
        b_tec.var_states = Var(b_tec.set_t_reduced, b_tec.set_grid_blocks, within= NonNegativeReals, bounds=(0, 1000000))
        b_tec.var_bhp = Var(b_tec.set_t_reduced, within=NonNegativeReals)
        cell_topwell = 2
        jac = self.fitted_performance.jacobian
        jac_inj = self.fitted_performance.jacobian_injection

        # Approximate bhp by identifying the top well cell in the states
        def init_approx_bhp(const, t_red):
            return b_tec.var_bhp[t_red] == b_tec.var_states[t_red, cell_topwell]
        b_tec.const_approx_bhp = Constraint(b_tec.set_t_reduced, rule=init_approx_bhp)

        # Calculate distance between states and training run
        search_range = 1
        s_search_indices = range(-search_range, search_range + 1)
        # TODO: add proper bounds to the distance variables
        b_tec.var_distance = Var(b_tec.set_t_reduced, s_search_indices, within=NonNegativeReals, bounds=(0, 1000000))
        b_tec.var_d_min = Var(b_tec.set_t_reduced, within=NonNegativeReals, bounds=(0, 1000000))
        # TODO: add distance calculations
        def init_distance_calc(const, t_red, t_search):
            if t_red + t_search >= 1 and t_red + t_search <= max(b_tec.set_t_reduced):
                return b_tec.var_distance[t_red, t_search] == t_red + t_search
            else:
                return b_tec.var_distance[t_red, t_search] == 20000
        b_tec.const_distance_calc = Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_distance_calc)

        # Find minimum distance per timestep
        def init_upper_bound_dmin(const, t_red, t_search):
            return b_tec.var_d_min[t_red] <= b_tec.var_distance[t_red, t_search]
        b_tec.const_upper_bound_dmin = Constraint(b_tec.set_t_reduced, s_search_indices, rule=init_upper_bound_dmin)


        self.big_m_transformation_required = 1

        b_tec.var_test = Var(b_tec.set_t_reduced, within=NonNegativeReals, bounds=(0, 1000000))
        alpha = 1
        beta = 2
        bhp_initial = 10  # in Pascal
        injection = b_tec.var_average_inj_rate
        J = {-1: 1, 0: 10, 1:200}
        def init_min_dist(dis, t_red, t_search):
            def init_lower_bound_dmin(const):
                return (
                    b_tec.var_d_min[t_red]
                    >= b_tec.var_distance[t_red, t_search]
                )

            dis.const_lower_bound = Constraint(rule=init_lower_bound_dmin)

            # TPWL equation
            def init_states_calc(const,cell):
                if t_red + t_search >= 1 and t_red + t_search <= max(b_tec.set_t_reduced):
                    if t_red == 1:
                        return  b_tec.var_states[t_red, cell] == sum(jac[t_red + t_search-1, cell-1, k-1] for k in b_tec.set_grid_blocks)
                    else:
                        return  b_tec.var_states[t_red, cell] == sum(jac[t_red + t_search-1, cell-1, k-1] for k in b_tec.set_grid_blocks)
                else:
                    return Constraint.Skip
            dis.const_states_calc = Constraint(b_tec.set_grid_blocks, rule=init_states_calc)

        b_tec.dis_min_distance = Disjunct(
            b_tec.set_t_reduced, s_search_indices, rule=init_min_dist
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t_red):
            return [b_tec.dis_min_distance[t_red, i] for i in s_search_indices]
        b_tec.disjunction_min_distance = Disjunction(b_tec.set_t_reduced, rule=bind_disjunctions)


        # Electricity consumption for compression
        b_tec.var_pwellhead = Var(b_tec.set_t_reduced, within=NonNegativeReals)
        b_tec.var_pratio = Var(b_tec.set_t_reduced, within=NonNegativeReals, bounds=[0, 100])
        # TODO add constraint on relationship bhp and wellhead pressure

        def init_pressure_ratio(const, t):
            return b_tec.var_pratio[t] == 5

        b_tec.const_pratio1 = Constraint(expr=b_tec.var_pratio[1] == 5)
        b_tec.const_pratio2 = Constraint(expr=b_tec.var_pratio[2] == 15)
        nr_segments =2
        b_tec.set_pieces = RangeSet(1, nr_segments)
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
                    return Constraint.Skip

            dis.const_output = Constraint(b_tec.set_t_for_reduced_period[t_red], rule=init_output)

            # Lower bound on the energy input (eq. 5)
            def init_input_low_bound(const):
                return (
                    pratio_range[ind - 1]
                    <= b_tec.var_pratio[t_red]
                )

            dis.const_input_on1 = Constraint(rule=init_input_low_bound)

            # Upper bound on the energy input (eq. 5)
            def init_input_up_bound(const):
                return (
                    b_tec.var_pratio[t_red]
                    <= pratio_range[ind]
                )

            dis.const_input_on2 = Constraint(rule=init_input_up_bound)

        b_tec.dis_input_output = Disjunct(
            b_tec.set_t_reduced, b_tec.set_pieces, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t_red):
            return [b_tec.dis_input_output[t_red, i] for i in b_tec.set_pieces]
        b_tec.disjunction_input_output = Disjunction(b_tec.set_t_reduced, rule=bind_disjunctions)




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
