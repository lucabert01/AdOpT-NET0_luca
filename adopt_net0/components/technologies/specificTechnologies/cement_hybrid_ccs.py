import pyomo.environ as pyo
import pyomo.gdp as gdp
import pandas as pd
import numpy as np
from pathlib import Path
from ...utilities import annualize, set_discount_rate
from ..technology import Technology
from warnings import warn
import logging

log = logging.getLogger(__name__)


class CementHybridCCS(Technology):
    """
    Cement plant with hybrid CCS

    The plant has an oxyfuel combustion in the calciner and post-combustion capture with MEA afterward. The size
    of the oxyfuel correspond to the size of the cement plant, as it is built-in in the calciner. This size is
    generally a variable, but it can be fixed. The size of MEA and amount of CO2 captured (per h) by the MEA are
    variables of the optimization. The CO2 output of the oxyfuel is processed by a CPU, while the one of the MEA
    is processed by a compressor. The output phase (gas, liquid, supercritical) should be specified in the json file.
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "output"
        self.size_based_on = "output"
        self.main_output_carrier = tec_data["Performance"][
            "main_output_carrier"
        ]

    def _define_size(self, b_tec):
        """
        Defines variables and parameters related to technology size.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        super(CementHybridCCS, self)._define_size(b_tec)

        b_tec.var_size_mea = pyo.Var(
            within=pyo.NonNegativeReals,
            bounds=[0, self.processed_coeff.time_independent["size_max_mea"]],
        )

        return b_tec

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits the technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(CementHybridCCS, self).fit_technology_performance(climate_data, location)

        performance_data_path = Path(__file__).parent.parent.parent.parent
        performance_data_path = (
            performance_data_path
            / "database/templates/technology_data/Industrial/CementHybridCCS_data/cement_sheet.xlsx"
        )

        performance_data_oxy_mea = pd.read_excel(
            performance_data_path, sheet_name="energy_oxy_mea", index_col=0
        )


        self.processed_coeff.time_independent["alpha_oxy"] = (
            performance_data_oxy_mea.loc["alpha_oxy", "value"]
        )
        self.processed_coeff.time_independent["el_recovery_oxy"] = (
            performance_data_oxy_mea.loc["el_recovery_oxy", "value"]
        )
        self.processed_coeff.time_independent["alpha_mea"] = (
            performance_data_oxy_mea.loc["alpha_mea", "value"]
        )
        self.processed_coeff.time_independent["beta_oxy"] = (
            performance_data_oxy_mea.loc["beta_oxy", "value"]
        )

        performance_data_cpu_compressor = pd.read_excel(
            performance_data_path, sheet_name="energy_cpu_compressor", index_col=0
        )

        self.processed_coeff.time_independent["el_cons_cpu"] = (
            performance_data_cpu_compressor.loc["el_cons_cpu"]
        )
        self.processed_coeff.time_independent["el_cons_compressor"] = (
            performance_data_cpu_compressor.loc["el_cons_compressor"]
        )

        valid_phases_compression = {"gas", "liquid", "supercritical"}

        if self.performance_data["co2_out_is_compressed"]:
            phase = self.performance_data["phase_of_co2_out"]
            if phase not in valid_phases_compression:
                raise ValueError(
                    f"Invalid value for 'phase_of_co2_out': '{phase}'. Must be one of {valid_phases_compression}."
                )

            self.processed_coeff.time_independent["alpha_oxy"] = (
                performance_data_oxy_mea.loc["alpha_oxy", "value"]
                + self.processed_coeff.time_independent["el_cons_cpu"][phase]
            )

            self.processed_coeff.time_independent["alpha_mea"] = (
                performance_data_oxy_mea.loc["alpha_mea", "value"]
                + self.processed_coeff.time_independent["el_cons_compressor"][phase]
            )

        self.processed_coeff.time_independent["size_max_mea"] = (
            self.processed_coeff.time_independent["size_max"]
            * self.performance_data["performance"]["tCO2_tclinker"]
            * (1 - self.performance_data["performance"]["CCR_oxy"])
            * self.performance_data["performance"]["CCR_mea"]
        )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(CementHybridCCS, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)
        emissions_clinker = self.performance_data["performance"][
            "tCO2_tclinker"
        ]
        CCR_oxy = self.performance_data["performance"]["CCR_oxy"]
        CCR_mea = self.performance_data["performance"]["CCR_mea"]

        # Output Bounds
        self.bounds["output"]["CO2captured"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * emissions_clinker
                * (CCR_oxy + (1 - CCR_oxy) * CCR_mea),
            )
        )

        self.bounds["output"]["clinker"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps),
            )
        )

        # Input Bounds
        self.bounds["input"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * (
                    emissions_clinker
                    * CCR_oxy
                    * self.processed_coeff.time_independent["alpha_oxy"]
                    + emissions_clinker
                    * (1 - CCR_oxy)
                    * CCR_mea
                    * self.processed_coeff.time_independent["alpha_mea"]
                ),
            )
        )


    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type CementHybridCCS

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(CementHybridCCS, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )


        # add additional constraints for performance type 2 (min. part load)
        if self.performance_function_type == 2:
            b_tec = self._performance_function_type_2(b_tec)



        # Size constraint
        prod_capacity_clinker = self.performance_data[
            "prod_capacity_clinker"
        ]
        emissions_clinker = self.performance_data["performance"][
            "tCO2_tclinker"
        ]
        alpha_oxy = self.processed_coeff.time_independent["alpha_oxy"]
        el_recovery_oxy = self.processed_coeff.time_independent["el_recovery_oxy"]
        beta_oxy = self.processed_coeff.time_independent["beta_oxy"]
        alpha_mea = self.processed_coeff.time_independent["alpha_mea"]
        CCR_oxy = self.performance_data["performance"]["CCR_oxy"]
        CCR_mea = self.performance_data["performance"]["CCR_mea"]



        if self.performance_data["size_is_fixed"]:

            def init_size_clinker(const):
                return b_tec.var_size == prod_capacity_clinker

            b_tec.const_size_clinker = pyo.Constraint(rule=init_size_clinker)
            warn(
                f"The clinker capacity of {self.name} is currently fixed at {prod_capacity_clinker} t/h"
            )

        def init_size_constraint_mea(const, t):
            return b_tec.var_co2_captured_mea[t] <= b_tec.var_size_mea

        b_tec.const_size_mea = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint_mea
        )

        def init_size_mea_max_constraint(const):
            return (
                b_tec.var_size_mea
                <= b_tec.var_size * emissions_clinker * (1 - CCR_oxy) * CCR_mea
            )

        b_tec.const_size_mea_max = pyo.Constraint(rule=init_size_mea_max_constraint)

        def init_mea_operation_constraint(const, t):
            return (
                b_tec.var_co2_captured_mea[t]
                <= self.output[t, "clinker"]
                * emissions_clinker
                * (1 - CCR_oxy)
                * CCR_mea
            )

        b_tec.const_mea_operation = pyo.Constraint(
            self.set_t_performance, rule=init_mea_operation_constraint
        )

        def init_size_clinker_max_constraint(const, t):
            return self.output[t, "clinker"] <= b_tec.var_size

        b_tec.const_size_clinker_max = pyo.Constraint(
            self.set_t_performance, rule=init_size_clinker_max_constraint
        )

        # input-output correlations
        def init_input_output(const, t, car_input):
            if car_input == "electricity":
                return pyo.Constraint.Skip  # handled by disjunction above

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_input_carriers, rule=init_input_output
        )

        def init_disjunct_mea_active(dis, t):
            """MEA is installed (var_size_mea > 0)"""

            def init_elec(const):
                return (
                        self.input[t, "electricity"]
                        == self.output[t, "clinker"] * emissions_clinker * CCR_oxy * alpha_oxy
                        + b_tec.var_co2_captured_mea[t] * alpha_mea
                )

            dis.const_elec = pyo.Constraint(rule=init_elec)

            def init_size_bound(const):
                return b_tec.var_size_mea >= 0.001  # var_size_mea > 0

            dis.const_size_active = pyo.Constraint(rule=init_size_bound)

        def init_disjunct_mea_inactive(dis, t):
            """MEA is not installed (var_size_mea == 0)"""

            def init_elec(const):
                return (
                        self.input[t, "electricity"]
                        == self.output[t, "clinker"] * emissions_clinker * CCR_oxy * (alpha_oxy-el_recovery_oxy)
                )

            dis.const_elec = pyo.Constraint(rule=init_elec)

            def init_size_bound(const):
                return b_tec.var_size_mea == 0

            dis.const_size_inactive = pyo.Constraint(rule=init_size_bound)

        b_tec.dis_mea_active = gdp.Disjunct(self.set_t_performance, rule=init_disjunct_mea_active)
        b_tec.dis_mea_inactive = gdp.Disjunct(self.set_t_performance, rule=init_disjunct_mea_inactive)

        def bind_disjunction_mea(dis, t):
            return [b_tec.dis_mea_active[t], b_tec.dis_mea_inactive[t]]

        b_tec.disjunction_mea = gdp.Disjunction(self.set_t_performance, rule=bind_disjunction_mea)




        def init_output_output(const, t):
            return (
                self.output[t, "CO2captured"]
                == self.output[t, "clinker"] * emissions_clinker * CCR_oxy
                + b_tec.var_co2_captured_mea[t]
            )

        b_tec.const_output_output = pyo.Constraint(
            self.set_t_performance, rule=init_output_output
        )

        return b_tec

    def _define_input(self, b_tec, data: dict):
        """
        Defines input to a technology

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        # Technology related data
        c = self.processed_coeff.time_independent

        def init_input_bounds(bounds, t, car):
            return tuple(
                self.bounds["input"][car][self.sequence[t - 1] - 1, :]
                * self.processed_coeff.time_independent["size_max"]
            )

        b_tec.var_input = pyo.Var(
            self.set_t_global,
            b_tec.set_input_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_input_bounds,
        )

        return b_tec

    def _define_output(self, b_tec, data: dict):
        """
        Defines output to a technology

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        # Technology related data
        c = self.processed_coeff.time_independent

        def init_output_bounds(bounds, t, car):
            return tuple(
                self.bounds["output"][car][self.sequence[t - 1] - 1, :]
                * self.processed_coeff.time_independent["size_max"]
            )

        b_tec.var_output = pyo.Var(
            self.set_t_global,
            b_tec.set_output_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )

        b_tec.var_co2_captured_mea = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=[0, self.processed_coeff.time_independent["size_max_mea"]],
        )

        return b_tec

    def _define_emissions(self, b_tec):
        """
        Defines Emissions

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        emissions_clinker = self.performance_data["performance"][
            "tCO2_tclinker"
        ]

        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        def init_tec_emissions_pos(const, t):
            """emissions_pos = output * emissionfactor"""
            return (
                self.output[t, "clinker"] * emissions_clinker
                - self.output[t, "CO2captured"]
                == b_tec.var_tec_emissions_pos[t]
            )

        b_tec.const_tec_emissions_pos = pyo.Constraint(
            self.set_t_global, rule=init_tec_emissions_pos
        )

        def init_tec_emissions_neg(const, t):
            return b_tec.var_tec_emissions_neg[t] == 0

        b_tec.const_tec_emissions_neg = pyo.Constraint(
            self.set_t_global, rule=init_tec_emissions_neg
        )

        return b_tec

    def _define_capex_parameters(self, b_tec, data):
        """
        Defines the capex parameters. In this case, it is only the decommissioning costs

        :param b_tec:
        :param data:
        :return:
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )
        capex_data_path = Path(__file__).parent.parent.parent.parent
        capex_data_path = (
                capex_data_path
                / "database/templates/technology_data/Industrial/CementHybridCCS_data/cement_sheet.xlsx"
        )



        phase = self.performance_data["phase_of_co2_out"]

        capex_cpu_oxy_data = pd.read_excel(
            capex_data_path, sheet_name="capex_cpu_oxyfuel", index_col=0
        )
        capex_compressor_mea_data = pd.read_excel(
            capex_data_path, sheet_name="capex_compressor_mea", index_col=0
        )
        bp_y_capex_cpu_oxy = capex_cpu_oxy_data[phase].tolist()
        bp_y_capex_compressor_mea = capex_compressor_mea_data[
            phase].tolist()
        economics["piecewise_capex"]["bp_y"] = np.add(
            economics["piecewise_capex"]["bp_y"],
            bp_y_capex_cpu_oxy,
        )
        economics["other_economics"]["piecewise_capex_MEA"]["bp_y"] = np.add(
            economics["other_economics"]["piecewise_capex_MEA"]["bp_y"],
            bp_y_capex_compressor_mea,
        )

        if self.existing and not self.decommission == "impossible":
            b_tec.para_decommissioning_cost_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=annualization_factor * economics["decommission_cost"],
                mutable=True,
            )

        return b_tec

    def _define_capex_variables(self, b_tec, data: dict):
        """
        Defines variables related to technology capex.

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )


        def calculate_max_capex_oxy():
            max_capex = (
                                    max(economics["piecewise_capex"]["bp_y"])) * annualization_factor
            return (0, max_capex)

        def calculate_max_capex_mea():
            max_capex = (
                                max(economics["other_economics"]["piecewise_capex_MEA"]["bp_y"])) * annualization_factor
            return (0, max_capex)

        def calculate_max_capex():
            bounds_mea = calculate_max_capex_mea()
            bounds_oxy = calculate_max_capex_oxy()
            return tuple(map(sum, zip(bounds_mea, bounds_oxy)))

        b_tec.var_capex_oxy = pyo.Var(bounds=calculate_max_capex_oxy())
        b_tec.var_capex_mea = pyo.Var(bounds=calculate_max_capex_mea())
        b_tec.var_capex_aux = pyo.Var(bounds=calculate_max_capex())
        b_tec.var_capex = pyo.Var()

        return b_tec



    def _define_capex_constraints(self, b_tec, data):
        """
        Defines constraints related to capex.
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )
        b_tec.para_unit_capex = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["unit_capex"],
            mutable=True,
        )

        # capex oxyfuel as a piecewise function
        bp_x_oxy = economics["piecewise_capex"]["bp_x"]
        bp_y_annual_oxy = [
            y * annualization_factor
            for y in economics["piecewise_capex"]["bp_y"]
        ]
        # capex mea as piecewise or linear
        bp_x_mea = economics["other_economics"]["piecewise_capex_MEA"]["bp_x"]
        bp_y_annual_mea = [
            y * annualization_factor
            for y in economics["other_economics"]["piecewise_capex_MEA"]["bp_y"]
        ]

        self.big_m_transformation_required = 1
        if self.performance_data["size_is_fixed"]:
            size = self.performance_data["prod_capacity_clinker"]
            b_tec.const_capex_oxy = pyo.Constraint(
                expr=b_tec.var_capex_oxy == np.interp(size, bp_x_oxy, bp_y_annual_oxy)
            )
            b_tec.const_capex_mea = pyo.Piecewise(
                b_tec.var_capex_mea,
                b_tec.var_size_mea,
                pw_pts=bp_x_mea,
                pw_constr_type="EQ",
                f_rule=bp_y_annual_mea,
                pw_repn="SOS2",
            )

            # capex tot
            b_tec.const_capex_aux = pyo.Constraint(
                expr=b_tec.var_capex_mea + b_tec.var_capex_oxy == b_tec.var_capex_aux
            )
        else:

            b_tec.const_capex_oxy = pyo.Piecewise(
                b_tec.var_capex_oxy,
                b_tec.var_size,
                pw_pts=bp_x_oxy,
                pw_constr_type="EQ",
                f_rule=bp_y_annual_oxy,
                pw_repn="SOS2",
            )


            b_tec.const_capex_mea = pyo.Piecewise(
                b_tec.var_capex_mea,
                b_tec.var_size_mea,
                pw_pts=bp_x_mea,
                pw_constr_type="EQ",
                f_rule=bp_y_annual_mea,
                pw_repn="SOS2",
            )

            # capex tot
            b_tec.const_capex_aux = pyo.Constraint(
                expr=b_tec.var_capex_mea + b_tec.var_capex_oxy == b_tec.var_capex_aux
            )

        # capex
        if self.existing:
            if self.decommission == "impossible":
                # technology cannot be decommissioned
                b_tec.const_capex = pyo.Constraint(expr=b_tec.var_capex == 0)
            else:
                b_tec.const_capex = pyo.Constraint(
                    expr=b_tec.var_capex
                    == (b_tec.para_size_initial - b_tec.var_size)
                    * b_tec.para_decommissioning_cost_annual
                )
        else:
            b_tec.const_capex = pyo.Constraint(
                expr=b_tec.var_capex == b_tec.var_capex_aux
            )

        return b_tec

    def _define_opex(self, b_tec, data):
        """
        Defines variable and fixed OPEX

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )
        emissions_clinker = self.performance_data["performance"][
            "tCO2_tclinker"
        ]
        CCR_oxy = self.performance_data["performance"]["CCR_oxy"]

        # VARIABLE OPEX
        b_tec.para_opex_var_oxy = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["other_economics"]["opex_var_oxy"],
            mutable=True,
        )

        b_tec.para_opex_var_mea = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["other_economics"]["opex_var_MEA"],
            mutable=True,
        )
        b_tec.var_opex_variable = pyo.Var()



        hour_factors = data["hour_factors"]
        nr_timesteps_averaged = data["nr_timesteps_averaged"]

        def init_opex_variable(const):
            return (
                    sum(
                        (
                                b_tec.var_output[t, self.main_output_carrier]
                                * emissions_clinker * CCR_oxy
                                * nr_timesteps_averaged
                                * hour_factors[t - 1]
                                * b_tec.para_opex_var_oxy
                        )
                        + (
                                b_tec.var_co2_captured_mea[t]
                                * nr_timesteps_averaged
                                * hour_factors[t - 1]
                                * b_tec.para_opex_var_mea
                        )
                        for t in self.set_t_global
                    )
                    == b_tec.var_opex_variable
            )

        b_tec.const_opex_variable = pyo.Constraint(rule=init_opex_variable
        )

        # FIXED OPEX
        b_tec.para_opex_fixed_oxy = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["other_economics"]["opex_fixed_oxy"],
            mutable=True,
        )

        b_tec.para_opex_fixed_mea = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["other_economics"]["opex_fixed_MEA"],
            mutable=True,
        )

        b_tec.var_opex_fixed = pyo.Var()
        b_tec.const_opex_fixed = pyo.Constraint(
            expr=(b_tec.var_capex_mea / annualization_factor)
            * b_tec.para_opex_fixed_mea
            + (b_tec.var_capex_oxy / annualization_factor) * b_tec.para_opex_fixed_oxy
            == b_tec.var_opex_fixed
        )
        return b_tec

    def _performance_function_type_2(self, b_tec):
        """
        Sets the minimum part load constraint based on output with
        performance type 2.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]
        min_part_load = coeff_ti["min_part_load"]

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_performance, domain=pyo.NonNegativeReals, bounds=(0, 1)
            )

        if min_part_load == 0:
            warn(
                "Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for "
                + self.name
            )

        # define disjuncts
        s_indicators = range(0, 2)

        def init_output(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # technology on

                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                def init_min_partload(const):
                    return (
                        self.output[t, self.main_output_carrier]
                        >= min_part_load * b_tec.var_size * rated_capacity
                    )

                dis.const_min_partload = pyo.Constraint(rule=init_min_partload)

        b_tec.dis_output = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_output[t, i] for i in s_indicators]

        b_tec.disjunction_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        return b_tec





    def write_results_tec_design(self, h5_group, model_block):
        """
        Function to report technology design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(CementHybridCCS, self).write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset("size_mea", data=[model_block.var_size_mea.value])
        h5_group.create_dataset("capex_mea", data=[model_block.var_capex_mea.value])
        h5_group.create_dataset("capex_oxy", data=[model_block.var_capex_oxy.value])

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(CementHybridCCS, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "CO2_to_mea",
            data=[model_block.var_co2_captured_mea[t].value for t in self.set_t_full],
        )
