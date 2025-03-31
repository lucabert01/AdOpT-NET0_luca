import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from ..utilities import fit_piecewise_function
from ..technology import Technology

import logging

log = logging.getLogger(__name__)


class CementHybridCCS(Technology):
    """
    Cement plant with hybrid CCS

    The plant has an oxyfuel combustion in the calciner and post-combustion capture with MEA afterward. The size
    of the oxyfuel is fixed, while the size and capture rate of the MEA are variables of the optimization
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.component_options.emissions_based_on = "output"
        self.component_options.size_based_on = "output"
        self.component_options.main_output_carrier = tec_data["Performance"][
            "main_output_carrier"
        ]

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
            / "database/templates/technology_data/Industrial/CementHybridCCS_data/performance_cost_cementHybridCCS.xlsx"
        )

        performance_data = pd.read_excel(
            performance_data_path, sheet_name="performance", index_col=0
        )
        capex_data_oxy = pd.read_excel(
            performance_data_path, sheet_name="cost_oxy", index_col=0
        )
        capex_data_mea = pd.read_excel(
            performance_data_path, sheet_name="cost_mea", index_col=0
        )
        # TODO: make a function that cleans data (cement output either 0 or at full capacity), converts CO2 to clinker and daily to hourly

        prod_capacity_clinker = self.input_parameters.performance_data[
            "prod_capacity_clinker"
        ]

        if prod_capacity_clinker < 100:
            plant_size_type = "small"
        else:
            plant_size_type = "large"

        self.processed_coeff.time_independent["plant_size_type"] = plant_size_type
        self.processed_coeff.time_independent["alpha_oxy"] = performance_data.loc[
            "alpha_oxy", "value"
        ]
        self.processed_coeff.time_independent["alpha_mea"] = performance_data.loc[
            "alpha_mea", "value"
        ]
        self.processed_coeff.time_independent["beta_oxy"] = performance_data.loc[
            "beta_oxy", "value"
        ]
        self.processed_coeff.time_independent["capex_data_oxy"] = capex_data_oxy
        self.processed_coeff.time_independent["capex_data_mea"] = capex_data_mea

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(CementHybridCCS, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)
        prod_capacity_clinker = self.input_parameters.performance_data[
            "prod_capacity_clinker"
        ]
        emissions_clinker = self.input_parameters.performance_data["performance"][
            "tCO2_tclinker"
        ]
        CCR_oxy = self.input_parameters.performance_data["performance"]["CCR_oxy"]
        CCR_mea = self.input_parameters.performance_data["performance"]["CCR_mea"]

        # Output Bounds
        self.bounds["output"]["CO2captured"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * prod_capacity_clinker
                * emissions_clinker
                * (CCR_oxy + (1 - CCR_oxy) * CCR_mea),
            )
        )

        self.bounds["output"]["clinker"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps) * prod_capacity_clinker,
            )
        )

        # Input Bounds
        self.bounds["input"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * (
                    prod_capacity_clinker
                    * emissions_clinker
                    * CCR_oxy
                    * self.processed_coeff.time_independent["alpha_oxy"]
                    + prod_capacity_clinker
                    * emissions_clinker
                    * (1 - CCR_oxy)
                    * CCR_mea
                    * self.processed_coeff.time_independent["alpha_mea"]
                ),
            )
        )
        self.bounds["input"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * (
                    prod_capacity_clinker
                    * emissions_clinker
                    * CCR_oxy
                    * self.processed_coeff.time_independent["beta_oxy"]
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

        # Size constraint
        prod_capacity_clinker = self.input_parameters.performance_data[
            "prod_capacity_clinker"
        ]
        emissions_clinker = self.input_parameters.performance_data["performance"][
            "tCO2_tclinker"
        ]
        alpha_oxy = self.processed_coeff.time_independent["alpha_oxy"]
        beta_oxy = self.processed_coeff.time_independent["beta_oxy"]
        alpha_mea = self.processed_coeff.time_independent["alpha_mea"]
        emission_capacity_max = prod_capacity_clinker * emissions_clinker
        CCR_oxy = self.input_parameters.performance_data["performance"]["CCR_oxy"]
        CCR_mea = self.input_parameters.performance_data["performance"]["CCR_mea"]

        b_tec.var_co2_captured_mea = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=[0, emission_capacity_max * (1 - CCR_oxy) * CCR_mea],
        )

        b_tec.var_size_mea = pyo.Var(
            within=pyo.NonNegativeReals,
            bounds=[0, emission_capacity_max * (1 - CCR_oxy)],
        )

        def init_size_clinker(const):
            return b_tec.var_size == prod_capacity_clinker

        b_tec.const_size_clinker = pyo.Constraint(rule=init_size_clinker)

        def init_size_constraint_mea(const, t):
            return b_tec.var_co2_captured_mea[t] <= b_tec.var_size_mea * CCR_mea

        b_tec.const_size_mea = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint_mea
        )

        def init_size_mea_max_constraint(const):
            return b_tec.var_size_mea <= b_tec.var_size * emissions_clinker * (
                1 - CCR_oxy
            )

        b_tec.const_size_mea_max = pyo.Constraint(rule=init_size_mea_max_constraint)

        def init_mea_operation_constraint(const, t):
            return (
                b_tec.var_co2_captured_mea[t]
                <= self.output[t, "clinker"] * (1 - CCR_oxy) * CCR_mea
            )

        b_tec.const_mea_operation = pyo.Constraint(
            self.set_t_performance, rule=init_mea_operation_constraint
        )

        # input-output correlations
        def init_input_output(const, t, car_input):
            b = 1
            if car_input == "heat":
                a = 1
                return (
                    self.input[t, car_input]
                    == self.output[t, "clinker"]
                    * emissions_clinker
                    * CCR_oxy
                    * beta_oxy
                )
            elif car_input == "electricity":
                return (
                    self.input[t, car_input]
                    == self.output[t, "clinker"]
                    * emissions_clinker
                    * CCR_oxy
                    * alpha_oxy
                    + b_tec.var_co2_captured_mea[t] * alpha_mea
                )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_input_carriers, rule=init_input_output
        )

        def init_output_output(const, t):
            a = 1
            return (
                self.output[t, "CO2captured"]
                == self.output[t, "clinker"] * emissions_clinker * CCR_oxy
                + b_tec.var_co2_captured_mea[t]
            )

        b_tec.const_output_output = pyo.Constraint(
            self.set_t_performance, rule=init_output_output
        )

        # define emissions

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
            return tuple(self.bounds["input"][car][self.sequence[t - 1] - 1, :])

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
            return tuple(self.bounds["output"][car][self.sequence[t - 1] - 1, :])

        b_tec.var_output = pyo.Var(
            self.set_t_global,
            b_tec.set_output_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )
        return b_tec

    def _define_emissions(self, b_tec):
        """
        Defines Emissions

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        c = self.processed_coeff.time_independent
        technology_model = self.component_options.technology_model
        emissions_clinker = self.input_parameters.performance_data["performance"][
            "tCO2_tclinker"
        ]

        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        # TODO var_emission_pos is not showing up in the results
        def init_tec_emissions_pos(const, t):
            """emissions_pos = output * emissionfactor"""
            a = 1
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

    # TODO define capex
    # def _define_capex_variables(self, b_tec, data: dict):
    #     """
    #     Defines variables related to technology capex.
    #
    #     :param b_tec: pyomo block with technology model
    #     :param dict data: dict containing model information
    #     :return: pyomo block with technology model
    #     """
    #     config = data["config"]
    #
    #     economics = self.economics
    #     discount_rate = set_discount_rate(config, economics)
    #     fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
    #     annualization_factor = annualize(
    #         discount_rate, economics.lifetime, fraction_of_year_modelled
    #     )
    #
    #     capex_model = set_capex_model(config, economics)
    #
    #     def calculate_max_capex():
    #         if capex_model == 1:
    #             max_capex = (
    #                 b_tec.para_size_max
    #                 * economics.capex_data["unit_capex"]
    #                 * annualization_factor
    #             )
    #             bounds = (0, max_capex)
    #         elif capex_model == 2:
    #             max_capex = (
    #                 b_tec.para_size_max
    #                 * max(economics.capex_data["piecewise_capex"]["bp_y"])
    #                 * annualization_factor
    #             )
    #             bounds = (0, max_capex)
    #         elif capex_model == 3:
    #             max_capex = (
    #                 b_tec.para_size_max * economics.capex_data["unit_capex"]
    #                 + economics.capex_data["fix_capex"]
    #             ) * annualization_factor
    #             bounds = (0, max_capex)
    #         else:
    #             bounds = None
    #         return bounds
    #
    #     # CAPEX auxilliary (used to calculate theoretical CAPEX)
    #     # For new technologies, this is equal to actual CAPEX
    #     # For existing technologies it is used to calculate fixed OPEX
    #     b_tec.var_capex_aux = pyo.Var(bounds=calculate_max_capex())
    #
    #     b_tec.var_capex = pyo.Var()
    #
    #     return b_tec
    #
    # def _define_capex_parameters(self, b_tec, data):
    #     """
    #     Defines the capex parameters
    #
    #     For capex model 1:
    #     - para_unit_capex
    #     - para_unit_capex_annual
    #
    #     For capex model 2: defined with constraints
    #     For capex model 3:
    #     - para_unit_capex
    #     - para_fix_capex
    #     - para_unit_capex_annual
    #     - para_fix_capex_annual
    #
    #     :param b_tec:
    #     :param data:
    #     :return:
    #     """
    #     config = data["config"]
    #     economics = self.economics
    #     discount_rate = set_discount_rate(config, economics)
    #     fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
    #     annualization_factor = annualize(
    #         discount_rate, economics.lifetime, fraction_of_year_modelled
    #     )
    #
    #     capex_model = set_capex_model(config, economics)
    #
    #     if capex_model == 1:
    #         b_tec.para_unit_capex = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=economics.capex_data["unit_capex"],
    #             mutable=True,
    #         )
    #         b_tec.para_unit_capex_annual = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=annualization_factor * economics.capex_data["unit_capex"],
    #             mutable=True,
    #         )
    #
    #     elif capex_model == 2:
    #         # This is defined in the constraints
    #         pass
    #     elif capex_model == 3:
    #         b_tec.para_unit_capex = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=economics.capex_data["unit_capex"],
    #             mutable=True,
    #         )
    #         b_tec.para_fix_capex = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=economics.capex_data["fix_capex"],
    #             mutable=True,
    #         )
    #         b_tec.para_unit_capex_annual = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=annualization_factor * economics.capex_data["unit_capex"],
    #             mutable=True,
    #         )
    #         b_tec.para_fix_capex_annual = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=annualization_factor * economics.capex_data["fix_capex"],
    #             mutable=True,
    #         )
    #     else:
    #         # Defined in the technology subclass
    #         pass
    #
    #     if self.existing and not self.component_options.decommission == "impossible":
    #         b_tec.para_decommissioning_cost_annual = pyo.Param(
    #             domain=pyo.Reals,
    #             initialize=annualization_factor * economics.decommission_cost,
    #             mutable=True,
    #         )
    #
    #     return b_tec
    #
    # def _define_capex_constraints(self, b_tec, data):
    #     """
    #     Defines constraints related to capex.
    #     """
    #     config = data["config"]
    #     economics = self.economics
    #     discount_rate = set_discount_rate(config, economics)
    #     fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
    #     annualization_factor = annualize(
    #         discount_rate, economics.lifetime, fraction_of_year_modelled
    #     )
    #
    #     capex_model = set_capex_model(config, economics)
    #
    #     if capex_model == 1:
    #         b_tec.const_capex_aux = pyo.Constraint(
    #             expr=b_tec.var_size * b_tec.para_unit_capex_annual
    #             == b_tec.var_capex_aux
    #         )
    #     elif capex_model == 2:
    #         self.big_m_transformation_required = 1
    #         bp_x = economics.capex_data["piecewise_capex"]["bp_x"]
    #         bp_y_annual = [
    #             y * annualization_factor
    #             for y in economics.capex_data["piecewise_capex"]["bp_y"]
    #         ]
    #         b_tec.const_capex_aux = pyo.Piecewise(
    #             b_tec.var_capex_aux,
    #             b_tec.var_size,
    #             pw_pts=bp_x,
    #             pw_constr_type="EQ",
    #             f_rule=bp_y_annual,
    #             pw_repn="SOS2",
    #         )
    #     elif capex_model == 3:
    #         self.big_m_transformation_required = 1
    #         s_indicators = range(0, 2)
    #
    #         if self.existing:
    #             b_tec.const_capex_aux = pyo.Constraint(
    #                 expr=b_tec.var_size * b_tec.para_unit_capex_annual
    #                 + b_tec.para_fix_capex_annual
    #                 == b_tec.var_capex_aux
    #             )
    #         else:
    #
    #             def init_installation(dis, ind):
    #                 if ind == 0:  # tech not installed
    #                     dis.const_capex_aux = pyo.Constraint(
    #                         expr=b_tec.var_capex_aux == 0
    #                     )
    #                     dis.const_not_installed = pyo.Constraint(
    #                         expr=b_tec.var_size == 0
    #                     )
    #                 else:  # tech installed
    #                     dis.const_capex_aux = pyo.Constraint(
    #                         expr=b_tec.var_size * b_tec.para_unit_capex_annual
    #                         + b_tec.para_fix_capex_annual
    #                         == b_tec.var_capex_aux
    #                     )
    #
    #             b_tec.dis_installation = gdp.Disjunct(
    #                 s_indicators, rule=init_installation
    #             )
    #
    #             def bind_disjunctions(dis):
    #                 return [b_tec.dis_installation[i] for i in s_indicators]
    #
    #             b_tec.disjunction_installation = gdp.Disjunction(rule=bind_disjunctions)
    #
    #     else:
    #         # Defined in the technology subclass
    #         pass
    #
    #     # CAPEX
    #     if self.existing:
    #         if self.component_options.decommission == "impossible":
    #             # technology cannot be decommissioned
    #             b_tec.const_capex = pyo.Constraint(expr=b_tec.var_capex == 0)
    #         else:
    #             b_tec.const_capex = pyo.Constraint(
    #                 expr=b_tec.var_capex
    #                 == (b_tec.para_size_initial - b_tec.var_size)
    #                 * b_tec.para_decommissioning_cost_annual
    #             )
    #     else:
    #         b_tec.const_capex = pyo.Constraint(
    #             expr=b_tec.var_capex == b_tec.var_capex_aux
    #         )
    #
    #     return b_tec

    def write_results_tec_design(self, h5_group, model_block):
        """
        Function to report technology design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(CementHybridCCS, self).write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset("size_mea", data=[model_block.var_size_mea.value])

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
