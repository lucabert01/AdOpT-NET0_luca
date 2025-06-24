from blib2to3.pygram import initialize

from ..component import ModelComponent
from ..utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
    determine_variable_scaling,
    determine_constraint_scaling,
    get_attribute_from_dict,
)

import pandas as pd
import math
import copy
import pyomo.environ as pyo
import pyomo.gdp as gdp

import logging

log = logging.getLogger(__name__)


class Compressor(ModelComponent):
    """
    Class to read and manage data for compressors

    **Set declarations:**

    **Parameter declarations:**

    The following is a list of declared pyomo parameters.

    - para_name: name of the compressor, formatted as type_component1_component2_(existing)
    - para_active: equals 1 if the compressor is active (a pressure increase is needed); otherwise equals to 0
    - para_output_component: name of the component that outputs the carrier to the compressor
    - para_output_pressure: pressure of the carrier at the output of the component that feeds the compressor
    - para_input_component: name of the component that receives the carrier as input from the compressor
    - para_input_pressure: pressure of the carrier at the input of the component fed by the compressor
    - para_carrier: carrier that flows through the compressor
    - para_unit_capex: investment costs per unit
    - para_unit_capex_annual: Unit CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - para_fix_capex: fixed costs independent of size
    - para_fix_capex_annual: Fixed CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - para_opex_variable: operational cost EUR/output or input
    - para_opex_fixed: fixed opex as fraction of up-front capex

    **Variable declarations:**

    - var_flow: flow that is processed by the compressor
    - var_consumption_energy: energy consumption required by the compressor to raise the pressure (only for if active)
    - var_size: size of the compressor (only if active)
    - var_capex: annualized investment of the compressor (only if active)
    - var_opex_variable: variable operation costs, defined for each time slice
    - var_opex_fixed: fixed operational costs as fraction of up-front CAPEX
    - var_capex_aux: auxiliary variable to calculate the fixed opex of existing compressor

    **Network constraint declarations**



    """

    def __init__(self, compr_data: dict):
        """
        Initializes compressor class from compressor data

        :param dict compr_data: compressor data
        """
        super().__init__(compr_data)

        # Modelling attributes
        self.input = None
        self.output = None
        self.set_t_full = None
        self.set_t_performance = None
        self.set_t_global = None
        self.sequence = None
        self.compression_active = None

        # General information
        self.energy_consumption = {}

        # TODO: definition of input/output
        self.output_component = compr_data["connection_info"]["components"][0]
        self.input_component = compr_data["connection_info"]["components"][1]
        self.output_pressure = compr_data["connection_info"]["pressure"][0]
        self.input_pressure = compr_data["connection_info"]["pressure"][1]
        # to be fixed
        self.carrier = compr_data["carrier"]
        self.output_type = compr_data["connection_info"]["type"][0]
        self.input_type = compr_data["connection_info"]["type"][1]
        self.output_existing = compr_data["connection_info"]["existing"][0]
        self.input_existing = compr_data["connection_info"]["existing"][1]
        self.name_compressor = (
            f"{self.carrier}_Compressor_{self.output_component}_{self.input_component}"
        )

        if self.output_existing == 1 and self.input_existing == 1:
            self.name_compressor = self.name_compressor + "_existing"

    def fit_compressor_performance(self):
        """
        Fits compressor performance (bounds and coefficients).
        """

        # TODO: double check on this

        # what do we need here?
        # from other classes: there are some parameter time independent that are saved here in self

        # to be fixed (gamma)
        # self.performance_data["compression_energy"] = 5
        time_independent = {}

        self.energy_consumption = self.performance_data["energyconsumption"]

        # Size
        time_independent["size_min"] = self.size_min
        time_independent["size_max"] = self.size_max

        # energy
        # self.processed_coeff.time_independent["compression_energy"] = (
        #     self.performance_data["compression_energy"]
        # )

        if self.output_pressure >= self.input_pressure:
            self.compression_active = 0
        else:
            self.compression_active = 1

        if self.output_existing == 1 and self.input_existing == 1:
            self.existing = 1
        else:
            self.existing = 0

        self.processed_coeff.time_independent = time_independent

    def construct_compressor_model(
        self, b_compr, data: dict, set_t_full, set_t_clustered
    ):
        """
        Construct the compressor model with all required parameters, variable, sets,...

        :param b_compr: pyomo block with compressor model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with compressor model
        """

        # LOG
        log_msg = f"\t - Adding Compressor {self.name}"
        print(log_msg)
        log.info(log_msg)

        # compressor data
        config = data["config"]

        # SET T
        self.set_t_full = set_t_full

        # MODELING TYPICAL DAYS
        technologies_modelled_with_full_res = config["optimization"]["typicaldays"][
            "technologies_with_full_res"
        ]["value"]

        if config["optimization"]["typicaldays"]["N"]["value"] == 0:
            # everything with full resolution
            self.modelled_with_full_res = True
            self.set_t_performance = set_t_full
            self.set_t_global = set_t_full
            self.sequence = list(self.set_t_performance)

        elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
            # everything with reduced resolution
            self.modelled_with_full_res = False
            self.set_t_performance = set_t_clustered
            self.set_t_global = set_t_clustered
            self.sequence = list(self.set_t_performance)

        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            # resolution of balances is full, so interactions with them also need to
            # be full resolution
            self.set_t_global = set_t_full

        # Coefficients
        if self.modelled_with_full_res:
            if config["optimization"]["timestaging"]["value"] == 0:
                self.processed_coeff.time_dependent_used = (
                    self.processed_coeff.time_dependent_full
                )
            else:
                self.processed_coeff.time_dependent_used = (
                    self.processed_coeff.time_dependent_averaged
                )
        else:
            self.processed_coeff.time_dependent_used = (
                self.processed_coeff.time_dependent_clustered
            )

        # GENERAL TECHNOLOGY CONSTRAINTS
        b_compr = self._define_output_component(b_compr)  # can I delete it?
        b_compr = self._define_input_component(b_compr)  # can I delete it?
        b_compr = self._define_output_pressure(b_compr)  # can I delete it?
        b_compr = self._define_input_pressure(b_compr)  # can I delete it?
        b_compr = self._define_carrier(b_compr)
        b_compr = self._define_existing(b_compr)
        b_compr = self._define_flow(b_compr)
        b_compr = self._define_compressor_name(b_compr)
        b_compr = self._define_compressor_active(b_compr)
        if self.compression_active == 1:
            b_compr = self._define_energyconsumption_parameters(b_compr)
            b_compr = self._define_energy_consumption(b_compr, data)
            b_compr = self._define_opex_var(b_compr, data)
            b_compr = self._define_size(b_compr)
            # if self.existing == 0:
            b_compr = self._define_capex_parameters(b_compr, data)
            b_compr = self._define_capex_variables(b_compr, data)
            b_compr = self._define_capex_constraints(b_compr, data)
            b_compr = self._define_opex_fixed(b_compr, data)

        # TODO: understand what to do with decommission

        # # EXISTING TECHNOLOGY CONSTRAINTS
        # if self.existing and self.decommission == "only_complete":
        #     b_compr = self._define_decommissioning_at_once_constraints(b_compr)

        # # CLUSTERED DATA
        # if (config["optimization"]["typicaldays"]["N"]["value"] == 0) or (
        #     config["optimization"]["typicaldays"]["method"]["value"] == 1
        # ):
        #     # input/output to calculate performance is the same as var_input
        #     if b_compr.find_component("var_input"):
        #         self.input = b_compr.var_input
        #     if b_compr.find_component("var_output"):
        #         self.output = b_compr.var_output
        # elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        #     if self.technology_model in technologies_modelled_with_full_res:
        #         # input/output to calculate performance is the same as var_input
        #         if b_compr.find_component("var_input"):
        #             self.input = b_compr.var_input
        #         if b_compr.find_component("var_output"):
        #             self.output = b_compr.var_output
        #     else:
        #         # input/output to calculate performance has lower resolution
        #         b_tec = self._define_auxiliary_vars(b_compr, data)
        #         if b_tec.find_component("var_input"):
        #             self.input = b_tec.var_input_aux
        #         if b_tec.find_component("var_output"):
        #             self.output = b_tec.var_output_aux

        # AGGREGATE ALL VARIABLES
        # self._aggregate_input(b_compr)
        # self._aggregate_output(b_compr)
        # self._aggregate_cost(b_compr)

        return b_compr

    def _define_compressor_name(self, b_compr):
        """
        Defines the name of the component as carrier_component1_component2(_existing)

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.para_name = pyo.Param(initialize=[self.name_compressor], within=pyo.Any)
        return b_compr

    def _define_compressor_active(self, b_compr):
        """
        Defines if compressor is active (1 for active, 0 for no active)

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.para_active = pyo.Param(
            initialize=self.compression_active, within=pyo.Any
        )
        return b_compr

    def _define_output_component(self, b_compr):
        """
        Defines the component which has the carrier as output

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.para_output_component = pyo.Param(
            initialize=[self.output_component], within=pyo.Any
        )
        return b_compr

    def _define_output_pressure(self, b_compr):
        """
        Defines the pressure from output component

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.para_output_pressure = pyo.Param(
            initialize=self.output_pressure, within=pyo.Any
        )
        return b_compr

    def _define_input_component(self, b_compr):
        """
        Defines the component which has the carrier as input

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.para_input_component = pyo.Param(
            initialize=[self.input_component], within=pyo.Any
        )
        return b_compr

    def _define_input_pressure(self, b_compr):
        """
        Defines the pressure of input component

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.para_input_pressure = pyo.Param(
            initialize=self.input_pressure, within=pyo.Any
        )
        return b_compr

    def _define_carrier(self, b_compr):
        """
        Defines the carrier of the compressor

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """

        b_compr.para_carrier = pyo.Param(initialize=[self.carrier], within=pyo.Any)
        return b_compr

    def _define_existing(self, b_compr):
        """
        Defines if the compressor is existing

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """

        b_compr.para_existing = pyo.Param(initialize=self.existing, within=pyo.Any)
        return b_compr

    def _define_flow(self, b_compr):
        """
        Defines variable for compressor flow.

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """

        b_compr.var_flow = pyo.Var(
            self.set_t_global,
            within=pyo.NonNegativeReals,  # to be fixed here correctly if we want bounds, otherwise clear the line
        )

        return b_compr

    def _define_capex_parameters(self, b_compr, data):
        """
        Defines the capex parameters

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return:
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        b_compr.para_unit_capex = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["unit_capex"],
            mutable=True,
        )
        b_compr.para_fix_capex = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["fix_capex"],
            mutable=True,
        )
        b_compr.para_unit_capex_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics["unit_capex"],
            mutable=True,
        )
        b_compr.para_fix_capex_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics["fix_capex"],
            mutable=True,
        )

        if self.existing and not self.decommission == "impossible":
            b_compr.para_decommissioning_cost_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=annualization_factor * economics["decommission_cost"],
                mutable=True,
            )
        return b_compr

    def _define_capex_variables(self, b_compr, data: dict):
        """
        Defines variables related to compressor capex.

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """
        config = data["config"]

        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        def calculate_max_capex():
            max_capex = (
                b_compr.para_size_max * economics["unit_capex"] + economics["fix_capex"]
            ) * annualization_factor
            bounds = (0, max_capex)

            return bounds

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new compressor, this is equal to actual CAPEX
        # For existing compressor it is used to calculate fixed OPEX
        # b_compr.var_capex_aux = pyo.Var(bounds=calculate_max_capex())
        b_compr.var_capex_aux = pyo.Var()

        b_compr.var_capex = pyo.Var()

        return b_compr

    def _define_capex_constraints(self, b_compr, data: dict):
        """
        Defines constraints related to compressor capex.

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # # CAPEX
        # self.big_m_transformation_required = 1
        # s_indicators = range(0, 2)
        #
        # if self.existing == 1:
        #     b_compr.const_capex_aux = pyo.Constraint(
        #         expr=b_compr.var_size * b_compr.para_unit_capex_annual
        #         + b_compr.para_fix_capex_annual
        #         == b_compr.var_capex_aux
        #     )
        #
        # else:
        #
        #     def init_installation(dis, ind):
        #         if ind == 0:  # compressor not installed
        #             dis.const_capex_aux = pyo.Constraint(
        #                 expr=b_compr.var_capex_aux == 0
        #             )
        #             dis.const_not_installed = pyo.Constraint(expr=b_compr.var_size == 0)
        #         else:  # tech installed
        #             dis.const_capex_aux = pyo.Constraint(
        #                 expr=b_compr.var_size * b_compr.para_unit_capex_annual
        #                 + b_compr.para_fix_capex_annual
        #                 == b_compr.var_capex_aux
        #             )
        #
        #     b_compr.dis_installation = gdp.Disjunct(
        #         s_indicators, rule=init_installation
        #     )
        #
        #     def bind_disjunctions(dis):
        #         return [b_compr.dis_installation[i] for i in s_indicators]
        #
        #     b_compr.disjunction_installation = gdp.Disjunction(rule=bind_disjunctions)
        #
        # if self.existing == 1:
        #     if self.decommission == "impossible":
        #         # compressor cannot be decommissioned
        #         b_compr.const_capex = pyo.Constraint(expr=b_compr.var_capex == 0)
        #     else:
        #         b_compr.const_capex = pyo.Constraint(
        #             expr=b_compr.var_capex
        #             == (b_compr.para_size_initial - b_compr.var_size)
        #             * b_compr.para_decommissioning_cost_annual
        #         )
        # else:
        #     b_compr.const_capex = pyo.Constraint(
        #         expr=b_compr.var_capex == b_compr.var_capex_aux
        #     )

        b_compr.const_capex_aux = pyo.Constraint(
            expr=b_compr.var_size * b_compr.para_unit_capex_annual
            == b_compr.var_capex_aux
            # + b_compr.para_fix_capex_annual
        )
        # b_compr.const_capex_aux = pyo.Constraint(
        #     expr=b_compr.var_size * b_compr.para_unit_capex_annual
        #     + b_compr.para_fix_capex_annual
        #          == b_compr.var_capex_aux
        # )

        if self.existing == 0:
            b_compr.const_capex = pyo.Constraint(
                expr=b_compr.var_capex == b_compr.var_capex_aux
            )

        return b_compr

    def _define_energyconsumption_parameters(self, b_compr):
        """
        Constructs constraints for compressor energy consumption

        :param b_compr: pyomo compressor block
        :return: pyomo compressor block
        """
        # Set of consumed carriers
        b_compr.set_consumed_carriers = pyo.Set(
            initialize=list(self.energy_consumption.keys())
        )

        self.pressure_per_stage = self.performance_data["max_pressure_per_stage"]
        self.isentropic_efficiency = self.performance_data["isentropic_efficiency"]
        self.heat_coefficient = self.performance_data["heat_coefficient"]
        self.mean_compressibility_factor = self.performance_data[
            "mean_compressibility_factor"
        ]

        # Consumption from compressor
        b_compr.var_consumption_energy = pyo.Var(
            self.set_t_global,
            b_compr.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
        )

        return b_compr

    def _define_energy_consumption(self, b_compr, data):
        """
        Defines compressor energy consumption

        :param b_compr: pyomo compressor block
        :return: pyomo compressor block
        """
        n_stages = math.ceil(
            math.log(self.input_pressure / self.output_pressure)
            / math.log(self.pressure_per_stage)
        )

        R = 8.314  # kJ/kmol/K
        T_in = 298.15  # K

        energy_consumption = (
            self.mean_compressibility_factor
            / 120
            / 2
            * T_in
            * (R / 1000)
            * n_stages
            * (self.heat_coefficient / (self.heat_coefficient - 1))
            * (1 / self.isentropic_efficiency)
            * (
                (self.input_pressure / self.output_pressure)
                ** ((self.heat_coefficient - 1) / (n_stages * self.heat_coefficient))
                - 1
            )
        )  # MW_el/MW_H2

        def init_compr_energy(b, t, car):
            """
            Define energy for compression in MW
            """
            return (
                b_compr.var_consumption_energy[t, car]
                == b_compr.var_flow[t] * energy_consumption
            )  # MW_el

        b_compr.const_compress_energy = pyo.Constraint(
            self.set_t_global, b_compr.set_consumed_carriers, rule=init_compr_energy
        )

        return b_compr

    def _define_size(self, b_compr):
        """
        Defines variables and parameters related to compressor size.

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        coeff_ti = self.processed_coeff.time_independent

        b_compr.para_size_min = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_min"], mutable=True
        )
        b_compr.para_size_max = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_max"], mutable=True
        )

        b_compr.var_size = pyo.Var(within=pyo.NonNegativeReals)

        def sizing_rule(b, t, car):
            return b_compr.var_size >= b_compr.var_consumption_energy[t, car]

        if self.existing == 0:
            b_compr.const_size = pyo.Constraint(
                self.set_t_global, b_compr.set_consumed_carriers, rule=sizing_rule
            )

        return b_compr

    def _define_opex_var(self, b_compr, data: dict):
        """
        Defines variable and fixed OPEX

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # VARIABLE OPEX
        b_compr.para_opex_variable = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_variable"], mutable=True
        )
        b_compr.var_opex_variable = pyo.Var(self.set_t_global)

        def init_opex_variable(const, t):
            """opexvar_{t} = Input_{t, maincarrier} * opex_{var}"""
            return (
                b_compr.var_opex_variable[t]
                == b_compr.para_opex_variable * b_compr.var_flow[t]
            )

        b_compr.const_opex_variable = pyo.Constraint(
            self.set_t_global, rule=init_opex_variable
        )

        return b_compr

    def _define_opex_fixed(self, b_compr, data: dict):

        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # FIXED OPEX
        b_compr.para_opex_fixed = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_fixed"], mutable=True
        )
        b_compr.var_opex_fixed = pyo.Var()

        b_compr.const_opex_fixed = pyo.Constraint(
            expr=(b_compr.var_capex_aux / annualization_factor)
            * b_compr.para_opex_fixed
            == b_compr.var_opex_fixed
        )

        return b_compr

    def _define_decommissioning_at_once_constraints(self, b_compr):
        """
        Defines constraints to ensure that a technology can only be decommissioned as a whole.

        This function creates a disjunction formulation that enforces
        full-plant decommissioning decisions, meaning that either the technology is fully installed
        or fully decommissioned, with no partial decommissioning allowed.

        :param b_tec: The block representing the technology.

        :return: The modified technology block with added decommissioning constraints.
        """

        # Full plant decommissioned only
        self.big_m_transformation_required = 1
        s_indicators = range(0, 2)

        def init_decommission_full(dis, ind):
            if ind == 0:  # tech not installed
                dis.const_decommissioned = pyo.Constraint(expr=b_compr.var_size == 0)
            else:  # tech installed
                dis.const_installed = pyo.Constraint(
                    expr=b_compr.var_size == b_compr.para_size_initial
                )

        b_compr.dis_decommission_full = gdp.Disjunct(
            s_indicators, rule=init_decommission_full
        )

        def bind_disjunctions(dis):
            return [b_compr.dis_decommission_full[i] for i in s_indicators]

        b_compr.disjunction_decommission_full = gdp.Disjunction(rule=bind_disjunctions)

        return b_compr

    def write_results_compressor_design(self, h5_group, model_block):
        """
        Function to report compressor design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        if self.compression_active == 1:
            h5_group.create_dataset("compressor", data=[self.name])
            h5_group.create_dataset("existing", data=[self.existing])
            h5_group.create_dataset(
                "max_flow",
                data=[max(model_block.var_flow[t].value for t in self.set_t_global)],
            )
            h5_group.create_dataset("size", data=[model_block.var_size.value])
            if self.existing == 0:
                h5_group.create_dataset("capex", data=[model_block.var_capex.value])
                # h5_group.create_dataset("opex_variable",
                #                         data=[sum(model_block.var_opex_variable[t] for t in self.set_t_global)])
            else:
                return
        else:
            return
        # h5_group.create_dataset(
        #     "capex_tot",
        #     data=[
        #         (
        #             model_block.var_capex.value + model_block.var_capex_ccs.value
        #             if hasattr(model_block, "var_capex_ccs")
        #             else 0
        #         )
        #     ],
        # )
        # h5_group.create_dataset(
        #     "opex_variable",
        #     data=[
        #         sum(
        #             (
        #                 model_block.var_opex_variable[t].value
        #                 + model_block.var_opex_variable_ccs.value
        #                 if hasattr(model_block, "var_opex_variable_ccs")
        #                 else 0
        #             )
        #             for t in self.set_t_global
        #         )
        #     ],
        # )
        # h5_group.create_dataset(
        #     "opex_fixed",
        #     data=[
        #         (
        #             model_block.var_opex_fixed.value
        #             + model_block.var_opex_fixed_ccs.value
        #             if hasattr(model_block, "var_opex_fixed_ccs")
        #             else 0
        #         )
        #     ],
        # )
        # h5_group.create_dataset(
        #     "emissions_pos",
        #     data=[
        #         sum(
        #             model_block.var_tec_emissions_pos[t].value
        #             for t in self.set_t_global
        #         )
        #     ],
        # )
        # h5_group.create_dataset(
        #     "emissions_neg",
        #     data=[
        #         sum(
        #             model_block.var_tec_emissions_neg[t].value
        #             for t in self.set_t_global
        #         )
        #     ],
        # )
        # if self.ccs_possible:
        #     h5_group.create_dataset("size_ccs", data=[model_block.var_size_ccs.value])
        #     h5_group.create_dataset("capex_tec", data=[model_block.var_capex.value])
        #     h5_group.create_dataset("capex_ccs", data=[model_block.var_capex_ccs.value])
        #     h5_group.create_dataset(
        #         "opex_fixed_ccs", data=[model_block.var_opex_fixed_ccs.value]
        #     )
        #
        # h5_group.create_dataset(
        #     "para_unitCAPEX", data=[model_block.para_unit_capex.value]
        # )
        # if hasattr(model_block, "para_fix_capex"):
        #     h5_group.create_dataset(
        #         "para_fixCAPEX", data=[model_block.para_fix_capex.value]
        #     )

    def write_results_compressor_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        # if model_block.find_component("var_flow"):
        #     for car in model_block.set_carrier:
        #         h5_group.create_dataset(
        #             f"{car}_input",
        #             data=[
        #                 model_block.var_flow[t].value for t in self.set_t_global
        #             ],
        #         )

        h5_group.create_dataset(
            "flow", data=[model_block.var_flow[t].value for t in self.set_t_global]
        )
        # for car in model_block.set_consumed_carriers:
        #     h5_group.create_dataset(
        #         "max_eletricity",
        #         data=[max(model_block.var_consumption_energy[t, car].value for t in self.set_t_global)]
        #     )
        for car in model_block.set_consumed_carriers:
            h5_group.create_dataset(
                "energy consumption",
                data=[
                    model_block.var_consumption_energy[t, car].value
                    for t in self.set_t_global
                ],
            )
        # h5_group.create_dataset(
        #     "emissions_pos",
        #     data=[
        #         model_block.var_tec_emissions_pos[t].value for t in self.set_t_global
        #     ],
        # )
        # h5_group.create_dataset(
        #     "emissions_neg",
        #     data=[
        #         model_block.var_tec_emissions_neg[t].value for t in self.set_t_global
        #     ],
        # )
        # if model_block.find_component("var_x"):
        #     h5_group.create_dataset(
        #         "var_x",
        #         data=[
        #             0 if x is None else x
        #             for x in [
        #                 model_block.var_x[t].value for t in self.set_t_performance
        #             ]
        #         ],
        #     )
        # if model_block.find_component("var_y"):
        #     h5_group.create_dataset(
        #         "var_y",
        #         data=[
        #             0 if x is None else x
        #             for x in [
        #                 model_block.var_y[t].value for t in self.set_t_performance
        #             ]
        #         ],
        #     )
        # if model_block.find_component("var_z"):
        #     h5_group.create_dataset(
        #         "var_z",
        #         data=[
        #             0 if x is None else x
        #             for x in [
        #                 model_block.var_z[t].value for t in self.set_t_performance
        #             ]
        #         ],
        #     )
        #
        # if model_block.find_component("set_carriers_ccs"):
        #     for car in model_block.set_carriers_ccs:
        #         h5_group.create_dataset(
        #             f"{car}_var_input_ccs",
        #             data=[
        #                 model_block.var_input_ccs[t, car].value
        #                 for t in self.set_t_performance
        #             ],
        #         )
        #     for car in model_block.set_output_carriers_ccs:
        #         h5_group.create_dataset(
        #             f"{car}_var_output_ccs",
        #             data=[
        #                 model_block.var_output_ccs[t, car].value
        #                 for t in self.set_t_performance
        #             ],
        #         )
