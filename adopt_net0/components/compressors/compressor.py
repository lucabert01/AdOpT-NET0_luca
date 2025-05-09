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
import copy
import pyomo.environ as pyo
import pyomo.gdp as gdp

import logging

log = logging.getLogger(__name__)


class Compressor(ModelComponent):
    """
    Class to read and manage compression features

    """

    def __init__(self, compr_data: dict):
        """
        Initializes compression class from compressor data

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

        # TODO: definition of input/output
        self.output_component = compr_data["connection_info"]["components"][0]
        self.input_component = compr_data["connection_info"]["components"][1]
        self.output_pressure = compr_data["connection_info"]["pressure"][0]
        self.input_pressure = compr_data["connection_info"]["pressure"][1]
        # to be fixed
        self.input_carrier = compr_data["carrier"]
        self.output_type = compr_data["connection_info"]["type"][0]
        self.input_type = compr_data["connection_info"]["type"][1]

    def fit_compressor_performance(self):
        """
        Fits compressor performance (bounds and coefficients).
        """

        # what do we need here?
        # from other classes: there are some parameter time independent that are saved here in self

        # to be fixed (gamma)
        self.performance_data["compression_energy"] = 5
        # time_independent = {}

        # energy
        self.processed_coeff.time_independent["compression_energy"] = (
            self.performance_data["compression_energy"]
        )

        if self.output_pressure >= self.input_pressure:
            self.compression_active = 0
        else:
            self.compression_active = 1

    def construct_compressor_model(
        self, b_compr, data: dict, set_t_full, set_t_clustered
    ):

        # LOG
        log_msg = f"\t - Adding Compressor {self.name}"
        print(log_msg)
        log.info(log_msg)

        # compressor data
        config = data["config"]

        # SET T
        self.set_t_full = set_t_full

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

        # # CALCULATE BOUNDS
        # self._calculate_bounds()

        # GENERAL TECHNOLOGY CONSTRAINTS
        b_compr = self._define_output_component(b_compr)
        b_compr = self._define_input_component(b_compr)
        b_compr = self._define_output_pressure(b_compr)
        b_compr = self._define_input_pressure(b_compr)
        b_compr = self._define_carrier(b_compr)
        b_compr = self._define_flow(b_compr, data)
        # if self.compression_active == 1:
        #     b_compr = self._define_size(b_compr)
        #     b_compr = self._define_energy(b_compr, data)
        #     b_compr = self._define_capex_parameters(b_compr, data)
        #     b_compr = self._define_capex_variables(b_compr, data)
        #     b_compr = self._define_capex_constraints(b_compr, data)
        #     b_compr = self._define_opex(b_compr, data)

        # EXISTING TECHNOLOGY CONSTRAINTS
        # if self.existing and self.component_options.decommission == "only_complete":
        #     b_compr = self._define_decommissioning_at_once_constraints(b_compr)

        # CLUSTERED DATA
        if (config["optimization"]["typicaldays"]["N"]["value"] == 0) or (
            config["optimization"]["typicaldays"]["method"]["value"] == 1
        ):
            # input/output to calculate performance is the same as var_input
            if b_compr.find_component("var_input"):
                self.input = b_compr.var_input
            if b_compr.find_component("var_output"):
                self.output = b_compr.var_output
        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            # input/output to calculate performance has lower resolution
            b_compr = self._define_auxiliary_vars(b_compr, data)
            if b_compr.find_component("var_input"):
                self.input = b_compr.var_input_aux
            if b_compr.find_component("var_output"):
                self.output = b_compr.var_output_aux

        # AGGREGATE ALL VARIABLES
        # self._aggregate_input(b_compr)
        # self._aggregate_output(b_compr)
        # self._aggregate_cost(b_compr)

        return b_compr

    def _define_output_component(self, b_compr):
        """
        Defines the component which has the carrier as output

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.set_output_component = pyo.Set(initialize=[self.output_component])
        return b_compr

    def _define_output_pressure(self, b_compr):
        """
        Defines the pressure from output component

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.set_output_pressure = pyo.Param(initialize=self.output_pressure)
        return b_compr

    def _define_input_component(self, b_compr):
        """
        Defines the component which has the carrier as output

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.set_input_component = pyo.Set(initialize=[self.input_component])
        return b_compr

    def _define_input_pressure(self, b_compr):
        """
        Defines the pressure to input component

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        b_compr.set_input_pressure = pyo.Param(initialize=self.input_pressure)
        return b_compr

    def _define_carrier(self, b_compr):
        """
        Defines the carrier

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        # to be fixed correctly
        b_compr.set_input_carrier = pyo.Set(initialize=[self.input_carrier])
        return b_compr

    def _define_flow(self, b_compr, data: dict):
        """
        Defines compressor flow.

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """

        b_compr.var_flow = pyo.Var(
            self.set_t_global,
            # within=pyo.NonNegativeReals  # to be fixed here correctly if we want bounds, otherwise clear the line
        )

        return b_compr

    def _define_size(self, b_compr):
        """
        Defines variables and parameters related to compressor size.

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """

        if self.existing:
            b_compr.para_size_initial = pyo.Param(
                within=pyo.NonNegativeIntegers, initialize=b_compr.var_flow * 10
            )

        if self.existing and self.component_options.decommission == "impossible":
            # Decommissioning is not possible, size fixed
            b_compr.var_size = pyo.Param(
                within=pyo.NonNegativeIntegers, initialize=b_compr.var_flow * 10
            )
        else:
            # Size is variable
            b_compr.var_size = pyo.Var(
                within=pyo.NonNegativeIntegers,
                bounds=(1, 100),
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
        b_compr.para_unit_capex_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics["unit_capex"],
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
            max_capex = 1000
            bounds = (0, max_capex)
            return bounds

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new compressor, this is equal to actual CAPEX
        # For existing compressor it is used to calculate fixed OPEX
        b_compr.var_capex_aux = pyo.Var(bounds=calculate_max_capex())

        b_compr.var_capex = pyo.Var()
        return b_compr

    def _define_capex_constraints(self, b_compr, data):
        """
        Defines constraints related to compressor capex.
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # b_compr.const_capex_aux = pyo.Constraint(
        #     expr=b_compr.var_size * b_compr.para_unit_capex_annual
        #     == b_compr.var_capex_aux
        # )
        b_compr.const_capex_aux = pyo.Constraint(expr=0 == b_compr.var_capex_aux)

        # CAPEX
        if self.existing:
            if self.component_options.decommission == "impossible":
                # technology cannot be decommissioned
                b_compr.const_capex = pyo.Constraint(expr=b_compr.var_capex == 0)
            else:
                # b_compr.const_capex = pyo.Constraint(
                #     expr=b_compr.var_capex
                #     == (b_compr.para_size_initial - b_compr.var_size)
                #     * b_compr.para_decommissioning_cost_annual
                # )
                b_compr.const_capex = pyo.Constraint(expr=b_compr.var_capex == 0)
        else:
            # b_compr.const_capex = pyo.Constraint(
            #     expr=b_compr.var_capex == b_compr.var_capex_aux
            # )
            b_compr.const_capex = pyo.Constraint(expr=b_compr.var_capex == 0)
        return b_compr

    def _define_energy(self, b_compr, data):
        """
        Defines compressor energy

        :param b_netw: pyomo compressor block
        :return: pyomo network block
        """
        # energy_cons = self.input_parameters.performance_data["compression_energy"]
        b_compr.var_compress_energy = pyo.Var(self.set_t_global)

        # to be fixed
        def init_compr_energy(b, t):
            return b_compr.var_compress_energy[t] == b_compr.var_flow[t] * 0

        b_compr.const_compress_energy = pyo.Constraint(
            self.set_t_global, rule=init_compr_energy
        )

        return b_compr

    def _define_opex(self, b_compr, data):
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
            opex_variable_based_on = b_compr.var_flow[t]
            # return (
            #     opex_variable_based_on * b_compr.para_opex_variable
            #     == b_compr.var_opex_variable[t]
            # )
            return 0 == b_compr.var_opex_variable[t]

        b_compr.const_opex_variable = pyo.Constraint(
            self.set_t_global, rule=init_opex_variable
        )

        # FIXED OPEX
        b_compr.para_opex_fixed = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_fixed"], mutable=True
        )
        b_compr.var_opex_fixed = pyo.Var()
        # b_compr.const_opex_fixed = pyo.Constraint(
        #     expr=(b_compr.var_capex_aux / annualization_factor)
        #     * b_compr.para_opex_fixed
        #     == b_compr.var_opex_fixed
        # )
        b_compr.const_opex_fixed = pyo.Constraint(expr=0 == b_compr.var_opex_fixed)
        return b_compr
