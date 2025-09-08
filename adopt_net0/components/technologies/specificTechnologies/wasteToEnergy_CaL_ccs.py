import pyomo.environ as pyo
import pyomo.gdp as gdp
import pandas as pd
import numpy as np
from pathlib import Path
from ...utilities import annualize, set_discount_rate
from ..technology import Technology
from warnings import warn
import logging
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)


class WasteToEnergyCaLCCS(Technology):
    """
    Waste to Energy with combined heat and power and a calcium looping (CaL) unit for CCS

    CaL is assumed to be an extra line of the WtE facility. Size of CCS is always in CO2 out (captured). The cost are
    only the ones of the CaL unit, i.e. the costs of the WtE are not included.
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.size_based_on = "input"
        self.emissions_based_on = "input"
        self.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]

    def _define_size(self, b_tec):
        """
        Calculates the bounds of the variables used
        """
        super(WasteToEnergyCaLCCS, self)._define_size(b_tec)

        emission_factor = self.performance_data["emission_factor"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]
        th_input_CaL = self.performance_data["th_input_CaL"]
        ccr = self.performance_data["capture_rate"]
        lhv_RDF = self.performance_data["LHV_RDF"]
        size_max_cal = b_tec.para_size_max.value * emission_factor * ccr * (
                    1 + th_input_CaL / lhv_RDF * emission_factor_RDF * ccr)

        b_tec.var_size_cal = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, size_max_cal),
        )

        return b_tec

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(WasteToEnergyCaLCCS, self)._calculate_bounds()

        other_data_path = Path(__file__).parent.parent.parent.parent
        other_data_path = (
                other_data_path
                / "database/templates/technology_data/Industrial/WasteCaL_data/wasteCaL_sheet.xlsx"
        )

        emission_factor_data = pd.read_excel(
            other_data_path, sheet_name="emission_factor_waste", index_col=0
        )

        possible_concentrations = emission_factor_data.columns.tolist()
        co2_concentration = self.performance_data["co2_concentration"]

        interp = interp1d(possible_concentrations, emission_factor_data.loc["emission_factor_tco2_twaste"], kind="linear",
                                          fill_value="extrapolate")
        emission_factor = interp(co2_concentration)
        self.performance_data["emission_factor"] = emission_factor
        time_steps = len(self.set_t_performance)
        th_efficiency = self.performance_data["th_efficiency"]
        el_efficiency = self.performance_data["el_efficiency"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]
        el_efficiency_CaL = self.performance_data["el_efficiency_CaL"]
        th_input_CaL = self.performance_data["th_input_CaL"]
        ccr = self.performance_data["capture_rate"]
        lhv = self.performance_data["LHV"]
        lhv_RDF = self.performance_data["LHV_RDF"]
        bound_factor_size_max_cal = emission_factor * ccr * (
                1 + th_input_CaL / lhv_RDF * emission_factor_RDF * ccr)
        #
        # # Input Bounds
        # self.bounds["input"]["wasteIn"] = np.column_stack(
        #     (
        #         np.zeros(shape=(time_steps)),
        #         np.ones(shape=time_steps)
        #         ,
        #     )
        # )
        #
        # self.bounds["input"]["wasteInRDF"] = np.column_stack(
        #     (
        #         np.zeros(shape=(time_steps)),
        #         np.ones(shape=time_steps)
        #         * emission_factor * ccr * th_input_CaL / lhv_RDF
        #         ,
        #     )
        # )
        # # Output Bounds
        # self.bounds["output"]["heat"] = np.column_stack(
        #     (
        #         np.zeros(shape=(time_steps)),
        #         np.ones(shape=time_steps)
        #         * th_efficiency * lhv,
        #     )
        # )
        #
        # self.bounds["output"]["electricity"] = np.column_stack(
        #     (
        #         np.zeros(shape=(time_steps)),
        #         np.ones(shape=time_steps)
        #         * (el_efficiency * lhv
        #         + bound_factor_size_max_cal*lhv_RDF*el_efficiency_CaL),
        #     )
        # )
        # self.bounds["output"]["wasteProcessed"] = np.column_stack(
        #     (
        #         np.zeros(shape=(time_steps)),
        #         np.ones(shape=time_steps)
        #         ,
        #     )
        # )
        # self.bounds["output"]["CO2captured"] = np.column_stack(
        #     (
        #         np.zeros(shape=(time_steps)),
        #         np.ones(shape=time_steps)* (emission_factor * ccr
        #         + emission_factor * ccr * th_input_CaL / lhv_RDF * emission_factor_RDF * ccr)
        #         ,
        #     )
        # )


        # Input Bounds
        self.bounds["input"]["wasteIn"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                ,
            )
        )

        self.bounds["input"]["wasteInRDF"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * 10000
                ,
            )
        )
        # Output Bounds
        self.bounds["output"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * 10000,
            )
        )

        self.bounds["output"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * 10000,
            )
        )
        self.bounds["output"]["wasteProcessed"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                ,
            )
        )
        self.bounds["output"]["CO2captured"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)* 10000
                ,
            )
        )


    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type WasteToEnergyCaLCCS

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(WasteToEnergyCaLCCS, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        th_efficiency = self.performance_data["th_efficiency"]
        el_efficiency = self.performance_data["el_efficiency"]
        emission_factor = self.performance_data["emission_factor"]
        el_efficiency_CaL = self.performance_data["el_efficiency_CaL"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]
        th_input_CaL = self.performance_data["th_input_CaL"]
        ccr = self.performance_data["capture_rate"]
        lhv = self.performance_data["LHV"]
        lhv_RDF = self.performance_data["LHV_RDF"]
        size_max_cal = b_tec.para_size_max*emission_factor*ccr*(1+ th_input_CaL/lhv_RDF* emission_factor_RDF * ccr)

        def init_size_cal(const, t):
            return self.output[t, "CO2captured"] <= b_tec.var_size_cal

        b_tec.const_size_max_cal = pyo.Constraint(
            self.set_t_performance, rule=init_size_cal
        )

        def init_size_waste_max(const, t):
            return self.input[t, "wasteIn"] <= b_tec.var_size

        b_tec.const_size_max = pyo.Constraint(
            self.set_t_performance, rule=init_size_waste_max
        )

        b_tec.var_el_cal = pyo.Var(
            self.set_t_performance,
            domain=pyo.NonNegativeReals,
            bounds=(0, size_max_cal*th_input_CaL*el_efficiency_CaL)
        )

        b_tec.var_el_wte = pyo.Var(
            self.set_t_performance,
            domain=pyo.NonNegativeReals,
            bounds=(0, b_tec.para_size_max*lhv*el_efficiency)
        )

        #Force the el production from CaL to be linked to the CO2 capturing (RDF is linked to CO2captured later)
        def init_el_cal(const, t):
            return b_tec.var_el_cal[t] == self.input[t, "wasteInRDF"]*lhv_RDF*el_efficiency_CaL

        b_tec.const_el_cal = pyo.Constraint(self.set_t_performance, rule=init_el_cal)

        def init_el_wte(const, t):
            return b_tec.var_el_wte[t] <= self.input[t, "wasteIn"] * lhv * el_efficiency

        b_tec.const_el_wte = pyo.Constraint(self.set_t_performance, rule=init_el_wte)


        def init_total_output_wte(const, t):
            return (
                self.output[t, "heat"]/th_efficiency + b_tec.var_el_wte[t]/el_efficiency
                == self.input[t, "wasteIn"]*lhv
            )


        b_tec.const_total_output_wte = pyo.Constraint(
            self.set_t_performance, rule=init_total_output_wte
        )

        def init_input_output(const, t, car):
            if car == "wasteProcessed":
                return (
                        self.output[t, car]
                        == self.input[t, "wasteIn"]
                )
            if car == "heat":
                return (
                    self.output[t, car]
                    <= self.input[t, "wasteIn"] * lhv * th_efficiency
                )
            if car == "electricity":
                return (
                    self.output[t, car]
                    == b_tec.var_el_wte[t]
                    + b_tec.var_el_cal[t]
                )
            if car == "CO2captured":
                return (
                    self.output[t, car] * th_input_CaL
                    == self.input[t, "wasteInRDF"] * lhv_RDF
                )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_output_carriers, rule=init_input_output
        )



        def init_max_co2_captured(const, t):
            return (self.output[t, "CO2captured"] <= self.input[t, "wasteIn"] * emission_factor * ccr
                    * (1+th_input_CaL/lhv_RDF*emission_factor_RDF*ccr))

        b_tec.const_max_captured = pyo.Constraint(self.set_t_performance, rule=init_max_co2_captured)


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
        return b_tec


    def _define_emissions(self, b_tec):
        """
        Defines Emissions

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        c = self.processed_coeff.time_independent
        technology_model = self.technology_model
        emission_factor = self.performance_data["emission_factor"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]

        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        def init_tec_emissions_pos(const, t):
            return (
                self.input[t, "wasteIn"] * emission_factor
                + self.input[t, "wasteInRDF"] * emission_factor_RDF
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
        b_tec.para_unit_capex = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["unit_capex"],
            mutable=True,
        )

        capex_data_path = Path(__file__).parent.parent.parent.parent
        capex_data_path = (
                capex_data_path
                / "database/templates/technology_data/Industrial/WasteCaL_data/wasteCaL_sheet.xlsx"
        )

        capex_data = pd.read_excel(
            capex_data_path, sheet_name="capex_eur", index_col=0
        )

        self.economics["bp_x_capex_cal"] = capex_data.index.tolist()
        possible_concentrations = capex_data.columns.tolist()
        co2_concentration = self.performance_data["co2_concentration"]
        bp_y_capex_cal_adjusted = []
        for s in self.economics["bp_x_capex_cal"]:
            capex_interpolated = interp1d(possible_concentrations, capex_data.loc[s], kind="linear", fill_value="extrapolate")
            bp_y_capex_cal_adjusted.append(capex_interpolated(co2_concentration))

        self.economics["bp_y_capex_cal"] = bp_y_capex_cal_adjusted


        if self.existing and not self.decommission == "impossible":
            b_tec.para_decommissioning_cost_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=annualization_factor * economics.decommission_cost,
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



        def calculate_max_capex():
            max_capex = (
                                max(self.economics["bp_y_capex_cal"])
                        ) * annualization_factor
            bounds = (0, max_capex)
            return bounds


        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX

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


        # Capex calcium looping as a piecewise function
        self.big_m_transformation_required = 1
        bp_x = economics["bp_x_capex_cal"]
        bp_y_annual = [
            y * annualization_factor
            for y in economics["bp_y_capex_cal"]
        ]
        b_tec.const_capex_cal = pyo.Piecewise(
            b_tec.var_capex_aux,
            b_tec.var_size_cal,
            pw_pts=bp_x,
            pw_constr_type="EQ",
            f_rule=bp_y_annual,
            pw_repn="SOS2",
        )


        # CAPEX
        if self.existing:
            if self.decommission == "impossible":
                # technology cannot be decommissioned
                b_tec.const_capex = pyo.Constraint(expr=b_tec.var_capex == 0)
            else:
                raise ValueError(
                    f"Decommissioning option '{self.decommission}' is not valid for {self.name}."
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

        # VARIABLE OPEX
        b_tec.para_opex_variable = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_variable"], mutable=True
        )
        b_tec.var_opex_variable = pyo.Var()


        hour_factors = data["hour_factors"]
        nr_timesteps_averaged = data["nr_timesteps_averaged"]
        def init_opex_variable(const):
            """opexvar = sum(Input_{t, maincarrier}) * opex_{var}"""
            return (
                sum(
                    (
                        b_tec.var_output[t, "CO2captured"]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                    )
                    * b_tec.para_opex_variable
                    for t in self.set_t_global
                )
                == b_tec.var_opex_variable
            )

        b_tec.const_opex_variable = pyo.Constraint(rule=init_opex_variable)

        # FIXED OPEX
        b_tec.para_opex_fixed = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["opex_fixed"],
            mutable=True,
        )


        b_tec.var_opex_fixed = pyo.Var()
        b_tec.const_opex_fixed = pyo.Constraint(
            expr=(b_tec.var_capex / annualization_factor)
            * b_tec.para_opex_fixed
            == b_tec.var_opex_fixed
        )
        return b_tec


    def write_results_tec_design(self, h5_group, model_block):
        """
        Function to report technology design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(WasteToEnergyCaLCCS, self).write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset(
            "size_cal", data=[model_block.var_size_cal.value]
        )

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(WasteToEnergyCaLCCS, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "el_cal",
            data=[model_block.var_el_cal[t].value for t in self.set_t_full],
        )
        h5_group.create_dataset(
            "el_wte_only",
            data=[model_block.var_el_wte[t].value for t in self.set_t_full],
        )