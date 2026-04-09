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


class WasteToEnergy(Technology):
    """
    Waste to Energy with combined heat and power

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

        self.size_based_on = "input"
        self.emissions_based_on = "input"
        self.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]


    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits the technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(WasteToEnergy, self).fit_technology_performance(climate_data, location)


        lhv_hourly = []
        emission_factor_hourly = []
        if self.performance_data["waste_prop_interp"]["waste_as_function_of_co2_conc"]:
            waste_prop = self.performance_data["waste_prop_interp"]

            x_values_conc = np.array(waste_prop["ref_co2_conc"])
            y_values_lhv = np.array(waste_prop["ref_lhv"])
            y_values_ef = np.array(waste_prop["ref_emission_factor"])
            function_lhv = interp1d(x_values_conc, y_values_lhv, fill_value="extrapolate")
            function_ef = interp1d(x_values_conc, y_values_ef, fill_value="extrapolate")
            if not self.performance_data.get("ccs"):
                raise ValueError(
                    f'You need to define the "ccs" block in the JSON file of {self.name} if you set waste_as_function_of_co2_conc=1')

            if self.performance_data["ccs"]["co2_concentration_is_hourly"]:

                co2_concentration = climate_data[f"co2_concentration_{self.name}"]

                for t in range(len(co2_concentration)):
                    lhv_hourly.append(function_lhv(co2_concentration.iloc[t]))
                    emission_factor_hourly.append(function_ef(co2_concentration.iloc[t]))

                self.processed_coeff.time_dependent_full["lhv_msw_hourly"] = lhv_hourly
                self.processed_coeff.time_dependent_full["emission_factor_msw_hourly"] = emission_factor_hourly
                self.processed_coeff.time_independent["max_lhv_msw"] = max(lhv_hourly)
                self.processed_coeff.time_independent["max_emission_factor_msw"] = max(emission_factor_hourly)
            else:
                co2_concentration = self.ccs_data["co2_concentration"]
                self.processed_coeff.time_independent["max_lhv_msw"] = function_lhv(co2_concentration)
                self.processed_coeff.time_independent["max_emission_factor_msw"] = function_ef(co2_concentration)
        else:
            self.processed_coeff.time_independent["max_lhv_msw"] = self.performance_data["LHV"]
            self.processed_coeff.time_independent["max_emission_factor_msw"] = self.performance_data["emission_factor"]

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(WasteToEnergy, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)
        th_efficiency = self.performance_data["th_efficiency"]
        el_efficiency = self.performance_data["el_efficiency"]
        max_lhv = self.processed_coeff.time_independent["max_lhv_msw"]

        # Output Bounds
        self.bounds["output"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * th_efficiency * max_lhv,
            )
        )

        self.bounds["output"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * el_efficiency * max_lhv,
            )
        )
        self.bounds["output"]["wasteProcessed"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                ,
            )
        )

        # Input Bounds
        self.bounds["input"]["wasteIn"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                ,
            )
        )


    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type WasteToEnergy

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(WasteToEnergy, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )


        th_efficiency = self.performance_data["th_efficiency"]
        el_efficiency = self.performance_data["el_efficiency"]
        lhv = self.processed_coeff.time_independent["max_lhv_msw"]
        lhv_msw_hourly = self.processed_coeff.time_dependent_full["lhv_msw_hourly"]


        def init_size_waste_max(const, t):
            return self.input[t, "wasteIn"] <= b_tec.var_size

        b_tec.const_size_max = pyo.Constraint(
            self.set_t_performance, rule=init_size_waste_max
        )



        def init_input_output(const, t, car):
            if car == "wasteProcessed":
                return (
                        self.output[t, car]
                        == self.input[t, "wasteIn"]
                )
            if car == "heat":
                if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                    return (
                        self.output[t, car]
                        <= self.input[t, "wasteIn"] * lhv_msw_hourly[t-1] * th_efficiency
                    )
                else:
                    return (
                            self.output[t, car]
                            <= self.input[t, "wasteIn"] * lhv * th_efficiency
                    )
            if car == "electricity":
                if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                    return (
                        self.output[t, car]
                        <= self.input[t, "wasteIn"] * lhv_msw_hourly[t-1] * el_efficiency
                    )
                else:
                    return (
                        self.output[t, car]
                        <= self.input[t, "wasteIn"] * lhv * el_efficiency
                    )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_output_carriers, rule=init_input_output
        )

        def init_total_output(const, t):
            if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                return (
                    self.output[t, "heat"]/th_efficiency + self.output[t, "electricity"]/el_efficiency
                    == self.input[t, "wasteIn"]*lhv_msw_hourly[t-1]
                )
            else:
                return (
                        self.output[t, "heat"] / th_efficiency + self.output[t, "electricity"] / el_efficiency
                        == self.input[t, "wasteIn"] * lhv
                )


        b_tec.const_total_output = pyo.Constraint(
            self.set_t_performance, rule=init_total_output
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
        return b_tec




    def _define_ccs_performance(self, b_tec, data: dict):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """

        coeff_ti = self.ccs_component.processed_coeff.time_independent
        coeff_td = self.ccs_component.processed_coeff.time_dependent_full

        emission_factor = self.processed_coeff.time_independent["max_emission_factor_msw"]
        emission_factor_hourly = self.processed_coeff.time_dependent_full["emission_factor_msw_hourly"]

        capture_rate = coeff_ti["capture_rate"]

        # Initialize the size of CCS as in _define_size (size given in mass flow of CO2 entering the CCS object)
        b_tec.para_size_min_ccs = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=self.ccs_component.size_min,
            mutable=True,
        )
        b_tec.para_size_max_ccs = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=self.ccs_component.size_max,
            mutable=True,
        )

        # Size CCS
        b_tec.var_size_ccs = pyo.Var(
            within=pyo.NonNegativeReals,
            bounds=(0, b_tec.para_size_max_ccs),
        )

        # TODO: maybe make the full set of all carriers as an intersection between this set and  the others?

        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        def init_input_bounds(bounds, t, car):
            return tuple(
                self.ccs_component.bounds["input"][car][self.sequence[t - 1] - 1, :]
                * coeff_ti["size_max"]
            )

        b_tec.var_input_ccs = pyo.Var(
            self.set_t_global,
            b_tec.set_input_carriers_ccs,
            within=pyo.NonNegativeReals,
            bounds=init_input_bounds,
        )

        def init_output_bounds(bounds, t, car):
            return tuple(
                self.ccs_component.bounds["output"][car][self.sequence[t - 1] - 1, :]
                * coeff_ti["size_max"]
            )

        b_tec.var_output_ccs = pyo.Var(
            self.set_t_global,
            b_tec.set_output_carriers_ccs,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )

        # Input-output correlation
        def init_input_output_ccs(const, t):
            if self.emissions_based_on == "output":
                if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                    return (
                        b_tec.var_output_ccs[t, "CO2captured"]
                        <= capture_rate
                        * emission_factor_hourly[t-1]
                        * b_tec.var_output[t, self.main_output_carrier]
                    )
                else:
                    return (
                            b_tec.var_output_ccs[t, "CO2captured"]
                            <= capture_rate
                            * emission_factor
                            * b_tec.var_output[t, self.main_output_carrier]
                    )
            else:
                if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                    return (
                        b_tec.var_output_ccs[t, "CO2captured"]
                        <= capture_rate
                        * emission_factor_hourly[t-1]
                        * b_tec.var_input[t, self.main_input_carrier]
                    )
                else:
                    return (
                            b_tec.var_output_ccs[t, "CO2captured"]
                            <= capture_rate
                            * emission_factor
                            * b_tec.var_input[t, self.main_input_carrier]
                    )

        b_tec.const_input_output_ccs = pyo.Constraint(
            self.set_t_global, rule=init_input_output_ccs
        )

        def init_size_output_ccs(const, t):
            return b_tec.var_output_ccs[t, "CO2captured"] <= b_tec.var_size_ccs

        b_tec.const_size_output_ccs = pyo.Constraint(
            self.set_t_global, rule=init_size_output_ccs
        )

        # Electricity and heat demand CCS
        def init_input_ccs(const, t, car):
            if self.performance_data["ccs"].get("co2_concentration_is_hourly", False):
                return (
                    b_tec.var_input_ccs[t, car]
                    == coeff_td["input_ratios"][car][t-1]
                    * b_tec.var_output_ccs[t, "CO2captured"]
                    / capture_rate
                )
            else:
                return (
                    b_tec.var_input_ccs[t, car]
                    == coeff_ti["input_ratios"][car]
                    * b_tec.var_output_ccs[t, "CO2captured"]
                    / capture_rate
                )

        b_tec.const_input_el = pyo.Constraint(
            self.set_t_global, b_tec.set_input_carriers_ccs, rule=init_input_ccs
        )

        return b_tec

    def _define_ccs_emissions(self, b_tec):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        emission_factor = self.processed_coeff.time_independent["max_emission_factor_msw"]
        emission_factor_hourly = self.processed_coeff.time_dependent_full["emission_factor_msw_hourly"]

        # Emissions
        if self.emissions_based_on == "output":

            def init_tec_emissions_pos(const, t):
                if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                    return (
                        b_tec.var_output[t, self.main_output_carrier]
                        * emission_factor_hourly[t-1]
                        - b_tec.var_output_ccs[t, "CO2captured"]
                        == b_tec.var_tec_emissions_pos[t]
                    )
                else:
                    return (
                            b_tec.var_output[t, self.main_output_carrier]
                            * emission_factor
                            - b_tec.var_output_ccs[t, "CO2captured"]
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

        elif self.emissions_based_on == "input":

            def init_tec_emissions_pos(const, t):
                if self.performance_data["ccs"]["co2_concentration_is_hourly"]:
                    return (
                        b_tec.var_input[t, self.main_input_carrier]
                        * emission_factor_hourly[t-1]
                        - b_tec.var_output_ccs[t, "CO2captured"]
                        == b_tec.var_tec_emissions_pos[t]
                    )
                else:
                    return (
                            b_tec.var_input[t, self.main_input_carrier]
                            * emission_factor
                            - b_tec.var_output_ccs[t, "CO2captured"]
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

