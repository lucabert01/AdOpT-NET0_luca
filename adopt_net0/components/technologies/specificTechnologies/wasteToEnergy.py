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

        self.component_options.emissions_based_on = "input"
        self.component_options.size_based_on = "input"
        self.component_options.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]




    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(WasteToEnergy, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)
        th_efficiency = self.input_parameters.performance_data["th_efficiency"]
        el_efficiency = self.input_parameters.performance_data["el_efficiency"]
        lhv = self.input_parameters.performance_data["LHV"]

        # Output Bounds
        self.bounds["output"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * th_efficiency * lhv,
            )
        )

        self.bounds["output"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * el_efficiency * lhv,
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
        self.bounds["input"]["wasteFuel"] = np.column_stack(
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


        th_efficiency = self.input_parameters.performance_data["th_efficiency"]
        el_efficiency = self.input_parameters.performance_data["el_efficiency"]
        lhv = self.input_parameters.performance_data["LHV"]


        def init_size_waste_max(const, t):
            return self.input[t, "wasteFuel"] <= b_tec.var_size

        b_tec.const_size_max = pyo.Constraint(
            self.set_t_performance, rule=init_size_waste_max
        )



        def init_input_output(const, t, car):
            if car == "wasteProcessed":
                return (
                        self.output[t, car]
                        == self.input[t, "wasteFuel"]
                )
            if car == "heat":
                return (
                    self.output[t, car]
                    <= self.input[t, "wasteFuel"] * lhv * th_efficiency
                )
            if car == "electricity":
                return (
                    self.output[t, car]
                    <= self.input[t, "wasteFuel"] * lhv * el_efficiency
                )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_output_carriers, rule=init_input_output
        )

        def init_total_output(const, t):
            return (
                self.output[t, "heat"]/th_efficiency + self.output[t, "electricity"]/el_efficiency
                == self.input[t, "wasteFuel"]*lhv
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




