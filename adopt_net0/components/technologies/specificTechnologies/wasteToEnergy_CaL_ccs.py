import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pathlib import Path
from ...utilities import annualize, set_discount_rate
from ..technology import Technology
from scipy.interpolate import interp1d


class WasteToEnergyCaLCCS(Technology):
    """
    Waste-to-energy plant with an add-on calcium looping (CaL) unit for CO2 capture.

    No electricity/heat output and no hourly LHV/CO2-concentration variation --
    everything is a fixed, scalar design value. RDF consumption needed to run the
    calciner is derived from the amount of CO2 captured via a fixed ratio
    (Performance.rdf_per_tCO2, t RDF per t CO2 captured); RDF's own emissions feed
    into the raw-emissions/capture balance, but RDF is not itself a node-connected
    carrier -- its cost (and CaL's electricity revenue, and any other variable cost)
    are folded into Economics.opex_variable_cal [EUR/t CO2 captured, can be negative
    if net revenue].

    CAPEX is a piecewise function of CaL design capacity (var_size_cal, t CO2/h
    captured), interpolated from wasteCaL_sheet.xlsx's "capex_eur" sheet by
    Performance.design_co2_concentration -- this part is unchanged from the original,
    more detailed version of this technology.
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.size_based_on = "input"
        self.emissions_based_on = "input"
        self.main_input_carrier = tec_data["Performance"]["main_input_carrier"]

    def _capture_factor(self):
        """
        CO2captured[t] <= capture_factor * wasteIn[t], derived from:
          CO2captured = (wasteIn*emission_factor + rdf_per_tCO2*CO2captured*emission_factor_RDF) * capture_rate
        solved for CO2captured (RDF's own captured-share emissions are self-referential,
        since more captured CO2 requires burning more RDF, which itself emits CO2 that
        also needs capturing).
        """
        emission_factor = self.performance_data["emission_factor"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]
        rdf_per_tCO2 = self.performance_data["rdf_per_tCO2"]
        ccr = self.performance_data["capture_rate"]

        denominator = 1 - ccr * rdf_per_tCO2 * emission_factor_RDF
        if denominator <= 0:
            raise ValueError(
                f"{self.name}: capture_rate * rdf_per_tCO2 * emission_factor_RDF >= 1 "
                f"-- RDF's own captured-share emissions would exceed the CO2 captured, "
                f"which is not physically sensible. Check these three values."
            )
        return ccr * emission_factor / denominator

    def _define_size(self, b_tec):
        """
        Defines variables and parameters related to technology size.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        super()._define_size(b_tec)

        size_max_cal = b_tec.para_size_max.value * self._capture_factor()

        b_tec.var_size_cal = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, size_max_cal),
        )

        return b_tec

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super()._calculate_bounds()

        time_steps = len(self.set_t_performance)
        capture_factor = self._capture_factor()

        # Input Bounds
        self.bounds["input"]["wasteIn"] = np.column_stack(
            (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
        )

        # Output Bounds
        self.bounds["output"]["waste"] = np.column_stack(
            (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
        )
        self.bounds["output"]["CO2captured"] = np.column_stack(
            (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)) * capture_factor)
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
        super().construct_tech_model(b_tec, data, set_t_full, set_t_clustered)

        emission_factor = self.performance_data["emission_factor"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]
        rdf_per_tCO2 = self.performance_data["rdf_per_tCO2"]
        ccr = self.performance_data["capture_rate"]

        # Pins var_size (max wasteIn throughput) to the node's real waste
        # capacity, mirroring cement_hybrid_ccs.py's size_is_fixed/
        # prod_capacity_clinker mechanism -- needed because this technology is
        # assigned "new" (see ALWAYS_NEW_TECHNOLOGIES in defined_functions.py,
        # so its capex reflects a real investment rather than being zeroed out
        # by the existing-technology capex constraint), which would otherwise
        # leave var_size as a free variable bounded only by the technology's
        # generic size_min/size_max, unrelated to this specific node's real
        # throughput. See update_wastecal_ccs_capacities() for how
        # prod_capacity_wte gets set per node.
        if self.performance_data.get("size_is_fixed"):
            prod_capacity_wte = self.performance_data["prod_capacity_wte"]

            def init_size_wte(const):
                return b_tec.var_size == prod_capacity_wte

            b_tec.const_size_wte = pyo.Constraint(rule=init_size_wte)

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

        def init_waste_output(const, t):
            return self.output[t, "waste"] == self.input[t, "wasteIn"]

        b_tec.const_waste_output = pyo.Constraint(
            self.set_t_performance, rule=init_waste_output
        )

        # CO2captured <= (wasteIn*emission_factor + rdf_per_tCO2*CO2captured*emission_factor_RDF) * ccr
        def init_max_co2_captured(const, t):
            return (
                self.output[t, "CO2captured"]
                * (1 - ccr * rdf_per_tCO2 * emission_factor_RDF)
                <= self.input[t, "wasteIn"] * emission_factor * ccr
            )

        b_tec.const_max_captured = pyo.Constraint(
            self.set_t_performance, rule=init_max_co2_captured
        )

        return b_tec

    def _define_emissions(self, b_tec):
        """
        Defines Emissions

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        emission_factor = self.performance_data["emission_factor"]
        emission_factor_RDF = self.performance_data["emission_factor_RDF"]
        rdf_per_tCO2 = self.performance_data["rdf_per_tCO2"]

        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        def init_tec_emissions_pos(const, t):
            # NOTE: uses the raw b_tec.var_input/var_output (always indexed over
            # self.set_t_global), not the self.input/self.output alias -- under
            # typicaldays method 2 with this technology not in
            # technologies_with_full_res, self.input/self.output alias to
            # var_input_aux/var_output_aux, which are only indexed over the smaller
            # self.set_t_performance (clustered) set and would raise a KeyError here.
            return (
                b_tec.var_input[t, "wasteIn"] * emission_factor
                + rdf_per_tCO2 * b_tec.var_output[t, "CO2captured"] * emission_factor_RDF
                - b_tec.var_output[t, "CO2captured"]
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
        Defines the capex parameters: a piecewise CaL capex curve interpolated from
        wasteCaL_sheet.xlsx by design_co2_concentration, plus decommissioning costs.

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

        possible_concentrations = capex_data.columns.tolist()
        co2_concentration = self.performance_data["design_co2_concentration"]
        bp_y_capex_cal_adjusted = []
        for s in capex_data.index.tolist():
            capex_interpolated = interp1d(possible_concentrations, capex_data.loc[s], kind="linear", fill_value="extrapolate")
            bp_y_capex_cal_adjusted.append(capex_interpolated(co2_concentration))

        self.economics["bp_y_capex_cal"] = bp_y_capex_cal_adjusted

        # convert bp_x from kmol_fluegas/h to tCO2_out/h
        mm_co2 = 44.01
        fraction_emissions_wte = self.performance_data["fraction_emissions_wte"]
        capture_rate = self.performance_data["capture_rate"]
        convert_to_co2_out = co2_concentration * mm_co2 / 1000 / fraction_emissions_wte * capture_rate
        self.economics["bp_x_capex_cal"] = (capex_data.index * convert_to_co2_out).tolist()

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
        def calculate_max_capex():
            config = data["config"]
            economics = self.economics
            discount_rate = set_discount_rate(config, economics)
            fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
            annualization_factor = annualize(
                discount_rate, economics["lifetime"], fraction_of_year_modelled
            )
            max_capex = max(self.economics["bp_y_capex_cal"]) * annualization_factor
            return (0, max_capex)

        # CAPEX auxiliary (used to calculate theoretical CAPEX)
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
        Defines variable and fixed OPEX. Variable opex is a single blended rate
        [EUR/t CO2 captured] covering RDF cost, CaL electricity revenue (negative),
        and any other variable cost -- see Economics.opex_variable_cal.

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
        b_tec.para_opex_variable_cal = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_variable_cal"], mutable=True
        )
        b_tec.var_opex_variable = pyo.Var()

        hour_factors = data["hour_factors"]
        nr_timesteps_averaged = data["nr_timesteps_averaged"]

        def init_opex_variable(const):
            """opexvar = sum(CO2captured_t) * opex_variable_cal"""
            return (
                sum(
                    b_tec.var_output[t, "CO2captured"]
                    * nr_timesteps_averaged
                    * hour_factors[t - 1]
                    for t in self.set_t_global
                )
                * b_tec.para_opex_variable_cal
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
            expr=(b_tec.var_capex_aux / annualization_factor)
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
        super().write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset(
            "size_cal", data=[model_block.var_size_cal.value]
        )
