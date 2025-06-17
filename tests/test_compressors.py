import warnings

import pyomo.core.base.param
import pytest
from pathlib import Path
from pyomo.environ import ConcreteModel, Set, Constraint, TerminationCondition
import json
import numpy as np

from tests.utilities import (
    make_climate_data,
    make_data_for_testing,
    run_model,
)
from adopt_net0.components.compressors.compressor import Compressor
from adopt_net0.data_management.utilities import open_json
from adopt_net0.components.utilities import annualize
from adopt_net0.components.utilities import perform_disjunct_relaxation


def define_compressor(
    compressor_name: str,
    load_path: Path,
    perf_type: int = None,
    capex_model: int = None,
    existing: int = 0,
    size_initial: float = 0,
    decommission: str = "impossible",
):
    """
    Reads compressor data and fits it

    :param str compressor_name: name of the compressor.
    :param Path load_path: Path to load from
    :param int perf_type: performance function type (for generic conversion tecs)
    :param int capex_model: capex model (1,2,3,4)
    :param int existing: is technology existing or not,
    :param float size_initial: initial size of existing technology,
    :param str decommission: type of decommissioning "impossible", "continuous", "only_complete"
    :return: Technology class
    """
    # Compressor Class Creation
    with open(load_path / (compressor_name + ".json")) as json_file:
        compr = json.load(json_file)
    compr["name"] = compressor_name

    if perf_type:
        compr["Performance"]["performance_function_type"] = perf_type
    if capex_model:
        compr["Economics"]["capex_model"] = capex_model

    compr = Compressor(compr)

    if existing:
        compr.existing = existing
        compr.size_initial = size_initial
        compr.decommission = decommission

    compr.fit_technology_performance(climate_data, location)

    return compr
