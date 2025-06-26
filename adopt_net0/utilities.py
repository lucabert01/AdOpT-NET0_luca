from pyomo.environ import SolverFactory


def get_gurobi_parameters(solveroptions: dict):
    """
    Initiates the gurobi solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory(solveroptions["solver"]["value"], solver_io="python")
    solver.options["TimeLimit"] = solveroptions["timelim"]["value"] * 3600
    solver.options["MIPGap"] = solveroptions["mipgap"]["value"]
    solver.options["MIPFocus"] = solveroptions["mipfocus"]["value"]
    solver.options["Threads"] = solveroptions["threads"]["value"]
    solver.options["NodefileStart"] = solveroptions["nodefilestart"]["value"]
    solver.options["Method"] = solveroptions["method"]["value"]
    solver.options["Heuristics"] = solveroptions["heuristics"]["value"]
    solver.options["Presolve"] = solveroptions["presolve"]["value"]
    solver.options["BranchDir"] = solveroptions["branchdir"]["value"]
    solver.options["LPWarmStart"] = solveroptions["lpwarmstart"]["value"]
    solver.options["IntFeasTol"] = solveroptions["intfeastol"]["value"]
    solver.options["FeasibilityTol"] = solveroptions["feastol"]["value"]
    solver.options["Cuts"] = solveroptions["cuts"]["value"]
    solver.options["NumericFocus"] = solveroptions["numericfocus"]["value"]

    return solver


def get_glpk_parameters(solveroptions: dict):
    """
    Initiates the glpk solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory("glpk")

    return solver


def get_set_t(config: dict, model_block):
    """
    Returns the correct set_t for different clustering options

    :param dict config: config dict
    :param model_block: pyomo block holding set_t_full and set_t_clustered
    :return: set_t
    """
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        return model_block.set_t_full
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        return model_block.set_t_clustered
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        return model_block.set_t_full


def get_hour_factors(config: dict, data, period: str) -> list:
    """
    Returns the correct hour factors to use for global balances

    :param dict config: config dict
    :param data: DataHandle
    :return: hour factors
    """
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        return [1] * len(data.topology["time_index"]["full"])
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        return data.k_means_specs[period]["factors"]
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        return [1] * len(data.topology["time_index"]["full"])


def get_nr_timesteps_averaged(config: dict) -> int:
    """
    Returns the correct number of timesteps averaged

    :param dict config: config dict
    :return: nr_timesteps_averaged
    """
    if config["optimization"]["timestaging"]["value"] != 0:
        nr_timesteps_averaged = config["optimization"]["timestaging"]["value"]
    else:
        nr_timesteps_averaged = 1

    return nr_timesteps_averaged


def determine_flow_existing_compressors(compr, compressor, b_period, node):
    component_output_bound = float("inf")
    component_input_bound = float("inf")
    type_component = [compr.para_type_output, compr.para_type_input]
    if type_component[0].value == "Technology":
        var_output = (
            b_period.node_blocks[node].tech_blocks_active[compressor[1]].var_output
        )
        component_output_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[0].value == "Network":
        component_output_bound = next(
            iter(b_period.network_block[compressor[1]].para_size_initial.values())
        )
    elif type_component[0].value == "Exchange":
        pass

    if type_component[1].value == "Technology":
        var_output = (
            b_period.node_blocks[node].tech_blocks_active[compressor[2]].var_output
        )
        component_input_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[1].value == "Network":
        component_input_bound = next(
            iter(b_period.network_block[compressor[2]].para_size_initial.values())
        )
    elif type_component[1].value == "Exchange":
        pass

    size = min(component_output_bound, component_input_bound)

    return size
