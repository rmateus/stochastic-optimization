from itertools import product
from copy import deepcopy
from . import SDPPolicy


def grid_search(grid: dict, policy: SDPPolicy.SDPPolicy, n_iterations: int, ordered: bool = False):
    if len(grid) != 2 and ordered:
        ordered = False
        print("Warning: Grid search for ordered parameters only works if there are only two parameters.")
    best_performance = 0.0
    best_parameters = None
    performance_list = []
    parameter_list = []
    params = grid.keys()
    for v in product(*grid.values()):
        if ordered:
            if v[0] >= v[1]:
                continue

        # Do a deep copy so all parameter sets get the same random numbers
        policy_copy = deepcopy(policy)

        for param, value in zip(params, v):
            policy_copy.__setattr__(param, value)

        performance = policy_copy.run_policy(n_iterations=n_iterations)

        performance_list.append(performance)
        parameter_list.append(dict(zip(params, v)))
        if performance > best_performance:
            best_performance = performance
            best_parameters = dict(zip(params, v))

    return {
        "best_parameters": best_parameters,
        "best_performance": best_performance,
        "parameter_list": parameter_list,
        "performance_list": performance_list,
    }
