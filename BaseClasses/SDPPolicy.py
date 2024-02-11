from copy import copy, deepcopy
from abc import ABC, abstractmethod
import pandas as pd
from . import SDPModel


class SDPPolicy(ABC):
    def __init__(self, model: SDPModel, policy_name: str = ""):
        self.model = model
        self.policy_name = policy_name
        self.results = pd.DataFrame()
        self.performance = pd.NA

    @abstractmethod
    def get_decision(self, state, t):
        """
        Returns the decision made by the policy based on the given state.

        Args:
            state (namedtuple): The current state of the system.
            t (float): The current time step.

        Returns:
            dict: The decision made by the policy.
        """
        pass

    def run_policy(self, n_iterations: int = 1):
        """
        Runs the policy over the time horizon [0,T] for a specified number of iterations and return the mean performance.

        Args:
            n_iterations (int): The number of iterations to run the policy. Default is 1.

        Returns:
            None
        """
        result_list = []
        # Note: the random number generator is not reset when calling copy().
        # When calling deepcopy(), it is reset (then all iterations are exactly the same).
        for i in range(n_iterations):
            model_copy = copy(self.model)
            while model_copy.is_finished() is False:
                state_t = model_copy.state
                decision_t = model_copy.build_decision(self.get_decision(state_t, model_copy.t))

                # Logging
                results_dict = {"N": i, "t": model_copy.t, "obj": model_copy.objective}
                results_dict.update(state_t._asdict())
                results_dict.update(decision_t._asdict())
                result_list.append(results_dict)

                state_t_plus_1 = model_copy.step(decision_t)

            results_dict = {"N": i, "t": model_copy.t, "obj": model_copy.objective}
            results_dict.update(state_t_plus_1._asdict())
            result_list.append(results_dict)

        # Logging
        self.results = pd.DataFrame.from_dict(result_list)
        # t_end per iteration
        self.results["t_end"] = self.results.groupby("N")["t"].transform("max")

        # performance of one iteration is the cumulative objective at t_end
        self.performance = self.results.loc[self.results["t"] == self.results["t_end"], ["N", "obj"]]
        self.performance = self.performance.set_index("N")

        # For reporting, convert cumulative objective to contribution per time
        self.results["obj"] = self.results.groupby("N")["obj"].diff().shift(-1)
        self.results = self.results.rename(columns={"obj": "C_t"})

        return self.performance.mean().iloc[0]
