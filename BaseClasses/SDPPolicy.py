from copy import copy
from abc import ABC, abstractmethod
import pandas as pd
from . import SDPModel


class SDPPolicy(ABC):
    def __init__(self, model: SDPModel, policy_name: str = ""):
        self.model = model
        self.policy_name = policy_name
        self.results = pd.DataFrame()

    @abstractmethod
    def get_decision(self, state):
        """
        Returns the decision made by the policy based on the given state.

        Args:
            state (namedtuple): The current state of the system.

        Returns:
            dict: The decision made by the policy.
        """
        pass

    def run_policy(self, n_iterations: int = 1):
        """
        Runs the policy over the time horizon [0,T] for a specified number of iterations.

        Args:
            n_iterations (int): The number of iterations to run the policy. Default is 1.

        Returns:
            None
        """
        result_list = []
        # Note: the random number generator is not reset when calling copy()
        # (only when calling deepcopy())
        for i in range(n_iterations):
            model_copy = copy(self.model)
            while model_copy.is_finished() is False:
                state_n = model_copy.state
                decision_n = model_copy.build_decision(self.get_decision(state_n))

                # Logging
                results_dict = {"N": i, "t": model_copy.t, "obj": model_copy.objective}
                results_dict.update(state_n._asdict())
                results_dict.update(decision_n._asdict())
                result_list.append(results_dict)

                state_n_plus_1 = model_copy.step(decision_n)

            results_dict = {"N": i, "t": model_copy.t, "obj": model_copy.objective}
            results_dict.update(state_n_plus_1._asdict())
            result_list.append(results_dict)

        self.results = pd.DataFrame.from_dict(result_list)
