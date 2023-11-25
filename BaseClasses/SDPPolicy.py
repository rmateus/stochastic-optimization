from collections import namedtuple
from copy import copy
from abc import ABC, abstractmethod
from SDPModel import SDPModel


class SDPPolicy(ABC):
    def __init__(
        self, model: SDPModel, policy_name: str = "", policy_parameters: dict = {}
    ):
        self.model = model
        self.policy_name = policy_name
        self.PolicyParameters = namedtuple("PolicyParameters", policy_parameters)
        self.policy_parameters = self.PolicyParameters(**policy_parameters)
        # TO DO: logging functionality:
        # log states, decisions, objectives for all t, all iterations

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
        # Note: the random number generator is not reset when calling copy()
        # (only when calling deepcopy())
        for i in range(n_iterations):
            model_copy = copy(self.model)
            while model_copy.is_finished() is False:
                decision = model_copy.build_decision(
                    self.get_decision(model_copy.state)
                )
                state = model_copy.step(decision)
