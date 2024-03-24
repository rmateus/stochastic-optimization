import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from BaseClasses.SDPPolicy import SDPPolicy
from math import sqrt, log
import numpy as np


class UCB(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "UCB", theta: float = 1):
        self.theta = theta
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        # this method implements the Upper Confidence Bound policy
        # N.B: can't implement this at time t=0 (from t=1 at least).
        # Also can't divide by zero, which means we need each drug to have been tested at least once.
        #
        # Note that state has a list of 3 entries, for each key(type of drug) in the dictionary
        # {"drug" : [mu_empirical, beta, number of times drug given to patient]}
        obj_approx = {}
        for s in state._fields:
            mu, beta, N = getattr(state, s)
            obj_approx[s] = mu + self.theta * sqrt(log(t + 1) / (N + 1))

        optimal_decision = max(obj_approx, key=obj_approx.get)

        return {"choice": optimal_decision}


class IE(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "IE", theta: float = 1):
        self.theta = theta
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        obj_approx = {}
        for s in state._fields:
            mu, beta, N = getattr(state, s)
            sigma = 1 / sqrt(beta)
            obj_approx[s] = mu + self.theta * sigma

        optimal_decision = max(obj_approx, key=obj_approx.get)

        return {"choice": optimal_decision}


class PureExploitation(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "PureExploitation"):
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        obj_approx = {}
        for s in state._fields:
            mu, beta, N = getattr(state, s)
            obj_approx[s] = mu

        optimal_decision = max(obj_approx, key=obj_approx.get)

        return {"choice": optimal_decision}


class PureExploration(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "PureExploration", seed: int = 42):
        self.prng = np.random.RandomState(seed)
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        optimal_decision = self.prng.choice(state._fields)

        return {"choice": optimal_decision}
