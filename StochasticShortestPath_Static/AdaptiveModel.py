"""
Stochastic Shortest Path Extension
Using point estimates

"""
from collections import namedtuple
from AdaptiveUtil import printFormatedDict
import numpy as np


class StaticModel:
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, params, g, V_t):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - need to contain at least information to populate initial state using s_names
        :param exog_info_fn: function -
        :param transition_fn: function -
        :param objective_fn: function -
        :param seed: int - seed for random number generator
        """

        self.init_args = params
        self.prng = np.random.RandomState(self.init_args["seed"])
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple("State", state_names)
        self.Decision = namedtuple("Decision", x_names)
        self.g = g
        self.V_t = V_t

        self.exog_info = self.g

        # Constructing the initial state
        self.init_state = {
            "CurrentNode": self.init_args["start_node"],
            "CurrentNodeLinksCost": self.exog_info_fn(self.init_args["start_node"]),
        }
        self.state = self.build_state(self.init_state)
        print("Initial State")
        self.print_State()

        # value of objective function
        self.obj = 0.0

        # current iteration
        self.n = 1
        # The stepsize will be set outside the constructor
        self.theta_step = 1

        # policy function, given by Bellman's equation
        self.policy = None

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])

    def print_State(self):
        print(
            " CurrentNode: {} and costs on its edges: ".format(self.state.CurrentNode)
        )
        print(printFormatedDict(self.state.CurrentNodeLinksCost))

    def update_VFA(self, vhat):
        self.V_t[str(self.state.CurrentNode)] = (1 - self.alpha()) * self.V_t[
            str(self.state.CurrentNode)
        ] + self.alpha() * vhat
        return self.V_t[str(self.state.CurrentNode)]

    def exog_info_fn(self, i):
        return {
            j: self.prng.uniform(self.g.lower[(i, j)], self.g.upper[(i, j)])
            for j in self.g.edges[i]
        }

    def transition_fn(self, decision):
        self.obj = self.obj + self.state.CurrentNodeLinksCost[decision.NextNode]
        self.state = self.build_state(
            {
                "CurrentNode": decision.NextNode,
                "CurrentNodeLinksCost": self.exog_info_fn(decision.NextNode),
            }
        )
        return self.state

    def objective_fn(self):
        return self.obj

    def alpha(self):
        if self.init_args["stepsize_rule"] == "Constant":
            return self.theta_step
        else:
            return self.theta_step / (self.theta_step + self.n - 1)
