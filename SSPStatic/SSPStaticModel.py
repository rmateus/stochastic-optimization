import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
import pandas as pd
import networkx as nx
import numpy as np


class SSPStatic(SDPModel):
    def __init__(
        self,
        # S0: dict,
        # t0: float = 0,
        # T: float = 1,
        seed: int = 42,
        G: nx.Graph = None,
        origin: int = None,
        destination: int = None,
        edge_weight: str = "travel_time",
        lower_bound: float = 0.8,
        upper_bound: float = 2.0,
        stepsize_rule: str = "constant",
    ) -> None:

        self.G = G
        self.origin = origin
        self.path = [self.origin]
        self.destination = destination
        self.edge_weight = edge_weight
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        state_names = ["CurrentNode", "CurrentNodeLinkCosts"]
        decision_names = ["NextNode"]
        T = G.number_of_nodes()
        t0 = 0

        cost_dict = {}
        for edge in self.G.out_edges(origin):
            edge_data = self.G.edges[edge + (0,)]
            left = edge_data[self.edge_weight] * self.lower_bound
            right = edge_data[self.edge_weight] * self.upper_bound
            mode = edge_data[self.edge_weight]
            cost_dict[edge] = np.random.RandomState(seed).triangular(left=left, mode=mode, right=right)

        S0 = {"CurrentNode": origin, "CurrentNodeLinkCosts": cost_dict}

        super().__init__(state_names, decision_names, S0, t0, T, seed)

        shortest_paths = nx.shortest_path(G, target=self.destination, weight=edge_weight)
        self.V_t = {}
        for node in self.G.nodes:
            if node in shortest_paths:
                self.V_t[node] = self.calc_path_length(shortest_paths[node])

        self.stepsize_rule = stepsize_rule

        # TODO: catch if meaningless parameters for graph are passed

    def calc_path_length(self, list_of_nodes):
        path_length = 0.0
        current_node = list_of_nodes[0]
        for i in range(1, len(list_of_nodes)):
            path_length += self.G.edges[(current_node, list_of_nodes[i], 0)][self.edge_weight]
            current_node = list_of_nodes[i]
        return path_length

    def exog_info_fn(self, decision):
        cost_dict = {}
        i = decision.NextNode
        for edge in self.G.out_edges(i):
            edge_data = self.G.edges[edge + (0,)]
            left = edge_data[self.edge_weight] * self.lower_bound
            right = edge_data[self.edge_weight] * self.upper_bound
            mode = edge_data[self.edge_weight]
            cost_dict[edge] = self.prng.triangular(left=left, mode=mode, right=right)

        return {"CurrentNodeLinkCosts": cost_dict}

    def is_finished(self):
        """
        Check if the model run (episode) is finished.
        This is the case when we reached the destination.

        Returns:
            bool: True if the run is finished, False otherwise.
        """
        return self.state.CurrentNode == self.destination

    def transition_fn(self, decision, exog_info: dict):
        return {"CurrentNode": decision.NextNode}

    def objective_fn(self, decision, exog_info: dict):
        # print(self.state.CurrentNodeLinkCosts)
        return self.state.CurrentNodeLinkCosts[(self.state.CurrentNode, decision.NextNode)]

    def update_VFA(self, vhat):
        # TODO: implement VFA update/learning procedure
        self.V_t[str(self.state.CurrentNode)] = (1 - self.alpha()) * self.V_t[
            str(self.state.CurrentNode)
        ] + self.alpha() * vhat
        return self.V_t[str(self.state.CurrentNode)]

    def alpha(self):
        if self.stepsize_rule == "constant":
            return self.theta_step
        else:
            return self.theta_step / (self.theta_step + self.n - 1)
