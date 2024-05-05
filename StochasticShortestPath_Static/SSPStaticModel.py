import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
import networkx as nx
import numpy as np
from collections import namedtuple


class SSPStatic(SDPModel):
    def __init__(
        self,
        seed: int = 42,
        G: nx.Graph = None,
        origin: int = None,
        destination: int = None,
        edge_weight: str = "travel_time",
        lower_bound: float = 0.8,
        upper_bound: float = 2.0,
        T: int = 300,
    ) -> None:

        # Weighted digraph
        self.G = G
        self.origin = origin
        self.path = [self.origin]
        self.actual_costs = []
        self.destination = destination
        self.edge_weight = edge_weight

        # Edge weight will follow a triangular distribution
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        state_names = ["CurrentNode", "CurrentNodeLinkCosts"]
        decision_names = ["NextNode"]
        t0 = 0

        # Create random sample out of origin node
        self.prng = np.random.RandomState(seed)
        S0 = self.sample_initial_state()

        super().__init__(state_names, decision_names, S0, t0, T, seed)

        self.reset_VFA()

        # TODO: catch if meaningless parameters for graph are passed

    def sample_initial_state(self):
        # Create random link costs for origin node
        Decision = namedtuple("Decision", "NextNode")
        start_decision = Decision(self.origin)
        S0 = self.exog_info_fn(decision=start_decision)
        S0["CurrentNode"] = self.origin

        return S0

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

            # This would be a more realistic stochastic model:
            # Travel time is random, but proportional to nominal travel time of the edge.
            # edge_data = self.G.edges[edge + (0,)]
            # left = edge_data[self.edge_weight] * self.lower_bound
            # right = edge_data[self.edge_weight] * self.upper_bound
            # mode = edge_data[self.edge_weight]
            # if np.abs(left - right) < 1e-4:
            #    cost_dict[edge] = mode

            # Just choose a random number between 0 and 20s according to a triangular distribution
            left = 0
            right = 10
            mode = 5

            cost_dict[edge] = self.prng.triangular(left=left, mode=mode, right=right)

        return {"CurrentNodeLinkCosts": cost_dict}

    def reset_VFA(self):
        # Initialize VFA with deterministic shortest paths from all nodes to target node
        shortest_path = nx.shortest_path(self.G, target=self.destination, weight=self.edge_weight)
        self.V_t = {}
        for node in self.G.nodes:
            if node in shortest_path:
                self.V_t[node] = self.calc_path_length(shortest_path[node])
            else:
                self.V_t[node] = np.inf

    def reset(self, reset_prng: bool = False):
        # Note: VFA is not reset
        # Sample a new initial state on reset (random costs from starting node)
        S0 = self.sample_initial_state()
        self.initial_state = self.build_state(S0)
        super().reset(reset_prng)
        self.path = [self.origin]
        self.actual_costs = []

    def is_finished(self):
        """
        Check if the model run (episode) is finished.
        This is the case when we reached the destination or the maximum number of nodes have been visited.

        Returns:
            bool: True if the run is finished, False otherwise.
        """
        if self.t == self.T:
            self.objective = np.nan

        return self.state.CurrentNode == self.destination or self.t >= self.T

    def transition_fn(self, decision, exog_info: dict):
        return {"CurrentNode": decision.NextNode}

    def objective_fn(self, decision, exog_info: dict):
        return self.state.CurrentNodeLinkCosts[(self.state.CurrentNode, decision.NextNode)]

    def update_VFA(self, vhat, alpha):
        self.V_t[self.state.CurrentNode] = (1 - alpha) * self.V_t[self.state.CurrentNode] + alpha * vhat
        return self.V_t[self.state.CurrentNode]
