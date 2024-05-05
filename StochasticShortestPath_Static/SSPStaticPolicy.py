import sys

sys.path.append("../")
from BaseClasses import SDPModel
from BaseClasses.SDPModel import SDPModel
from BaseClasses.SDPPolicy import SDPPolicy


class SSPStaticPolicy(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "", theta_step: float = 1.0):
        self.theta_step = theta_step
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        i = state.CurrentNode
        costs = {
            j: state.CurrentNodeLinkCosts[(i, j)] + self.model.V_t[j] for j in self.model.G.successors(i)
        }

        next_node = min(costs, key=costs.get)

        return {"NextNode": next_node}

    def train_value_function_paths(self, n_iterations: int = 1):

        V_t_origin = []
        for i in range(n_iterations):
            self.model.reset()

            V_t_origin.append(self.model.V_t[self.model.origin])

            # Create one path with the current value function approximation
            self.run_policy(1)

            # Read sampled costs from path
            k = len(self.model.path) - 1
            vhats = {self.model.path[k]: 0.0}
            while k > 0:
                k -= 1
                vhats[self.model.path[k]] = self.model.actual_costs[k] + vhats[self.model.path[k + 1]]

            # Update value function approximations for nodes on the path
            alpha = self.theta_step / (self.theta_step + i)
            for node in vhats.keys():
                self.model.V_t[node] = (1 - alpha) * self.model.V_t[node] + alpha * vhats[node]

    def train_value_function(self, n_iterations: int = 1):

        V_t_origin = []
        for i in range(n_iterations):
            alpha = self.theta_step / (self.theta_step + i)
            self.model.reset()

            V_t_origin.append(self.model.V_t[self.model.origin])

            while self.model.is_finished() is False:
                state_t = self.model.state
                decision_t = self.model.build_decision(self.get_decision(state_t, self.model.t, self.model.T))

                actual_costs = state_t.CurrentNodeLinkCosts[(state_t.CurrentNode, decision_t.NextNode)]
                vhat = actual_costs + self.model.V_t[decision_t.NextNode]

                self.model.update_VFA(vhat, alpha)
                self.model.step(decision_t)

        return V_t_origin
