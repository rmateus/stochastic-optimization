import sys

sys.path.append("../")
from BaseClasses import SDPModel
from BaseClasses.SDPModel import SDPModel
from BaseClasses.SDPPolicy import SDPPolicy


class SSPStaticPolicy(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = ""):
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        i = state.CurrentNode
        costs = {
            j: state.CurrentNodeLinkCosts[(i, j)] + self.model.V_t[j] for j in self.model.G.successors(i)
        }
        next_node = min(costs, key=costs.get)
        self.model.path.append(next_node)

        return {"NextNode": next_node}
