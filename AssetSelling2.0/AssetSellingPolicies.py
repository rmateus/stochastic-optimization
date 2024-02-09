import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from BaseClasses.SDPPolicy import SDPPolicy


class SellLowPolicy(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "", theta_low: float = 10):
        super().__init__(model, policy_name)
        self.theta_low = theta_low

    def get_decision(self, state):
        lower_limit = self.theta_low
        new_decision = {"sell": 1, "hold": 0} if state.price < lower_limit else {"sell": 0, "hold": 1}
        return new_decision


class HighLowPolicy(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "", theta_low: float = 10, theta_high: float = 30):
        super().__init__(model, policy_name)
        self.theta_low = theta_low
        self.theta_high = theta_high

    def get_decision(self, state):
        lower_limit = self.theta_low
        upper_limit = self.theta_high
        new_decision = (
            {"sell": 1, "hold": 0}
            if state.price < lower_limit or state.price > upper_limit
            else {"sell": 0, "hold": 1}
        )
        return new_decision


class TrackPolicy(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "", theta: float = 10):
        super().__init__(model, policy_name)
        self.theta = theta

    def get_decision(self, state):
        theta = self.theta

        new_decision = (
            {"sell": 1, "hold": 0}
            if state.price >= state.price_smoothed + theta
            or state.price <= max(0, state.price_smoothed - theta)
            else {"sell": 0, "hold": 1}
        )
        return new_decision
