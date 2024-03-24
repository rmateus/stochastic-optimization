import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
import pandas as pd


class AssetSellingModel(SDPModel):
    def __init__(
        self,
        S0: dict,
        t0: float = 0,
        T: float = 1,
        seed: int = 42,
        alpha: float = 0.7,
        var: float = 2,
        bias_df: pd.DataFrame = None,
        upstep: float = 1,
        downstep: float = -1,
    ) -> None:
        state_names = ["price", "bias", "price_smoothed", "resource"]

        # Set default values for helper states
        if "bias" not in S0:
            S0["bias"] = "Neutral"
        if "price_smoothed" not in S0:
            S0["price_smoothed"] = S0["price"]
        if "resource" not in S0:
            S0["resource"] = 1

        decision_names = ["sell"]
        super().__init__(state_names, decision_names, S0, t0, T, seed)
        self.alpha = alpha
        self.var = var
        if bias_df is None:
            self.bias_df = pd.DataFrame(
                {"Up": [0.9, 0.1, 0], "Neutral": [0.2, 0.6, 0.2], "Down": [0, 0.1, 0.9]}
            )
            self.bias_df.index = ["Up", "Neutral", "Down"]
        else:
            self.bias_df = bias_df
        self.upstep = upstep
        self.downstep = downstep

    def is_finished(self):
        """
        Check if the model run (episode) is finished.
        This is either the case when the time is over or we no longer hold the asset.

        Returns:
            bool: True if the run is finished, False otherwise.
        """
        hold_asset = self.state.resource
        return super().is_finished() or not hold_asset

    def exog_info_fn(self, decision):
        """
        Generates exogenous information for the asset selling model.

        Args:
            decision: The decision made (not used).

        Returns:
            A dictionary containing the updated price and bias.

        Notes:
            - The change in price is assumed to be normally distributed with mean bias and given variance.
            - The bias changes in every step according to given parameters.
            - The new price is set to 0 whenever the random process gives a negative price.
        """
        biasprob = self.bias_df[self.state.bias]

        coin = self.prng.uniform()
        if coin < biasprob["Up"]:
            new_bias = "Up"
            bias = self.upstep
        elif coin >= biasprob["Up"] and coin < biasprob["Neutral"]:
            new_bias = "Neutral"
            bias = 0
        else:
            new_bias = "Down"
            bias = self.downstep

        price_delta = self.prng.normal(bias, self.var)
        updated_price = self.state.price + price_delta
        new_price = 0.0 if updated_price < 0.0 else updated_price

        return {
            "price": new_price,
            "bias": new_bias,
        }

    def transition_fn(self, decision, exog_info):
        alpha = self.alpha
        new_resource = 0 if decision.sell == 1 else self.state.resource
        new_price_smoothed = (1 - alpha) * self.state.price_smoothed + alpha * exog_info["price"]

        return {"resource": new_resource, "price_smoothed": new_price_smoothed}

    def objective_fn(self, decision, exog_info):
        sell_size = 1 if decision.sell == 1 and self.state.resource != 0 else 0
        return self.state.price * sell_size


class AssetSellingModelHistorical(AssetSellingModel):
    def __init__(
        self,
        hist_data: pd.DataFrame,
        alpha: float = 0.7,
    ) -> None:
        super().__init__(S0={"price": 0.0}, alpha=alpha)
        self.T = 100
        self.hist_data = hist_data

    def reset(self, reset_prng: bool = False):
        # Get the subset of the historical data that corresponds to the current episode
        self.episode_data = self.hist_data.loc[self.hist_data["N"] == self.episode_counter, :]
        self.episode_data = self.episode_data["price"].tolist()
        self.episode_data.pop(0)
        self.T = len(self.episode_data)
        super().reset(reset_prng)

    def exog_info_fn(self, decision):
        return {"price": self.episode_data.pop(0), "bias": "Neutral"}
