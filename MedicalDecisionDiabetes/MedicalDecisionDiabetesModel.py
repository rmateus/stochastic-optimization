import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel


class MedicalDecisionDiabetesModel(SDPModel):
    def __init__(
        self,
        mu_truth: dict,
        sigma_W: float,
        S0: dict,
        t0: float = 0,
        T: float = 1,
        seed: int = 42,
    ) -> None:
        state_names = list(S0.keys())
        self.sigma_W = sigma_W
        self.beta_W = 1 / self.sigma_W**2
        self.mu_truth = mu_truth

        # For each drug, add the number of times the drug has been prescribed as a state
        for state in S0:
            if len(S0[state]) < 3:
                S0[state].append(0)
            if len(S0[state]) != 3:
                print(f"Need to provide prior mu and sigma for drug {state}!")
                # TODO: proper error handling
                return
            else:
                # S0 contains mu and sigma, but we model the states as mu and beta
                mu, sigma, N = S0[state]
                S0[state] = [mu, 1 / sigma**2, N]

        # Create one sample of the truth
        self.mu_truth_sample = {}
        for state in self.mu_truth:
            self.mu_truth_sample[state] = self.mu_truth[state].rvs()

        decision_names = ["choice"]

        super().__init__(state_names, decision_names, S0, t0, T, seed)

    def reset(self, reset_prng: bool = False):
        super().reset(reset_prng)

        # When the model is reset, create a new sample of the truth
        for state in self.mu_truth:
            self.mu_truth_sample[state] = self.mu_truth[state].rvs()

    # this function gives the exogenous information that is dependent on a random process
    # In our case, exogeneous information: W^(n+1) = mu_x + eps^(n+1),
    # Where eps^(n+1) is normally distributed with mean 0 and known variance (here s.d. 0.05)
    # W^(n+1)_x : reduction in A1C level
    # self.prng.normal takes two values, mu and sigma.
    def exog_info_fn(self, decision):
        x = decision.choice
        W = self.prng.normal(self.mu_truth_sample[x], self.sigma_W)

        return {"reduction": W}

    # this function takes in the decision and exogenous information to return\
    # the new mu and beta values corresponding to the decision.
    def transition_fn(self, decision, exog_info):
        # For all states except one the state values do not change.
        new_state = {state: getattr(self.state, state) for state in self.state_names}

        # Update the state for the drug that was prescribed in this step
        x = decision.choice
        mu_x, beta_x, N_x = getattr(self.state, x)
        mu_x = (beta_x * mu_x + self.beta_W * exog_info["reduction"]) / (beta_x + self.beta_W)
        beta_x = beta_x + self.beta_W
        N_x += 1  # count of no. times drug x was given.

        new_state[x] = [mu_x, beta_x, N_x]

        return new_state

    # contribution is W (reduction in A1C level)
    def objective_fn(self, decision, exog_info):
        W = exog_info["reduction"]
        return W
