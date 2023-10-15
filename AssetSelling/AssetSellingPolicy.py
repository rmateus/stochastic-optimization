"""
Asset selling policy class

"""
from collections import namedtuple
import math
from copy import copy
import numpy as np
import pandas as pd


class AssetSellingPolicy:
    """
    Base class for decision policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy

        :param model: the AssetSellingModel that the policy is being implemented on
        :param policy_names: list(str) - list of policies
        """
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple("Policy", policy_names)

    def build_policy(self, info):
        """
        this function builds the policies depending on the parameters provided

        :param info: dict - contains all policy information
        :return: namedtuple - a policy object
        """
        return self.Policy(*[info[k] for k in self.policy_names])

    def sell_low_policy(self, state, info_tuple):
        """
        this function implements the sell-low policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        new_decision = (
            {"sell": 1, "hold": 0}
            if state.price < lower_limit
            else {"sell": 0, "hold": 1}
        )
        return new_decision

    def high_low_policy(self, state, info_tuple):
        """
        this function implements the high-low policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        upper_limit = info_tuple[1]
        new_decision = (
            {"sell": 1, "hold": 0}
            if state.price < lower_limit or state.price > upper_limit
            else {"sell": 0, "hold": 1}
        )
        return new_decision

    def track_policy(self, state, info_tuple):
        """
        this function implements the track policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """

        theta = info_tuple[0]

        new_decision = (
            {"sell": 1, "hold": 0}
            if state.price >= state.price_smoothed + theta
            or state.price <= max(0, state.price_smoothed - theta)
            else {"sell": 0, "hold": 1}
        )
        return new_decision

    def run_policy(self, policy_info, policy, time):
        """
        this function runs the model with a selected policy

        :param param_list: list of policy parameters in tuple form (read in from an Excel spreadsheet)
        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :return: float - calculated contribution
        """
        model_copy = copy(self.model)

        df_log = pd.DataFrame(columns=["t"] + self.model.state_variable)
        df_log.loc[0] = [time] + list(self.model.state)
        k = 0
        while model_copy.state.resource != 0 and time < model_copy.initial_args["T"]:
            # Update policy parameters
            p = self.build_policy(policy_info)

            # make decision based on chosen policy
            if policy == "sell_low":
                decision = self.sell_low_policy(model_copy.state, p.sell_low)
            elif policy == "high_low":
                decision = self.high_low_policy(model_copy.state, p.high_low)
            elif policy == "track":
                decision = (
                    {"sell": 0, "hold": 1}
                    if time < 2
                    else self.track_policy(model_copy.state, p.track)
                )

            if time == model_copy.initial_args["T"] - 1:
                decision = {"sell": 1, "hold": 0}

            x = model_copy.build_decision(decision)

            # step the model forward one iteration
            model_copy.step(x)

            # increment time
            time += 1
            k += 1

            # log state
            df_log.loc[k] = [time] + list(model_copy.state)

        contribution = model_copy.objective

        return contribution, time, df_log

    def grid_search_theta_values(
        self,
        policy,
        low_min,
        low_max,
        high_min,
        high_max,
        track_min,
        track_max,
        increment_size,
    ):
        """
        this function gives a list of theta values needed to run a full grid search

        :param low_min: the minimum value/lower bound of theta_low
        :param low_max: the maximum value/upper bound of theta_low
        :param high_min: the minimum value/lower bound of theta_high
        :param high_max: the maximum value/upper bound of theta_high
        :param increment_size: the increment size over the range of theta values
        :return: list - list of theta values
        """

        if policy == "sell_low" or policy == "high_low":
            theta_low_values = np.linspace(
                low_min, low_max, math.floor((low_max - low_min) / increment_size) + 1
            )
            if policy == "high_low":
                theta_high_values = np.linspace(
                    high_min,
                    high_max,
                    math.floor((high_max - high_min) / increment_size) + 1,
                )
            else:
                theta_high_values = [None]
        elif policy == "track":
            theta_low_values = np.linspace(
                track_min,
                track_max,
                math.floor((track_max - track_min) / increment_size) + 1,
            )
            theta_high_values = [None]

        theta_values = []
        for x in theta_low_values:
            for y in theta_high_values:
                theta = (x, y)
                theta_values.append(theta)

        return theta_values, theta_high_values, theta_low_values

    def vary_theta(self, policy_info, policy, time, theta_values):
        """
        this function calculates the contribution for each theta value in a list

        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :param theta_values: list - list of all possible thetas to be tested
        :return: list - list of contribution values corresponding to each theta
        """
        contribution_values = []

        for theta in theta_values:
            t = time
            policy_info_copy = policy_info.copy()
            policy_info_copy.update({policy: theta})
            contribution = self.run_policy(policy_info_copy, policy, t)
            contribution_values.append((contribution[0], contribution[1]))

        return contribution_values
