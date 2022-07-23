import sys
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import math


class OptimalPortfolio:
    def __init__(
        self,
        target_return: float = None,
        target_risk: float = None,
        rf: float = None,
        start_weight: list = None,
        start_time: int = 10,
        **kwargs: np.array
    ):

        if (target_risk is None) & (target_return is None) & (rf is None):
            sys.exit()

        self.start_time = start_time
        self.stocks = np.array(list(kwargs.values()))[:, 0:start_time]

        self.return_matrix = np.apply_along_axis(
            lambda arr: np.diff(arr) / arr[:-1], 1, self.stocks
        )

        self.target_return = self.define_parameters(target_return)
        self.rf = self.define_parameters(rf)
        self.target_risk = self.define_parameters(target_risk)

        if start_weight is None:
            self.start_weight = [1 / len(kwargs)] * len(kwargs)

    @staticmethod
    def define_parameters(param):
        if param is None:
            return
        elif param >= 1:
            return param / 100
        else:
            return param

    def variance_covariance_matrix(self):
        return np.cov(self.return_matrix)

    def min_risk_mean_variance(self, solver="SLSQP"):
        var_cov = self.variance_covariance_matrix()

        def objective_function(w):
            w = np.array(w)
            return w.T @ var_cov @ w

        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w: np.sum(w @ self.return_matrix) - self.target_return,
            },
        )

        try:
            optimized_weight = np.array(
                minimize(
                    objective_function,
                    np.array([1 / len(self.start_weight)] * len(self.start_weight)),
                    method=solver,
                    constraints=cons,
                ).x
            )

        except ValueError:
            solver = input(
                'Unkwown solver: Please choose a known one. Tap "exit" if you want to reset the parameters'
            )
            if solver == "exit":
                sys.exit()
            return self.min_risk_mean_variance(solver=solver)

        return optimized_weight, self.target_return

    def efficient_ptf(self, solver="SLSQP"):
        ret_matrix = self.return_matrix

        def objective_function(w):
            w = np.array(w)
            return -w @ np.mean(ret_matrix, 1)

        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w: w.T @ self.variance_covariance_matrix() @ w
                - self.target_risk,
            },
        )

        try:
            optimized_weight = np.array(
                minimize(
                    objective_function,
                    self.start_weight,
                    method=solver,
                    constraints=cons,
                ).x
            )
            expected_return = optimized_weight @ np.mean(self.return_matrix, 1)

        except ValueError:
            solver = input(
                'Unkwown solver: Please choose a known one. Tap "exit" if you want to reset the parameters'
            )
            if solver == "exit":
                sys.exit()
            return self.min_risk_mean_variance(solver=solver)

        return optimized_weight, expected_return

    def tangancy_ptf(self, solver="SLSQP"):
        var_cov = self.variance_covariance_matrix()
        ret_matrix = self.return_matrix

        def objective_function(w):
            w = np.array(w)
            return -(w @ np.mean(ret_matrix, 1) - self.rf) * (w.T @ var_cov @ w) ** (
                -1 / 2
            )

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        # bounds = tuple((0,10) for x in range(0,ret.shape[0]))
        try:
            optimized_weight = np.array(
                minimize(
                    objective_function,
                    self.start_weight,
                    method=solver,
                    constraints=cons,
                ).x
            )
            sharpe_ratio = (
                optimized_weight @ np.mean(self.return_matrix, 1) - self.rf
            ) * (optimized_weight.T @ var_cov @ optimized_weight) ** (-1 / 2)

        except ValueError:
            solver = input(
                'Unkwown solver: Please choose a known one. Tap "exit" if you want to reset the parameters'
            )
            if solver == "exit":
                sys.exit()
            return self.min_risk_mean_variance(solver=solver)

        return optimized_weight, sharpe_ratio


class Backtest(OptimalPortfolio):
    def __init__(
        self,
        balancing_period,
        portfolio_matrix,
        weights,
        technics_of_rebalancing,
        target_return: float = None,
        target_risk: float = None,
        rf: float = None,
        **kwargs: np.array
    ):
        self.balancing_period = balancing_period
        self.portfolio_matrix = portfolio_matrix.T
        self.technics_of_rebalancing = technics_of_rebalancing
        self.start_weight = weights.reshape(len(weights), 1)
        self.real_ptf = self.portfolio_matrix @ self.start_weight
        self.ret = []
        super().__init__(target_return, target_risk, rf, self.start_weight, **kwargs)

    def new_ptf(self):
        return self.portfolio_matrix @ self.start_weight

    def reblance_ptf(self):
        if self.technics_of_rebalancing == "mv":
            self.start_weight = self.min_risk_mean_variance()[0]
        elif self.technics_of_rebalancing == "tangency":
            self.start_weight = self.tangancy_ptf()[0]
        if self.technics_of_rebalancing == "efficient":
            self.start_weight = self.efficient_ptf()[0]
        self.real_ptf = self.new_ptf()
        return self.real_ptf, self.start_weight

    def tracking_profit(self, calibration_of_var_cov="historical", rebal_opt=None):

        if rebal_opt is not None:
            self.balancing_period = rebal_opt

        rebal = self.balancing_period
        ret = []
        for countdown in range(0, len(self.real_ptf)):

            ret.append(
                self.real_ptf[countdown] / self.real_ptf[rebal - self.balancing_period]
            )
            print("The P&L of your porfolio is: " + str(round(math.prod(ret), 2)))
            if countdown + 1 == rebal:
                self.start_time = countdown + 1

                if calibration_of_var_cov == "historical":
                    start_index_varcov = 0
                elif calibration_of_var_cov == "dynamic":
                    start_index_varcov = self.start_time - self.balancing_period

                self.return_matrix = np.apply_along_axis(
                    lambda arr: np.diff(arr) / arr[:-1],
                    1,
                    self.stocks[:, start_index_varcov : self.start_time].copy(),
                )
                rebal += self.balancing_period
                self.real_ptf, self.start_weight = self.reblance_ptf()
                print(
                    "The portfolio has been rebalanced, here are the new weights: "
                    + str(self.start_weight)
                )

        print(
            "The equally weighted portfolio has provided a return of: "
            + str(
                round(self.stocks[:, -1].mean() / self.stocks[:, 0].mean() - 1, 2) * 100
            )
            + "%"
        )
        print(
            "The strategy has yielded a return of: "
            + str(round((math.prod(ret) - 1) * 100, 2))
            + "%"
        )
        return ret

    def calibration(
        self, lower_bound=5, upper_bound=10, calibration_of_var_cov="historical"
    ):

        result_calibration = []
        for i in range(lower_bound, upper_bound + 1):
            self.balancing_period = i
            result_calibration.append(
                math.prod(self.tracking_profit(calibration_of_var_cov))
            )

        best_parameter = result_calibration.index(max(result_calibration)) + lower_bound

        return best_parameter, result_calibration
