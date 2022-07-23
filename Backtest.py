import sys
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import math
from PortfolioOpt import OptimalPortfolio


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

if __name__ == "__main__":
    
    tsla = yf.Ticker("TSLA").history(period="2y").Close.to_numpy()[0:350]
    gold = yf.Ticker("GC=F").history(period="2y").Close.to_numpy()[0:350]
    sg = yf.Ticker("GLE.PA").history(period="2y").Close.to_numpy()[0:350]
    google = yf.Ticker("GOOG").history(period="2y").Close.to_numpy()[0:350]

    opt = OptimalPortfolio(
        target_return=0.05,
        target_risk=0.05,
        rf=0.0,
        start_time=10,
        tsla=tsla,
        gold=gold,
        sg=sg,
        google=google,
    )

    stocks = opt.stocks
    ret = opt.return_matrix
    var_cov = opt.variance_covariance_matrix()
    mv, target = opt.min_risk_mean_variance()
    weights, exp_ret = opt.efficient_ptf()
    weights_t, sharpe = opt.tangancy_ptf()
    old_ptf = stocks.T @ weights.reshape(len(weights), 1)

    back = Backtest(
        5,
        stocks,
        weights,
        "mv",
        target_return=0.05,
        target_risk=0.05,
        rf=0.0,
        tsla=tsla,
        gold=gold,
        sg=sg,
        google=google,
    )
    new_ptf, new_w = back.reblance_ptf()

    profit_loss = math.prod(back.tracking_profit(calibration_of_var_cov="dynamic"))

    # param, param_list = back.calibration(upper_bound=20, calibration_of_var_cov='dynamic')

    # profit_loss_calibrated = math.prod(back.tracking_profit(calibration_of_var_cov='dynamic', rebal_opt=param))
