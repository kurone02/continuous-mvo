import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from env import TradingEnv

class CapitalMarketLine:
    def __init__(self, intercept: float, slope: float):
        self.intercept = intercept
        self.slope = slope

    def plot(self, xmax: float = 1.0):
        """Plot the Capital Market Line."""
        x = np.linspace(0, xmax, 100)
        y = self.intercept + self.slope * x
        plt.plot(x, y, label='Capital Market Line')
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return')
        plt.title('Capital Market Line')
        plt.legend()
        plt.show()

    def __str__(self):
        return f"CapitalMarketLine(intercept={self.intercept}, slope={self.slope})"

class ContinuousMeanVarianceOptimization:
    """
    Continuous Mean-Variance Optimization for portfolio allocation.
    Based on Zhou et al. (2000): "Continuous-Time Mean-Variance Portfolio Selection:
    A Stochastic LQ Framework"
    """
    def __init__(self,
                 env: TradingEnv,
                 initial_funds: float = 1_000_000,
                 target_return: float = 0.10,):
        self.env = env
        self.initial_funds = initial_funds
        self.target_return = target_return
        self.target_wealth = initial_funds * (1 + target_return)
        if self.env.use_sde:
            raise NotImplementedError("SDE model is not implemented yet.")
        
        self.B = self.env.nominal_annual_appreciation - self.env.nominal_annual_interest_rate.values
        self.sigma = self.env.historical_stocks_returns.cov() * 252
        self.rho = self.B @ np.linalg.inv(self.sigma) @ self.B.T
        self.slope = np.sqrt(np.exp(self.rho) - 1)
        self.intercept = self.initial_funds * np.exp(self.env.nominal_annual_interest_rate.values[0])
        self.sigma_T = (self.target_wealth - self.intercept) / self.slope
        self.capital_market_line = CapitalMarketLine(self.intercept, self.slope)
        self.gamma = (self.target_wealth - self.initial_funds * np.exp(self.env.nominal_annual_interest_rate.values[0] - self.rho)) / (1 - np.exp(-self.rho))

    def get_control(self, t: float=0.0, x: float=1.0):
        """
        u(t, x) in Zhou et al. (2000) is the control variable (optimal portfolio weights).
        x is the wealth at time t.
        """
        T = self.env.num_trading_years
        if t > T:
            raise ValueError("Time t must be less than or equal to T.")
        
        if self.env.use_sde:
            raise NotImplementedError("SDE model is not implemented yet.")
        
        # Calculate the control variable (optimal portfolio weights)
        # Using the formula from Zhou et al. (2000)
        control = np.linalg.inv(self.sigma) @ self.B.T * (self.gamma * np.exp(-self.env.nominal_annual_interest_rate.values[0] * (T - t)) - x)
        # Calculate the portfolio weights
        portfolio_weights = control / x
        bonds_weight = 1 - np.sum(portfolio_weights)
        return portfolio_weights, bonds_weight
    