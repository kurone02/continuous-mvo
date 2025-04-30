import numpy as np
import pandas as pd
import yfinance as yf

class TradingEnv:
    def __init__(self, 
                 bonds_ticker: str, 
                 stocks_tickers: list[str],
                 historical_start_date: str,
                 historical_end_date: str,
                 start_trading_date: str,
                 end_trading_date: str,
                 initial_funds: float = 1_000_000,
                 use_sde: bool = False,):
        
        self.bonds_ticker = bonds_ticker
        self.stocks_tickers = stocks_tickers
        self.historical_start_date = historical_start_date
        self.historical_end_date = historical_end_date
        self.start_trading_date = start_trading_date
        self.end_trading_date = end_trading_date
        self.initial_funds = initial_funds
        self.current_funds = self.initial_funds
        self.use_sde = use_sde

        self.num_trading_years = (
            pd.to_datetime(self.end_trading_date) - pd.to_datetime(self.start_trading_date)
        ).days / 365
        # self.num_trading_years = int(self.num_trading_years)

        self.historical_bonds_data = self._get_historical_data(bonds_ticker) / 100
        self.historical_stocks_data = [
            self._get_historical_data(ticker) for ticker in stocks_tickers
        ]
        self.historical_stocks_data = pd.concat(self.historical_stocks_data, axis=1)
        self.historical_stocks_returns = self.historical_stocks_data.pct_change().dropna()

        if not self.use_sde:
            self.nominal_annual_appreciation = self.historical_stocks_returns.mean() * 252
            self.nominal_annual_interest_rate = self.historical_bonds_data.mean()
        else:
            raise NotImplementedError("SDE model is not implemented yet.")

        self.trading_bonds_data = self._get_trading_data(bonds_ticker) / 100
        self.trading_stocks_data = [
            self._get_trading_data(ticker) for ticker in stocks_tickers
        ]
        self.trading_stocks_data = pd.concat(self.trading_stocks_data, axis=1)
        self.trading_stocks_returns = self.trading_stocks_data.pct_change().dropna()

        print(f"""\
            Bonds Ticker: {self.bonds_ticker}
            Stocks Tickers: {self.stocks_tickers}
            Historical Start Date: {self.historical_start_date}
            Historical End Date: {self.historical_end_date}
            Start Trading Date: {self.start_trading_date}
            End Trading Date: {self.end_trading_date}
            Use SDE: {self.use_sde}
            Nominal Annual Appreciation: {self.nominal_annual_appreciation if not self.use_sde else 'N/A'}
            Nominal Annual Interest Rate: {self.nominal_annual_interest_rate if not self.use_sde else 'N/A'}
        """)


    def _get_historical_data(self, ticker: str):
        """Get historical data for a given ticker."""
        data = yf.download(ticker, start=self.historical_start_date, end=self.historical_end_date, auto_adjust=False)['Adj Close']
        return data
    
    def _get_trading_data(self, ticker: str):
        """Get trading data for a given ticker."""
        data = yf.download(ticker, start=self.start_trading_date, end=self.end_trading_date, auto_adjust=False)['Adj Close']
        return data
    
    def get_wealth_after_trading(self, portfolio: np.ndarray, t0: float=0, t1: float=1):
        T = self.num_trading_years
        if t0 < 0 or t1 > T:
            raise ValueError("t0 and t1 must be between 0 and T")
        start_time = int(t0 / T * len(self.trading_stocks_data))
        end_time = int(t1 / T * len(self.trading_stocks_data))
        live_stock_data = self.trading_stocks_data.iloc[start_time:end_time]
        live_bond_data = self.trading_bonds_data.iloc[start_time:end_time]

        stock_growth = live_stock_data / live_stock_data.iloc[0].values
        bond_growth = live_bond_data / live_bond_data.iloc[0].values
        market_growth = np.concatenate([stock_growth.values, bond_growth.values], axis=1)
        wealth_over_time = market_growth @ portfolio
        
        wealth_over_time = wealth_over_time
        return wealth_over_time