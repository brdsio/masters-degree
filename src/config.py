from typing import Tuple
import datetime
import pandas as pd


def get_data() -> (
    Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        datetime.date,
        datetime.date,
    ]
):
    """
    Load and preprocess financial data.

    Returns:
    - cdi (pd.DataFrame): Brazilian risk-free rate data.
    - ibov (pd.DataFrame): Data for the IBOVESPA index.
    - tickers (pd.DataFrame): Ticker dataframe.
    - prices (pd.DataFrame): Price dataframe.
    - prices_pivot (pd.DataFrame): Pivot table of last prices.
    - prices_trading (pd.DataFrame): Pivot table of average prices.
    - date_begin (datetime.date): Start date for the backtest.
    - date_end (datetime.date): End date for the backtest.
    """
    cdi = pd.read_csv("data/cdi.csv")  # this is the brazilian risk free rate
    ibov = pd.read_csv("data/ibov.csv")
    tickers = pd.read_csv("data/tickers.csv")

    prices = pd.read_csv("data/prices.csv")
    prices["date"] = pd.to_datetime(prices["date"])

    prices = prices.merge(tickers)

    prices_pivot = prices.pivot_table(
        index="date", columns="ticker", values="last_price"
    )
    prices_trading = prices.pivot_table(
        index="date", columns="ticker", values="average_price"
    )

    date_begin = datetime.date(2009, 12, 30)
    date_end = datetime.date(2017, 12, 30)

    return (
        cdi,
        ibov,
        tickers,
        prices,
        prices_pivot,
        prices_trading,
        date_begin,
        date_end,
    )
