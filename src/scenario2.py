from typing import List, Dict
import datetime
from decimal import Decimal
import pandas as pd
import numpy as np


from src.kmedoids import KMedoidScenario2
from src.backtest import calculate_cumret, calculate_sharpe_ratio
from src.config import get_data


def main():
    (
        cdi,
        ibov,
        tickers,
        prices,
        prices_pivot,
        prices_trading,
        date_begin,
        date_end,
    ) = get_data()
    """
    Main function to run k-medoids and backtest.

    Returns:
    - results (List[Dict[str, float]]): List of dictionaries containing backtest results.
    """
    # Load financial data using the get_data function
    (
        cdi,
        ibov,
        tickers,
        prices,
        prices_pivot,
        prices_trading,
        date_begin,
        date_end,
    ) = get_data()

    # Initialize an empty list to store the results
    results = []

    # Iterate through different values of 'k' (clusters) for k-medoids
    for k in range(3, 21):
        # Initialize and fit the k-medoids model
        km = KMedoidScenario2(
            prices_pivot[prices_pivot.index >= datetime.datetime(2007, 12, 30)],
            tickers,
            debug=False,
        )
        km.fit(k=k, window=13)
        km.pesos()

        # Extract and preprocess the allocation data within the specified date range
        pesos = km.pesos_rebals[
            (km.pesos_rebals.date >= date_begin) & (km.pesos_rebals.date <= date_end)
        ].copy()
        pesos["date"] = pd.to_datetime(pesos["date"])

        # Calculate cumulative returns and turnover
        cumret, turnover = calculate_cumret(
            pesos.pivot_table(index="date", columns="ticker", values="peso").fillna(0),
            prices_pivot,
            prices_trading,
        )

        # Calculate various performance metrics
        mean_turnover = turnover[1:].mean() * 100
        cumret = cumret.dropna().loc[:date_end]

        ret = cumret.tail(1).sub(1).mul(100).round(1).values[0][0]

        volatility = (
            cumret.dropna()
            .loc[:date_end]
            .pct_change()
            .std()
            .mul(np.sqrt(252))
            .mul(100)
            .round(1)
            .values[0]
        )
        max_drawdown = (
            cumret.div(cumret.expanding().max())
            .sub(1)
            .mul(100)
            .min()
            .round(1)
            .values[0]
        )
        sharpe = calculate_sharpe_ratio(cumret, cdi)

        # Store the results in a dictionary
        dict_result = {
            "k": k,
            "return": round(Decimal(ret), 1),
            "volatility": round(Decimal(volatility), 1),
            "turnover": round(Decimal(mean_turnover), 1),
            "drawdown": round(Decimal(max_drawdown), 1),
            "sharpe": round(Decimal(sharpe), 1),
        }

        # Append the results dictionary to the results list
        results.append(dict_result)

    return results
