from itertools import groupby
from typing import List, Dict
import random
import datetime
from decimal import Decimal
import pandas as pd
import numpy as np


from src.kmedoids import KMedoidScenario1
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
        random.seed(123)

        for _ in list(range(1, 11)):
            # Initialize and fit the k-medoids model
            km = KMedoidScenario1(
                prices_pivot[prices_pivot.index >= datetime.datetime(2007, 1, 30)],
                tickers,
                debug=False,
            )
            km.fit(k=k, window=13)
            km.pesos()

            # Extract and preprocess the allocation data within the specified date range
            pesos = km.pesos_rebals[
                (km.pesos_rebals.date >= date_begin)
                & (km.pesos_rebals.date <= date_end)
            ].copy()
            pesos["date"] = pd.to_datetime(pesos["date"])

            # Calculate cumulative returns and turnover
            cumret, turnover = calculate_cumret(
                pesos.pivot_table(index="date", columns="ticker", values="peso").fillna(
                    0
                ),
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

    # Sort the data by 'k' and 'return'
    sorted_data = sorted(results, key=lambda x: (x["k"], x["return"]), reverse=True)

    # Use groupby to group the data by 'k'
    grouped_data = groupby(sorted_data, key=lambda x: x["k"])

    # Initialize dictionaries to store the results
    highest_returns = {}
    lowest_returns = {}
    mean_returns = {}
    mean_volatilities = {}
    mean_turnovers = {}
    mean_drawdowns = {}
    mean_sharpes = {}

    for k, group in grouped_data:
        group_list = list(group)

        if len(group_list) > 0:
            highest_return_dict = max(group_list, key=lambda x: x["return"])
            lowest_return_dict = min(group_list, key=lambda x: x["return"])
            mean_return = round(
                Decimal(sum(item["return"] for item in group_list) / len(group_list)), 1
            )
            mean_volatility = round(
                Decimal(
                    sum(item["volatility"] for item in group_list) / len(group_list)
                ),
                1,
            )
            mean_turnover = round(
                Decimal(sum(item["turnover"] for item in group_list) / len(group_list)),
                1,
            )
            mean_drawdown = round(
                Decimal(sum(item["drawdown"] for item in group_list) / len(group_list)),
                1,
            )
            mean_sharpe = round(
                Decimal(sum(item["sharpe"] for item in group_list) / len(group_list)), 1
            )
        else:
            highest_return_dict = None
            lowest_return_dict = None
            mean_return = None

        highest_returns[k] = highest_return_dict
        lowest_returns[k] = lowest_return_dict
        mean_returns[k] = mean_return
        mean_volatilities[k] = mean_volatility
        mean_turnovers[k] = mean_turnover
        mean_drawdowns[k] = mean_drawdown
        mean_sharpes[k] = mean_sharpe

    stats_results = []
    for k, mean_return in mean_returns.items():
        stats_k = {
            "k": k,
            "result_type": "mean",
            "return": mean_return,
            "volatility": mean_volatilities[k],
            "turnover": mean_turnovers[k],
            "drawdown": mean_drawdowns[k],
            "sharpe": mean_sharpes[k],
        }
        stats_results.append(stats_k)

        stats_k = {
            "k": k,
            "result_type": "lowest",
            "return": lowest_return_dict["return"],
            "volatility": lowest_return_dict["volatility"],
            "turnover": lowest_return_dict["turnover"],
            "drawdown": lowest_return_dict["drawdown"],
            "sharpe": lowest_return_dict["sharpe"],
        }
        stats_results.append(stats_k)

        stats_k = {
            "k": k,
            "result_type": "highest",
            "return": highest_return_dict["return"],
            "volatility": highest_return_dict["volatility"],
            "turnover": highest_return_dict["turnover"],
            "drawdown": highest_return_dict["drawdown"],
            "sharpe": highest_return_dict["sharpe"],
        }
        stats_results.append(stats_k)

    return stats_results
