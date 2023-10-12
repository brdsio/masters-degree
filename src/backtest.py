import numpy as np
import pandas as pd


def calculate_sharpe_ratio(cumret, cdi):
    df_cumret = (
        cumret.merge(
            cdi.set_index("date"), how="left", left_index=True, right_index=True
        )
        .pct_change()
        .add(1)
        .cumprod()
    )

    df_period_return = df_cumret.tail(1).sub(1)

    sharpe = (df_period_return.cumret - df_period_return.cdi).values[0] / (
        df_cumret.cumret.pct_change().std() * np.sqrt(252)
    )
    return sharpe


def calculate_cumret(weights_input, price_close, price_trading):
    date_begin = weights_input.index.min()
    weights_input = weights_input.copy()

    df_date = pd.DataFrame(price_close.index)
    df_date["next"] = df_date.date.shift(-1)

    weights_input.index = df_date[df_date.date.isin(weights_input.index)].next.values
    price_close = price_close.loc[date_begin : weights_input.index.max()][
        weights_input.columns
    ].copy()
    price_trading = price_trading.loc[date_begin : weights_input.index.max()][
        weights_input.columns
    ].copy()

    is_rebalancing = price_close.index.isin(weights_input.index)

    return_day_mtm = price_close.values / price_close.shift(1).values - 1
    returns_day = price_trading.values / price_close.shift(1).values - 1
    returns_day[~is_rebalancing] = return_day_mtm[~is_rebalancing]

    returns_trading = price_close.values / price_trading.values - 1
    returns_trading[np.isnan(returns_trading)] = 0
    returns_day[np.isnan(returns_day)] = 0

    weights = np.zeros([is_rebalancing.shape[0], weights_input.shape[1]])
    weights_trading = weights.copy()

    weights_rebalancing = weights.copy()
    weights_rebalancing[is_rebalancing] = weights_input.values

    for i, rebalacing in enumerate(is_rebalancing):
        if rebalacing:
            weight_day = weights_rebalancing[i]
            return_day = returns_trading[
                i
            ]  # in case if you want to trade with a vwap price or open price for example
        else:
            weight_day = weights[i - 1]
            return_day = returns_day[i]

        weights[i] = weight_day * (1 + return_day) / (1 + sum(weight_day * return_day))
        weights_trading[i] = (
            weights[i - 1]
            * (1 + returns_day[i])
            / (1 + sum(weights[i - 1] * returns_day[i]))
        )

    weights_trading[~is_rebalancing] = 0

    return_backtest = np.sum(weights[:-1] * returns_day[1:], axis=1)
    return_backtest = np.insert(return_backtest, 0, 0)
    return_backtest_trading = np.sum(weights_rebalancing * returns_trading, axis=1)

    returns = returns = (1 + return_backtest) * (1 + return_backtest_trading) - 1

    turnover_daily = np.abs((weights_rebalancing - weights_trading)).sum(axis=1)
    turnover = turnover_daily[is_rebalancing]

    cumret = pd.DataFrame(
        (1 + returns).cumprod(),
        index=price_close.index,
        columns=["cumret"],
    )
    return cumret, turnover
