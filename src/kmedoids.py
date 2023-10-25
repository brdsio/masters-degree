import datetime
import random
import copy

import pandas as pd
import numpy as np

import src.ibov_constituents as mi
from src.helper import correlDist, find_nearest
from abc import ABC, abstractmethod


class KMedoid(ABC):
    def __init__(self, prices):
        self.prices = prices

        self._main()

    def _main(self) -> None:
        self._get_rebalancing_date()
        return None

    def _get_rebalancing_date(self) -> None:
        date = pd.DataFrame(self.prices.index)
        date["next"] = date.date.shift(-1)

        self.rebalancing_dates = pd.DataFrame(
            date[date.date.dt.month != date.next.dt.month].date
        )
        return None

    @abstractmethod
    def fit(self):
        pass

    def _calculate_intragroup_allocation(self):
        pass

    def _calculate_intergroup_allocation(self):
        pass

    def _calculate_groups(self, matrix, centroids):
        print("Calculando grupos ", centroids)

        C = {}
        clusters = matrix[centroids].idxmin(axis=1)
        for centroid in centroids:
            C[centroid] = clusters[clusters == centroid].index.values
        return C

    def _calcular_fuzzy(self, C, dist, m=2):
        exp = 2 / (m - 1)
        # exp = 1
        grupos_fuzzy = {}

        grupos = list(C.keys())
        for g in grupos:
            papeis_grupo = list(C[g])

            distancias = dist.loc[papeis_grupo, grupos]
            distancias.loc[g] = (distancias.loc[g].index == g) + 0

            fkdist = (1 / distancias).replace([pd.np.inf], [0]).pow(exp)
            fkdist = fkdist.divide(fkdist.sum(axis=1), axis="rows")

            grupos_fuzzy[g] = fkdist

        return grupos_fuzzy


class KMedoidScenario4(KMedoid):
    def __init__(self, prices, tickers):
        self.tickers = tickers
        super().__init__(prices)

    def fit(self, k=3, window=3, max_iter=10):
        self.k = k
        grupos = {}
        precos_mes = {}

        for i in range(window, self.rebalancing_dates.shape[0]):
            filtro_datas = self.rebalancing_dates[i - window : i]
            precos_filtro = self.prices.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            # I do this because this particular stock was behaving like a fixed income instrument given a merge acquisition.
            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)
            returns2 = precos_filtro.pct_change()

            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter, returns, returns2)
            grupos[filtro_datas.date.max().date()] = fuzzy
            precos_mes[filtro_datas.date.max().date()] = precos_filtro

        self.precos_mes = precos_mes
        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter, returns, returns2):
        centroids = (
            (returns.std() * np.sqrt(252)).sort_values()[:k].index.tolist()
        )  # genial

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            C = self._calculate_groups(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix, returns2)
        return grupos_fuzzy

    def _calcular_fuzzy(self, C, dist, returns, m=2):
        preco = (1 + returns).cumprod().fillna(1)

        retorno_periodo = (
            preco[preco.index == preco.index.max()]
            .stack()
            .reset_index()
            .rename(columns={0: "retorno"})
        )

        grupos_fuzzy = {}

        grupos = list(C.keys())
        for g in grupos:
            # print(g)
            dft = pd.DataFrame(C[g], columns=["ticker"])
            dft = dft.merge(retorno_periodo, how="left", on="ticker")
            dft["peso"] = dft.retorno / dft.retorno.sum()
            grupos_fuzzy[g] = dft

        return grupos_fuzzy

    def pesos(self):
        pesos_rebals = []
        for date, values in self.grupos_fuzzy.items():
            lista_pesos = []
            grupos = values.keys()

            for g in grupos:
                pesos = values[g].set_index("ticker").peso
                pesos = (pesos / (pesos.sum())) * 1 / self.k
                pesos.name = "peso"
                pesos = pesos.reset_index().rename(columns={"index": "ticker"})
                lista_pesos.append(pesos)
            pesos = pd.concat(lista_pesos)
            pesos = pd.merge(pesos, self.tickers, how="left", on="ticker")
            pesos["date"] = date

            pesos_rebals.append(pesos)

        self.pesos_rebals = pd.concat(pesos_rebals)


class KMedoidScenario1(KMedoid):
    def __init__(self, prices, tickers):
        self.tickers = tickers

        super().__init__(prices)

    def pesos(self):
        pesos_rebals = []
        for date, values in self.grupos_fuzzy.items():
            lista_pesos = []
            grupos = values.keys()
            for g in grupos:
                pesos = values[g][g]
                pesos = (pesos / (pesos.sum())) * 1 / self.k
                pesos.name = "peso"
                pesos = pesos.reset_index().rename(columns={"index": "ticker"})
                lista_pesos.append(pesos)
            pesos = pd.concat(lista_pesos)
            pesos = pd.merge(pesos, self.tickers, how="left", on="ticker")
            pesos["date"] = date

            pesos_rebals.append(pesos)

        self.pesos_rebals = pd.concat(pesos_rebals)

    def fit(self, k=3, window=3, max_iter=10):
        self.k = k
        grupos = {}

        for i in range(window, self.rebalancing_dates.shape[0]):
            filtro_datas = self.rebalancing_dates[i - window : i]
            precos_filtro = self.prices.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            # I do this because this particular stock was behaving like a fixed income instrument given a merge acquisition.
            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)

            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter)
            grupos[filtro_datas.date.max().date()] = fuzzy

        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter):
        centroids = random.sample(list(matrix.columns), k)

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            # 2. Associate each data point to the closest medoid.
            C = self._calculate_groups(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix)
        return grupos_fuzzy


class KMedoidScenario2(KMedoid):
    def __init__(self, prices, tickers):
        self.tickers = tickers

        super().__init__(prices)

    def pesos(self):
        pesos_rebals = []
        for date, values in self.grupos_fuzzy.items():
            lista_pesos = []
            grupos = values.keys()
            for g in grupos:
                pesos = values[g][g]
                pesos = (pesos / (pesos.sum())) * 1 / self.k
                pesos.name = "peso"
                pesos = pesos.reset_index().rename(columns={"index": "ticker"})
                lista_pesos.append(pesos)
            pesos = pd.concat(lista_pesos)
            pesos = pd.merge(pesos, self.tickers, how="left", on="ticker")
            pesos["date"] = date

            pesos_rebals.append(pesos)

        self.pesos_rebals = pd.concat(pesos_rebals)

    def fit(self, k=3, window=3, max_iter=10):
        self.k = k
        grupos = {}

        for i in range(window, self.rebalancing_dates.shape[0]):
            filtro_datas = self.rebalancing_dates[i - window : i]
            print("Datas ", filtro_datas.date.min(), filtro_datas.date.max())
            precos_filtro = self.prices.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            # I do this because this particular stock was behaving like a fixed income instrument given a merge acquisition.
            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)

            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter, returns)
            grupos[filtro_datas.date.max().date()] = fuzzy

        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter, returns):
        centroids = (
            (returns.std() * np.sqrt(252)).sort_values()[:k].index.tolist()
        )  # genial

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            # 2. Associate each data point to the closest medoid.
            C = self._calculate_groups(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix)
        return grupos_fuzzy


class KMedoidScenario3(KMedoid):
    def __init__(self, prices, tickers):
        self.tickers = tickers

        super().__init__(prices)

    def fit(self, k=3, window=3, max_iter=10):
        self.k = k
        grupos = {}
        precos_mes = {}

        for i in range(window, self.rebalancing_dates.shape[0]):
            filtro_datas = self.rebalancing_dates[i - window : i]
            precos_filtro = self.prices.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            # I do this because this particular stock was behaving like a fixed income instrument given a merge acquisition.
            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)

            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter, returns)
            grupos[filtro_datas.date.max().date()] = fuzzy
            precos_mes[filtro_datas.date.max().date()] = precos_filtro

        self.precos_mes = precos_mes
        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter, returns):
        centroids = (
            (returns.std() * np.sqrt(252)).sort_values()[:k].index.tolist()
        )  # genial

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            C = self._calculate_groups(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix)
        return grupos_fuzzy

    def pesos(self):
        pesos_rebals = []
        for date, values in self.grupos_fuzzy.items():
            # print(date)

            qtd_dias = len(self.precos_mes[date])

            retorno_periodo = self.precos_mes[date].pct_change(qtd_dias - 1)
            retorno_periodo = (
                retorno_periodo.stack().reset_index().rename(columns={0: "retorno"})
            )
            grupos = values.keys()
            retorno_grupos = {}
            for g in grupos:
                frame_temp = pd.DataFrame(values[g].index)
                frame_temp = frame_temp.merge(retorno_periodo, how="left", on="ticker")
                frame_temp["valor_inicial"] = 1.0
                frame_temp["valor_final"] = frame_temp["valor_inicial"] * (
                    1 + frame_temp["retorno"]
                )

                retorno_grupo = (
                    frame_temp["valor_final"].sum() / frame_temp["valor_inicial"].sum()
                )
                retorno_grupos[g] = retorno_grupo

            lista_pesos = []
            for g in grupos:
                soma_retornos_grupos = sum(retorno_grupos.values())

                proporcao_alocacao_grupo = retorno_grupos[g] / soma_retornos_grupos

                pesos = values[g][g]
                pesos = (pesos / (pesos.sum())) * proporcao_alocacao_grupo
                pesos.name = "peso"
                pesos = pesos.reset_index().rename(columns={"index": "ticker"})
                lista_pesos.append(pesos)

            pesos = pd.concat(lista_pesos)
            pesos = pd.merge(pesos, self.tickers, how="left", on="ticker")
            pesos["date"] = date

            pesos_rebals.append(pesos)

        self.pesos_rebals = pd.concat(pesos_rebals)


class KMedoidScenario5(KMedoid):
    def __init__(self, prices, tickers):
        self.tickers = tickers

        super().__init__(prices)

    def fit(self, k=3, window=3, max_iter=10):
        self.k = k
        grupos = {}
        precos_mes = {}

        for i in range(window, self.rebalancing_dates.shape[0]):
            filtro_datas = self.rebalancing_dates[i - window : i]
            print("Datas ", filtro_datas.date.min(), filtro_datas.date.max())
            precos_filtro = self.prices.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            # I do this because this particular stock was behaving like a fixed income instrument given a merge acquisition.
            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)
            returns2 = precos_filtro.pct_change()

            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter, returns, returns2)
            grupos[filtro_datas.date.max().date()] = fuzzy
            precos_mes[filtro_datas.date.max().date()] = precos_filtro

        self.precos_mes = precos_mes
        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter, returns, returns2):
        centroids = (
            (returns.std() * np.sqrt(252)).sort_values()[:k].index.tolist()
        )  # genial

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            C = self._calculate_groups(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix, returns2)
        return grupos_fuzzy

    def _calcular_fuzzy(self, C, dist, returns, m=2):
        preco = (1 + returns).cumprod().fillna(1)
        qtd_dias = len(preco)
        retorno_periodo = (
            preco[preco.index == preco.index.max()]
            .stack()
            .reset_index()
            .rename(columns={0: "retorno"})
        )

        grupos_fuzzy = {}

        grupos = list(C.keys())
        for g in grupos:
            dft = pd.DataFrame(C[g], columns=["ticker"])
            dft = dft.merge(retorno_periodo, how="left", on="ticker")
            dft["peso"] = dft.retorno / dft.retorno.sum()

            grupos_fuzzy[g] = dft

        return grupos_fuzzy

    def pesos(self):
        pesos_rebals = []
        for date, values in self.grupos_fuzzy.items():
            qtd_dias = len(self.precos_mes[date])

            retorno_periodo = self.precos_mes[date].pct_change(qtd_dias - 1)
            retorno_periodo = (
                retorno_periodo.stack()
                .reset_index()
                .rename(columns={0: "retorno_periodo"})
            )
            grupos = values.keys()
            retorno_grupos = {}
            for g in grupos:
                # print(values)
                frame_temp = pd.DataFrame(values[g])
                frame_temp = frame_temp.merge(retorno_periodo, how="left", on="ticker")
                frame_temp["valor_inicial"] = 1.0
                frame_temp["valor_final"] = frame_temp["valor_inicial"] * (
                    1 + frame_temp["retorno_periodo"]
                )

                retorno_grupo = (
                    frame_temp["valor_final"].sum() / frame_temp["valor_inicial"].sum()
                )
                retorno_grupos[g] = retorno_grupo

            lista_pesos = []
            for g in grupos:
                soma_retornos_grupos = sum(retorno_grupos.values())

                proporcao_alocacao_grupo = retorno_grupos[g] / soma_retornos_grupos

                pesos = values[g][["ticker", "peso"]].copy()
                pesos["peso"] = pesos["peso"] * proporcao_alocacao_grupo
                lista_pesos.append(pesos)

            pesos = pd.concat(lista_pesos)
            pesos = pd.merge(pesos, self.tickers, how="left", on="ticker")
            pesos["date"] = date
            pesos_rebals.append(pesos)

        self.pesos_rebals = pd.concat(pesos_rebals)
