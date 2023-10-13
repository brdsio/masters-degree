import datetime
import random
import copy

import pandas as pd
import numpy as np

import src.ibov_constituents as mi
from src.helper import correlDist, find_nearest


class KMedoidScenario2(object):
    def __init__(self, precos, tickers, debug=True):
        self.precos = precos
        self.tickers = tickers

        self._calcular_meses()
        self.debug = debug

    def _calcular_meses(self):
        datas = pd.DataFrame(self.precos.index)
        datas["amanha"] = datas.date.shift(-1)
        self.meses = pd.DataFrame(
            datas[datas.date.dt.month != datas.amanha.dt.month].date
        )

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

        for i in range(window, self.meses.shape[0]):
            filtro_datas = self.meses[i - window : i]
            if self.debug == True:
                print("Datas ", filtro_datas.date.min(), filtro_datas.date.max())
            precos_filtro = self.precos.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)
            # print(filtro_datas.date.max())
            # print("len(tickers_filtro)", len(tickers_filtro))
            # print("len(precos_filtro)", len(precos_filtro.columns))
            # print("----------")
            # print(tickers_filtro)
            # print(precos_filtro.columns)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)

            cov = returns.cov()
            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter, returns)
            grupos[filtro_datas.date.max().date()] = fuzzy

        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter, returns):
        centroids = []

        dt = pd.DataFrame(matrix.sum(axis=1), columns=["distancia"])

        dt["quantile"] = pd.qcut(dt.distancia, q=k, labels=False)
        dt = dt.sort_values(by="distancia")

        for q in dt["quantile"].unique():
            pos = find_nearest(dt.distancia, dt[dt["quantile"] == q].distancia.mean())
            centroids.append(dt.index[pos])

        centroids = random.sample(list(matrix.columns), k)
        # centroids = list(matrix.columns)[:k]

        # centroids = list(matrix.columns)[-k:]

        pcs = (1 + returns).cumprod().fillna(1)
        # print(pcs.head())
        # print(pcs.tail())
        # centroids = pcs.pct_change(pcs.shape[0]-1).dropna().stack().reset_index().sort_values(
        #    by=0, ascending=False)[:k].ticker.values.tolist()
        # print(pcs.pct_change(pcs.shape[0]-1).dropna().stack().reset_index().sort_values(
        #    by=0, ascending=False))
        # print((returns.std() * np.sqrt(252)).sort_values().reset_index())
        centroids = (
            (returns.std() * np.sqrt(252)).sort_values()[:k].index.tolist()
        )  # genial

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            # 2. Associate each data point to the closest medoid.
            C = self._calcular_grupos(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                if self.debug:
                    print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix)
        return grupos_fuzzy

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

    def _calcular_grupos(self, matrix, centroids):
        if self.debug:
            print("Calculando grupos ", centroids)

        C = {}
        clusters = matrix[centroids].idxmin(axis=1)
        for centroid in centroids:
            C[centroid] = clusters[clusters == centroid].index.values
        return C


class KMedoidScenario3(object):
    def __init__(self, precos, tickers, debug=True):
        self.precos = precos
        self.tickers = tickers

        self._calcular_meses()
        self.debug = debug

    def _calcular_meses(self):
        datas = pd.DataFrame(self.precos.index)
        datas["next"] = datas.date.shift(-1)
        self.meses = pd.DataFrame(
            datas[datas.date.dt.month != datas.next.dt.month].date
        )

    def fit(self, k=3, window=3, max_iter=10):
        self.k = k
        grupos = {}
        precos_mes = {}

        for i in range(window, self.meses.shape[0]):
            filtro_datas = self.meses[i - window : i]
            if self.debug == True:
                print("Datas ", filtro_datas.date.min(), filtro_datas.date.max())
            precos_filtro = self.precos.loc[
                filtro_datas.date.min() : filtro_datas.date.max()
            ]

            tickers_filtro = mi.ibov_constituents[
                datetime.date(
                    filtro_datas.date.max().year, filtro_datas.date.max().month, 1
                )
            ]
            precos_filtro = precos_filtro[tickers_filtro]

            if filtro_datas.date.max().date() == datetime.date(2017, 10, 31):
                precos_filtro = precos_filtro.drop(["CPFE3"], axis=1)

            returns = precos_filtro.pct_change()[1:].dropna(axis=1)

            cov = returns.cov()
            corr = returns.corr()

            dist = correlDist(corr)

            fuzzy = self._calcular_fuzzy_kmedoids(k, dist, max_iter, returns)
            grupos[filtro_datas.date.max().date()] = fuzzy
            precos_mes[filtro_datas.date.max().date()] = precos_filtro

        self.precos_mes = precos_mes
        self.grupos_fuzzy = grupos

    def _calcular_fuzzy_kmedoids(self, k, matrix, max_iter, returns):
        centroids = []

        dt = pd.DataFrame(matrix.sum(axis=1), columns=["distancia"])

        dt["quantile"] = pd.qcut(dt.distancia, q=k, labels=False)
        dt = dt.sort_values(by="distancia")

        for q in dt["quantile"].unique():
            pos = find_nearest(dt.distancia, dt[dt["quantile"] == q].distancia.mean())
            centroids.append(dt.index[pos])

        centroids = random.sample(list(matrix.columns), k)

        pcs = (1 + returns).cumprod().fillna(1)
        centroids = (
            (returns.std() * np.sqrt(252)).sort_values()[:k].index.tolist()
        )  # genial

        centroids_swap = copy.copy(centroids)

        for i in range(max_iter):
            C = self._calcular_grupos(matrix, centroids)

            for centroid, data in C.items():
                new_centroid = np.argmin(np.sum(matrix.loc[data, data], axis=1))
                centroids_swap[centroids_swap.index(centroid)] = new_centroid

            if np.array_equal(centroids, centroids_swap):
                if self.debug:
                    print("Convergiu", i)
                break

            centroids = copy.copy(centroids_swap)
        else:
            print("Halted")

        grupos_fuzzy = self._calcular_fuzzy(C, matrix)
        return grupos_fuzzy

    def _calcular_grupos(self, matrix, centroids):
        if self.debug:
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

    def pesos(self):
        pesos_rebals = []
        for date, values in self.grupos_fuzzy.items():
            # print(date)

            qtd_dias = len(self.precos_mes[date])

            retorno_periodo = self.precos_mes[date].pct_change(qtd_dias - 1)
            retorno_periodo = (
                retorno_periodo.stack().reset_index().rename(columns={0: "retorno"})
            )
            # print(values)
            grupos = values.keys()
            retorno_grupos = {}
            for g in grupos:
                # print(g)
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

            # print(retorno_grupos)
            lista_pesos = []
            for g in grupos:
                soma_retornos_grupos = sum(retorno_grupos.values())

                proporcao_alocacao_grupo = retorno_grupos[g] / soma_retornos_grupos
                # print(g, proporcao_alocacao_grupo)

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
