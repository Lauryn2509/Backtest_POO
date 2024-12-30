import pandas as pd, numpy as np, statsmodels.api as sm
from scipy.optimize import minimize
from typing import Dict
from enum import Enum
import math
import sys
import os
from backtest.strategy import Strategy


class Momentum(Strategy):
    """
    Classe qui implémente une stratégie Momentum basée sur les rendements logarithmiques.
    """

    def __init__(self, calculation_window : int, nbPositionInPortfolio: int,
                  shortSell : bool, isEquiponderated : bool):
        """
        Initialise la classe avec la fréquence de rebalancement.
        :param frequency: Nombre de périodes entre chaque rebalancement.
        """
        self.calculation_window = calculation_window
        self.shortSell = shortSell
        self.nbPositionInPortfolio = nbPositionInPortfolio
        self.isEquiponderated = isEquiponderated

    def get_position(self, historical_data):
        # Calcul des rendements logarithmiques sur toute la période
        returns_log = pd.Series(
            np.log(historical_data.iloc[-1] / historical_data.iloc[0]),
            index=historical_data.columns
        )

        position = pd.Series(dtype='float64')  # Initialisation explicite

        # Initialisation de la série position
        position = pd.Series(0.0, index=returns_log.index)

        # Trier les rendements dans l'ordre décroissant
        sorted_titles = returns_log.sort_values(ascending=False)

        # Sélectionner les meilleurs et pires titres
        top_titles = sorted_titles[:math.ceil(self.nbPositionInPortfolio / 2)]
        bottom_titles = sorted_titles[-math.floor(self.nbPositionInPortfolio / 2):]

        if not self.isEquiponderated:  # Si la stratégie n'est pas équipondérée
            # Ajuster les rendements pour les top titres
            adjustment_factor_top = abs(top_titles.min())
            # important pour gèrer les rendements de différents signes 
            adjusted_top_scores = top_titles + adjustment_factor_top
            total_top_score = adjusted_top_scores.sum()

            if total_top_score != 0:  # Éviter les divisions par zéro
                position[top_titles.index] = (adjusted_top_scores / total_top_score) * 2

            # Ajuster les rendements pour les bottom titres
            adjustment_factor_bottom = abs(bottom_titles.max())            
            # important pour gèrer les rendements de différents signes
            adjusted_bottom_scores = bottom_titles - adjustment_factor_bottom
            total_bottom_score = abs(adjusted_bottom_scores.sum())

            if total_bottom_score != 0:  # Éviter les divisions par zéro
                position[bottom_titles.index] = (adjusted_bottom_scores / total_bottom_score) * 1
        else:  # Si la stratégie est équipondérée
            # Poids positifs pour les top titres
            positive_weight = 2 / top_titles.size
            position[top_titles.index] = positive_weight

            # Poids négatifs pour les bottom titres
            if self.shortSell:
                negative_weight = -1 / bottom_titles.size
                position[bottom_titles.index] = negative_weight
        return position