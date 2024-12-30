import abc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Union
import pandas as pd, numpy as np, statsmodels.api as sm

class Strategy(metaclass=abc.ABCMeta):

    """
    Classe abstraite pour définir une stratégie de position.
    """
    @abc.abstractmethod
    def __init__(self, calculation_window : int = None) -> None:
        self.calculation_window = calculation_window

    @abc.abstractmethod
    def get_position(self, historical_data : pd.DataFrame, current_position: np.ndarray) -> np.ndarray:
        """
        Méthode abstraite pour lire un fichier CSV et retourner un vecteur de position.
        """
        pass

    @staticmethod
    def fit(endogenous_variable:np.ndarray, exogeneous_variables:Union[np.array, pd.DataFrame]) -> sm.regression.linear_model.RegressionResultsWrapper :
        """
        Méthode statique qui effectue une régression linéaire classique par les moindres carrés
        """
        results_OLS = sm.OLS(endogenous_variable, exogeneous_variables).fit()
        return results_OLS
