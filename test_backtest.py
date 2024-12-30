import unittest
import pandas as pd
import numpy as np
from backtest import Strategy, Backtest

class MockStrategy(Strategy):
    """ 
    Création d'une "fausse stratégie" pour tester la classe Backtest. 
    Cela permet ainsi d'obtenir un vecteur fictif de parts.  
    """
    def __init__(self, calculation_window):
        self.calculation_window = calculation_window

    def get_position(self, historical_data):
        return np.ones(historical_data.shape[1])/historical_data.shape[1]

class TestBacktest(unittest.TestCase):
    def setUp(self):
        """
        Mise en place d'un backtest et création d'une stratégie sur des données fictives
        """
        data = pd.DataFrame(np.random.randn(100, 4), columns=['actif1', 'actif2', 'actif3', 'actif4'])
        data = data.abs()    # On raisonne sur des prix (positifs). 
        strategy = MockStrategy(calculation_window=10)
        self.backtest = Backtest(data=data, strategy=strategy, frequence_rebalancement=5, initial_value=100, transaction_cost=0.001)

    def test_initialization(self):
        """
        Vérification sur l'intialisation : il faut que les données soient bien un Dataframe, 
        la stratégie une classe héritée de Strategy, et que la fréqeunce de rebalancement corresponde bien
        à celle définie plus haut
        """
        self.assertIsInstance(self.backtest.data, pd.DataFrame)
        self.assertIsInstance(self.backtest.strategy, Strategy)
        self.assertEqual(self.backtest.frequence_rebalancement, 5)

    def test_get_position(self):
        """
        Vérifie que le portefeuille est totalement investi (somme des parts égale à 1).
        Il faut tenir compte de la fenêtre minimale de calcul : la somme des parts doit être 
        nulle avant d'avoir suffisamment de données.
        """
        positions = self.backtest.get_position(self.backtest.strategy)
        total_weights = positions.sum(axis=1)
        # Avant d'avoir suffisamment de données, rien n'est investi
        before_window = total_weights[:self.backtest.strategy.calculation_window]
        pd.testing.assert_series_equal(before_window, pd.Series(0.0, index=before_window.index))
        # Après, la somme des parts doit être de 1.
        after_window = total_weights[self.backtest.strategy.calculation_window:]
        pd.testing.assert_series_equal(after_window, pd.Series(1.0, index=after_window.index))

    def test_calculate_returns(self):
        """
        On vérifie que la méthode de calcul des rendements fournit bien une série.
        """
        positions = pd.DataFrame(1, index=self.backtest.data.index, columns=self.backtest.data.columns)
        self.backtest.calculate_market_value(positions)
        returns = self.backtest.calculate_returns()
        self.assertIsInstance(returns, pd.Series)


if __name__ == '__main__':
    unittest.main()
