import unittest
import pandas as pd
import numpy as np
from backtest import Strategy, Backtest, Result

class MockStrategy(Strategy):
    """ 
    Création d'une "fausse stratégie" pour tester les résultats. 
    Cela permet ainsi d'obtenir un vecteur fictif de parts.  
    """
    def __init__(self, calculation_window):
        self.calculation_window = calculation_window

    def get_position(self, historical_data):
        return np.ones(historical_data.shape[1])/historical_data.shape[1]


class TestResult(unittest.TestCase):
    def setUp(self):
        """
        Mise en place d'un backtest et des résultats et création d'une stratégie sur des données fictives
        """
        data = pd.DataFrame(np.random.randn(376, 4), columns=['actif1', 'actif2', 'actif3', 'actif4'])
        data = data.abs()   # On raisonne sur des prix (positifs). 
        # Nous devons ajouter des dates en indice du DataFrame. Elles sont en effet utilisées pour divers calculs. 
        # Pour simplifier les calculs, nous faisons en sorte qu'une année sépare la première date de calcul et la dernière date
        start_date = "2021-12-22"
        date_range = pd.date_range(start=start_date, periods=len(data))
        data.insert(0, "date", date_range) ; data.set_index("date", inplace = True)
        strategy = MockStrategy(calculation_window=10)
        self.backtest = Backtest(data=data, strategy=strategy, frequence_rebalancement=5, initial_value=100, transaction_cost=0.001)
        self.result = Result(self.backtest, risk_free_rate=0.02, mar=0.03)
       
    def test_calculate_sharpe_ratio(self):
        """
        On vérifie que la formule du ratio de Sharpe est correctement implémenté
        """
        self.result.annualized_performance = 0.1
        self.result.volatility = 0.2
        calculated_sharpe_ratio = self.result.calculate_sharpe_ratio(0.03)
        self.assertAlmostEqual(calculated_sharpe_ratio, (0.1 - 0.03)/0.2)

    def test_calculate_max_drawdown(self):
        """
        On vérifie que le calcul du max drawdown est correctement implémenté
        Avec nos données imposées, le max drawdown est de (90 - 120)/120 = -0.25
        """
        strategy_value = pd.Series([100, 120, 90, 140], index=pd.date_range("2022-01-01", periods=4))
        self.result.strategy_value = strategy_value
        max_drawdown = self.result.calculate_max_drawdown()
        self.assertAlmostEqual(max_drawdown, -0.25)  

    def test_types(self):
        """
        On s'assure que les éléments renvoyés ont bien le format souhaité. 
        """
        self.assertIsInstance(self.result.calculate_total_performance(), float)
        self.assertIsInstance(self.result.calculate_volatility(), float)
        self.assertIsInstance(self.result.calculate_sharpe_ratio(), float)
        self.assertIsInstance(self.result.calculate_sortino_ratio(), float)
        self.assertIsInstance(self.result.calculate_max_drawdown(), float)
        self.assertIsInstance(self.result.calculate_trade_stats()[0], int)
        self.assertIsInstance(self.result.calculate_trade_stats()[1], float)
        
if __name__ == '__main__':
    unittest.main()
