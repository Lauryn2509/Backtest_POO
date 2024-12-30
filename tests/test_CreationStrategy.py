import unittest
import pandas as pd
import numpy as np
from strategy_to_be_tested import Momentum
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from example.strategy_examples import Momentum  # On effectue les tests sur Momentum pour l'exemple

def create_test_strategy_instance():
    """
    On copie-colle la stratégie que l'on souhaite tester dans le fichier "strategy_to_be_tested.py".
    Par exemple, ici nous prenons la stratégie Momentum. 
    """
    return Momentum(calculation_window=90, nbPositionInPortfolio=4, shortSell=True, isEquiponderated=True)

class TestStrategy(unittest.TestCase):

    def test_strategy_implementation(self):
        """
        Teste si la stratégie implémente les méthodes abstraites
        """
        strategy = create_test_strategy_instance()
        self.assertTrue(hasattr(strategy, 'get_position'), "La méthode get_position doit être implémentée")

    def test_get_position_output_type(self):
        
        """
        Teste si get_position renvoie un type correct (Series)
        """
        strategy = create_test_strategy_instance()
        data = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))  # Données fictives
        result = strategy.get_position(data)
        self.assertIsInstance(result, pd.Series, "get_position doit renvoyer une Series")

    def test_positions_sum_to_one(self):
        """
        Teste si la somme des poids des positions est toujours égale à 1
        """
        strategy = create_test_strategy_instance()
        data = pd.DataFrame({
            'actif1': np.linspace(0.01, 0.02, 90),
            'actif2': np.linspace(0.02, 0.01, 90),
            'actif3': np.linspace(0.015, 0.015, 90),
            'actif4': np.linspace(0.012, 0.022, 90),
        })
        result = strategy.get_position(data)
        total_weight = result.sum()
        self.assertAlmostEqual(total_weight, 1.0, places=4, msg="La somme des poids doit être égale à 1")

if __name__ == '__main__':
    unittest.main()
