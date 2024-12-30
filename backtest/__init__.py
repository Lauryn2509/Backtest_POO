"""
Package BACKTEST
Ce package contient les modules pour effectuer des backtests, des stratégies, et analyser les résultats.
"""

# Import des classes 
from .backtest import Backtest
from .strategy import Strategy
from .result import Result

__all__ = ["Backtest", "Strategy", "Result"]


