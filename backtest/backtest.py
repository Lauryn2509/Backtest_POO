import pandas as pd
from typing import Tuple
from .result import Result
from .strategy import Strategy
   
class Backtest:
    """
    La classe Backtest permet de calculer a posteriori diverses mesures de performance et de
    visualiser la performance des stratégies sur une période donnée, étant donnée ladite stratégie,
    un univers d'investissement, une fréquence de rebalancement du portefeuille, une valeur initiale (investie) 
    du portefeuille, et des coûts de transaction (supposés fixes). 

    Attributs : 
    ---
        data : dataframe contenant les prix des actifs composant l'univers d'investissement
        strategy : instance de Strategy, définie par l'utilisateur et suivant un ensemble de fonctions obligatoires
        frequence_rebalancement : entier correspondant à la période séparant deux rebalancements de portefeuille. Naturellement,
            cette période doit être exprimée dans la même unité que les données. 
        initial_value : valeur initiale du portefeuille.
        transaction_cost : coûts de transaction proportionnels au nombre de trades. 
        market_value : valeur du portefeuille. Est initialement égal à initial_value.
    """
    def __init__(self, data: pd.DataFrame, strategy: Strategy, frequence_rebalancement : int, initial_value : float, transaction_cost : float) -> None:
        # Vérifications élémentaires
        if not isinstance(strategy, Strategy):
            raise TypeError("Votre stratégie doit être une instance de la classe abstraite Strategy")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Le jeu de données doit être préalablement converti en dataframe pandas")

        self.data : pd.DataFrame = data  
        self.strategy : Strategy = strategy 
        self.frequence_rebalancement : int = frequence_rebalancement
        self.initial_value : float = initial_value
        self.transaction_cost : float = transaction_cost
    
    def get_position(self, strategy : Strategy) -> pd.DataFrame:
        """
        Calcul des positions de la stratégie. Le fonctionnement est le suivant : 
            - on met à jour le jeu de données passé en paramètres de la fonction "get_position" de la stratégie en fonction
                de la fréquence de rebalancement et de la fenêtre de calcul
            - on récupère, à chaque date, les positions ainsi calculées (entre deux dates de rebalancement, les positions restent les mêmes)
        Le résultat renvoyé est un dataframe contenant les positions par actifs à chaque date.
        """
        # Initialisation de positions
        positions = pd.DataFrame(0.0, index=self.data.index, columns=self.data.columns)
        calculation_window = strategy.calculation_window
        length = self.data.shape[0]
        for idx in range(calculation_window, length, self.frequence_rebalancement):
            begin_idx = 0 if calculation_window== None else idx - calculation_window
            correct_frequence = self.frequence_rebalancement if (length - idx > self.frequence_rebalancement) else (length - idx)
            positions.iloc[idx:idx + correct_frequence, :] = strategy.get_position(self.data.iloc[begin_idx:idx, :])
        return positions

    def calculate_market_value(self, positions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calcule la valeur du portefeuille dans le temps à partir de la valeur initiale,
        des positions et des variations des prix des actifs.

        Résultat : 
        ---
            titles_values : valeur de marché des actifs
            portfolio_value : valeur du portefeuille
        """
        # Vérifications élémentaires
        if not positions.index.equals(self.data.index) or not positions.columns.equals(self.data.columns):
            raise ValueError("Les indices ou colonnes de `positions` et `data` ne correspondent pas.")
        if self.data.isna().any().any():
            raise ValueError("Les données contiennent des NaN. Vérifiez les entrées dans `self.data`.")
        if positions.isna().any().any():
            raise ValueError("Les positions contiennent des NaN. Vérifiez les entrées dans `positions`.")
        if (self.data <= 0).any().any():
            raise ValueError("Les données contiennent des valeurs négatives ou nulles, ce qui peut provoquer des erreurs dans les calculs.")
        
        # Initialisation 
        portfolio_value = pd.Series(index=self.data.index, dtype=float)
        portfolio_value.iloc[0] = self.initial_value  # La valeur initiale du portefeuille

        for t in range(len(positions)):
            # Gestion des périodes initiales où les positions sont nulles
            if t == 0:
                # Initialisation pour t=0
                portfolio_value.iloc[t] = self.initial_value
            else:
                # Si toutes les positions sont nulles, conserver la valeur précédente
                if (positions.iloc[t] == 0).all() or (positions.iloc[t - 1] == 0).all():
                    portfolio_value.iloc[t] = portfolio_value.iloc[t - 1]
                else:
                    # Calcul des variations de prix entre t-1 et t
                    price_change = self.data.iloc[t] / self.data.iloc[t - 1] - 1

                    # Calcul de la nouvelle valeur du portefeuille et des frais de transaction
                    weighted_price_change = (positions.iloc[t - 1] * price_change).sum()
                    trade_cost = (positions.iloc[t-1] - positions.iloc[t - 2]).abs().sum() * portfolio_value.iloc[t-1]*self.transaction_cost
                    portfolio_value.iloc[t]  =  portfolio_value.iloc[t - 1] * (1 + weighted_price_change) - trade_cost
        
        # Mise à jour des valeurs de marché
        self.portfolio_value = portfolio_value
        return portfolio_value
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calcule les rendements des actifs basés sur la market value.
        """
        # Vérifications élémentaires
        if self.portfolio_value is None:
            raise ValueError("`market_value` n'a pas encore été calculée. Appelez `calculate_market_value` d'abord.")
        if len(self.portfolio_value) < 2:
            raise ValueError("`market_value` doit contenir au moins deux lignes pour calculer les rendements.")
        valid_values = (self.portfolio_value != 0) & (self.portfolio_value.shift(1) != 0)
        returns = self.portfolio_value/self.portfolio_value.shift(1) - 1
        self.returns = returns[valid_values]
        return self.returns
        
    def execute_backtest(self, risk_free_rate : float, mar :float, backend:str = "plotly") -> None:
        """
        Fonction qui exécute le backtest et diffuse les résultats pour une stratégie donnée.
        """
        result = Result(self, risk_free_rate=risk_free_rate, mar=mar)

        # Affichage des statistiques de performance
        result.display_statistics()

        # Visualisation
        result.plot_results(backend)

        result.plot_positions_bar(backend)
