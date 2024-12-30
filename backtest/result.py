import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
from typing import Tuple, List

class Result:
    def __init__(self, backtest, risk_free_rate : float, mar : float) -> None:
        """
        La classe Result regroupe les résultats du backtester. Elle résulte ainsi de son exécution.

        Attributs : 
        ---
            backtest : instance du backtester concerné
            positions : dataframe des positions par actifs à chaque date
            strategy_value : série des valeurs du portefeuille à chaque date
            strategy_returns : série des rendements de la stratégie 
            annualization_factor : facteur d'actualisation utilisé pour le calcul de la volatilité annualisée
        Puis un ensemble de statistiques concernant la stratégie (performance, volatilité, nombre de trades,...)
        """
        self.backtest = backtest
        self.positions : pd.DataFrame = self.backtest.get_position(self.backtest.strategy)
        self.strategy_value : pd.Series = self.backtest.calculate_market_value(self.positions)
        self.strategy_returns : pd.Series = self.backtest.calculate_returns()  # Rendements de la stratégie
        diff_days : int = (self.strategy_returns.index[1] - self.strategy_returns.index[0]).days
        
        if diff_days == 1 or diff_days == 3: # prise en compte de la possibilité d'avoir des données qui commencent un vendredi
            self.annualization_factor = 252
        elif diff_days == 7:
            self.annualization_factor = 52
        elif 28 <= diff_days <= 31:
            self.annualization_factor = 12
        elif 360 <= diff_days <= 370:
            self.annualization_factor = 1
        else:
            raise ValueError("La périodicité des données n'est pas détectée.")
        
        # Statistiques
        self.total_performance : float = self.calculate_total_performance()
        self.annualized_performance : float = self.calculate_annualized_performance()
        self.volatility : float = self.calculate_volatility()
        self.sharpe_ratio : float = self.calculate_sharpe_ratio(risk_free_rate)
        self.sortino_ratio : float = self.calculate_sortino_ratio(mar)
        self.max_drawdown : float = self.calculate_max_drawdown()
        self.trade_count, self.win_rate = self.calculate_trade_stats()

    def calculate_total_performance(self) -> float:
        """Calcule la performance totale."""
        return self.strategy_value.iloc[-1] / self.strategy_value.iloc[0] - 1

    def calculate_annualized_performance(self) -> float:
        """Calcule la performance annualisée."""
        total_years = (self.strategy_value.index[-1] - self.strategy_value.index[self.backtest.strategy.calculation_window]).days / 365.25
        return np.sign(1+self.total_performance)*(abs(1 + self.total_performance)) ** (1 / total_years) - 1

    def calculate_volatility(self) -> float:
        """Calcule la volatilité annualisée."""
        return self.strategy_returns.std() * np.sqrt(self.annualization_factor)

    def calculate_sharpe_ratio(self, risk_free_rate=0.03) -> float:
        """Calcule le ratio de Sharpe."""
        excess_return = self.annualized_performance - risk_free_rate
        return excess_return / self.volatility if self.volatility != 0 else np.inf

    def calculate_sortino_ratio(self, mar=0.03) -> float:
        """Calcule le ratio de Sortino."""
        downside_returns = self.strategy_returns[self.strategy_returns < mar].mean().mean()
        downside_volatility = np.sqrt((downside_returns**2).mean())
        return (self.annualized_performance - mar) / downside_volatility if downside_volatility != 0 else np.inf

    def calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum."""
        rolling_max = self.strategy_value.cummax()
        drawdowns = self.strategy_value/rolling_max - 1
        return drawdowns.min() 
    
    def calculate_trade_stats(self) -> Tuple[int, float]:
        """
        Calcule le nombre de trades (changements de position) et le pourcentage de trades gagnants.
        Un trade est gagnant si :
        - Une vente est suivie d'une sous-performance (prix baisse après la vente).
        - Un achat est suivie d'une surperformance (prix monte après l'achat).
        """
        # Différences entre positions à deux périodes consécutives
        position_changes = self.positions.diff()

        # Variation des prix (returns sur une base simple)
        price_changes = self.backtest.data.pct_change()

        trade_wins = 0
        trades = 0

        for t in range(1, len(position_changes)):
            for asset in self.positions.columns:
                # Si la position a changé pour cet actif à la date t
                if position_changes.iloc[t, self.positions.columns.get_loc(asset)] != 0:
                    trades += 1  
                    # Si on vend 
                    if position_changes.iloc[t, self.positions.columns.get_loc(asset)] < 0:
                        # On gagne si le prix baisse après la vente
                        if price_changes.iloc[t, self.backtest.data.columns.get_loc(asset)] < 0:
                            trade_wins += 1  
                    # Si on achète
                    elif position_changes.iloc[t, self.positions.columns.get_loc(asset)] > 0:
                        # On gagne si le prix augmente après l'achat
                        if price_changes.iloc[t, self.backtest.data.columns.get_loc(asset)] > 0:
                            trade_wins += 1  

        # On évite la division par zéro pour le win rate
        total_trades = max(trades, 1)
        win_rate = trade_wins / total_trades
        return total_trades, win_rate

    def plot_results(self, backend : str ="matplotlib") -> None:
        """
        Affiche l'évolution de la valeur du portefeuille pour une stratégie donnée.
        Le choix du type de graphique est laissé à la discrétion de l'utilisateur. 
        """
        # Vérifications élémentaires
        if backend not in {"matplotlib", "seaborn", "plotly"}:
            raise ValueError(f"Backend non reconnu : {backend}")
        if isinstance(self.strategy_value, pd.Series):
            data = self.strategy_value.reset_index()
            data.columns = ['Date', 'PortfolioValue']  
        else:
            data = self.strategy_value

        # Trois possibilités pour l'utilisateur dans le choix de ses graphiques
        if backend == "matplotlib":
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['PortfolioValue'], label="Valeur du portefeuille", color="blue")
            plt.title("Evolution de la valeur du portefeuille")
            plt.xlabel("Date")
            plt.ylabel("Valeur du portefeuille")
            plt.legend()
            plt.grid()
            plt.show()

        elif backend == "seaborn":
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x="Date", y="PortfolioValue", color="blue", label="Valeur du portefeuille")
            plt.title("Evolution de la valeur du portefeuille")
            plt.xlabel("Date")
            plt.ylabel("Valeur du portefeuille")
            plt.legend()
            plt.show()

        elif backend == "plotly":

            warnings.filterwarnings(action='ignore', category=FutureWarning)
            try:
                # Conversion explicite des dates en datetime64 si nécessaire
                if not np.issubdtype(data['Date'].dtype, np.datetime64):
                    data['Date'] = pd.to_datetime(data['Date'])
                
                # Utiliser la colonne de dates sans conversion explicite en tableau NumPy
                fig = px.line(
                    data,
                    x="Date",  # Utilise directement la colonne 'Date'
                    y="PortfolioValue",
                    title="Evolution de la valeur du portefeuille",
                    labels={"PortfolioValue": "Valeur du portefeuille", "Date": "Date"}
                )
                
                # Mise à jour du format des ticks pour l'axe des X (dates)
                fig.update_layout(
                    xaxis=dict(
                        title="Date",
                        tickformat="%Y",  # Affiche uniquement l'année (YYYY). Pour des dates complètes, utilisez '%Y-%m-%d'
                        showgrid=True,
                    ),
                    yaxis=dict(title="Valeur du portefeuille"),
                    template="plotly_white"
                )
                
                fig.show()
            finally:
                # Réactiver les avertissements globalement
                warnings.filterwarnings(action='default', category=FutureWarning)

    def plot_positions_bar(self, backend : str ="matplotlib") -> None:
        """
        Affiche un graphique en barres horizontales des positions à la date finale.
        Le choix du type de graphique est laissé à la discrétion de l'utilisateur.
        """
        # Vérifications élémentaires
        if backend not in {"matplotlib", "seaborn", "plotly"}:
            raise ValueError(f"Backend non reconnu : {backend}")
        if not isinstance(self.positions, pd.DataFrame):
            raise ValueError("self.positions doit être un DataFrame avec des colonnes d'actifs et des lignes de dates.")

        # Préparer les données
        latest_positions = self.positions.iloc[-1]
        latest_positions = latest_positions[latest_positions != 0]  
        if latest_positions.empty:
            raise ValueError("Aucune position non nulle trouvée dans les dernières données.")
        latest_positions = latest_positions.reindex(latest_positions.abs().index)
        labels = latest_positions.index
        values = latest_positions.values

        # Trois possibilités pour l'utilisateur
        if backend == "matplotlib":
            plt.figure(figsize=(10, 6))
            plt.barh(labels, values, color=["blue" if v > 0 else "red" for v in values])
            plt.title("Répartition des positions à la date finale")
            plt.xlabel("Valeur des positions (part de la valeur totale du portefeuille)")
            plt.ylabel("Actifs")
            plt.grid(axis="x")
            plt.show()

        elif backend == "seaborn":
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            data = pd.DataFrame({"Actif": labels, "Valeur": values})
            data["Type"] = ["Long" if v > 0 else "Short" for v in values]
            sns.barplot(
                x="Valeur",
                y="Actif",
                hue="Type",
                dodge=False,
                data=data,
                palette={"Long": "blue", "Short": "red"}
            )
            plt.title("Répartition des positions à la date finale")
            plt.xlabel("Valeur des positions (part de la valeur totale du portefeuille)")
            plt.ylabel("Actifs")
            plt.legend(title="Type", loc="best")
            plt.show()

        elif backend == "plotly":
            fig = px.bar(
                x=values,
                y=labels,
                orientation="h",
                title="Répartition des positions à la date finale",
                labels={"x": "Valeur des positions (part de la valeur totale du portefeuille)", "y": "Actifs"},
                color=values,
                color_continuous_scale=["red", "blue"]
            )
            fig.update_layout(template="plotly_white")
            fig.show()

    def display_statistics(self) -> None:
        """
        Affiche les statistiques de performance pour une stratégie donnée.
        """
        stats = {
            "Performance totale": f"{self.total_performance:.2%}",
            "Performance annualisée": f"{self.annualized_performance:.2%}",
            "Volatilité annualisée": f"{self.volatility:.2%}",
            "Ratio de Sharpe": f"{self.sharpe_ratio:.2f}",
            "Ratio de Sortino": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Nombre de trades": self.trade_count,
            "Pourcentage de trades gagnants": f"{self.win_rate:.2%}"
        }

        for key, value in stats.items():
            print(f"{key}: {value}")

    @staticmethod
    def compare_results(*results, strategy_names : List[str]=None, backend : str ="matplotlib") -> None:
        """
        Compare les statistiques de plusieurs résultats de backtests. 
        Ajoute la possibilité d'afficher les noms des stratégies dans la légende.
        Cette fonction appelle d'autres fonctions graphiques. Elle prend en argument plusieurs
        instances de résultats, une liste de nom et une chaîne de caractères pour le backend. 
        """
        # Vérifications élémentaires
        if backend not in {"matplotlib", "seaborn", "plotly"}:
            raise ValueError(f"Backend non reconnu : {backend}. Choisissez parmi 'matplotlib', 'seaborn', ou 'plotly'.")

        # Gestion des noms des stratégies
        if strategy_names is None:
            strategy_names = [f"Stratégie {i+1}" for i in range(len(results))]
        elif len(strategy_names) != len(results):
            raise ValueError("Le nombre de noms de stratégies doit correspondre au nombre de résultats.")

        # Affichage des statistiques pour chaque stratégie
        Result.stats_comparison(results, strategy_names)
        # Graphique des performances au fil du temps
        Result.plot_portfolio_performance(results, strategy_names, backend)
        # Graphique rendement / risque des stratégies
        Result.plot_risk_return_strategy(results, strategy_names, backend)
        # Heatmap des corrélations des stratégies
        Result.plot_correlation_heatmap(results, strategy_names)
        # Graphique de l'évolution des drawdowns 
        Result.plot_max_drawdown(results, strategy_names, backend)

    @staticmethod
    def stats_comparison(results, strategy_names) -> None:
        """
        Affiche les statistiques des stratégies comparées.
        """
        comparison = pd.DataFrame([
            {
                "Performance totale": res.total_performance,
                "Performance annualisée": res.annualized_performance,
                "Volatilité": res.volatility,
                "Ratio de Sharpe": res.sharpe_ratio,
                "Ratio de Sortino": res.sortino_ratio,
                "Max Drawdown": res.max_drawdown
            }
            for res in results
        ], index=strategy_names)

    # Affichage des statistiques comparées
        print("\nComparaison des performances :\n")
        print(comparison)
    
    @staticmethod
    def plot_portfolio_performance(results, strategy_names, backend) -> None:
        """
        Affiche sur un même graphique l'évolution des valeurs des portefeuilles
        pour plusieurs stratégies. Le choix du type de graphique est laissé à la 
        discrétion de l'utilisateur. 
        """
        if backend == "plotly":
            fig = go.Figure()  # Initialisation unique pour Plotly

        for i, (res, name) in enumerate(zip(results, strategy_names)):
            # Extraction des dates et des valeurs du portefeuille
            dates = res.strategy_value.index
            portfolio_values = res.strategy_value

            # Trois choix de graphique pour l'utilisateur
            if backend == "matplotlib":
                plt.plot(dates, portfolio_values, label=name)
            elif backend == "seaborn":
                sns.lineplot(x=dates, y=portfolio_values, label=name)
            elif backend == "plotly":
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=portfolio_values,
                        mode='lines',
                        name=name
                    )
                )

        # Configuration et affichage des graphiques
        if backend in {"matplotlib", "seaborn"}:
            plt.title("Comparaison des Performances des Stratégies")
            plt.xlabel("Date")
            plt.ylabel("Evolution de la valeur du portefeuille")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif backend == "plotly":
            fig.update_layout(
                title="Evolution de la valeur du portefeuille",
                xaxis_title="Date",
                yaxis_title="Evolution de la valeur du portefeuille",
                template="plotly_white"
            )
            fig.show()


    @staticmethod
    def plot_risk_return_strategy(results, strategy_names, backend) -> None:
        """
        Crée un graphique représentant les stratégies dans un plan risque / rendement.
        Le choix du type de graphique est laissé à la discrétion de l'utilisateur. 
        """
        volatility = [res.volatility for res in results]
        annualized_perf = [res.annualized_performance for res in results]
        # Vérifications élémentaires
        if backend not in {"matplotlib", "seaborn", "plotly"}:
            raise ValueError(f"Backend non reconnu : {backend}.")

        # Trois choix de graphiques pour l'utilisateur
        if backend == "matplotlib":
            plt.figure(figsize=(8, 6))
            for i, name in enumerate(strategy_names):
                plt.scatter(volatility[i], annualized_perf[i], label=name, s=100)
            plt.xlabel("Volatilité Annualisée (%)")
            plt.ylabel("Performance Annualisée (%)")
            plt.title("Couple rendement - risque pour chaque stratégie")
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.show()

        elif backend == "seaborn":
            plt.figure(figsize=(8, 6))
            data = pd.DataFrame({
                "Volatilité": volatility,
                "Performance": annualized_perf,
                "Stratégies": strategy_names
            })
            sns.scatterplot(data=data, x="Volatilité", y="Performance", hue="Stratégies", s=100)
            plt.title("Couple rendement - risque pour chaque stratégie")
            plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='red', linestyle='--', linewidth=0.8)
            plt.show()

        elif backend == "plotly":
            fig = px.scatter(
                x=volatility,
                y=annualized_perf,
                text=strategy_names,
                title="Couple rendement - risque pour chaque stratégie",
                labels={"x": "Volatilité Annualisée (%)", "y": "Performance Annualisée (%)"}
            )
            fig.update_traces(marker=dict(size=12, opacity=0.8))
            fig.show()

    @staticmethod
    def plot_correlation_heatmap(results, strategy_names, backend="seaborn") -> None:
        """
        Trace une heatmap des corrélations entre les stratégies. 
        Le choix du type de graphique est laissé à la discrétion de l'utilisateur
        """
        # On doit d'abord mettre tous les rendements des stratégies dans un même dataframe.
        combined_returns = pd.DataFrame({
            name: res.strategy_returns for res, name in zip(results, strategy_names)
        })

        # Calcul de la matrice de corrélation
        corr_matrix = combined_returns.corr()

        # Trois possibilités pour l'utilisateur
        if backend == "seaborn":
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
            plt.title("Corrélation entre les stratégies")
            plt.show()

        elif backend == "plotly":
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="coolwarm",
                title="Corrélation entre les stratégies"
            )
            fig.update_layout(
                xaxis_title="Stratégies",
                yaxis_title="Stratégies",
                coloraxis_colorbar=dict(title="Corrélation")
            )
            fig.show()
        
        elif backend == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(corr_matrix, cmap="coolwarm")
            fig.colorbar(cax)
            ax.set_xticks(range(len(strategy_names)))
            ax.set_yticks(range(len(strategy_names)))
            ax.set_xticklabels(strategy_names, rotation=45, ha="left")
            ax.set_yticklabels(strategy_names)
            plt.title("Corrélation entre les stratégies", pad=20)
            plt.show()

        else:
            raise ValueError("Backend non reconnu. Choisissez parmi 'seaborn' ou 'plotly'.")

    @staticmethod
    def plot_max_drawdown(results, strategy_names, backend="plotly") -> None:
        """
        Affiche les drawdowns maximums des stratégies sur la période.
        Le choix du graphique est laissé à la discrétion de l'utilisateur.
        """
        backend = backend.lower()
        if backend not in {"matplotlib", "plotly", "seaborn"}:
            raise ValueError("Backend non reconnu. Choisissez parmi 'matplotlib', 'seaborn' ou 'plotly'.")

        drawdowns = {}

        # Calcul des drawdowns pour chaque stratégie
        for res, name in zip(results, strategy_names):
            portfolio_values = res.strategy_value
            rolling_max = portfolio_values.cummax()
            drawdown = (portfolio_values / rolling_max) - 1
            drawdowns[name] = drawdown

        drawdowns_df = pd.DataFrame(drawdowns)

        if backend == "matplotlib":
            plt.figure(figsize=(12, 8))
            for name in strategy_names:
                plt.plot(drawdowns_df.index, drawdowns_df[name], label=name)
            plt.title("Drawdowns des stratégies")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        elif backend == "seaborn":
            plt.figure(figsize=(12, 8))
            sns.set(style="whitegrid")
            for name in strategy_names:
                sns.lineplot(x=drawdowns_df.index, y=drawdowns_df[name], label=name)
            plt.title("Drawdowns des stratégies")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.legend()
            plt.tight_layout()
            plt.show()
    
        elif backend == "plotly":
            fig = go.Figure()
            for name in strategy_names:
                fig.add_trace(go.Scatter(x=drawdowns_df.index, y=drawdowns_df[name], mode='lines', name=name))
            fig.update_layout(
                title="Drawdowns des stratégies",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white"
            )
            fig.show()
    
