# Backtest Package

## Description

Le package `Backtest` est conçu pour effectuer des simulations historiques d'une stratégie d'investissement sur un ensemble de données. Il permet d'obtenir des statistiques de performance, des visualisations et des rendements.

Ce backtest sert à tester une stratégie de trading définie par l'utilisateur du package. Elle devra hériter de la classe abstraite `Strategy`, qui impose certaines méthodes essentielles comme `get_position`. Cela garantit une flexibilité maximale tout en maintenant une structure cohérente pour l'analyse.

## Fonctionnalités principales

- **Calcul des positions** : Génération des positions passées basées sur une stratégie définie par l'utilisateur ou dans les stratégies exemple.
- **Valeur de marché** : Calcul de la valeur de marché d'un portefeuille suivant cette stratégie à partir des positions et des prix des actifs.
- **Rendements** : Calcul des rendements logarithmiques du portefeuille suivant la stratégie.
- **Analyse de performance** : Affichage et visualisation des statistiques de performance.

## Classes principales

### Backtest

Cette classe est le coeur du package. Elle permet de réaliser un backtest complet en utilisant une stratégie prédéfinie et un ensemble de données.

#### Constructeur

```python
Backtest(data: pd.DataFrame, strategy: Strategy, frequence_rebalancement: int)
```

- **data** : `pd.DataFrame` contenant les données de prix historiques des actifs.
- **strategy** : Instance de la classe `Strategy` définissant les règles d'investissement.
- **frequence_rebalancement** : Fréquence de rebalancement des positions en nombre de jours.

#### Méthodes

- **get_position(strategy: Strategy)** :
  - Calcule les positions de la stratégie pour chaque date en utilisant une fenêtre glissante définie par la stratégie.

- **calculate_market_value(positions: pd.DataFrame)** :
  - Calcule la valeur de marché des positions du portefeuille.

- **calculate_returns() -> pd.DataFrame** :
  - Calcule les rendements logarithmiques basés sur la valeur de marché du portefeuille.

- **BACKTEST(risk_free_rate: float, mar: float)** :
  - Effectue le backtest complet en calculant les statistiques et en affichant les visualisations via la classe `Result`.

### Strategy

La classe `Strategy` est une classe abstraite définissant les règles générales pour implémenter une stratégie d'investissement. Les utilisateurs doivent créer des classes enfants qui héritent de `Strategy` et implémentent les méthodes nécessaires, notamment :

- **`__init__`** :
  - Permet de définir une fenêtre de calcul optionnelle (`calculation_window`).

- **`get_position`** :
  - Méthode obligatoire pour calculer les positions basées sur les données historiques et la position actuelle.

#### Méthode statique utile

- **fit** :
  - Permet d'effectuer une régression linéaire ordinaire (OLS) sur des variables endogènes et exogènes.

```python
@staticmethod
def fit(endogenous_variable:Union[np.array, pd.DataFrame], exogeneous_variables:Union[np.array, pd.DataFrame]) -> sm.regression.linear_model.RegressionResultsWrapper:
    results_OLS = sm.OLS(endogenous_variable, exogeneous_variables).fit()
    return results_OLS
```

### Result

La classe `Result` permet d'analyser et de visualiser les résultats du backtest. Elle calcule les métriques de performance et génère des graphiques interactifs ou statiques pour aider à interpréter les performances de la stratégie testée.

#### Constructeur

```python
Result(backtest: Backtest, risk_free_rate: float, mar: float)
```

- **backtest** : Instance de la classe `Backtest` utilisée pour générer les données.
- **risk_free_rate** : Taux sans risque utilisé pour calculer le ratio de Sharpe.
- **mar** : Rendement minimum acceptable utilisé pour le ratio de Sortino.

#### Méthodes principales

- **calculate_total_performance()** : Calcule la performance totale de la stratégie.
- **calculate_annualized_performance()** : Calcule la performance annualisée.
- **calculate_volatility()** : Calcule la volatilité annualisée.
- **calculate_sharpe_ratio(risk_free_rate: float)** : Calcule le ratio de Sharpe.
- **calculate_sortino_ratio(mar: float)** : Calcule le ratio de Sortino.
- **calculate_max_drawdown()** : Calcule le drawdown maximum.
- **calculate_trade_stats()** : Analyse les trades pour déterminer leur nombre et le pourcentage de succès.

#### Visualisations

- **plot_results(backend="matplotlib")** : Affiche la valeur du portefeuille au cours du temps. Options pour `seaborn` et `plotly`.
- **plot_risk_return(backend="seaborn")** : Génère un graphique rendement/risque pour chaque actif. Options pour `matplotlib`, `seaborn`, et `plotly`.

#### Statistiques

- **display_statistics()** : Affiche un résumé des métriques de performance principales :
  - Performance totale
  - Performance annualisée
  - Volatilité annualisée
  - Ratio de Sharpe
  - Ratio de Sortino
  - Drawdown maximum
  - Nombre de trades
  - Pourcentage de trades gagnants

- **compare_results(*results, backend="matplotlib")** : Compare les performances de plusieurs stratégies à l'aide d'un graphique interactif.

## Prérequis

Le package nécessite les bibliothèques suivantes :

- `pandas`
- `numpy`
- `plotly` (pour les visualisations via la classe `Result`)
- `statsmodels`
- `scipy`
- `matplotlib`
- `seaborn`

## Installation

### Installation via PyPI

Un package PyPI est disponible pour l'implémentation du `Backtest`. Vous pouvez l'installer directement depuis PyPI :

```bash
pip install backtest
```

### Installation via le dépôt Git

Vous pouvez également cloner le dépôt et l'installer localement :
(Il faudra changer cette partie quand on aura déposé le projet sur GitHub)

```bash
git clone https://github.com/username/backtest.git
cd backtest
pip install .
```

## Exemple d'utilisation

```python
import pandas as pd
from strategy import MyStrategy  # Classe personnalisée héritant de Strategy
from backtest import Backtest
from result import Result

# Charger les données
data = pd.read_csv("historical_prices.csv", index_col="Date", parse_dates=True)

# Définir une stratégie
strategy = MyStrategy(calculation_window=30)  # Fenêtre de calcul de 30 jours

# Initialiser le backtest
backtest = Backtest(data=data, strategy=strategy, frequence_rebalancement=10)

# Effectuer le backtest complet
backtest.BACKTEST(risk_free_rate=0.02, mar=0.05)

# Analyser les résultats
result = Result(backtest, risk_free_rate=0.02, mar=0.05)
result.display_statistics()
result.plot_results(backend="plotly")
result.plot_risk_return(backend="seaborn")
```

## Structure des fichiers

- `backtest.py` : Contient la classe principale `Backtest`.
- `strategy.py` : Contient la classe `Strategy` et ses implémentations personnalisées.
- `result.py` : Gère l'analyse des résultats et les visualisations via la classe `Result`.


## Licence

Hmm sous quelle license est ce projet ? Dauphine ?
