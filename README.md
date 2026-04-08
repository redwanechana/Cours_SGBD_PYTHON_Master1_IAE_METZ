# 📈 Prédiction de Cours Boursiers — ARIMA vs LSTM

### Master 1 Finance — Projet de groupe académique — IAE Metz 2026

### Redwane Chana, Walid Litouche

### Team Wared Capital 

> **Date de remise : 11 avril 2026**

---

## Objectif du projet

Ce projet compare deux méthodes de prédiction de cours boursiers sur un portefeuille d'actions technologiques américaines.

**L'objectif :** évaluer quelle méthode capture le mieux les tendances du marché et produit les prévisions les plus précises à court terme.

Nous comparons deux approches :

- **ARIMA** — un modèle statistique classique de séries temporelles
- **LSTM** — un réseau de neurones récurrent (deep learning)

---

## Portefeuille

| Ticker | Entreprise      | Secteur     |
|--------|-----------------|-------------|
| AAPL   | Apple           | Technologie |
| MSFT   | Microsoft       | Technologie |
| GOOGL  | Alphabet        | Technologie |
| AMZN   | Amazon          | E-Commerce  |
| TSLA   | Tesla           | Automobile  |

- **Action cible pour la prédiction :** AAPL (Apple)
- **Période :** Janvier 2019 – Décembre 2024
- **Répartition Train/Test :** 80% / 20%

---

## Source des données

Toutes les données sont téléchargées automatiquement depuis **Yahoo Finance** via la bibliothèque Python `yfinance`.
Aucun téléchargement manuel n'est nécessaire. Le script gère tout automatiquement.

---

## Structure du projet
```
projet/
│
├── DATA/                              # Fichiers de données et graphiques générés
│   ├── prices.csv
│   ├── prices_clean.csv
│   ├── returns.csv
│   ├── arima_results.csv
│   ├── lstm_results.csv
│   ├── arima_metrics.csv
│   ├── lstm_metrics.csv
│   ├── chart_01_prices.png
│   ├── chart_02_returns_distribution.png
│   ├── chart_03_arima_forecast.png
│   ├── chart_04_lstm_forecast.png
│   └── chart_05_model_comparison.png
│
├── SCRIPT/                            # Scripts Python
│   ├── 01_download_data.py            # Téléchargement des prix depuis Yahoo Finance
│   ├── 02_inspect_data.py             # Inspection et nettoyage des données
│   ├── 03_compute_returns.py          # Calcul des rendements journaliers
│   ├── 04_arima_prediction.py         # Entraînement et prédiction ARIMA
│   ├── 05_lstm_prediction.py          # Entraînement et prédiction LSTM
│   ├── 06_visualizations.py           # Génération des graphiques
│   └── main.py                        # Exécuter tout en une seule commande
│
├── PRESENTATION/                      # Slides et supports de présentation
├── app.py                             # Dashboard interactif Streamlit
├── requirements.txt                   # Dépendances Python
└── README.md                          # Ce fichier
```

---

## Installation

Assurez-vous que Python 3.9 ou supérieur est installé sur votre machine.

Installez les bibliothèques requises avec cette commande :
```
pip install -r requirements.txt
```

---

## Comment exécuter

### Option 1 — Tout exécuter d'un coup (recommandé)
```
cd SCRIPT
python main.py
```

### Option 2 — Exécuter les scripts un par un
```
cd SCRIPT
python 01_download_data.py
python 02_inspect_data.py
python 03_compute_returns.py
python 04_arima_prediction.py
python 05_lstm_prediction.py
python 06_visualizations.py
```

### Option 3 — Dashboard interactif
```
streamlit run app.py
```

Tous les fichiers générés (CSV et PNG) seront sauvegardés dans le dossier `DATA/`.

---

## Description des méthodes

### ARIMA (AutoRegressive Integrated Moving Average)

Modèle statistique classique pour la prévision de séries temporelles. Nous utilisons ARIMA(5,1,0) :
- **AR(5) :** utilise les 5 valeurs précédentes comme prédicteurs
- **I(1) :** différenciation d'ordre 1 pour rendre la série stationnaire
- **MA(0) :** pas de composante moyenne mobile

ARIMA fonctionne bien pour les tendances linéaires à court terme mais ne capture pas les patterns non-linéaires complexes.

### LSTM (Long Short-Term Memory)

Réseau de neurones récurrent (RNN) conçu pour les données séquentielles. Notre architecture :
- 2 couches LSTM (64 unités chacune) avec Dropout (20%)
- 1 couche Dense (32 unités, activation ReLU)
- Sortie : 1 valeur (prix du jour suivant)
- Fenêtre d'observation : 60 jours
- Entraînement : 50 époques, batch size 32

Le LSTM peut apprendre des dépendances à long terme et des patterns non-linéaires dans les données.

---

## Métriques d'évaluation

| Métrique | Description |
|----------|-------------|
| **MAE** (Mean Absolute Error) | Différence absolue moyenne entre prédictions et valeurs réelles |
| **RMSE** (Root Mean Squared Error) | Pénalise davantage les erreurs importantes |
| **MAPE** (Mean Absolute Percentage Error) | Erreur en pourcentage des valeurs réelles |

---

## Limites

- Les performances passées ne garantissent pas les résultats futurs
- ARIMA suppose des relations linéaires et la stationnarité
- Le LSTM est sensible aux hyperparamètres et nécessite plus de ressources
- Aucun des deux modèles ne prend en compte les facteurs externes (actualités, résultats financiers, macroéconomie)
- Le modèle est entraîné sur une seule action ; les résultats ne sont pas forcément généralisables

---

## Bibliothèques Python utilisées

| Bibliothèque | Version   | Utilisation                    |
|---------------|-----------|--------------------------------|
| yfinance      | >= 0.2.30 | Téléchargement Yahoo Finance   |
| pandas        | >= 2.0.0  | Manipulation de données        |
| numpy         | >= 1.24.0 | Calculs numériques             |
| matplotlib    | >= 3.7.0  | Visualisations                 |
| scikit-learn  | >= 1.3.0  | Métriques, normalisation       |
| statsmodels   | >= 0.14.0 | Modèle ARIMA                   |
| tensorflow    | >= 2.15.0 | Réseau de neurones LSTM        |
| streamlit     | >= 1.30.0 | Dashboard interactif           |

---

*Master 1 Finance — Projet Académique — IAE Metz*
