# Interface de prévision régionale du paludisme (Streamlit)

Interface web construite autour du pipeline `predictor_drs.py` (LightGBM, 14 DRS, données mensuelles 2021‑2024).
Elle sert à la fois d'outil d'utilisation (prévisions, exports) et de support de présentation du pipeline.

## Lancer l'application

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Structure

```
palu_dashboard/
├── app.py                     # point d'entrée + navigation
├── core/
│   ├── predictor.py           # VOTRE module de prévision (chemins rendus configurables, sinon inchangé)
│   └── services.py            # chargement (cache), météo (climatologie / année ref / import), scénarios, backtest, importances
├── views/                     # une page = un fichier
│   ├── _ui.py                 # CSS, cartes KPI, thème Plotly, graphique historique+prévision
│   ├── accueil.py             # vue d'ensemble : KPIs, carte, courbe nationale, résumé du pipeline
│   ├── exploration.py         # données historiques : séries, saisonnalité, météo, table
│   ├── prevision.py           # prévoir : DRS, horizon, hypothèse météo, résultats, exports
│   ├── scenarios.py           # scénarios climatiques comparés
│   ├── evaluation.py          # rétro‑prévision et métriques (MAE, RMSE, MAPE, R²)
│   └── modele.py              # schéma du pipeline, importance des variables, fiche technique
├── models/registry/           # drs_model.pkl, history_df_drs.pkl, feature_columns_drs.pkl
├── data/exemple_meteo_2025_climatologie.csv   # exemple de fichier météo importable
├── .streamlit/config.toml     # thème
└── requirements.txt
```

## Brancher sur votre pipeline

- **Artefacts** : déposez vos `.pkl` dans `models/registry/` (ou pointez `PALU_MODEL_DIR=/chemin/vers/registry`).
  L'interface lit la liste des DRS, la période et la climatologie directement depuis `history_df_drs.pkl` :
  après ré‑entraînement, il suffit de remplacer les fichiers.
- **Météo future** : trois options dans la page *Prévoir* — climatologie (défaut), rejouer une année passée,
  ou importer un CSV/Excel avec les colonnes `PERIODE, temperature_moyenne, temperature_max, temperature_min,
  precipitation, humidite, vent, rayonnement_solaire` (+ `DRS` optionnel pour un fichier multi‑régions).
  Un modèle pré‑rempli est téléchargeable depuis la page.
- **Reste du pipeline** (préparation, entraînement) : non inclus dans le zip reçu ; `core/predictor.py` est le seul
  point de contact avec le modèle. Si `predict_region` change de signature, adaptez `services.forecast_one`.

## Modification apportée à `predictor_drs.py`

Uniquement deux choses, sans changer la logique :
1. `MODEL_DIR` devient configurable (variable d'environnement `PALU_MODEL_DIR`, défaut `models/registry` du projet).
2. `predict_region(..., model=None, history=None)` accepte les objets déjà chargés pour éviter de relire les
   pickles à chaque prévision (Streamlit les met en cache).

## Notes

- La page *Évaluation* fait une rétro‑prévision avec le modèle fourni, entraîné sur toute la période : les scores
  sont donc optimistes. Pour un vrai hors‑échantillon, réentraînez jusqu'à la date de coupure et remplacez le `.pkl`.
- Testé avec Streamlit 1.61 / Python 3.12 ; compatible ≥ 1.36 (`st.navigation`).
