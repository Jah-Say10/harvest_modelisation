import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import services as S
from views import _ui

_ui.page_head("Pipeline & explicabilité", "Ce que le modèle apprend, et comment il produit une prévision")

tab1, tab2, tab3 = st.tabs(["Architecture du pipeline", "Importance des variables", "Fiche technique"])

# ------------------------------------------------------------------ Architecture
with tab1:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("#### De la donnée brute à la prévision")
        fig = go.Figure()
        boxes = [
            ("Historique mensuel\npar DRS", 0), ("Météo\n(7 variables)", 0),
            ("Ingénierie des\nvariables", 1), ("One‑hot DRS +\npassthrough", 2),
            ("LightGBM\n200 arbres", 3), ("Prévision\nmois M", 4), ("Réinjection\ncomme lag", 5),
        ]
        pos = {0: [(0.5, 3.6), (0.5, 1.4)], 1: [(2.3, 2.5)], 2: [(4.1, 2.5)], 3: [(5.9, 2.5)], 4: [(7.7, 2.5)], 5: [(7.7, 0.6)]}
        used = {k: 0 for k in pos}
        coords = []
        for label, col in boxes:
            x, y = pos[col][used[col]]
            used[col] += 1
            coords.append((x, y))
            color = _ui.OCHRE if "LightGBM" in label else (_ui.GREEN if "Prévision" in label else "#F3F6F4")
            fc = "white" if color != "#F3F6F4" else "#1B2A24"
            fig.add_shape(type="rect", x0=x - 0.75, x1=x + 0.75, y0=y - 0.5, y1=y + 0.5,
                          fillcolor=color, line=dict(color="#B8C4BE", width=1), layer="below")
            fig.add_annotation(x=x, y=y, text=label.replace("\n", "<br>"), showarrow=False,
                               font=dict(size=12, color=fc, family="Inter, sans-serif"))
        arrows = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        for a, b in arrows:
            (xa, ya), (xb, yb) = coords[a], coords[b]
            fig.add_annotation(x=xb - 0.75 if xb > xa else xb, y=yb if xb > xa else yb + 0.5,
                               ax=xa + 0.75 if xb > xa else xa, ay=ya if xb > xa else ya - 0.5,
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor=_ui.SLATE)
        # boucle récursive
        fig.add_annotation(x=2.3, y=2.0, ax=7.0, ay=0.6, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=_ui.OCHRE)
        fig.add_annotation(x=4.6, y=0.25, text="boucle récursive : la prévision du mois M devient lag_1 du mois M+1",
                           showarrow=False, font=dict(size=11, color=_ui.OCHRE))
        fig.update_xaxes(visible=False, range=[-0.5, 8.7])
        fig.update_yaxes(visible=False, range=[-0.2, 4.4])
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("#### Variables d'entrée du modèle")
        st.markdown(
            "<span class='badge'>Région</span> DRS (one‑hot, 14 modalités)<br><br>"
            "<span class='badge'>Météo</span> " + " · ".join(S.WEATHER_LABELS[v].split(" (")[0] for v in S.WEATHER_VARS) + "<br><br>"
            "<span class='badge o'>Historique des cas</span> lags 1, 2, 3, 6, 12 mois · moyennes glissantes 3 et 6 mois<br><br>"
            "<span class='badge'>Calendrier</span> mois, année, sin/cos du mois",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("**Cible** : *Cas confirmés (par TDR) palu — consultations externes*, par mois et par DRS.")
        st.markdown("**Contrainte** : ≥ 12 mois d'historique avant le premier mois prévu (lag 12).")

# ------------------------------------------------------------------ Importances
with tab2:
    imp = S.feature_importances()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        top = imp.head(20).iloc[::-1]
        fam_colors = {"Historique des cas (lags / moyennes glissantes)": _ui.OCHRE, "Météo": _ui.BLUE,
                      "Calendrier": _ui.GREEN, "Région (one-hot)": _ui.SLATE}
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                               marker_color=top["famille"].map(fam_colors),
                               hovertemplate="%{y}<br>%{x} splits<extra></extra>"))
        fig.update_layout(height=560, xaxis_title="Importance (nombre de divisions dans les arbres)",
                          margin=dict(l=10, r=10, t=30, b=10), title="20 variables les plus utilisées")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fam = imp.groupby("famille")["importance"].sum().reset_index()
        fig2 = px.pie(fam, values="importance", names="famille", hole=0.55,
                      color="famille", color_discrete_map=fam_colors)
        fig2.update_layout(height=340, showlegend=True, legend=dict(orientation="h", y=-0.15), title="Par famille de variables")
        fig2.update_traces(textinfo="percent")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            "**Lecture** : le modèle s'appuie d'abord sur la dynamique récente des cas (lag 1, lag 12 pour la "
            "saisonnalité annuelle), puis sur la météo — vent et humidité en tête — pour moduler la prévision. "
            "L'importance mesurée ici est le nombre de fois où une variable est utilisée pour diviser un nœud ; "
            "elle ne dit pas le sens de l'effet."
        )
    with st.expander("Table complète"):
        st.dataframe(imp, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ Fiche
with tab3:
    ms = S.model_summary()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Hyperparamètres LightGBM")
        st.dataframe(pd.DataFrame({"Paramètre": list(ms), "Valeur": [str(v) for v in ms.values()]}),
                     hide_index=True, use_container_width=True)
    with c2:
        st.markdown("#### Artefacts chargés")
        st.code(
            f"{S.P.MODEL_DIR}/\n"
            "├── drs_model.pkl            # sklearn Pipeline (ColumnTransformer + LGBMRegressor)\n"
            "├── history_df_drs.pkl       # historique 14 DRS × 48 mois\n"
            "└── feature_columns_drs.pkl  # liste des colonnes de features",
            language="text",
        )
        st.markdown("#### Utilisation programmatique")
        st.code(
            'from core.predictor import predict_region\n\n'
            'forecast = predict_region(\n'
            '    drs="DRS Kolda",\n'
            '    start_date="2025-01-01",\n'
            '    end_date="2025-12-01",\n'
            '    weather_df=weather_df,   # PERIODE + 7 variables météo\n'
            ')',
            language="python",
        )
    st.markdown("#### Limites et bonnes pratiques")
    st.markdown(
        "- **Météo future inconnue** : la qualité de la prévision dépend de l'hypothèse météo fournie "
        "(climatologie, prévision saisonnière, scénario).\n"
        "- **Prévision récursive** : les erreurs se propagent via les lags ; privilégier des horizons ≤ 12 mois.\n"
        "- **Effets régionaux** : un seul modèle national ; les petites régions (Fatick, Matam) sont plus bruitées "
        "en valeur relative.\n"
        "- **Ré‑entraînement** : à chaque nouvelle année de données, réentraîner et remplacer les artefacts dans "
        "`models/registry/` — l'interface s'adapte automatiquement (liste des DRS, période, climatologie)."
    )
