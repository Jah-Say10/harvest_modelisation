import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import services as S
from views import _ui

_ui.page_head(
    "Prévision régionale du paludisme",
    "Cas confirmés par TDR en consultations externes, par Direction Régionale de la Santé (DRS) — Sénégal",
)

h = S.get_history()
tmin, tmax = S.history_range()
last_year = tmax.year
tot_last = h.loc[h["PERIODE"].dt.year == last_year, S.TARGET].sum()
tot_prev = h.loc[h["PERIODE"].dt.year == last_year - 1, S.TARGET].sum()
delta = (tot_last - tot_prev) / tot_prev * 100 if tot_prev else 0
peak = h.groupby(h["PERIODE"].dt.month)[S.TARGET].sum().idxmax()
top_drs = h.loc[h["PERIODE"].dt.year == last_year].groupby("DRS")[S.TARGET].sum().idxmax()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    _ui.kpi("Régions couvertes", f"{h['DRS'].nunique()} DRS", "Toutes les régions du pays")
with c2:
    _ui.kpi("Historique", f"{h['PERIODE'].nunique()} mois",
            f"{S.label_period(tmin)} → {S.label_period(tmax)}")
with c3:
    _ui.kpi(f"Cas {last_year}", _ui.fmt_int(tot_last),
            f"{delta:+.0f} % vs {last_year - 1}")
with c4:
    _ui.kpi("Mois de pic (moy.)", S.MOIS_FR[peak - 1], "Saison de transmission Sept‑Nov")
with c5:
    _ui.kpi(f"DRS la plus touchée {last_year}", S.short_name(top_drs), accent=True)

st.write("")
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Où se concentrent les cas ?")
    year_map = st.select_slider("Année", options=sorted(h["PERIODE"].dt.year.unique()),
                                value=last_year, key="acc_year")
    agg = (h[h["PERIODE"].dt.year == year_map].groupby("DRS")[S.TARGET].sum()
           .reset_index().rename(columns={S.TARGET: "cas"}))
    agg["lat"] = agg["DRS"].map(lambda d: S.DRS_COORDS[d][0])
    agg["lon"] = agg["DRS"].map(lambda d: S.DRS_COORDS[d][1])
    agg["Région"] = agg["DRS"].map(S.short_name)
    fig = px.scatter_geo(
        agg, lat="lat", lon="lon", size="cas", color="cas", hover_name="Région",
        hover_data={"cas": ":,.0f", "lat": False, "lon": False},
        color_continuous_scale=["#CFE5DC", "#0F6E56", "#D28A2C", "#B7412E"],
        size_max=48, projection="mercator",
    )
    fig.update_geos(
        lataxis_range=[12.0, 16.9], lonaxis_range=[-17.9, -11.2],
        showcountries=True, countrycolor="#B8C4BE", showland=True, landcolor="#F7F9F8",
        showocean=True, oceancolor="#EAF2F5", showframe=False,
    )
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                      coloraxis_colorbar=dict(title="Cas", thickness=12))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Dynamique nationale")
    nat = h.groupby("PERIODE")[S.TARGET].sum().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=nat["PERIODE"], y=nat[S.TARGET], mode="lines",
                              fill="tozeroy", line=dict(color=_ui.GREEN, width=2.4),
                              fillcolor="rgba(15,110,86,0.10)", name="Cas (toutes DRS)",
                              hovertemplate="%{x|%b %Y}<br>%{y:,.0f} cas<extra></extra>"))
    fig2.update_layout(height=430, yaxis_title="Cas confirmés (TDR)", showlegend=False,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Comment fonctionne le pipeline ?")
a, b = st.columns([1.2, 1])
with a:
    steps = [
        ("Données mensuelles par DRS", "Cas confirmés (TDR) + 7 variables météo agrégées au mois "
         "(température, précipitations, humidité, vent, rayonnement)."),
        ("Ingénierie des variables", "Retards de la cible (1, 2, 3, 6, 12 mois), moyennes glissantes "
         "3 et 6 mois, calendrier (mois, année, encodage sinus/cosinus)."),
        ("Modèle LightGBM", "Un seul modèle national avec la région en one‑hot ; "
         "gradient boosting sur 200 arbres."),
        ("Prévision récursive", "Chaque mois prévu réalimente les lags du mois suivant, "
         "ce qui permet de projeter sur 12 mois et plus."),
        ("Restitution", "Cette interface : prévisions, scénarios climatiques, "
         "évaluation et export."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        st.markdown(f"<div class='step'><div class='n'>{i}</div><div><div class='t'>{t}</div>"
                    f"<div class='d'>{d}</div></div></div>", unsafe_allow_html=True)
with b:
    st.markdown("**Par où commencer ?**")
    st.page_link("views/prevision.py", label="Lancer une prévision", icon="🔮")
    st.page_link("views/scenarios.py", label="Comparer des scénarios climatiques", icon="🌦️")
    st.page_link("views/evaluation.py", label="Vérifier la qualité du modèle", icon="🎯")
    st.page_link("views/modele.py", label="Comprendre le modèle", icon="🧬")
    st.info("La prévision requiert une hypothèse météo pour les mois à venir. "
            "Par défaut l'application utilise la climatologie historique de chaque région ; "
            "vous pouvez importer vos propres prévisions météo.", icon="💡")
