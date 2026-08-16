import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import services as S
from views import _ui

_ui.page_head("Données historiques", "Explorer les séries de cas et de météo qui ont servi à entraîner le modèle")

h = S.get_history()
drs_all = S.get_drs_list()

tab1, tab2, tab3, tab4 = st.tabs(["Séries de cas", "Saisonnalité", "Météo", "Table"])

# ------------------------------------------------------------------ Séries
with tab1:
    sel = st.multiselect("Régions", drs_all, default=["DRS Kolda", "DRS Kedougou", "DRS Tambacounda", "DRS Diourbel"],
                         format_func=S.short_name, key="exp_sel")
    log = st.toggle("Échelle logarithmique", value=False,
                    help="Utile car les volumes varient de quelques cas (Fatick) à plusieurs milliers (Kolda).")
    if sel:
        sub = h[h["DRS"].isin(sel)].copy()
        sub["Région"] = sub["DRS"].map(S.short_name)
        fig = px.line(sub, x="PERIODE", y=S.TARGET, color="Région", markers=True,
                      labels={S.TARGET: "Cas confirmés (TDR)", "PERIODE": ""})
        fig.update_layout(height=460, yaxis_type="log" if log else "linear")
        fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y:,.0f} cas")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Volume total par région et par année**")
    piv = (h.assign(annee=h["PERIODE"].dt.year, Région=h["DRS"].map(S.short_name))
           .pivot_table(index="Région", columns="annee", values=S.TARGET, aggfunc="sum")
           .sort_values(h["PERIODE"].dt.year.max(), ascending=False))
    st.dataframe(piv.style.format("{:,.0f}").background_gradient(cmap="YlOrBr", axis=None),
                 use_container_width=True)

# ------------------------------------------------------------------ Saisonnalité
with tab2:
    st.markdown("Profil mensuel moyen (part de chaque mois dans le total annuel de la région). "
                "La saison de transmission suit l'hivernage avec un décalage de 1 à 3 mois.")
    prof = h.assign(m=h["PERIODE"].dt.month).groupby(["DRS", "m"])[S.TARGET].mean().unstack("m")
    prof = prof.div(prof.sum(axis=1), axis=0) * 100
    prof.index = prof.index.map(S.short_name)
    prof.columns = S.MOIS_FR
    fig = go.Figure(go.Heatmap(z=prof.values, x=prof.columns, y=prof.index,
                               colorscale=[[0, "#F7F9F8"], [0.5, "#D28A2C"], [1, "#B7412E"]],
                               hovertemplate="%{y} · %{x}<br>%{z:.1f} % des cas annuels<extra></extra>",
                               colorbar=dict(title="%", thickness=12)))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        d = st.selectbox("Zoom sur une région", drs_all, format_func=S.short_name, key="exp_season_drs")
    sub = h[h["DRS"] == d].assign(annee=h["PERIODE"].dt.year, m=h["PERIODE"].dt.month)
    fig2 = px.line(sub, x="m", y=S.TARGET, color="annee", markers=True,
                   labels={"m": "Mois", S.TARGET: "Cas", "annee": "Année"})
    fig2.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=S.MOIS_FR)
    fig2.update_layout(height=380, title=f"{S.short_name(d)} — une courbe par année")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------ Météo
with tab3:
    c1, c2 = st.columns([1, 1])
    with c1:
        d = st.selectbox("Région", drs_all, format_func=S.short_name, key="exp_w_drs")
    with c2:
        var = st.selectbox("Variable météo", S.WEATHER_VARS, format_func=lambda v: S.WEATHER_LABELS[v],
                           index=3, key="exp_w_var")
    sub = h[h["DRS"] == d].sort_values("PERIODE")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub["PERIODE"], y=sub[var], name=S.WEATHER_LABELS[var],
                         marker_color=_ui.BLUE, opacity=0.75, yaxis="y"))
    fig.add_trace(go.Scatter(x=sub["PERIODE"], y=sub[S.TARGET], name="Cas confirmés",
                             line=dict(color=_ui.GREEN, width=2.4), yaxis="y2"))
    fig.update_layout(
        height=430, title=f"{S.short_name(d)} — {S.WEATHER_LABELS[var]} vs cas",
        yaxis=dict(title=S.WEATHER_LABELS[var]),
        yaxis2=dict(title="Cas", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Corrélation (toutes régions) entre les variables météo et les cas du même mois / des mois suivants**")
    lag_max = 3
    rows = []
    for v in S.WEATHER_VARS:
        r = {"Variable": S.WEATHER_LABELS[v]}
        for L in range(0, lag_max + 1):
            shifted = h.groupby("DRS")[v].shift(L)
            r[f"décalage {L} mois"] = h[S.TARGET].corr(shifted)
        rows.append(r)
    corr = pd.DataFrame(rows).set_index("Variable")
    st.dataframe(corr.style.format("{:+.2f}").background_gradient(cmap="RdYlGn", vmin=-0.4, vmax=0.4, axis=None),
                 use_container_width=True)
    st.caption("Corrélation de Pearson. Un décalage de k mois compare les cas du mois t à la météo du mois t‑k.")

# ------------------------------------------------------------------ Table
with tab4:
    d = st.multiselect("Filtrer par région", drs_all, format_func=S.short_name, key="exp_tab_drs")
    sub = h if not d else h[h["DRS"].isin(d)]
    show = sub.rename(columns={S.TARGET: "Cas confirmés (TDR)"} | S.WEATHER_LABELS)
    show["PERIODE"] = show["PERIODE"].dt.strftime("%Y-%m")
    st.dataframe(show, use_container_width=True, height=480, hide_index=True)
    st.download_button("⬇️ Télécharger (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "historique_drs.csv", "text/csv")
