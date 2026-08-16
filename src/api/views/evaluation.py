import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import services as S
from views import _ui

_ui.page_head("Évaluation du modèle", "Rétro‑prévision : que prévoyait le modèle sur les mois déjà observés ?")

h = S.get_history()
drs_all = S.get_drs_list()
tmin, tmax = S.history_range()

st.markdown(
    "On coupe l'historique à une date, on prévoit récursivement les mois suivants avec la **météo réellement "
    "observée**, puis on compare aux cas observés. C'est un test de la mécanique de prévision récursive "
    "(propagation des erreurs de lag)."
)
st.warning(
    "Le modèle chargé a été entraîné sur l'ensemble de la période disponible : ces mois n'étaient donc pas "
    "totalement inconnus de lui. Les métriques ci‑dessous sont **optimistes** par rapport à une vraie "
    "validation hors‑échantillon. Pour une évaluation stricte, réentraînez le modèle jusqu'à la date de coupure "
    "et remplacez `drs_model.pkl`.", icon="⚠️",
)

with st.container(border=True):
    c1, c2, c3 = st.columns([1.5, 1.2, 1])
    with c1:
        d = st.selectbox("Région", drs_all, index=drs_all.index("DRS Kolda"), format_func=S.short_name, key="ev_drs")
    with c2:
        cut_options = pd.date_range(tmin + pd.DateOffset(months=11), tmax - pd.DateOffset(months=1), freq="MS")
        cutoff = st.select_slider("Dernier mois connu (date de coupure)", options=list(cut_options),
                                  value=pd.Timestamp(f"{tmax.year - 1}-12-01"),
                                  format_func=S.label_period, key="ev_cut")
    with c3:
        max_h = int((tmax.to_period("M") - cutoff.to_period("M")).n)
        horizon = st.slider("Horizon (mois)", 1, max(1, min(24, max_h)), min(12, max_h), key="ev_h")

bt = S.backtest(d, cutoff, horizon)
if bt.empty:
    st.info("Aucun mois observé après cette date de coupure.")
    st.stop()

m = S.metrics(bt["observe"], bt["prevu"])
k = st.columns(5)
labels = [("MAE", "cas / mois"), ("RMSE", "cas / mois"), ("MAPE (%)", "sur mois > 0"),
          ("R²", "sur la période"), ("Biais", "prévu − observé")]
for col, (key, sub) in zip(k, labels):
    v = m[key]
    txt = "—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if key == "R²" else f"{v:,.1f}")
    with col:
        _ui.kpi(key, txt, sub)

st.write("")
hist = h[(h["DRS"] == d) & (h["PERIODE"] <= cutoff)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["PERIODE"].tail(24), y=hist[S.TARGET].tail(24), name="Historique (connu)",
                         line=dict(color=_ui.GREEN, width=2.2), mode="lines+markers", marker=dict(size=5)))
fig.add_trace(go.Scatter(x=bt["PERIODE"], y=bt["observe"], name="Observé (caché au modèle)",
                         line=dict(color=_ui.GREEN, width=2.2, dash="dot"), mode="lines+markers",
                         marker=dict(size=7, symbol="circle-open", line_width=2)))
fig.add_trace(go.Scatter(x=bt["PERIODE"], y=bt["prevu"], name="Prévu (récursif)",
                         line=dict(color=_ui.OCHRE, width=2.6, dash="dash"), mode="lines+markers",
                         marker=dict(size=7, symbol="diamond")))
fig.add_vline(x=cutoff + pd.Timedelta(days=15), line_dash="dot", line_color=_ui.SLATE)
fig.add_annotation(x=cutoff + pd.Timedelta(days=15), y=1, yref="paper", text="coupure", showarrow=False,
                   xanchor="left", font=dict(color=_ui.SLATE, size=11))
fig.update_layout(height=430, title=f"{S.short_name(d)} — rétro‑prévision à partir de {S.label_period(cutoff)}",
                  yaxis_title="Cas confirmés (TDR)")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Erreur par horizon**")
    err = bt.assign(h=range(1, len(bt) + 1), erreur=bt["prevu"] - bt["observe"])
    fig2 = go.Figure(go.Bar(x=err["h"], y=err["erreur"], marker_color=np.where(err["erreur"] >= 0, _ui.OCHRE, _ui.BLUE),
                            hovertemplate="h+%{x}<br>%{y:+,.0f} cas<extra></extra>"))
    fig2.update_layout(height=300, xaxis_title="Mois après la coupure", yaxis_title="prévu − observé")
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    st.markdown("**Détail**")
    show = bt.copy()
    show["PERIODE"] = show["PERIODE"].map(S.label_period)
    show["écart"] = show["prevu"] - show["observe"]
    show["écart %"] = np.where(show["observe"] > 0, show["écart"] / show["observe"] * 100, np.nan)
    st.dataframe(show.style.format({"observe": "{:,.0f}", "prevu": "{:,.0f}", "écart": "{:+,.0f}", "écart %": "{:+.0f} %"}),
                 use_container_width=True, hide_index=True, height=300)

st.divider()
st.markdown("### Vue d'ensemble : toutes les régions")
st.caption("Même coupure et même horizon, appliqués à chaque DRS.")
if st.button("Calculer pour les 14 régions", key="ev_all"):
    rows = []
    prog = st.progress(0.0)
    for i, dd in enumerate(drs_all):
        b = S.backtest(dd, cutoff, horizon)
        mm = S.metrics(b["observe"], b["prevu"])
        rows.append({"Région": S.short_name(dd), "Observé": b["observe"].sum(), "Prévu": b["prevu"].sum(),
                     **{k: v for k, v in mm.items()}})
        prog.progress((i + 1) / len(drs_all))
    prog.empty()
    st.session_state["ev_all_df"] = pd.DataFrame(rows)

df_all = st.session_state.get("ev_all_df")
if df_all is not None:
    df_all = df_all.copy()
    df_all["Écart total %"] = (df_all["Prévu"] - df_all["Observé"]) / df_all["Observé"].replace(0, np.nan) * 100
    st.dataframe(
        df_all.style.format({"Observé": "{:,.0f}", "Prévu": "{:,.0f}", "MAE": "{:,.1f}", "RMSE": "{:,.1f}",
                             "MAPE (%)": "{:,.0f}", "R²": "{:.2f}", "Biais": "{:+,.1f}", "Écart total %": "{:+.0f} %"})
        .background_gradient(subset=["R²"], cmap="RdYlGn", vmin=-0.5, vmax=1),
        use_container_width=True, hide_index=True,
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df_all["Région"], y=df_all["Observé"], name="Observé", marker_color=_ui.GREEN))
    fig3.add_trace(go.Bar(x=df_all["Région"], y=df_all["Prévu"], name="Prévu", marker_color=_ui.OCHRE))
    fig3.update_layout(barmode="group", height=360, yaxis_title="Cas cumulés sur l'horizon")
    st.plotly_chart(fig3, use_container_width=True)
    st.download_button("⬇️ Télécharger les métriques (CSV)", df_all.to_csv(index=False).encode(), "evaluation_drs.csv")
