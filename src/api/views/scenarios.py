import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import services as S
from views import _ui

_ui.page_head("Scénarios climatiques", "Mesurer la sensibilité des prévisions aux conditions météo")

h = S.get_history()
drs_all = S.get_drs_list()
tmin, tmax = S.history_range()

with st.container(border=True):
    c1, c2, c3 = st.columns([1.5, 1.2, 1])
    with c1:
        d = st.selectbox("Région", drs_all, index=drs_all.index("DRS Kolda"), format_func=S.short_name, key="sc_drs")
    with c2:
        start = st.date_input("Premier mois", value=(tmax + pd.offsets.MonthBegin(1)).to_pydatetime(),
                              min_value=(tmin + pd.DateOffset(months=12)).to_pydatetime(), key="sc_start")
        start = pd.Timestamp(start).to_period("M").to_timestamp()
    with c3:
        horizon = st.slider("Horizon (mois)", 3, 24, 12, key="sc_h")
    dates = S.month_range(start, horizon)

    st.markdown("**Scénario de base** : climatologie de la région. Définissez jusqu'à deux scénarios alternatifs.")
    sc_cols = st.columns(2)
    adjustments = {}
    presets = {
        "Personnalisé": S.WeatherAdjust(),
        "Hivernage abondant (+30 % pluie, +5 pts humidité)": S.WeatherAdjust(precip_pct=30, humid_delta=5),
        "Sécheresse (−30 % pluie, −5 pts humidité, +1 °C)": S.WeatherAdjust(precip_pct=-30, humid_delta=-5, temp_delta=1),
        "Réchauffement (+2 °C)": S.WeatherAdjust(temp_delta=2),
        "Refroidissement (−1 °C, +10 % pluie)": S.WeatherAdjust(temp_delta=-1, precip_pct=10),
    }
    defaults = ["Hivernage abondant (+30 % pluie, +5 pts humidité)", "Sécheresse (−30 % pluie, −5 pts humidité, +1 °C)"]
    for i, col in enumerate(sc_cols):
        with col:
            st.markdown(f"**Scénario {chr(65 + i)}**")
            name = st.selectbox("Préréglage", list(presets), index=list(presets).index(defaults[i]), key=f"sc_preset{i}")
            base = presets[name]
            with st.expander("Ajuster finement", expanded=(name == "Personnalisé")):
                adj = S.WeatherAdjust(
                    precip_pct=st.slider("Précipitations (%)", -60, 80, int(base.precip_pct), 5, key=f"sc_p{i}"),
                    temp_delta=st.slider("Températures (°C)", -3.0, 4.0, float(base.temp_delta), 0.5, key=f"sc_t{i}"),
                    humid_delta=st.slider("Humidité (points)", -15, 15, int(base.humid_delta), 1, key=f"sc_hu{i}"),
                    wind_pct=st.slider("Vent (%)", -30, 30, int(base.wind_pct), 5, key=f"sc_w{i}"),
                    solar_pct=st.slider("Rayonnement (%)", -20, 20, int(base.solar_pct), 5, key=f"sc_s{i}"),
                )
            adjustments[f"Scénario {chr(65 + i)} — {name.split(' (')[0]}"] = adj

    run = st.button("🌦️ Comparer les scénarios", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Calcul des scénarios…"):
            base_w = S.weather_from_climatology(d, dates)
            results = {"Base (climatologie)": S.forecast_one(d, dates, base_w)}
            for name, adj in adjustments.items():
                results[name] = S.forecast_one(d, dates, adj.apply(base_w))
        st.session_state["sc_result"] = {"drs": d, "dates": dates, "results": results}
    except Exception as e:  # noqa: BLE001
        st.error(f"Erreur : {e}")

res = st.session_state.get("sc_result")
if not res:
    st.info("Choisissez vos scénarios puis cliquez sur **Comparer les scénarios**.", icon="👆")
    st.stop()

d, dates, results = res["drs"], res["dates"], res["results"]
base_total = results["Base (climatologie)"]["prediction"].sum()

cols = st.columns(len(results))
for col, (name, df) in zip(cols, results.items()):
    tot = df["prediction"].sum()
    with col:
        if name.startswith("Base"):
            _ui.kpi(name, _ui.fmt_int(tot), "cas prévus sur la période", accent=True)
        else:
            _ui.kpi(name, _ui.fmt_int(tot), f"{(tot - base_total) / base_total * 100:+.1f} % vs base" if base_total else "")

st.write("")
hist = h[(h["DRS"] == d) & (h["PERIODE"] < dates[0])]
extra = {k: v for k, v in results.items() if not k.startswith("Base")}
fig = _ui.history_forecast_chart(hist, results["Base (climatologie)"], S.TARGET,
                                 title=f"{S.short_name(d)} — scénarios", show_last_months=24, extra=extra)
fig.data[2].name = "Base (climatologie)"  # trace prévision
st.plotly_chart(fig, use_container_width=True)

# Écart mensuel vs base
st.markdown("#### Écart mensuel par rapport au scénario de base")
fig2 = go.Figure()
base_pred = results["Base (climatologie)"].set_index("PERIODE")["prediction"]
for i, (name, df) in enumerate(extra.items()):
    diff = df.set_index("PERIODE")["prediction"] - base_pred
    fig2.add_trace(go.Bar(x=diff.index, y=diff.values, name=name,
                          marker_color=[_ui.BLUE, _ui.RED, "#6B4E9B"][i % 3],
                          hovertemplate="%{x|%b %Y}<br>%{y:+,.0f} cas<extra></extra>"))
fig2.update_layout(barmode="group", height=320, yaxis_title="Δ cas vs base")
st.plotly_chart(fig2, use_container_width=True)

table = pd.DataFrame({name: df.set_index("PERIODE")["prediction"].round(0) for name, df in results.items()})
table.index = [S.label_period(t) for t in table.index]
st.dataframe(table.style.format("{:,.0f}"), use_container_width=True)

export = pd.concat([df.assign(scenario=name) for name, df in results.items()])
export["PERIODE"] = export["PERIODE"].dt.strftime("%Y-%m-%d")
_ui.download_buttons(export, f"scenarios_{S.short_name(d)}", S.to_excel_bytes({"scenarios": export}))

st.caption("Note : le modèle capture les relations météo‑cas apprises sur 2021‑2024. Les scénarios très éloignés "
           "des conditions observées (ex. +4 °C) extrapolent hors du domaine d'entraînement.")
