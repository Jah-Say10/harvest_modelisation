import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import services as S
from views import _ui

_ui.page_head("Prévoir les cas de paludisme", "Projection mensuelle par DRS avec le modèle LightGBM entraîné")

h = S.get_history()
drs_all = S.get_drs_list()
tmin, tmax = S.history_range()
default_start = (tmax + pd.offsets.MonthBegin(1)).to_pydatetime()

# ================================================================== Paramètres
with st.container(border=True):
    st.markdown("#### 1 · Périmètre")
    c1, c2, c3 = st.columns([2, 1.2, 1])
    with c1:
        mode = st.radio("Portée", ["Une région", "Plusieurs régions", "Toutes les régions"],
                        horizontal=True, key="pv_mode")
        if mode == "Une région":
            drs_sel = [st.selectbox("Région", drs_all, format_func=S.short_name, key="pv_drs1")]
        elif mode == "Plusieurs régions":
            drs_sel = st.multiselect("Régions", drs_all, format_func=S.short_name,
                                     default=["DRS Kolda", "DRS Kedougou", "DRS Tambacounda"], key="pv_drsn")
        else:
            drs_sel = drs_all
    with c2:
        start = st.date_input("Premier mois prévu", value=default_start,
                              min_value=(tmin + pd.DateOffset(months=12)).to_pydatetime(),
                              help="Le modèle a besoin d'au moins 12 mois d'historique avant ce mois. "
                                   "Choisir une date à l'intérieur de l'historique permet une rétro‑prévision.")
        start = pd.Timestamp(start).to_period("M").to_timestamp()
    with c3:
        horizon = st.slider("Horizon (mois)", 1, 24, 12, key="pv_h")
    dates = S.month_range(start, horizon)

    st.markdown("#### 2 · Hypothèse météo pour la période prévue")
    wsrc = st.radio(
        "Source", ["Climatologie historique", "Année de référence", "Importer un fichier", "Saisie manuelle"],
        horizontal=True, key="pv_wsrc",
        captions=["Moyenne mensuelle 2021‑2024 de chaque région",
                  "Reprendre la météo observée d'une année passée",
                  "CSV / Excel : PERIODE + 7 variables (+ DRS optionnel)",
                  "Éditer les valeurs à la main (une région)"],
    )
    ref_year = None
    uploaded = None
    manual_df = None
    if wsrc == "Année de référence":
        ref_year = st.selectbox("Année à rejouer", sorted(h["PERIODE"].dt.year.unique(), reverse=True), key="pv_ref")
    elif wsrc == "Importer un fichier":
        uploaded = st.file_uploader("Fichier météo", type=["csv", "xlsx"], key="pv_up")
        with st.expander("Format attendu / modèle à télécharger"):
            tpl = S.weather_from_climatology(drs_sel[0], dates)
            tpl.insert(0, "DRS", drs_sel[0])
            st.dataframe(tpl.head(3), hide_index=True, use_container_width=True)
            st.download_button("⬇️ Télécharger un modèle pré‑rempli (climatologie)",
                               pd.concat([S.weather_from_climatology(d, dates).assign(DRS=d) for d in drs_sel])
                               [["DRS", "PERIODE"] + S.WEATHER_VARS].to_csv(index=False).encode(),
                               "modele_meteo.csv", "text/csv")
    elif wsrc == "Saisie manuelle":
        if len(drs_sel) != 1:
            st.warning("La saisie manuelle n'est disponible que pour une seule région.")
        else:
            base = S.weather_from_climatology(drs_sel[0], dates)
            base["PERIODE"] = base["PERIODE"].dt.strftime("%Y-%m")
            st.caption("Valeurs pré‑remplies avec la climatologie ; modifiez librement.")
            manual_df = st.data_editor(base.rename(columns=S.WEATHER_LABELS), hide_index=True,
                                       use_container_width=True, key="pv_editor", disabled=["PERIODE"])
            manual_df = manual_df.rename(columns={v: k for k, v in S.WEATHER_LABELS.items()})
            manual_df["PERIODE"] = pd.to_datetime(manual_df["PERIODE"])

    run = st.button("🔮 Lancer la prévision", type="primary", use_container_width=True, disabled=not drs_sel)


# ================================================================== Exécution
def _weather_fn(d):
    if wsrc == "Climatologie historique":
        return S.weather_from_climatology(d, dates)
    if wsrc == "Année de référence":
        return S.weather_from_reference_year(d, dates, ref_year)
    if wsrc == "Importer un fichier":
        if uploaded is None:
            raise ValueError("Aucun fichier importé.")
        uploaded.seek(0)
        return S.parse_weather_csv(uploaded, drs=d)
    return manual_df


if run:
    try:
        with st.spinner("Prévision récursive en cours…"):
            fc = S.forecast_many(drs_sel, dates, _weather_fn)
        st.session_state["pv_result"] = {"fc": fc, "drs": drs_sel, "dates": dates,
                                         "wsrc": wsrc, "ref_year": ref_year}
    except Exception as e:  # noqa: BLE001
        st.error(f"Impossible de lancer la prévision : {e}")

res = st.session_state.get("pv_result")
if not res:
    st.info("Configurez la prévision ci‑dessus puis cliquez sur **Lancer la prévision**.", icon="👆")
    st.stop()

fc, drs_sel, dates = res["fc"], res["drs"], res["dates"]
fc["Région"] = fc["DRS"].map(S.short_name)
src_txt = res["wsrc"] + (f" ({res['ref_year']})" if res["ref_year"] else "")

st.markdown("---")
st.markdown(f"### Résultats &nbsp;<span class='badge'>{len(drs_sel)} région(s)</span>"
            f"<span class='badge'>{S.label_period(dates[0])} → {S.label_period(dates[-1])}</span>"
            f"<span class='badge o'>Météo : {src_txt}</span>", unsafe_allow_html=True)

# ---- KPIs
total = fc["prediction"].sum()
by_month = fc.groupby("PERIODE")["prediction"].sum()
peak_m = by_month.idxmax()
same_period_prev = h[(h["DRS"].isin(drs_sel)) & (h["PERIODE"].isin(dates - pd.DateOffset(years=1)))][S.TARGET].sum()
k1, k2, k3, k4 = st.columns(4)
with k1:
    _ui.kpi("Cas prévus (total)", _ui.fmt_int(total), f"sur {len(dates)} mois", accent=True)
with k2:
    _ui.kpi("Mois de pic", S.label_period(peak_m), f"{_ui.fmt_int(by_month.max())} cas")
with k3:
    if same_period_prev > 0:
        _ui.kpi("Vs même période un an avant", f"{(total - same_period_prev) / same_period_prev * 100:+.0f} %",
                f"{_ui.fmt_int(same_period_prev)} cas observés")
    else:
        _ui.kpi("Vs même période un an avant", "—", "historique indisponible")
with k4:
    top = fc.groupby("DRS")["prediction"].sum().idxmax()
    _ui.kpi("Région la plus touchée", S.short_name(top), f"{_ui.fmt_int(fc[fc['DRS']==top]['prediction'].sum())} cas")

st.write("")

# ---- Graphiques
if len(drs_sel) == 1:
    d = drs_sel[0]
    hist = h[h["DRS"] == d]
    hist = hist[hist["PERIODE"] < dates[0]]
    fig = _ui.history_forecast_chart(hist, fc, S.TARGET, title=f"{S.short_name(d)} — historique et prévision")
    # Si des observations existent sur la période prévue (rétro‑prévision), on les affiche
    obs = h[(h["DRS"] == d) & (h["PERIODE"].isin(dates))]
    if len(obs):
        fig.add_trace(go.Scatter(x=obs["PERIODE"], y=obs[S.TARGET], name="Observé (période prévue)",
                                 mode="markers", marker=dict(color=_ui.GREEN, size=8, symbol="circle-open", line_width=2)))
    st.plotly_chart(fig, use_container_width=True)
else:
    t1, t2, t3 = st.tabs(["Courbes", "Comparaison des régions", "Carte"])
    with t1:
        fig = px.line(fc, x="PERIODE", y="prediction", color="Région", markers=True,
                      labels={"prediction": "Cas prévus", "PERIODE": ""})
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        tot_r = fc.groupby("Région")["prediction"].sum().sort_values()
        fig = go.Figure(go.Bar(x=tot_r.values, y=tot_r.index, orientation="h", marker_color=_ui.OCHRE,
                               text=[_ui.fmt_int(v) for v in tot_r.values], textposition="outside"))
        fig.update_layout(height=max(320, 28 * len(tot_r) + 80), xaxis_title="Cas prévus sur la période",
                          margin=dict(l=10, r=60, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        agg = fc.groupby("DRS")["prediction"].sum().reset_index()
        agg["Région"] = agg["DRS"].map(S.short_name)
        agg["lat"] = agg["DRS"].map(lambda x: S.DRS_COORDS[x][0])
        agg["lon"] = agg["DRS"].map(lambda x: S.DRS_COORDS[x][1])
        fig = px.scatter_geo(agg, lat="lat", lon="lon", size="prediction", color="prediction", hover_name="Région",
                             hover_data={"prediction": ":,.0f", "lat": False, "lon": False}, size_max=48,
                             color_continuous_scale=["#F7E9D3", "#D28A2C", "#B7412E"], projection="mercator")
        fig.update_geos(lataxis_range=[12.0, 16.9], lonaxis_range=[-17.9, -11.2], showcountries=True,
                        countrycolor="#B8C4BE", showland=True, landcolor="#F7F9F8", showocean=True,
                        oceancolor="#EAF2F5", showframe=False)
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="Cas", thickness=12))
        st.plotly_chart(fig, use_container_width=True)

# ---- Table + export
st.markdown("#### Détail mensuel")
table = fc.pivot_table(index="PERIODE", columns="Région", values="prediction").round(0)
table.index = [S.label_period(t) for t in table.index]
if len(drs_sel) > 1:
    table["Total"] = table.sum(axis=1)
st.dataframe(table.style.format("{:,.0f}").background_gradient(cmap="YlOrBr", axis=None), use_container_width=True)

export = fc[["DRS", "PERIODE", "prediction"]].copy()
export["PERIODE"] = export["PERIODE"].dt.strftime("%Y-%m-%d")
export["prediction"] = export["prediction"].round(1)
sheets = {"previsions": export}
weather_used = pd.concat([_weather_fn(d).assign(DRS=d) for d in drs_sel]) if res["wsrc"] != "Importer un fichier" else None
if weather_used is not None:
    weather_used["PERIODE"] = pd.to_datetime(weather_used["PERIODE"]).dt.strftime("%Y-%m-%d")
    sheets["meteo_utilisee"] = weather_used[["DRS", "PERIODE"] + S.WEATHER_VARS]
_ui.download_buttons(export, f"previsions_{dates[0]:%Y%m}_{dates[-1]:%Y%m}", S.to_excel_bytes(sheets))

with st.expander("Comment lire ces résultats ?"):
    st.markdown(
        "- La prévision est **récursive** : la valeur prévue du mois M sert de lag pour le mois M+1. "
        "L'incertitude croît donc avec l'horizon ; au‑delà de 12 mois, considérez les valeurs comme indicatives.\n"
        "- Les cas prévus dépendent fortement de l'**hypothèse météo**. Comparez plusieurs hypothèses "
        "dans la page *Scénarios climatiques*.\n"
        "- Les valeurs sont bornées à 0 et non arrondies dans l'export."
    )
