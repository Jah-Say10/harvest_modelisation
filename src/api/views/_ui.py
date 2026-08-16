"""Éléments d'interface partagés : CSS, cartes KPI, thème Plotly, en-têtes."""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Palette
GREEN = "#0F6E56"       # historique / observé
OCHRE = "#D28A2C"       # prévision
OCHRE_LIGHT = "rgba(210,138,44,0.14)"
SLATE = "#5B6B66"
RED = "#B7412E"
BLUE = "#2F6690"
GRID = "#E6ECE9"

REGION_COLORS = [
    "#0F6E56", "#D28A2C", "#2F6690", "#B7412E", "#6B4E9B", "#3A9C7A",
    "#C25A8E", "#8A7A2C", "#1F8FA8", "#7A5230", "#4C6C3B", "#A23B7C",
    "#5B6B66", "#D6A03C",
]

# Template Plotly commun
_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#1B2A24"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12),
    colorway=REGION_COLORS,
)
pio.templates["palu"] = _tpl
pio.templates.default = "palu"


def inject_css():
    st.markdown(
        """
<style>
/* Titres */
h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 650 !important; letter-spacing: -0.01em; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

/* Bandeau de page */
.page-head { border-left: 5px solid #0F6E56; padding: .2rem 0 .2rem 1rem; margin-bottom: 1.2rem; }
.page-head h1 { margin: 0; font-size: 1.9rem; }
.page-head p  { margin: .25rem 0 0; color: #5B6B66; font-size: 1rem; }

/* Cartes KPI */
.kpi { background: #F3F6F4; border-radius: 12px; padding: .9rem 1.1rem; height: 100%;
       border: 1px solid #E6ECE9; }
.kpi .lab { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #5B6B66; }
.kpi .val { font-size: 1.7rem; font-weight: 700; color: #1B2A24; line-height: 1.15; margin-top: .15rem; }
.kpi .sub { font-size: .82rem; color: #5B6B66; margin-top: .2rem; }
.kpi.accent { background: #FBF3E7; border-color: #F0DFC2; }
.kpi.accent .val { color: #A66A16; }

/* Étapes du pipeline */
.step { display:flex; gap:.9rem; align-items:flex-start; padding:.75rem 0; border-bottom:1px dashed #E6ECE9; }
.step .n { flex:0 0 2rem; height:2rem; border-radius:50%; background:#0F6E56; color:white;
           display:flex; align-items:center; justify-content:center; font-weight:700; }
.step .t { font-weight:650; }
.step .d { color:#5B6B66; font-size:.92rem; }

/* Pied de barre latérale */
.sb-foot { position: fixed; bottom: 1rem; font-size: .75rem; color:#5B6B66; line-height:1.4; }

/* Badges */
.badge { display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.75rem;
         font-weight:600; background:#E8F1EE; color:#0F6E56; margin-right:.3rem; }
.badge.o { background:#FBF3E7; color:#A66A16; }
</style>
""",
        unsafe_allow_html=True,
    )


def page_head(title: str, subtitle: str = ""):
    st.markdown(
        f"<div class='page-head'><h1>{title}</h1>"
        + (f"<p>{subtitle}</p>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def kpi(label: str, value, sub: str = "", accent: bool = False):
    cls = "kpi accent" if accent else "kpi"
    st.markdown(
        f"<div class='{cls}'><div class='lab'>{label}</div>"
        f"<div class='val'>{value}</div>"
        + (f"<div class='sub'>{sub}</div>" if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", "\u202f")
    except (TypeError, ValueError):
        return "—"


def history_forecast_chart(hist: pd.DataFrame, fc: pd.DataFrame, target: str,
                           title: str = "", show_last_months: int = 36,
                           extra: dict | None = None) -> go.Figure:
    """Courbe historique (vert) + prévision (ocre pointillé) avec zone ombrée.
    extra : {nom: DataFrame(PERIODE, prediction)} pour superposer des scénarios."""
    hist = hist.sort_values("PERIODE").tail(show_last_months)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["PERIODE"], y=hist[target], name="Observé",
        mode="lines+markers", line=dict(color=GREEN, width=2.4), marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>Observé : %{y:,.0f}<extra></extra>",
    ))
    if fc is not None and len(fc):
        # relier le dernier point observé au premier prévu
        if len(hist):
            bridge_x = [hist["PERIODE"].iloc[-1], fc["PERIODE"].iloc[0]]
            bridge_y = [hist[target].iloc[-1], fc["prediction"].iloc[0]]
            fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines", showlegend=False,
                                     line=dict(color=OCHRE, width=2, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=fc["PERIODE"], y=fc["prediction"], name="Prévision",
            mode="lines+markers", line=dict(color=OCHRE, width=2.6, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="%{x|%b %Y}<br>Prévu : %{y:,.0f}<extra></extra>",
        ))
        fig.add_vrect(x0=fc["PERIODE"].iloc[0] - pd.Timedelta(days=15),
                      x1=fc["PERIODE"].iloc[-1] + pd.Timedelta(days=15),
                      fillcolor=OCHRE_LIGHT, line_width=0, layer="below")
    if extra:
        for i, (name, df) in enumerate(extra.items()):
            fig.add_trace(go.Scatter(
                x=df["PERIODE"], y=df["prediction"], name=name, mode="lines+markers",
                line=dict(width=2.2, color=[BLUE, RED, "#6B4E9B"][i % 3]),
                hovertemplate="%{x|%b %Y}<br>" + name + " : %{y:,.0f}<extra></extra>",
            ))
    fig.update_layout(title=title, yaxis_title="Cas confirmés (TDR)", height=430)
    return fig


def download_buttons(df: pd.DataFrame, base_name: str, excel_bytes: bytes | None = None):
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{base_name}.csv", mime="text/csv", use_container_width=True)
    if excel_bytes is not None:
        with c2:
            st.download_button("⬇️ Excel", excel_bytes, file_name=f"{base_name}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
