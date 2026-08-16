"""
Point d'entrée de l'interface.

Lancer :  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Paludisme · Prévisions régionales",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

from views import _ui  # noqa: E402  (après set_page_config)

_ui.inject_css()

pages = {
    "Tableau de bord": [
        st.Page("views/accueil.py", title="Vue d'ensemble", icon="🏠", default=True),
        st.Page("views/exploration.py", title="Données historiques", icon="📊"),
    ],
    "Prévision": [
        st.Page("views/prevision.py", title="Prévoir les cas", icon="🔮"),
        st.Page("views/scenarios.py", title="Scénarios climatiques", icon="🌦️"),
    ],
    "Modèle": [
        st.Page("views/evaluation.py", title="Évaluation", icon="🎯"),
        st.Page("views/modele.py", title="Pipeline & explicabilité", icon="🧬"),
    ],
}

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown(
        "<div class='sb-foot'>Modèle : LightGBM · 14 DRS · données 2021‑2024<br>"
        "Cible : cas confirmés (TDR) — consultations externes</div>",
        unsafe_allow_html=True,
    )

nav.run()
