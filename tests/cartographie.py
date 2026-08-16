import pandas as pd
import folium

# Charger les données
df = pd.read_csv("departements_senegal.csv")

# Carte centrée sur le Sénégal
m = folium.Map(
    location=[14.5, -14.5],
    zoom_start=7,
    tiles="OpenStreetMap"
)

# Une couleur par région
couleurs = {
    "Dakar": "red",
    "Diourbel": "blue",
    "Fatick": "green",
    "Kaffrine": "purple",
    "Kaolack": "orange",
    "Kedougou": "darkred",
    "Kolda": "darkblue",
    "Louga": "cadetblue",
    "Matam": "pink",
    "Saint-Louis": "gray",
    "Sedhiou": "lightgreen",
    "Tambacounda": "beige",
    "Thies": "darkgreen",
    "Ziguinchor": "darkpurple"
}

# Ajouter les districts
for _, row in df.iterrows():

    region = row["region"]
    district = row["departement"]

    popup = f"""
    <div style="width:300px">
        <h4>{district}</h4>
        <b>Région :</b> {region}<br>
        <b>Département :</b> {district}<br>
        <b>Latitude :</b> {row["latitude"]}<br>
        <b>Longitude :</b> {row["longitude"]}<br>
        <b>Adresse :</b> {row["adresse"]}
    </div>
    """

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup, max_width=350),
        tooltip=f"{district} — {region}",
        icon=folium.Icon(
            color=couleurs.get(region, "blue"),
            icon="plus-sign"
        )
    ).add_to(m)

# Sauvegarder
m.save("carte_districts_sanitaires_senegal.html")

print("Carte créée : carte_districts_sanitaires_senegal.html")