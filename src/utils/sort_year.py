import pandas as pd

def sort_data(aggregated_drs):
    # French month mapping
    mois_fr = {
        "Janvier": 1,
        "Février": 2,
        "Mars": 3,
        "Avril": 4,
        "Mai": 5,
        "Juin": 6,
        "Juillet": 7,
        "Août": 8,
        "Septembre": 9,
        "Octobre": 10,
        "Novembre": 11,
        "Décembre": 12
    }

    # Split month/year
    aggregated_drs[["MOIS", "ANNEE"]] = aggregated_drs["PERIODE"].str.split(" ", expand=True)

    # Convert to numeric month
    aggregated_drs["MOIS_NUM"] = aggregated_drs["MOIS"].map(mois_fr)

    # Create sortable datetime column
    aggregated_drs["DATE"] = pd.to_datetime({
        "year": aggregated_drs["ANNEE"].astype(int),
        "month": aggregated_drs["MOIS_NUM"],
        "day": 1
    })

    # Sort
    aggregated_drs = aggregated_drs.sort_values("DATE")

    # Optional: remove helper columns
    aggregated_drs = aggregated_drs.drop(columns=["MOIS", "ANNEE", "MOIS_NUM", "DATE"])
    
    return aggregated_drs

# ==============================
# 
# =============================

def sort_and_replace(data):
    # Convert French month names to English
    month_map = {
        "Janvier": "January",
        "Février": "February",
        "Mars": "March",
        "Avril": "April",
        "Mai": "May",
        "Juin": "June",
        "Juillet": "July",
        "Août": "August",
        "Septembre": "September",
        "Octobre": "October",
        "Novembre": "November",
        "Décembre": "December"
    }

    data["PERIODE"] = data["PERIODE"].replace(month_map, regex=True)
    data["PERIODE"] = pd.to_datetime(data["PERIODE"], format="%B %Y")

    data_sort = data.sort_values("PERIODE")
    data_sort = data_sort.set_index("PERIODE")
    
    return data_sort