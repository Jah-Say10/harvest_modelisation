import pandas as pd
import json

# Read data in ../data/raw/Base_PEC_MALARIA_CS_2021_2024.xlsx
data = pd.read_excel('../data/processed/Base_PEC_MALARIA_CS_2021_2024.xlsx')

# Group districts by DRS
drs_districts = (
    data.groupby("DRS")["DISTRICT"]
      .unique()
      .apply(list)
      .to_dict()
)

# Convert to JSON
drs_districts = json.dumps(drs_districts, indent=4, ensure_ascii=False)