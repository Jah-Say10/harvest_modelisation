import json
from pathlib import Path
from datetime import datetime


import json
from pathlib import Path
from datetime import datetime


def save_metadata(
    district,
    algorithm,
    rmse,
    training_rows,
    version,
    best_params,
    features,
    output_dir="../models/metadata",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metadata = {
        "district": district,
        "algorithm": algorithm,
        "rmse": round(float(rmse), 4),
        "trained_at": datetime.now().strftime("%Y-%m-%d"),
        "training_rows": training_rows,
        "version": version,
        "best_params": best_params,
        "features": features,
    }

    filename = f"{district}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(Path(output_dir) / filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata

# ==============================
# 
# =============================

def save_metadata_batch(
    type,
    metadata_list,
    filename=None,
    output_dir="../models/metadata",
):
    """
    Save metadata for multiple models into a single JSON file.

    Parameters
    ----------
    metadata_list : list[dict]
        List of metadata dictionaries.
    filename : str, optional
        Output filename. If None, a timestamped filename is generated.
    output_dir : str
        Directory where the file will be saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{type}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(Path(output_dir) / filename, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4)

    return metadata_list