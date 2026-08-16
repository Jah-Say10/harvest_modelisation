That structure is designed to separate your project into clear responsibilities so it stays manageable as you scale from a few models to potentially dozens or hundreds. Here's what each part is for and why it's useful.

```text
senegal_prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── registry/
│   └── metadata/
│
├── src/
│   ├── training/
│   │   ├── train_global.py
│   │   ├── train_region.py
│   │   └── train_district.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── batch_predict.py
│   │
│   ├── api/
│   │   └── app.py
│   │
│   └── utils/
│
├── configs/
│   ├── global.yaml
│   └── regions/
│
├── notebooks/
├── tests/
├── requirements.txt
└── docker-compose.yml
```

---

# Root Folder (`senegal_prediction/`)

This is your entire machine learning project.

Everything related to the prediction system lives here:

* datasets
* models
* training code
* prediction code
* API
* configuration
* tests

This makes the project easy to share and deploy.

---

# `data/`

This folder stores all datasets.

You should **never mix raw data with cleaned data**.

## `raw/`

Original datasets exactly as received.

Example:

```
raw/

    dakar.csv

    thies.csv

    kaolack.csv
```

These files should never be modified.

If something goes wrong during preprocessing, you still have the originals.

---

## `processed/`

After cleaning:

* removing missing values
* fixing dates
* encoding categories
* removing duplicates

save the cleaned datasets here.

Example:

```
processed/

    dakar_clean.csv

    thies_clean.csv
```

Training uses these datasets instead of the raw ones.

---

## `features/`

Sometimes the cleaned data is still not ready.

You may create new variables (feature engineering).

Example:

Original:

```
Rain
Temperature
Humidity
```

New features:

```
Rain_last_7_days
Average_temperature
Rainfall_index
```

These engineered datasets are stored here.

---

# `models/`

Contains everything related to trained models.

---

## `registry/`

This is where the actual trained models are saved.

Example:

```
registry/

    dakar.pkl

    thies.pkl

    kedougou.pkl

    tambacounda.pkl
```

or

```
registry/

    dakar.joblib

    thies.joblib
```

These are the files your application loads when making predictions.

---

## `metadata/`

Metadata means **information about the models**, not the models themselves.

Example JSON:

```json
{
    "district": "Thies",
    "algorithm": "XGBoost",
    "accuracy": 0.94,
    "trained_at": "2026-07-20",
    "training_rows": 45231,
    "version": "1.0.3"
}
```

This helps you answer questions like:

* Which algorithm trained this model?
* When was it trained?
* How accurate is it?
* Which version is currently deployed?

---

# `src/`

Contains all your Python source code.

Think of it as the engine of the project.

---

# `training/`

Everything related to building models.

---

## `train_global.py`

Trains **one model using data from all Senegal**.

Example:

```
All districts combined

↓

One large model
```

Useful for comparison or as a fallback if a district has little data.

---

## `train_region.py`

Trains one model for each region.

Example:

```
Dakar region

↓

Model

Saint-Louis region

↓

Model

Kaolack region

↓

Model
```

Since Senegal has 14 regions, this script could produce 14 models.

---

## `train_district.py`

Trains one model for every district.

Example:

```
District A

↓

Model

District B

↓

Model

District C

↓

Model
```

If there are 80 districts, this script produces 80 models automatically.

This script usually loops through all district datasets:

```python
for district in districts:
    train_model(district)
    save_model()
```

---

# `inference/`

Training builds models.

Inference uses them.

---

## `predictor.py`

Loads a model and predicts.

Example:

```python
predict(
    district="Thies",
    input_data=data
)
```

It:

* finds the correct model
* loads it
* returns predictions

Your app calls this file.

---

## `batch_predict.py`

Predicts for many records at once.

Instead of:

```
1 prediction
```

it handles

```
10,000 predictions
```

Useful for:

* monthly reports
* yearly forecasts
* scheduled jobs

---

# `api/`

This exposes the models to other applications.

---

## `app.py`

Usually a **FastAPI** or **Flask** application.

Example request:

```
POST /predict

{
    "district":"Thies",
    "temperature":31,
    "rain":12
}
```

The API:

1. receives the request
2. loads the Thies model
3. predicts
4. returns the result

Response:

```json
{
    "prediction": 152
}
```

This is what your web app or mobile app communicates with.

---

# `utils/`

Utility functions used throughout the project.

Examples:

```
save_model()

load_model()

read_config()

logging()

preprocessing()

evaluation_metrics()
```

Instead of rewriting the same code in multiple places, you keep shared functions here.

---

# `configs/`

Configuration files.

Instead of hardcoding values into Python, you keep them here.

---

## `global.yaml`

Settings that apply to the entire project.

Example:

```yaml
model: xgboost

random_state: 42

test_size: 0.2

save_models: true
```

Changing the YAML file changes the behavior without editing code.

---

## `regions/`

Region-specific settings.

Example:

```
regions/

    dakar.yaml

    thies.yaml

    kaolack.yaml
```

Maybe Dakar needs:

```yaml
max_depth: 8
```

while another region performs better with:

```yaml
max_depth: 5
```

Each region can have its own configuration.

---

# `notebooks/`

Jupyter notebooks used for experimentation.

Examples:

```
EDA.ipynb

Feature_Engineering.ipynb

Model_Comparison.ipynb
```

Use these for exploring data and testing ideas. Once the workflow is stable, move the reusable code into `src/`.

---

# `tests/`

Automated tests to make sure your code works correctly.

Examples:

* Does model loading work?
* Does preprocessing return expected columns?
* Does prediction return a valid value?
* Does the API respond correctly?

Example:

```python
def test_predict():
    prediction = predictor.predict(sample)
    assert prediction > 0
```

Testing helps catch bugs before deployment.

---

# `requirements.txt`

Lists all Python packages your project depends on.

Example:

```text
pandas
numpy
scikit-learn
xgboost
lightgbm
fastapi
uvicorn
joblib
```

Anyone can recreate the environment with:

```bash
pip install -r requirements.txt
```

---

# `docker-compose.yml`

Used to run the application and its supporting services in containers.

For example, it can start:

* your prediction API
* a database
* a caching service like Redis
* monitoring tools

This ensures the application runs consistently across development, testing, and production environments.

---

# How Everything Works Together

The overall workflow looks like this:

```text
Raw Data
    │
    ▼
data/raw
    │
    ▼
Cleaning & Preprocessing
    │
    ▼
data/processed
    │
    ▼
Feature Engineering
    │
    ▼
data/features
    │
    ▼
Training Scripts (src/training)
    │
    ▼
Trained Models
(models/registry)
    │
    ▼
Prediction Engine
(src/inference/predictor.py)
    │
    ▼
API (src/api/app.py)
    │
    ▼
Web or Mobile Application
```

This architecture follows common machine learning engineering practices. It separates data management, model training, model storage, prediction logic, and application code into distinct components. That separation makes the project easier to maintain, retrain, version, and scale as you add models for every district in Senegal.
