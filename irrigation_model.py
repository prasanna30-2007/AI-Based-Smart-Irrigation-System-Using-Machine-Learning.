
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/irrigation_data.csv"

FEATURES = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall_mm",
    "crop_type",
]

def train_model():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["irrigation_needed"]

    preprocessing = ColumnTransformer([
        ("crop", OneHotEncoder(handle_unknown="ignore"), ["crop_type"])
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model

def predict_irrigation(
    model, soil_moisture, temperature,
    humidity, rainfall_mm, crop_type
):
    sample = pd.DataFrame([{
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall_mm": rainfall_mm,
        "crop_type": crop_type,
    }])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    return prediction, float(max(probabilities))
  
