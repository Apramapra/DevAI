import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = FastAPI()

# Path to save/load the model
MODEL_PATH = "model.joblib"

# Pydantic model for optional training parameters
class TrainParams(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = None
    random_state: int = 42

# Pydantic model for prediction (optional)
class PredictRequest(BaseModel):
    features: list[float]  # length 4 for Iris

@app.get("/")
def read_root():
    return {"message": "ML training API is running"}

@app.post("/train")
def train_model(params: TrainParams = TrainParams()):
    """Train a RandomForest on the Iris dataset and save the model."""
    try:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=params.random_state
        )

        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Save model
        joblib.dump(model, MODEL_PATH)

        return {
            "status": "success",
            "accuracy": round(acc, 4),
            "model_saved": MODEL_PATH,
            "parameters": params.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
def model_info():
    """Check if a trained model exists."""
    if os.path.exists(MODEL_PATH):
        return {"model_exists": True, "path": MODEL_PATH}
    return {"model_exists": False}

@app.post("/predict")
def predict(request: PredictRequest):
    """Make a prediction using the latest trained model."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="No trained model found. Train first via /train.")
    model = joblib.load(MODEL_PATH)
    features = np.array(request.features).reshape(1, -1)
    if features.shape[1] != 4:
        raise HTTPException(status_code=400, detail="Iris model expects 4 features.")
    pred = int(model.predict(features)[0])
    class_names = load_iris().target_names
    return {"prediction": pred, "class_name": class_names[pred]}
