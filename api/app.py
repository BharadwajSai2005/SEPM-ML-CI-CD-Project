from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Heart disease model running"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"heart_disease": int(prediction[0])}