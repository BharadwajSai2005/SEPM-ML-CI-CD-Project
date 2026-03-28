import joblib
import numpy as np

def test_model_load():
    model = joblib.load("model.pkl")
    assert model is not None

def test_prediction():
    model = joblib.load("model.pkl")
    sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
    pred = model.predict(sample)
    assert pred in [[0], [1]]