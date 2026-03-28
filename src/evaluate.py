import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

model = joblib.load("model.pkl")

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print("Model accuracy:", acc)

# CI will fail if model underperforms
if acc < 0.75:
    raise Exception("Accuracy below acceptable threshold")