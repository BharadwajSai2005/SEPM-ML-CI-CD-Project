from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

dataset = fetch_ucirepo(id=45)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("data/heart.csv", index=False)

print("Heart disease dataset downloaded")