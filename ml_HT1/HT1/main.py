import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

model = joblib.load('materials/mod_log_cat_v1.pkl')
ohe = joblib.load('materials/OHE_v1.pkl')
model_feats = model.feature_names_in_
ohe_feats = ohe.feature_names_in_

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float
    max_rpm: float

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    vals_dict = item.__dict__
    vals = pd.Series(item.__dict__) 
    ohe_vals = [int(x[0]) for x in ohe.transform(np.array(vals[ohe_feats]).reshape(1,-1)).toarray().reshape(-1,1)]
    cat_vals = dict(zip(ohe.get_feature_names_out(), ohe_vals))

    vals_dict.update(cat_vals)
    vals = pd.Series(vals_dict)

    features = np.array(vals[model_feats]).reshape(1,-1)
    log_prediction = model.predict(features)[0]

    return np.exp(log_prediction)
    # return vals

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    results = []
    for item in items:
        results.append(predict_item(item))

    return results