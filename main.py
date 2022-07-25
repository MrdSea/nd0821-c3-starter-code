from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from model_train.base import model
from model_train.base.data import process_data
import pandas as pd
import pickle

class InputData(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }


app = FastAPI()

@app.get("/")
async def welcome():
    return {"message": "This is a FAST API GBM deploymenet. Use this app to predict income."}

@app.post("/predict")
async def predict(data: InputData):
    ##input = jsonable_encoder(data)
    ##df = pd.DataFrame(input, index=[0])
    input = {k: v for k, v in data.dict().items()}
    df = pd.DataFrame(input, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"]

    with open('../model/model.pkl', 'rb') as f:
        output_model = pickle.load(f)
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('../model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)

    X_test, y_test, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)

    predictions = model.inference(output_model, X_test)
    return predictions