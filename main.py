from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from model_train.base import model
from model_train.base.data import process_data
import pandas as pd
import pickle

class InputData(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias='capital-gain')
    capital_loss: float = Field(alias='capital-loss')
    hours_per_week: float = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


app = FastAPI()

@app.get("/")
async def welcome():
    return {"message": "This is a FAST API GBM deploymenet. Use this app to predict income."}

@app.post("/predict")
async def predict(data: InputData):
    input = jsonable_encoder(data)
    df = pd.DataFrame(input, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]

    with open('../model/model.pkl', 'rb') as f:
        output_model = pickle.load(f)
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('../model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)

    X_test, y_test, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)

    predictions = model.inference(output_model, X_test)
    return str(predictions[0])