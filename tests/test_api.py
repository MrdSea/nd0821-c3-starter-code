import sys
sys.path.append("..")
import pytest
from fastapi.testclient import TestClient
from main import app



@pytest.fixture()
def client():
    client = TestClient(app)
    return client

def test_get(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "This is a FAST API GBM deploymenet. Use this app to predict income."}

def test_post_no(client):
    example = {
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

    response = client.post("/predict",json=example)

    assert response.status_code == 200
    assert response.json() == '{"prediction": [" <=50K"]}'

def test_post_yes(client):
    example = {"age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"}

    response = client.post("/predict",json=example)

    assert response.status_code == 200
    assert response.json() == '{"prediction": [" >50K"]}'