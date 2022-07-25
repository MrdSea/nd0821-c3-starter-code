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

    response = client.post("/predict",json=example)

    assert response.status_code == 200
    assert response.json() == '0'

def test_post_yes(client):
    example = {"age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"}

    response = client.post("/predict",json=example)

    assert response.status_code == 200
    assert response.json() == '1'