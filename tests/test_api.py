from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_hello(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}


def test_predict_higher(client):
    data = {"age": 33,
            "workclass": "Private",
            "fnlwgt": 149184,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States"
            }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 1,
                               "prediction_label": ">50K"}


def test_predict_lower(client):
    data = {"age": 19,
            "workclass": "Private",
            "fnlwgt": 149184,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States"
            }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 0,
                               "prediction_label": "<=50K"}


def test_predict_wrong_body(client):
    data = {"age": 19,
            "workclass": "Private",
            "fnlwgt": 149184,
            }
    response = client.post("/predict", json=data)
    assert response.status_code == 422
