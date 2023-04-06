# Write a script that POSTS to the API using the requests module and returns both the result of model inference and the status code
# API endpoint: https://deployingmlprojectudacity.onrender.com/predict

import requests


ENDPOINT = "https://deployingmlprojectudacity.onrender.com/predict"


def predict(data):
    response = requests.post(ENDPOINT, json=data)
    return response.status_code, response.json()


if __name__ == "__main__":
    data = {'age': 33,
            'workclass': 'Private',
            'fnlwgt': 149184,
            'education': 'HS-grad',
            'education-num': 9,
            'marital-status': 'Never-married',
            'occupation': 'Prof-specialty',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 0,
            'capital-loss': 0,
            'hours-per-week': 60,
            'native-country': 'United-States'
            }
    status_code, response = predict(data)
    print('Status code:', status_code)
    print('Response:', response)
