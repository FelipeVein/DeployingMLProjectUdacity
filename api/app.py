"""
Main file for the FastAPI app.
"""

from ml.model import inference, load_model
from api.preprocessing import transform_data
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# add the parent directory to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)


# instantiate the app
app = FastAPI()

# load the model and the encoder
MODEL, ENCODER, LB = load_model(model_dir="model")


@app.get("/")
def hello_world():
    return JSONResponse(
        status_code=200,
        content={"message": "Hello World!"},
    )


class RequestBody(BaseModel):
    age: int = Field(..., example=39,
                     description="Age of the person")
    workclass: str = Field(..., example="State-gov",
                           description="Work class of the person")
    fnlwgt: int = Field(..., example=77516,
                        description="Final weight of the row")
    education: str = Field(..., example="Bachelors",
                           description="Education level of the person")
    education_num: int = Field(..., example=13,
                               alias="education-num",
                               description="Years of education completed")
    marital_status: str = Field(..., example="Never-married",
                                alias="marital-status",
                                description="Marital status of the person")
    occupation: str = Field(..., example="Adm-clerical",
                            description="Occupation of the person")
    relationship: str = Field(..., example="Not-in-family",
                              description="Relationship of the person to the householder")
    race: str = Field(..., example="White",
                      description="Race of the person")
    sex: str = Field(..., example="Male",
                     description="Gender of the person")
    capital_gain: int = Field(..., example=2174,
                              alias="capital-gain",
                              description="Capital Gain")
    capital_loss: int = Field(..., example=0,
                              alias="capital-loss",
                              description="Capital Loss")
    hours_per_week: int = Field(..., example=40,
                                alias="hours-per-week",
                                description="Hours worked per week")
    native_country: str = Field(..., example="United-States",
                                alias="native-country",
                                description="Native country of the person")


@app.post('/predict')
def predict(request: RequestBody):
    # transform the data
    x_data = transform_data(request, ENCODER, LB)
    # make the prediction
    preds = inference(MODEL, x_data)
    # transform the prediction
    preds_inv = LB.inverse_transform(preds)
    # return the prediction
    return JSONResponse(
        status_code=200,
        content={"prediction": float(preds[0]),
                 "prediction_label": preds_inv[0]},
    )
