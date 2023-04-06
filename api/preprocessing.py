import pandas as pd
from ml.data import process_data
from ml.constants import CATEGORICAL_FEATURES

def transform_data(requestBody, encoder, lb):
    # requestBody is not a dict, it is a class from FastAPI
    # https://fastapi.tiangolo.com/tutorial/request-body/
    # transform it to a dict
    requestBody = requestBody.dict(by_alias=True)
    # create a dataframe from the dict
    df = pd.DataFrame(requestBody, index=[0])
    x_data, _, encoder, lb = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return x_data