import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.constants import CATEGORICAL_FEATURES, LABEL, DATA_FILE
from ml.model import compute_model_metrics, inference, save_model, train_model, load_model

def training_pipeline():
    # Load data
    df_train, df_test = load_data()
    # Process data
    X_train, y_train, encoder, lb = process_data(
        df_train, categorical_features=CATEGORICAL_FEATURES, label=LABEL
    )
    X_test, y_test, encoder, lb = process_data(
        df_test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)
    # Validate model
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")
    # Save model
    save_model(model, encoder, lb)
    
    return {"precision": precision, "recall": recall, "fbeta": fbeta,
            "model": model, "encoder": encoder, "lb": lb, 
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def load_data():
    """ Load the data from the data source.
    Returns
    -------
    train : pd.DataFrame
        Training data.
    test : pd.DataFrame
        Test data.
    """
    # Load data from data source
    df = pd.read_csv(DATA_FILE)
    # Split data into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def prediction_pipeline(X, process=True):
    """ Run the prediction pipeline.
    Parameters
    ----------
    X : pd.DataFrame
        Data to be used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Load model
    model, encoder, lb = load_model()
    if process:
        # Process data
        X, _, _, _ = process_data(
            X,
            categorical_features=CATEGORICAL_FEATURES,
            label=LABEL,
            training=False,
            encoder=encoder,
            lb=lb,
        )
    # Run inference
    preds = inference(model, X)
    return preds
