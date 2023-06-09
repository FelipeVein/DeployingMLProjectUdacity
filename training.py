from ml.pipelines import training_pipeline, prediction_pipeline, slice_performance

# run training pipeline
print("Running training pipeline...")
output = training_pipeline()
# run prediction pipeline
X_test = output["X_test"]
print("Running prediction pipeline...")
preds = prediction_pipeline(X_test[0:5, :], process=False)
print(preds)
# run slice performance
print("Running slice performance...")
slice_performance()
