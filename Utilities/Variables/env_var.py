from pyspark.ml.classification import  RandomForestClassificationModel, GBTClassificationModel, LogisticRegressionModel



load_model_dict = {'GBTClassifier': GBTClassificationModel,
                   'RandomForestClassifier': RandomForestClassificationModel,
                   'LogisticRegression':LogisticRegressionModel}

exc_cols_ = ['HeartDisease_Yes', 'capture_date']

