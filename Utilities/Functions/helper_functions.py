from pyspark.sql import SparkSession, types as T, functions as F
import pyspark.sql.types  as st
import numpy as np
import re
# Model Score record Table
def Model_scores( df , prediction_col , lable_col, label_dict, prob_col, model_verion, model_name):
    """  Function to extract out model performance on outputs
    df  = input dataframe, 
    prediction_col  = predicted column in binary (1,0), 
    lable_col = Y label column, 
    label_dict = Label mapping dictionary 0 = Non 1 = Yes, 
    prob_col = probability of Y predicted by model, 
    model_verion = version of model, 
    model_name = Model Name (RF, GBT, LR)
    """
    predictions_and_labels = df.rdd.map(lambda row: (row[prediction_col] , float(row[lable_col])))
    multi_class_object = MulticlassMetrics(predictions_and_labels)
    # Label Markings
    label_category = []
    label_name = []
    label_value = []
    labels = df.rdd.map(lambda lp: lp[lable_col]).distinct().collect()
    for label in sorted(labels):
        metrics_dict= {"precision": multi_class_object.precision(label),
                       "recall": multi_class_object.recall(label),
                       "F1 Measure" : multi_class_object.fMeasure(float(label), beta=1.0)}
        metric_key = [key for key in metrics_dict]
        metric_val = [metrics_dict[value] for value in metric_key ]
        label_name.extend(metric_key)
        label_value.extend(metric_val)
        label_category.extend( [label_dict[label]] * len(metrics_dict) )
  
    # Weighted Metrics
    Weighted_scores =  {
        "Weighted recall" : multi_class_object.weightedRecall,
        "Accuracy" : multi_class_object.accuracy, 
        "Weighted precision" :  multi_class_object.weightedPrecision,
        "Weighted F(1) Score" : multi_class_object.weightedFMeasure(),
        "Weighted F(0.5) Scores" : multi_class_object.weightedFMeasure(beta=0.5),
        "Weighted false positive rate" : multi_class_object.weightedFalsePositiveRate}
    w_metric_key = [key for key in Weighted_scores]
    w_metric_val = [Weighted_scores[value] for value in w_metric_key ]

    # appending info to main list 
    label_name.extend(w_metric_key)
    label_value.extend(w_metric_val)
    label_category.extend( ["Weighted_Overall"] * len(Weighted_scores))
    
    #   ROC_AUC AND PR Score of the Model
    y_val_prob = df.select(prob_col).rdd.flatMap(lambda x: x).collect()
    y_val_label = df.select(lable_col).rdd.flatMap(lambda x: x).collect()
    auc_score = roc_auc_score(y_val_label,y_val_prob)
    pr_score = average_precision_score(y_val_label,y_val_prob)
    roc_pr_list = [auc_score,pr_score]
    metric_name = ['roc_auc', 'avg_pr']
    
    # appending info to main list 
    label_name.extend(metric_name)
    label_value.extend(roc_pr_list)
    label_category.extend( ["Weighted_Overall"] * len(roc_pr_list))

  # tranforming to Pandas Df, because of format consistency
    final_df = pd.DataFrame({
        'Model_version' : model_verion,
        'Label_category': label_category,
        'Metric_name': label_name,
        'Metric_value': label_value, 
        'classifier': model_name
    })
    spark_df = spark.createDataFrame(final_df).withColumn("capture_date",F.current_date())
    return spark_df


# Extract probabolity from pyspark probability column
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None
extract_prob_udf = F.udf(extract_prob, st.DoubleType())



# Split of Train and Test
def train_test_split_spark(df, dep_var,repartition_col ,train_split = 0.7, test_split = 0.3):
    df = df.repartition(100, repartition_col)

    #Separate out classes into different dataframes
    zero  = df.filter(dep_var+"=0.0")
    ones = df.filter(dep_var+ "=1.0")
    
    #Do train / test random split of each class dataframe
    (trainingData_np, testData_np) = zero.randomSplit([train_split, test_split], seed=42)
    (trainingData_ap, testData_ap) = ones.randomSplit([train_split, test_split], seed=42)
    
    #Combine all of the separate train class dataframes into a train dataframe
    trainingData = trainingData_np.union(trainingData_ap).orderBy(F.rand())
    testData = testData_np.union(testData_ap).orderBy(F.rand())
    return trainingData, testData


# Makes list of Input features cols
def input_features(df, exclude_cols):
    col_list = df.columns
    input_features = [feature for feature in col_list if feature not in exclude_cols]
    return input_features

# function to create input features 
def create_features_and_transform(df, input_features):
    assembler = VectorAssembler(
        inputCols= input_features,
        outputCol="features")
    df_transform = assembler.transform(df)
    return df_transform


# Function to apply model experiments.
# addtional params retain_best_model = True, select_model_metric = 'avg_pr'
def Binary_model_Experiment( model_list, train_df, test_df, y_label, input_feature, configs_, label_dictionary, 
                            save_model = True, model_location = None ):
  schema = st.StructType([ \
                          st.StructField("Model_version", st.IntegerType(),True), \
                          st.StructField("Label_category",st.StringType(),True), \
                          st.StructField("Metric_name",   st.StringType(),True), \
                          st.StructField("Metric_value",  st.DoubleType(), True), \
                          st.StructField("classifier",    st.StringType(), True), \
                          st.StructField("capture_date",  st.DateType(), True) \
                          ])
  model_name_ = [] 
  master_log_df = spark.createDataFrame(data =[], schema=schema)
  for model_ in model_list:
    iter_model = model_(labelCol= y_label, featuresCol= input_feature)
    Mtype = type(iter_model).__name__
    model_name_.append(Mtype)
    print("Iter Model :",Mtype)
    _evaluator = BinaryClassificationEvaluator(labelCol= y_label, rawPredictionCol= "rawPrediction",
                                               metricName = configs_.eval_metric)
    # create hyper parameter grid to evaluate cross validation Model Config is added in the retrian config part.
    _paramGrid = SelectParameters(ModelType = Mtype, model_ =  iter_model)
    # Create K-fold CrossValidator
    _cv = CrossValidator(estimator= iter_model, estimatorParamMaps= _paramGrid, 
                                evaluator=_evaluator, numFolds= configs_.cv_folds)
    _cvModel = _cv.fit(train_df)
    _bestModel = _cvModel.bestModel
    testPrediction = _bestModel.transform(test_df)
    testPrediction = testPrediction.withColumn("prob_raw", extract_prob_udf(F.col("probability")))
    if save_model:
      _bestModel.write().overwrite().save(os.path.join(model_location,Mtype))
    # Model Scores
    # label_dictionary = {0.0: "Non ALK+ NSCLC", 1.0: "ALK+ NSCLC"}
    Iter_model_scores = Model_scores(df = testPrediction,  prediction_col = "prediction" ,
                                     lable_col = y_label,prob_col = "prob_raw",
                                     label_dict = label_dictionary,model_name = Mtype,
                                     model_verion = 1 ) # Can be made Dynamically changable once 
    master_log_df = master_log_df.union(Iter_model_scores)
  
  master_log_df_pivot = master_log_df.groupBy("Model_version", "Label_category", "Metric_name", 
                                               "capture_date" ).pivot("classifier", model_name_).sum("Metric_value")
  return {'train_results': master_log_df_pivot}
