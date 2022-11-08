# Model Parameters are only set for three model architecture GBTClassifier, RandomForestClassifier, LogisticRegression

def SelectParameters(ModelType, model_):
  if ModelType == 'GBTClassifier':
    param_grid_settings = ParamGridBuilder() \
                            .addGrid(model_.stepSize, [0.1] ) \
                            .addGrid(model_.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 10, num = 2)]) \
                            .addGrid(model_.maxBins, [int(x) for x in np.linspace(start = 10, stop = 20, num = 2)]) \
                            .addGrid(model_.maxIter, [int(x) for x in np.linspace(start = 100, stop = 150, num = 2)]) \
                            .build()
  if ModelType == 'RandomForestClassifier':
    param_grid_settings = ParamGridBuilder() \
                            .addGrid(model_.numTrees, [int(x) for x in np.linspace(start = 100, stop = 150, num = 1)]) \
                            .addGrid(model_.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 2)]) \
                            .addGrid(model_.maxBins, [int(x) for x in np.linspace(start = 20, stop = 60, num = 2)]) \
                            .build()
  if ModelType == 'LogisticRegression':
    param_grid_settings = ParamGridBuilder() \
                            .addGrid(model_.regParam, [0.1, 0.5, 2.0]) \
                            .addGrid(model_.elasticNetParam, [0.0 , 0.1, 0.5]) \
                            .addGrid(model_.maxIter, [int(x) for x in np.linspace(start = 5, stop = 30, num = 5)]) \
                            .build()
  return param_grid_settings
# regParam : ridge Regularization. L2 
# elasticNetParam: Lasso Regulatization. L1 

#  regParam
# 0.0 , 0.1, 0.5 elasticNetParam

# train configs
class train_config:
  cv_folds = 2
  eval_metric = 'areaUnderPR'
