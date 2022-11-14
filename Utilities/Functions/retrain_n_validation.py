# sensitivity speccificty & precision from the

def sensivity_specificity(y, y_pred):
    rang = np.linspace(0,1,num = 101)
    Sensitivity = []
    Specificity = []
    Accuracy = []
    precision = []
    for i in rang:
#         print("Running at Cut-off: ", i)
        pred__test =  [1 if x > i else 0 for x in y_pred]
        accuracy =  np.sum(np.array(pred__test) == np.array(y)) / len(y) 
        a = confusion_matrix(y, pred__test)
        tp = a[1, 1]
        fn = a[1,0]
        tpr = tp/(tp + fn)
        tn = a[0,0]
        fp = a[0,1]
        tnr = tn/ (tn + fp)
        prcn = tp/(tp + fp)
        Sensitivity.append(tpr)
        Specificity.append(tnr)
        Accuracy.append(accuracy)
        precision.append(prcn)
    df_return = pd.DataFrame({
        'Cut-Off': np.round(rang,2),
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'precision': precision,
        'Accuracy': Accuracy
    })
    return df_return


def recall_precision_selection(model_, model_location, df, return_graph = True, 
                               select_metric = 'precision',threshold = 0.4, print_cut_off = True):
  """
  model_: Model Name, 
  model_location: Model Location from the Saved Models, 
  df: Input Data Frame, 
  return_graph = True, return Precicion VS recall Graph for Decision Making 
  select_metric = {'precision' or 'Sensitivity'} as model objective,
  threshold = 0.4 of selected metric precision or recall, 
  print_cut_off = True print model cutoff
  """
  iter_model = load_model_dict[model_].load(os.path.join(model_location,model_))
  test_df_ = iter_model.transform(df)
  test_df_ = test_df_.withColumn("prob_raw", extract_prob_udf(F.col("probability")))
  prob = test_df_.select('prob_raw').rdd.flatMap(lambda x: x).collect()
  actual = test_df_.select('HeartDisease_Yes').rdd.flatMap(lambda x: x).collect()
  recall_pr = sensivity_specificity(actual, prob)
  if select_metric == 'precision':
    selected_threshold_new = recall_pr[(recall_pr['precision'] >= threshold)]
    max_recall_new = np.max(selected_threshold_new['Sensitivity'])
    precison_ = threshold
    recall_ = max_recall_new
    # Selecting the Threshold Value
    new_selected_threshold_value = np.max(selected_threshold_new['Cut-Off'][selected_threshold_new['Sensitivity']==max_recall_new])
  if select_metric == 'Sensitivity':
    selected_threshold_new = recall_pr[(recall_pr['Sensitivity'] >= threshold)]
    max_precision_new = np.max(selected_threshold_new['precision'])
    precison_ = max_precision_new
    recall_ = threshold
    # Selecting the Threshold Value
    new_selected_threshold_value = np.max(selected_threshold_new['Cut-Off'][selected_threshold_new['precision']==max_precision_new])
  if print_cut_off:
    print('To attain ',select_metric, ' {0} '.format(threshold),
          'the {0} Model threshold cut-off has to be set to '.format(model_), new_selected_threshold_value)
  if return_graph:
    fig = px.line(recall_pr, x='Cut-Off', y =['Sensitivity', 'precision',])
    fig.update_layout(showlegend=True, legend=dict(orientation="h"),  plot_bgcolor='white',
                      autosize=False,width=600,height=400)
    fig.show()
  output_obj = {'Model_name':model_,'selected_metric': select_metric, 
                'precision': np.str(precison_),'recall': np.str(recall_), 
                'cut-off': np.str(new_selected_threshold_value)}
  return output_obj


# Loading Saved Model
def loadModel(input_model, location = os.path.join(os.getcwd(),'output/content/saved_models')):
  model_location = os.path.join(location, input_model) 
  model_ = load_model_dict[input_model].load(model_location)
  return model_
