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
