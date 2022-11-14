
from pyspark.ml.pipeline import Transformer
import sys
import os
from pyspark.sql.functions import udf
import pyspark.sql.types  as st
from pyspark.sql import SparkSession, types as T, functions as F
import numpy as np
sys.path.insert(1, os.path.join(os.getcwd(), 'ML_Ops_Practices/Utilities/Functions'))
# from pipeline_func import one_hot_encoder_pipeline, ordinal_label_mapping_pipeline, extract_regex_expr_pipeline, drop_columns_pipeline

# Feature Engineering

# Add Class here.
# Listing Two categories
two_categories = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
                  'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']

# Mapping only Yes Category in Encoding for Heart Disease
Heart_disease =  {'HeartDisease': ['No'], 
                 'Smoking': ['No'],            
                 'AlcoholDrinking': ['No'],
                 'Stroke':['No'],
                'PhysicalHealth': ['No'],
                 'MentalHealth': ['No'],
                 'DiffWalking' : ['No'], 
                 'Sex': ['Male'],
                 'PhysicalActivity': ['No'],
                 'Asthma' :  ['No'],
                 'KidneyDisease' :['No'],
                 'SkinCancer': ['No']}

#  Setup Ordinal Piece
# Numerical Mapping to Categorical columns
ordinal_dict = {"GenHealth": {'Excellent': 5.0,'Very good': 4.5,
                              'Good': 4,'Fair': 3,'Poor': 2},
                'Diabetic' : { 'No, borderline diabetes': 1 , 'No': 0, 'Yes': 3} }

# Pipeline Steps
# excluding categories from One Hot Encoding 
excludng_categories = {'Diabetic' :  ['No, borderline diabetes', 'No', 'Yes'],
                       'Race': ['Other']}

# Step 1 : One Hot encoding for two level categories Yes or No's
step_1_one_hot_enc=  one_hot_encoder_pipeline(col_list = two_categories, exclude_cat = Heart_disease, drop_cols = two_categories,
                                              drop_orignal =True,drop_last_col= True)
# Step 2 : One Hot Encoding for Female who are diabetice during Pregnancy, as it is non comparable to Normal Diabetic condtions.  
# Excluding Others from Race as it's not interpretable from Business Perspective.
step_2_diabetic_enc_pregnancy = one_hot_encoder_pipeline(col_list = ['Diabetic', 'Race'],exclude_cat = excludng_categories,
                                                         drop_orignal =True, drop_cols = ['Race'], drop_last_col= True)
# Ordinal Mapping (for Genhealth Diabetics)
step_3_ordinal_mapping = ordinal_label_mapping_pipeline(mapping_ = ordinal_dict, drop_orignal =True,
                                                        replace_na = True, replace_na_val = 0)

# Step 4 Regex Expresion, to Extract out  first characters from the age categorical binning column
exp_ = '^.{0,2}'
step_4_regex = extract_regex_expr_pipeline(cols_ = ['AgeCategory'] , expr = exp_, first_value = True)

# Step 5  Dropping unessary Columns. 
step_5_drop_columns = drop_columns_pipeline( cols = ['capture_date'])


# Addtional transformation.        
create_feature_pipeline = create_features_and_transform_pipeline(exc_cols = exc_cols_)

