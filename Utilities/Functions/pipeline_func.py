import pyspark
from pyspark.ml.pipeline import Transformer
from pyspark.sql.functions import udf
import pyspark.sql.types  as st
from pyspark.sql import SparkSession, types as T, functions as F
import numpy as np
import re

# Functions
# count the categories of Cat columns.
def extract_col_datatype(df):
  dict_ = {}
  for i in df.columns:
    dict_[i] = df.schema[i].dataType
  return dict_

def data_type_col(df, data_type = 'StringType'):
  col_dict = extract_col_datatype(df)
  col_list = []
  for col in col_dict:
    if (str(col_dict[col]) == data_type):
       col_list.append(col)
  return col_list

# Regex Condition to remove space and special characters. 
def make_alpha_num (x):
  x = re.sub('[^A-Za-z0-9_]+', '', x)
  return x

# Pipeline Functions
# Function for One Hot Encoding, to convert column to binary one hot encoding. 
def one_hot_encoder(df,col_list, drop_cols, exclude_cat = None, drop_orignal =True, drop_last_col= True): # col_list is a python list
  """  
  df = Input Data frame, 
  col_list = Columns that have to convereted to One hot encoding (1,0)
  exclude_cat = this is custom dictionary input function that which will exclude specific categories for encoding from the column. (Optional)
  drop_orignal =True, drops original Categorical Column.
  drop_last_col= True, drops one of the two categories expalining each other, for example "Male" encoding will auto encode for "Female"
  """
  if exclude_cat != None:
    exclude_cols = np.array(list(exclude_cat))
  
  for col_ in col_list:
    iter_col = np.array([col_])
    if (exclude_cat != None):
      for _col in exclude_cat:
        if col_ == _col:
          execlude_cols = np.array(exclude_cat[col_])
          all_cats = np.array(df.select(col_).distinct().rdd.flatMap(lambda x: x).collect())
          type_bool = np.isin(all_cats, execlude_cols)
          types = sorted(list(all_cats[~type_bool]))
          break
    
    if (exclude_cat != None) & ((~np.isin(iter_col,exclude_cols))[0]):
      types = sorted(df.select(col_).distinct().rdd.flatMap(lambda x: x).collect())
    if exclude_cat == None:
      types = sorted(df.select(col_).distinct().rdd.flatMap(lambda x: x).collect())
    types_expr = [F.when(F.col(col_) == ty, 1).otherwise(0).alias(col_+"_"+ make_alpha_num(ty.replace(" ", "_"))) for ty in types]
    if (drop_last_col) & (len(types_expr) > 1):
      types_expr = types_expr[: (len(types_expr)-1)] # Dropping Last Column as Not Being Necessary which will avoid multi collinearity.
    df = df.select("*", *types_expr)
  # Dropping all the orignal Columns if True
  try:
    drop_cols
  except:
    drop_cols = col_list
  if drop_orignal:
    df = df.drop(*drop_cols)
  return df

# Function to plug function within pipeline
class one_hot_encoder_pipeline(Transformer):
    """Function To calculate apply One Encoding in pipeline Functions"""
    def __init__(self, col_list, drop_cols, exclude_cat = None,drop_orignal =True, drop_last_col= True):
        self.col_list = col_list #the name of your columns
        self.exclude_cat = exclude_cat
        self.drop_orignal =drop_orignal
        self.drop_last_col = drop_last_col
        self.drop_cols  = drop_cols
    def this():
        #define an unique ID
        this(Identifiable.randomUID("one_hot_encoder"))
# transform
    def _transform(self, df):
        return one_hot_encoder(df = df,
                               col_list =     self.col_list,
                               drop_cols =    self.drop_cols,
                               exclude_cat =  self.exclude_cat,
                               drop_orignal = self.drop_orignal, 
                               drop_last_col= self.drop_last_col)

def col_translate(mapping):
    def translate_(col):
        return mapping.get(col)
    return udf(translate_, st.StringType())

# Function for Ordinal 
def ordinal_label_mapping(df, mapping_, drop_orignal =True, replace_na = True, replace_na_val = 0):
  """  
  df = Input Data frame, 
  mapping_ = Mapping dictionary which has column categories mapped to Numerical value e.g. bad: 0, good: 1, excellent: 2 
  replace_na = any NA values found (categories which doesn't have mappping key)
  replace_na_val = value to impute for null values.
  drop_orignal =True, drops original Categorical Column.
  """
  cols = [i for i in mapping_]
  for dicts in cols:
    iter_dict = mapping_[dicts]
    df = df.withColumn(dicts+"_transformed", col_translate(iter_dict)(dicts))
    if replace_na:
      df = df.fillna({dicts+"_transformed": replace_na_val})
      df = df.withColumn(dicts+"_transformed", F.col(dicts+"_transformed").cast(st.FloatType()))
  if drop_orignal:
    df = df.drop(*cols)
  return df

class ordinal_label_mapping_pipeline(Transformer):
    """Function To calculate apply ordinal mapping pipeline Functions"""
    def __init__(self, mapping_, drop_orignal =True, replace_na = True, replace_na_val = 0):
        self.mapping_ = mapping_ #the name of your columns
        self.drop_orignal = drop_orignal
        self.replace_na =replace_na
        self.replace_na_val = replace_na_val
    def this():
        #define an unique ID
        this(Identifiable.randomUID("ordinal_label_mapping"))
# transform
    def _transform(self, df):
        return ordinal_label_mapping(df = df,
                                     mapping_       = self.mapping_,
                                     drop_orignal   = self.drop_orignal,
                                     replace_na     = self.replace_na,
                                     replace_na_val = self.replace_na_val)
        

# Custom Regex Functions
def extract_regex_expr(str_ , expr_ , first_value = True, convert_float = True):
  """Integer will be applicable only on single string"""
  iter_str_ = re.findall(expr_, str_)
  # First Value
  if first_value:
    iter_str_ = iter_str_[0]
    if convert_float:
      iter_str_ = np.float(iter_str_)
  return iter_str_

extract_regex_expr_f = udf(extract_regex_expr, st.FloatType())

def apply_regex_expr(df, cols_ , expr, first_value = True, convert_float = True, drop_orignal = True):
  """Apply Regex Expression in Spark DF
  cols_ = columns where regex to be applied.
  expr = regex expression.
  first_value = out of multiple expressions captured by regex, dilter the first one.
  convert_float = True, captured value to be converted to float 
  drop_orignal = True, drop orignal data column.
  """ 
  df = df.withColumn('expr_column', F.lit(expr))
  df = df.withColumn('first_value', F.lit(first_value).cast(st.BooleanType()))
  df = df.withColumn('convert_float', F.lit(convert_float).cast(st.BooleanType()))

  for col in cols_:
    df = df.withColumn(col+"_transformed", extract_regex_expr_f(df[col] ,df['expr_column']
                                                                ,df['first_value'], df['convert_float'] 
                                                                ))
  # Drop Working Columns
  df = df.drop(*['expr_column', 'first_value', 'convert_float'])
  # If Drop Orginal is True. 
  if drop_orignal:
    df = df.drop(*cols_)
  return df
  

class extract_regex_expr_pipeline(Transformer):
    """Function To calculate apply regex expr_ in pipeline Functions"""
    def __init__(self, cols_, expr,first_value = True, convert_float = True, drop_orignal = True):
        self.cols_ = cols_ #the name of your columns
        self.drop_orignal = drop_orignal
        self.expr = expr
        self.first_value = first_value
        self.convert_float = convert_float
    def this():
        #define an unique ID
        this(Identifiable.randomUID("Regex_Expression_Corpus"))
# transform
    def _transform(self, df):
        return apply_regex_expr(df = df,
                                     cols_         =  self.cols_,
                                     expr          =  self.expr, 
                                     first_value   =  self.first_value,
                                     convert_float =  self.convert_float,
                                     drop_orignal  =  self.drop_orignal)
        
# Change Datatype. 
def change_datatype(df, data_dict):
  for col_ in data_dict:
    df = df.withColumn(col_, F.col(col_).cast(data_dict[col_]))
  return df

class change_datatype_pipeline(Transformer):
    """Function To change datatypes in pipeline Functions"""
    def __init__(self, data_dict):
        self.data_dict = data_dict
    def this():
        #define an unique ID
        this(Identifiable.randomUID("Change_data_types"))
# transform
    def _transform(self, df):
        return apply_regex_expr(df = df,
                                data_dict     =  self.data_dict)

# Dropping Columns
class drop_columns_pipeline(Transformer):
    """Function To drop columns in pipeline Functions"""
    def __init__(self, cols):
        self.cols = cols
    def this():
      #define an unique ID
      this(Identifiable.randomUID("Drop Columns"))
    # transform
    def _transform(self, df):
        return df.drop(*self.cols) 
        

# Function to plug function within pipeline
class create_features_and_transform_pipeline(Transformer):
    """Function To create feature for model input"""
    def __init__(self, exc_cols):
       self.exc_cols = exc_cols
    def this():
        #define an unique ID
        this(Identifiable.randomUID("create_and_Transform"))
    def _transform(self, df):
        return create_features_and_transform(df = df,
                                             input_features =  input_features(df, exclude_cols = self.exc_cols))

 # To know dimensions of Spark Dataframe (Rowns and Coloumns) similar to pandas shape function.
def spark_shape(self):
    return (self.count(), len(self.columns))
pyspark.sql.dataframe.DataFrame.shape = spark_shape


#  Pipeline to apply inference on the transformed Dataset.
def model_inference(model_, df,  apply_cutoff=False, model_threshold = 0.5):
  """ 
  model_ = Trained Model, 
  df = Input Data frame, 
  y_label = dependant Variables
  apply_cutoff=False,  cut_off apply custom cutoff on the model 
  model_threshold = 0.5 default model threshold at 0.5
  """
  validation_predicted_prod = model_.transform(df)
  validation_predicted_prod = validation_predicted_prod.withColumn("prob_raw", extract_prob_udf(F.col("probability")))

  if apply_cutoff: # Condtion if prediction to be applied on 
    validation_predicted_prod = validation_predicted_prod.withColumn( "prediction_with_threshold" ,
                                                                     F.when(F.col("prob_raw") >= model_threshold , 1.0)\
                                                                     .otherwise(0.0))
  else:
    validation_predicted_prod = validation_predicted_prod.withColumn("prediction_with_threshold" , F.col('prediction'))
  return validation_predicted_prod

# Adding Function into Pipeline.
class model_inference_pipeline(Transformer):
    """Function To calculate apply ordinal mapping pipeline Functions"""
    def __init__(self,model_,  apply_cutoff=False, model_threshold = 0.5):
        self.model_ = model_ #the name of your columns
        self.apply_cutoff = apply_cutoff
        self.model_threshold = model_threshold
    def this():
        #define an unique ID
        this(Identifiable.randomUID("Model_Inference_generate_scores"))
    # transform
    def _transform(self, df):
        return model_inference(df = df,
                               model_       = self.model_,
                               apply_cutoff     = self.apply_cutoff,
                               model_threshold =   self.model_threshold)


#  Performance Evaluation code itegrated within Pipeline
class model_scores_pipeline(Transformer):
  """Model scores function to be integrated within pipeline"""
  def __init__(self,prediction_col, lable_col, label_dict, prob_col, model_verion, model_name):
        self.prediction_col = prediction_col # Y Label
        self.lable_col = lable_col 
        self.label_dict = label_dict
        self.prob_col = prob_col
        self.model_verion = model_verion
        self.model_name = model_name
  def this():
    #define an unique ID
    this(Identifiable.randomUID("Model_scores"))
  def _transform(self, df):
        return Model_scores(df = df , 
                            prediction_col = self.prediction_col , 
                            lable_col = self.lable_col, 
                            label_dict = self.label_dict, 
                            prob_col = self.prob_col, 
                            model_verion = self.model_verion, 
                            model_name = self.model_name)

    #  Feature Score Pipeline piece

# Function to record Data Drift Scores
def input_feature_scores(df, cat_var, cont_var, label_feature, model_version):
    feature_category = []
    feature_name = []
    metrics_value = []
    score_type = []
    
    # Chisquare test for Categorical Variables
    print('------------------- Calculating Chi scores-------------------')
    chisqaure_assembler = VectorAssembler(
      inputCols = cat_var,
      outputCol = 'cat_feature')
    chisquare_vector_df1 = chisqaure_assembler.transform(df).select( "cat_feature", label_feature)
    chi_square_result =  list((ChiSquareTest.test(chisquare_vector_df1, "cat_feature", label_feature).head().pValues ))
    
    feature_category.extend( ["Categorical"] * len(cat_var) )
    score_type.extend( ["Chi-Square test - pValue"] * len(cat_var) )
    feature_name.extend(cat_var)
    metrics_value.extend(chi_square_result)
    
    # Anova Test for Continous Variable
    print('------------------- Calculating Anova scores-------------------')
    pvalue = []
    for i in cont_var:
        my_df = df.select(i,label_feature).toPandas()
        model = ols('{0} ~ {1}'.format(i,label_feature),                 # Model formula
                  data = my_df).fit()
        anova_result = sm.stats.anova_lm(model, typ=2)
        pvalue.append(anova_result['PR(>F)'][0])
   
    feature_category.extend( ["Continous"] * len(cont_var) )
    score_type.extend( ["Anova test - pValue"] * len(cont_var) )
    feature_name.extend(cont_var)
    metrics_value.extend(pvalue)

    final_df = pd.DataFrame({
        'Model_version' : model_version,
        'Label_category': feature_category,
        'Feature_name': feature_name,
        'Metric_name': score_type,
        'Metric_value':  np.round(np.array(metrics_value) * 1.0, 3) 
    })
    spark_df = spark.createDataFrame(final_df).withColumn("capture_date",F.current_date())
    spark_df =  spark_df.withColumn( "Metric_value" , F.when( F.col("Metric_value") == 'NaN', None).otherwise(F.col("Metric_value")))
    return spark_df


#  Performance Evaluation code itegrated within Pipeline
class input_feature_pipeline(Transformer):
  """Model scores function to be integrated within pipeline"""
  def __init__(self,cat_var, cont_var, label_feature, model_version):
        self.cat_var = cat_var # Y Label
        self.cont_var = cont_var 
        self.label_feature = label_feature
        self.model_version = model_version
  def this():
    #define an unique ID
    this(Identifiable.randomUID("Feature_score"))
  def _transform(self, df):
        return input_feature_scores(df = df , 
                            cat_var = self.cat_var , 
                            cont_var = self.cont_var, 
                            label_feature = self.label_feature, 
                            model_version = self.model_version)
