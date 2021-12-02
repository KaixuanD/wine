
from pyspark import SparkContext
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import trim
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import avg
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder 
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import PipelineModel
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, Imputer, VectorAssembler, SQLTransformer
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier 
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import OneHotEncoder,StringIndexer, VectorAssembler


spark = SparkSession.builder.appName('wine_quality_model').getOrCreate()

def return_parsed_df(path):
    temp_df = spark.read.csv(path,header='true', inferSchema='true', sep=';')
    temp_df_pd = temp_df.toPandas()
    temp_df_pd.columns = temp_df_pd.columns.str.strip('""')
    return spark.createDataFrame(temp_df_pd)


df2 = spark.read.option("header",True).csv(str(sys.argv[1]) + "TrainingDataset.csv")

wine_train_df = return_parsed_df(str(sys.argv[1]) + 'TrainingDataset.csv')

wine_test_df = return_parsed_df(str(sys.argv[1]) + 'ValidationDataset.csv')


pd.DataFrame(wine_train_df.take(5), columns = wine_train_df.columns)


wine_train_df = wine_train_df.dropna()


wine_test_df = wine_test_df.dropna()


all_col_exc_quality = [c for c in wine_train_df.columns if c != 'quality']


stages = []

label_stringIdx = StringIndexer(inputCol = 'quality', outputCol = 'label')
stages += [label_stringIdx]

assembler = VectorAssembler(inputCols=all_col_exc_quality, 
                            outputCol="features")

stages += [assembler]


pipeline = Pipeline(stages = stages)

pipelineModel = pipeline.fit(wine_train_df)

pd.DataFrame(wine_train_df.take(5), columns = wine_train_df.columns)

dataDF = pipelineModel.transform(wine_train_df)

Data_test_df = pipelineModel.transform(wine_test_df)

Data_test_df.limit(3).toPandas().head()

dataDF.limit(10).toPandas().head()

selectedCols = ['label', 'features'] + wine_train_df.columns

Data_test_df = Data_test_df.select(selectedCols)

dataDF = dataDF.select(selectedCols)

pd.DataFrame(dataDF.take(5), columns = dataDF.columns)

pd.DataFrame(Data_test_df.take(5), columns = Data_test_df.columns)

print("Training Dataset Count: " + str(dataDF.count()))

print("Test Dataset Count: " + str(Data_test_df.count()))

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

rf_model = rf.fit(dataDF)

predictions = rf_model.transform(Data_test_df)

predictions.select('features', 'label', 'prediction').show(20)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)

print("Accuracy = %g" % (accuracy))

print("Test Error = %g" % (1.0 - accuracy))

print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))

stages += [rf]

pipeline = Pipeline(stages = stages)

params = ParamGridBuilder().build()
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params,evaluator=evaluator,numFolds=5)


cvModel = cv.fit(wine_train_df)
predictions = cvModel.transform(wine_test_df)
predictions_pandas = predictions.toPandas()
print('Test Area Under PR: ', evaluator.evaluate(predictions))

cvModel.write().overwrite().save(str(sys.argv[2]) + 'wine_quality_model.model')


f1 = f1_score(predictions_pandas.label, predictions_pandas.prediction, average='weighted')

print("F1 Score : ", f1)







