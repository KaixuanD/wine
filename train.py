
import random
import sys
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#  SPARK INITIALIZING
spark = SparkSession.builder.appName("predict").getOrCreate()
spark.sparkContext.setLogLevel("Error")
print("================SPARK VERSION==============")
print(spark.version)


# Read data
print("Reading data from {}...".format(sys.argv[1]))
traintb = spark.read.format("csv").load(sys.argv[1], header=True, sep=";")
traintb.show(5, False)

traintb = traintb.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")
traintb.show(5, False)

traintb = traintb \
        .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
        .withColumn("volatile_acidity", col("volatile_acidity").cast(DoubleType())) \
        .withColumn("citric_acid", col("citric_acid").cast(DoubleType())) \
        .withColumn("residual_sugar", col("residual_sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free_sulfur_dioxide", col("free_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("total_sulfur_dioxide", col("total_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("label", col("label").cast(IntegerType()))

features = traintb.columns
features = features[:-1]

va = VectorAssembler(inputCols=features, outputCol="features")
va_df = va.transform(traintb)
va_df = va_df.select(["features", "label"])
traintb = va_df


print("Training please wait")

layers = [11, 9, 9, 9,9, 11]
tr = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=64, stepSize=0.030, solver='l-bfgs')
trModel = tr.fit(traintb)

print("Saving file".format(sys.argv[2]))
trModel.write().overwrite().save(sys.argv[2])
print("Succseefull ---Close.")
