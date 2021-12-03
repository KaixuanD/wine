import random
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#  SPARK INITIALIZING
spark = SparkSession.builder.appName("predict").getOrCreate()
spark.sparkContext.setLogLevel("Error")
print("================SPARK VERSION==============")
print(spark.version)

# READ DATA
print("Reading data from {}...".format(sys.argv[1]))
traintb = spark.read.format("csv").load(sys.argv[1], header=True, sep=";")
traintb.show(5)

# modify column names.
traintb = traintb.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                       "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")
traintb.show(5)

# Extract feature names.
features = traintb.columns
features = features[:-1]

# make sure the data in proper types.
traintb = traintb .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
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


va = VectorAssembler(inputCols=features, outputCol="features")
va_df = va.transform(traintb)
va_df = va_df.select(["features", "label"])
traintb = va_df

print("Loading model")
trModel = MultilayerPerceptronClassificationModel.load(sys.argv[2])
print("Processing predictions")
predictions = trModel.transform(traintb)

print("Evaluating")
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print("F1 = %g " % f1)
print("==========================Finish=======================.")
