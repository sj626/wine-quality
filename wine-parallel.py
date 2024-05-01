from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from io import BytesIO
import os
import subprocess
import tarfile

# Initialize Spark
APP_NAME = "Wine-Test"
conf = SparkConf().setAppName(APP_NAME)
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName(APP_NAME).getOrCreate()

# Retrieve training data from S3 bucket and preprocess
traindf = spark.read.csv("s3://wine-bucket-testing/TrainingDataset.csv", header=True, inferSchema=True)
traindf = traindf.withColumnRenamed("fixed acidity", "fixed_acidity") \
    .withColumnRenamed("volatile acidity", "volatile_acidity") \
    .withColumnRenamed("citric acid", "citric_acid") \
    .withColumnRenamed("residual sugar", "residual_sugar") \
    .withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide") \
    .withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide") \
    .withColumn("quality", traindf["quality"].cast(IntegerType()))

# Create features vector
feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
traindf = assembler.transform(traindf)

# Train Random Forest classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='quality')
rf_model = rf.fit(traindf)

# Save the model to S3 as a tar.gz file
rf_model.write().overwrite().save("s3://sjolaosho.tar.gz/RFMLModel.tar.gz")

# Stop Spark Context
sc.stop()
