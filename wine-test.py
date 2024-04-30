from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
import boto3
import tarfile
import subprocess
import sys

// CREATE SPARK APP

APP_NAME = "Wine-Test"
conf = SparkConf().setAppName(APP_NAME)
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName(APP_NAME).getOrCreate()

client = boto3.client('s3')

arguments = sys.argv
if len(arguments) > 1:
        filename = arguments[1]
        filepath = "./"+filename

if len(arguments) <= 1:
        // VALIDATE FILE
        testdataresponse = client.get_object(Bucket='',Key='ValidationDataset.csv')
        testdsString = testdataresponse['Body'].read().decode('ascii')
        testdsString = testdsString.replace('"','')
        testdsString = testdsString.replace('b','')
        testdsList = testdsString.split('\r\n')

        testtupledata = []
        for r in testdsList:
                temp = r.split(';')
                testtupledata.append(tuple(temp))

        testcolumns = list(testtupledata[0])
        testtupledata = testtupledata[1:len(testtupledata)-1]

        testdf = spark.createDataFrame(testtupledata,testcolumns)
else:
        //TEST DATA
        testdf = spark.read.option("header", True).option("delimiter",";").csv(filepath)

columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide'\
                , 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

testdf = testdf.toDF("fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide",\
                        "total_sulfur_dioxide","density","pH","sulphates","alcohol","quality")

testdf = testdf.withColumn('fixed_acidity', testdf.fixed_acidity.cast(FloatType()))\
                                .withColumn('volatile_acidity', testdf.volatile_acidity.cast(FloatType()))\
                                .withColumn('citric_acid', testdf.citric_acid.cast(FloatType()))\
                                .withColumn('residual_sugar', testdf.residual_sugar.cast(FloatType()))\
                                .withColumn('chlorides', testdf.chlorides.cast(FloatType()))\
                                .withColumn('free_sulfur_dioxide', testdf.free_sulfur_dioxide.cast(FloatType()))\
                                .withColumn('total_sulfur_dioxide', testdf.total_sulfur_dioxide.cast(FloatType()))\
                                .withColumn('density', testdf.density.cast(FloatType()))\
                                .withColumn('pH', testdf.pH.cast(FloatType()))\
                                .withColumn('sulphates', testdf.sulphates.cast(FloatType()))\
                                .withColumn('alcohol', testdf.alcohol.cast(FloatType()))\
                                .withColumn('quality', testdf.quality.cast(IntegerType()))

// MODEL PREDICTION 
vatestdf = VectorAssembler(inputCols=columns,outputCol='features')
testdf = vatestdf.transform(testdf)

// RETRIEVE MODEL FROM S3 
tarloc='sjolaosho.tar.gz'
client.download_file(Bucket='arn:aws:s3:::wine-bucket-testing', Key='RFMLModel.tar.gz', Filename=tarloc)
tar = tarfile.open(tarloc, "r:gz")
tar.extractall('./extracted/')

// MODEL
s3model = RandomForestClassificationModel.load('./extracted/modeltozip/rfmodel')
rfModelTest = s3model.transform(testdf)

// MODEL EVALUATION
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")

s3rfaccuracy = evaluator.evaluate(rfModelTest)

print("************************************************************************")
print("F1 Score for Random Forest Classifier for test data = %g " % s3rfaccuracy)
print("************************************************************************")

display= rfModelTest['quality','prediction']
display.show(display.count())

# Removing the extracted files from the environment
subprocess.run("rm sjolaosho.tar.gz", shell=True)
subprocess.run("rm -r extracted/", shell=True)

sc.stop()
