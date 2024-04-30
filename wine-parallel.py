"""
    UCID:   mk2246
    Desc:   Wine Quality Prediction Parallel Training Application.
"""

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from io import BytesIO
import os
import boto3
import subprocess
import tarfile

"""
    Create Spark Application
"""

APP_NAME = "Wine Quality Predictor Application"
conf = SparkConf().setAppName(APP_NAME)
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName(APP_NAME).getOrCreate()

"""
    Retrieve the training data set from S3 bucket and preprocess the dataset for training.
"""

client = boto3.client('s3')
traindataresponse = client.get_object(Bucket='mk2246-ml-files',Key='TrainingDataset.csv')
traindsString=traindataresponse['Body'].read().decode('ascii')
traindsString = traindsString.replace('"','')
traindsList = traindsString.split('\r\n')
tupledata = []
for r in traindsList:
        temp = r.split(';')
        tupledata.append(tuple(temp))
columns = list(tupledata[0])
tupledata = tupledata[1:len(tupledata)-1]
traindf = spark.createDataFrame(tupledata,columns)

traindf = traindf.toDF("fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide",\
                        "total_sulfur_dioxide","density","pH","sulphates","alcohol","quality")

traindf = traindf.withColumn('fixed_acidity', traindf.fixed_acidity.cast(FloatType()))\
                                .withColumn('volatile_acidity', traindf.volatile_acidity.cast(FloatType()))\
                                .withColumn('citric_acid', traindf.citric_acid.cast(FloatType()))\
                                .withColumn('residual_sugar', traindf.residual_sugar.cast(FloatType()))\
                                .withColumn('chlorides', traindf.chlorides.cast(FloatType()))\
                                .withColumn('free_sulfur_dioxide', traindf.free_sulfur_dioxide.cast(FloatType()))\
                                .withColumn('total_sulfur_dioxide', traindf.total_sulfur_dioxide.cast(FloatType()))\
                                .withColumn('density', traindf.density.cast(FloatType()))\
                                .withColumn('pH', traindf.pH.cast(FloatType()))\
                                .withColumn('sulphates', traindf.sulphates.cast(FloatType()))\
                                .withColumn('alcohol', traindf.alcohol.cast(FloatType()))\
                                .withColumn('quality', traindf.quality.cast(IntegerType()))

columns= ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide",\
                        "total_sulfur_dioxide","density","pH","sulphates","alcohol"]
vatraindf = VectorAssembler(inputCols=columns,outputCol='features')
traindf = vatraindf.transform(traindf)

"""
    Train the training data set to the Random Forest Classification Model.
"""
rf = RandomForestClassifier(featuresCol='features', labelCol='quality')
rfModel = rf.fit(traindf)

"""
    Save the Random Forest Classification Model to S3 as a tar.gz
"""
path = '/rfmodel/'
rfModel.write().overwrite().save(path)

mkdir_cmd = "mkdir modeltozip"
subprocess.run(mkdir_cmd, shell=True)
copy_cmd = "hadoop fs -get /rfmodel /home/ec2-user/modeltozip"
subprocess.run(copy_cmd, shell=True)

with tarfile.open("RFMLModel.tar.gz", "w:gz") as tarhandle:
        for root, dirs, files in os.walk('./modeltozip/'):
                for f in files:
                        tarhandle.add(os.path.join(root, f))

client.upload_file(
        Filename="RFMLModel.tar.gz",
        Bucket='mk2246-ml-files',
        Key='RFMLModel.tar.gz'
)

"""
    Retrieve validation data set from S3 Bucket and prepare the dataframe.
"""
testdataresponse = client.get_object(Bucket='mk2246-ml-files',Key='ValidationDataset.csv')
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


columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide'\
                    , 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
"""
    Model application to validation data set.
"""

vatestdf = VectorAssembler(inputCols=columns,outputCol='features')
testdf = vatestdf.transform(testdf)

# Apply model from locally trained, to the validation dataset
rfTest = rfModel.transform(testdf)

# Apply model from S3 to validation dataset
tarloc='mlmodel.tar.gz'
client.download_file(Bucket='mk2246-ml-files', Key='RFMLModel.tar.gz', Filename=tarloc)
tar = tarfile.open(tarloc, "r:gz")
tar.extractall('./extracted/')

rm_cmd = "hadoop fs -rm -r /rfmodel"
subprocess.run(rm_cmd,shell=True)
copy_cmd_to_hdfs = "hadoop fs -put ./extracted/modeltozip/ /model"
subprocess.run(copy_cmd_to_hdfs, shell=True)

s3model = RandomForestClassificationModel.load('/model/rfmodel')

rfModelTest = s3model.transform(testdf)

"""
    Model Evaluation for stats like F1-Score and Accuracy.
"""
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
rfaccuracy = evaluator.evaluate(rfTest)
print("************************************************************************")
print("F1 Score for Random Forest Classifier for local model = %g " % rfaccuracy)
print("************************************************************************")

s3rfaccuracy = evaluator.evaluate(rfModelTest)

print("*****************************************************************************************************")
print("F1 Score for Random Forest Classifier for validation dataset for s3 saved model = %g " % s3rfaccuracy)
print("*****************************************************************************************************")

# Remove extracted files from environment
subprocess.run("rm -r extracted/", shell=True)
subprocess.run("rm -r modeltozip/", shell=True)
subprocess.run("rm mlmodel.tar.gz", shell=True)
subprocess.run("rm RFMLModel.tar.gz", shell=True)
subprocess.run("hadoop fs -rm -r /model", shell=True)

sc.stop()
