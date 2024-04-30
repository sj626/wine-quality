from flask import Flask, Blueprint, render_template, request, redirect, flash
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
import csv
import io
import pandas

winepredict = Blueprint('winepredict', __name__)

@winepredict.route("/", methods=["GET","POST"])
def wineprediction():
    score=0
    predictHtml=""
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', "warning")
            return redirect(request.url)

        if not file.filename.endswith('.csv'):
            flash('Selected file is not a csv file!', "warning")
            return redirect(request.url)
        if file.filename:
            stream = io.TextIOWrapper(file.stream._file, "UTF8", newline=None)
            fileDataDict = csv.DictReader(stream)
            tempfileDataDict = fileDataDict
            tupledata = []
            header=""
            for r in tempfileDataDict:
                columns = list(r.keys())[0]
                header = columns
                temp = ""
                for v in r.values():
                    temp = temp.join(v)
                    temp = temp.split(';')
                    tupledata.append(tuple(temp))

            header = header.replace('"','')
            header = header.split(';')            

        """
            Create Spark Application
        """

        APP_NAME = "Wine Quality Predictor Application"
        conf = SparkConf().setAppName(APP_NAME)
        sc = SparkContext(conf=conf)

        spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
        testdf = spark.createDataFrame(data=tupledata,schema=header)
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
        
        #Model application to validation data set.
        columns= ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide",\
                        "total_sulfur_dioxide","density","pH","sulphates","alcohol"]
        vatestdf = VectorAssembler(inputCols=columns,outputCol='features')
        testdf = vatestdf.transform(testdf)

        s3model = RandomForestClassificationModel.load('./views/extracted')
        rfModelTest = s3model.transform(testdf)
        predictTable=rfModelTest.select("fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide",\
                        "total_sulfur_dioxide","density","pH","sulphates","alcohol","quality","prediction").toPandas()
        predictHtml=predictTable.to_html()

        #Model Evaluation for stats like F1-Score and Accuracy.
        
        evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
        s3rfaccuracy = evaluator.evaluate(rfModelTest) 
        print("*****************************************************************************************************")
        print("F1 Score for Random Forest Classifier for validation dataset for s3 saved model = %g " % s3rfaccuracy)
        print("*****************************************************************************************************")
        score=s3rfaccuracy
        sc.stop()

    return render_template("upload.html", score=score, table=predictHtml)
