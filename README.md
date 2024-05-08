ucid - sj626
Wine Quality Classification with Apache Spark

This project demonstrates how to perform wine quality classification using Apache Spark. The classification model is trained using a Random Forest classifier, and the trained model is then used to predict the quality of wine samples.
Prerequisites

    Python 3.x
    Apache Spark
    pyspark
    boto3 (for AWS S3 interaction)
    tarfile (for working with tar.gz files)

Setup

    Install the required Python packages:
    pip install pyspark boto3
Ensure you have Apache Spark installed. You can download it from the Apache Spark website and follow the installation instructions.

Clone this repository:
git clone https://github.com/example/wine-quality-classification.git

Set up the AWS EC2 instances:

Launch 4 EC2 instances for model training and 1 EC2 instance for prediction.

Install Java, Spark, and Docker on all instances.

Prepare the dataset:

Upload the TrainingDataset.csv and ValidationDataset.csv files to AWS S3.

Ensure that the EC2 instances have the necessary permissions to access the S3 bucket.

Develop the Spark ML model training application:

Create a new Maven project in your preferred IDE.

Add the necessary Spark dependencies to the pom.xml file.

Implement the Spark ML model training code in Java:

Usage
Training the Model

To train the Random Forest classification model, run the train_model.py script. This script reads the training data from an AWS S3 bucket, preprocesses it, trains the model, and saves the trained model back to S3.

python train_model.py

Evaluating the Model


Ensure that you have configured your AWS credentials properly to access the S3 bucket where the data and models are stored. Also, make sure to replace the placeholder values in the scripts (e.g., bucket names, file paths) with your actual AWS S3 bucket information.
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityTraining {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("WineQualityTraining");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("WineQualityTraining").getOrCreate();

        // Load the training dataset from S3
        Dataset<Row> trainingData = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("s3://<your-s3-bucket>/TrainingDataset.csv");

        // Prepare the feature columns
        String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");

        // Split the data into training and validation sets
        Dataset<Row>[] splits = trainingData.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainData = splits[0];
        Dataset<Row> validationData = splits[1];

        // Create a LogisticRegression model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily("multinomial")
                .setLabelCol("quality");

        // Create a pipeline with feature transformation and model
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});

        // Train the model
        PipelineModel model = pipeline.fit(trainData);

        // Evaluate the model on the validation dataset
        Dataset<Row> predictions = model.transform(validationData);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 score on validation data: " + f1Score);

        // Save the trained model
        model.write().overwrite().save("s3://<your-s3-bucket>/wineQualityModel");

        jsc.close();
    }
}
Package the application into a JAR file.
Step - 2
Train the model on AWS EC2 instances:

Copy the application JAR file to all 4 EC2 instances.

SSH into each EC2 instance and submit the Spark job using the spark-submit command:

spark-submit --class WineQualityTraining --master spark://<master-node-ip>:7077 <application-jar-file>
he model training will be distributed across the 4 EC2 instances.

Develop the Spark ML model prediction application:

Create a new Maven project for the prediction application.

Add the necessary Spark dependencies to the pom.xml file.

Implement the Spark ML model prediction code in Java:
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("WineQualityPrediction");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("WineQualityPrediction").getOrCreate();

        // Load the trained model
        PipelineModel model = PipelineModel.load("s3://<your-s3-bucket>/wineQualityModel");

        // Load the test dataset from S3
        Dataset<Row> testData = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("s3://<your-s3-bucket>/TestDataset.csv");

        // Make predictions on the test dataset
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model performance on the test dataset
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 score on test data: " + f1Score);

        jsc.close();
    }
}
Package the application into a JAR file.
Step - 3 
Create a Docker container for the prediction application:

Create a Dockerfile in the prediction application project

FROM openjdk:8-jre-slim
COPY target/<prediction-application-jar-file> /app/
CMD ["java", "-jar", "/app/<prediction-application-jar-file>"]
Build the Docker image

docker build -t wine-quality-prediction .
Push the Docker image to a container registry (e.g., Amazon ECR).

Run the prediction application on a single EC2 instance:

SSH into the EC2 instance.

Pull the Docker image from the container registry.

Run the Docker container:

docker run -it wine-quality-prediction
Explanation:
The prediction application will load the trained model from S3, make predictions on the test dataset, and display the F1 score.
The model training will be distributed across 4 EC2 instances, leveraging Spark's parallel processing capabilities.



The trained model will be saved to S3 for later use in the prediction application.



The prediction application will be packaged as a Docker container, making it easy to deploy and run on any environment.



The F1 score will be displayed as the output of the prediction application, indicating the performance of the trained model on the test dataset.


