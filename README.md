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

Set up your AWS credentials. This project uses AWS S3 to store and retrieve data and models. You can set up your AWS credentials using AWS CLI or by configuring environment variables.

Usage
Training the Model

To train the Random Forest classification model, run the train_model.py script. This script reads the training data from an AWS S3 bucket, preprocesses it, trains the model, and saves the trained model back to S3.

python train_model.py

Evaluating the Model

To evaluate the trained model on a test dataset, run the evaluate_model.py script. This script retrieves the trained model from S3, reads the test data from S3 or a local file, applies the model to make predictions, and evaluates the performance of the model.
python evaluate_model.py [optional_test_data_filename]

If you provide a test data filename as an argument, the script will use that file for evaluation. Otherwise, it will use the default test dataset stored in the S3 bucket.

Note

Ensure that you have configured your AWS credentials properly to access the S3 bucket where the data and models are stored. Also, make sure to replace the placeholder values in the scripts (e.g., bucket names, file paths) with your actual AWS S3 bucket information.


