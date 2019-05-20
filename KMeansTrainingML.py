from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import time

if __name__ == "__main__":

    # Create the SparkSession
    spark_session = SparkSession\
        .builder\
        .appName("Spark ML KMeans example")\
        .master("local[4]")\
        .getOrCreate()

    # Take the initial time
    initial_time = time.time()

    # Read the dataset
    data_frame = spark_session\
        .read\
        .format("libsvm")\
        .load("data/classificationDataLibsvm.txt")

    # 70% for training and 30% for testing
    (training_data, test_data) = data_frame.randomSplit([0.7, 0.3])

    # Calculate the time the machine spend before training
    time_before_training = time.time()
    print("Time before training: " + str(time_before_training - initial_time))

    # Create the KMeans model with K = 2
    kmeans = KMeans\
        .setK(value=2)\
        .setSeed(1)

    # Fit the model
    kmeans_model = kmeans.fit(training_data)

    # Calculate the time the machine is training
    training_time = time.time()
    print("Training time: " + str(training_time - time_before_training))

    # Make the prediction
    prediction = kmeans_model.transform(test_data)

    # Calculate the time the machine is testing with the model fitted
    testing_time = time.time()
    print("Test time: " + str(testing_time - training_time))

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    print("Test Error = %g " % (1.0 - accuracy))

    # Stop the SparkSession
    spark_session.stop()
