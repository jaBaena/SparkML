from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import time

if __name__ == "__main__":

    # Create the SparkSession
    spark_session = SparkSession\
        .builder\
        .appName("Spark ML decision tree example with performance marks")\
        .master("local[4]")\
        .getOrCreate()

    # Take the initial time
    initial_time = time.time()

    # Read the dataset (it is randomly created for testing the performance)
    data_frame = spark_session\
        .read\
        .format("libsvm")\
        .load("data/classification.100000.4.txt")

    # 70% for training and 30% for testing
    (training_data, test_data) = data_frame.randomSplit([0.7, 0.3])

    # Calculate the time the machine spend before training
    time_before_training = time.time()
    print("Time before training: " + str(time_before_training - initial_time))

    # Create the  decision_tree model
    decision_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # Fit the model
    model = decision_tree.fit(training_data)

    # Calculate the time the machine is training
    training_time = time.time()
    print("Training time: " + str(training_time - time_before_training))

    # Make the prediction
    prediction = model.transform(test_data)

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
