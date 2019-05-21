from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import time

# Dataset found in: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#phishing


if __name__ == "__main__":

    # Create the SparkSession.
    spark_session = SparkSession\
        .builder\
        .appName("Spark ML Random Forest example")\
        .master("local[4]")\
        .getOrCreate()

    # Define the initial time.
    initial_time = time.time()

    # Load the data with 'libsvm' format.
    data_frame = spark_session\
        .read\
        .format("libsvm")\
        .load("data/phishing")

    # Split the dataset into training data and testing data. We use the 75% for training.
    (training_data, test_data) = data_frame.randomSplit([0.75, 0.25])

    # Let's see the quantity of training data, and another visualizations of it.
    print("training data: " + str(training_data.count()))
    training_data.printSchema()
    training_data.show()

    # The same for the testing data.
    print("test data: " + str(test_data.count()))
    test_data.printSchema()
    test_data.show()

    # Check the performance of the time before training, the training time (line 50), and the testing time (line 55).
    time_before_training = time.time()
    print("Time before training: " + str(time_before_training - initial_time))

    # Select the model
    random_forest = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)

    # Train the model with the training data.
    model = random_forest.fit(training_data)
    training_time = time.time()
    print("Training time: " + str(training_time - time_before_training))

    # Make the predictions with testing data.
    prediction = model.transform(test_data)
    testing_time = time.time()
    print("Test time: " + str(testing_time - training_time))

    # Visualizations of predictions.
    prediction.printSchema()
    prediction.show()

    # Select (prediction, true label) and compute test error.
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    print("Test Error = %g " % (1.0 - accuracy))

    # As we can see we have great results with this model we could save it with the next snippet:
    # model.save("RF20Trees")

    # And for load that model we need the following snippet:
    # model = RandomForestClassifier.load("RF20Trees")

    # It is important to stop the SparkSession.
    spark_session.stop()
