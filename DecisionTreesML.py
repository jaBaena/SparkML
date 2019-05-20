from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":

    # Create the SparkSession
    spark_session = SparkSession\
        .builder\
        .appName("Spark ML decision tree example")\
        .master("local[4]")\
        .getOrCreate()

    # Read the dataset
    data_frame = spark_session\
        .read\
        .format("libsvm")\
        .load("data/classificationDataLibsvm.txt")

    data_frame.show()

    # 70% for training and 30% for testing
    (training_data, test_data) = data_frame.randomSplit([0.7, 0.3])

    print("training data: " + str(training_data.count()))
    training_data.printSchema()
    training_data.show()

    print("test data: " + str(test_data.count()))
    test_data.printSchema()
    test_data.show()

    # Create the  decision_tree model
    decision_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # Fit the model
    model = decision_tree.fit(training_data)

    # Make a prediction
    prediction = model.transform(test_data)
    prediction.printSchema()
    prediction.show()

    # We can see the difference between the prediction and the real values
    prediction\
        .select("prediction", "label", "features")\
        .show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    print("Test Error = %g " % (1.0 - accuracy))

    # Stop the SparkSession
    spark_session.stop()
