import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession

if __name__ == "__main__":
    sparkSession = SparkSession\
        .builder\
        .getOrCreate()

    # Python list
    dense_vector1 = [1.0, 0.0, 3.5, 0.0, 5.1]

    # NumPy array
    dense_vector2 = np.array([1.0, 0.0, 3.5, 0.0, 5.1])

    # Vector
    dense_vector3 = Vectors.dense([1.0, 0.0, 3.5, 0.0, 5.1])

    sparse_vector = Vectors.sparse(5, [0, 2, 4], [1.0, 3.5, 5.1])

    print("Vector 1 (Python list) : " + str(dense_vector1))
    print("Vector 2 (NumPy Array) : " + str(dense_vector2))
    print("Vector 3 (Vectors) : " + str(dense_vector3))
    print("Vector 1 (Vectors): " + str(sparse_vector))

    labeled_point = LabeledPoint(1.0, dense_vector1)
    labeled_point2 = LabeledPoint(0.0, Vectors.sparse(5, [2, 4], [5.2, 6.2]))

    print("Labeled point (Python list): " + str(labeled_point))
    print("Labeled point (Sparse vector): " + str(labeled_point2))

    sparkSession.stop()