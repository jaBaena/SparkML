import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class DataTypes {
  public static void main(String[] args) {
    Logger.getLogger("org").setLevel(Level.OFF) ;

    // Create a SparkSession.
    SparkSession sparkSession = SparkSession
            .builder()
            .appName("SparkML datatypes")
            .master("local[2]")
            .getOrCreate();

    Vector denseVector = new DenseVector(new double[]{1.0, 0.0, 3.2}) ;
    Vector denseVector2 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0);

    Vector sparseVector = new SparseVector(3, new int[]{0, 2}, new double[]{1.0, 3.5}) ;
    Vector sparseVector2 = Vectors.sparse(3, new int[]{0, 2}, new double[]{1.0, 3.5}) ;

    System.out.println("Vector 1 (dense): " + denseVector) ;
    System.out.println("Vector 2 (dense): " + denseVector2) ;
    System.out.println("Vector 1 (sparse): " + sparseVector) ;
    System.out.println("Vector 2 (sparse): " + sparseVector2) ;

    LabeledPoint labeledPoint = new LabeledPoint(1.0, denseVector) ;
    LabeledPoint labeledPoint2 = new LabeledPoint(0.0, Vectors.sparse(5, new int[]{2, 4}, new double[]{5.2, 6.2})) ;

    System.out.println("Labeled point 1: " + labeledPoint) ;
    System.out.println("Labeled point 2: " + labeledPoint2) ;

    sparkSession.stop();
  }
}
