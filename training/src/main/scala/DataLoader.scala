// ============================================================
// DataLoader.scala — Load & explore the Telco Churn dataset
// ============================================================
// In web dev terms, this is like a "data access layer" or a
// database service module. It reads the raw CSV, cleans it,
// and returns a typed DataFrame ready for ML processing.
//
// Key Scala concepts used here:
//   object  = a singleton (like a static class / module)
//   def     = function definition
//   val     = const (immutable variable)
//   Option  = nullable wrapper (like TypeScript's `string | null`)
//   =>      = arrow function (like JS arrow =>)
// ============================================================

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object DataLoader {

  /**
   * Load the Telco Customer Churn CSV and return a cleaned DataFrame.
   *
   * What this does:
   *   1. Reads the CSV file (like fs.readFileSync but for big data)
   *   2. Fixes the TotalCharges column (it's a string in the raw CSV!)
   *   3. Converts the Churn column from "Yes"/"No" to 1.0/0.0
   *   4. Drops the customerID column (it's not a feature)
   *
   * @param spark  The SparkSession (like a database connection pool)
   * @param path   Path to the CSV file
   * @return       Cleaned DataFrame ready for preprocessing
   */
  def loadAndClean(spark: SparkSession, path: String): DataFrame = {
    println("\n" + "=" * 60)
    println("STEP 1: Loading dataset")
    println("=" * 60)

    // Read CSV with headers and automatic type inference
    // This is like: const data = Papa.parse(csv, { header: true })
    val rawDf = spark.read
      .option("header", "true")       // first row is column names
      .option("inferSchema", "true")   // auto-detect types (int, string, etc.)
      .csv(path)

    println(s"  Loaded ${rawDf.count()} rows, ${rawDf.columns.length} columns")
    println("\n  Schema (column types):")
    rawDf.printSchema()

    // --- Fix TotalCharges ---
    // In the raw CSV, TotalCharges is stored as a STRING (not a number!)
    // because some rows have " " (space) instead of a number.
    // We need to: cast to Double → replace nulls with 0.0
    //
    // In JS terms: parseFloat(row.TotalCharges) || 0
    val cleanedDf = rawDf
      .withColumn("TotalCharges",
        when(col("TotalCharges") === " ", lit(0.0))
          .otherwise(col("TotalCharges").cast(DoubleType))
      )
      // Drop customerID — it's just an ID, not a predictive feature
      // Like removing _id from a MongoDB document before analysis
      .drop("customerID")

    println("  ✅ TotalCharges converted to Double (spaces → 0.0)")
    println("  ✅ customerID column dropped")

    cleanedDf
  }

  /**
   * Print exploration statistics about the dataset.
   * Returns a summary string that we'll log as an MLflow artifact.
   *
   * This is Phase 2a — understanding the data before building models.
   */
  def explore(df: DataFrame): String = {
    println("\n" + "=" * 60)
    println("STEP 2: Data Exploration")
    println("=" * 60)

    val totalRows = df.count()
    val totalCols = df.columns.length

    // --- Class distribution ---
    // How many customers churned vs stayed?
    // This tells us if the dataset is "imbalanced" (e.g., 90% No, 10% Yes)
    println("\n  Churn Distribution:")
    val churnDist = df.groupBy("Churn").count()
      .withColumn("percentage", round(col("count") / lit(totalRows) * 100, 2))
    churnDist.show()

    // Collect distribution for the summary
    val distRows = churnDist.collect()
    val distSummary = distRows.map { row =>
      s"    ${row.getString(0)}: ${row.getLong(1)} (${row.getDouble(2)}%)"
    }.mkString("\n")

    // --- Null counts per column ---
    println("  Null counts per column:")
    val nullCounts = df.columns.map { colName =>
      val nullCount = df.filter(col(colName).isNull || col(colName) === "").count()
      (colName, nullCount)
    }
    val nullSummary = nullCounts.map { case (name, count) =>
      f"    $name%-25s $count%d"
    }.mkString("\n")
    nullCounts.foreach { case (name, count) =>
      if (count > 0) println(f"    $name%-25s $count%d nulls")
    }
    if (nullCounts.forall(_._2 == 0)) println("    No nulls found! ✅")

    // --- Numeric feature statistics ---
    println("\n  Numeric Feature Statistics:")
    val numericCols = Seq("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")
    df.select(numericCols.map(col): _*).describe().show()

    // --- Sample rows ---
    println("  Sample rows (first 5):")
    df.show(5, truncate = false)

    // Build summary text for MLflow artifact
    val summary = s"""
      |============================================================
      |Dataset Exploration Summary
      |============================================================
      |Total rows:    $totalRows
      |Total columns: $totalCols
      |
      |Churn Distribution:
      |$distSummary
      |
      |Null Counts:
      |$nullSummary
      |
      |Columns: ${df.columns.mkString(", ")}
      |
      |Numeric columns: ${numericCols.mkString(", ")}
      |Categorical columns: ${df.columns.diff(numericCols :+ "Churn").mkString(", ")}
      |============================================================
    """.stripMargin

    summary
  }
}
