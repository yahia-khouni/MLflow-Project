// ============================================================
// Preprocessor.scala — Feature Engineering Pipeline
// ============================================================
// This builds a Spark ML Pipeline that transforms raw data
// into the numeric format that ML algorithms need.
//
// Think of it like a data transformation middleware chain:
//   Raw strings → Indexed numbers → One-hot vectors → 
//   Single feature vector → Scaled features
//
// It's like Express.js middleware:
//   app.use(parseJSON)      → StringIndexer
//   app.use(validateInput)  → OneHotEncoder
//   app.use(normalizeData)  → VectorAssembler + StandardScaler
//
// Spark ML Pipeline stages are applied in order, just like
// middleware. Each stage reads from input columns and writes
// to output columns.
// ============================================================

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._

object Preprocessor {

  // All the categorical (string) columns in the dataset.
  // These need to be converted to numbers because ML algorithms
  // can only work with numbers, not strings like "Male"/"Female".
  val categoricalCols: Array[String] = Array(
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
  )

  // Numeric columns that go directly into the feature vector
  val numericCols: Array[String] = Array(
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
  )

  /**
   * Build the preprocessing pipeline stages.
   *
   * The pipeline has 5 types of stages:
   *
   * 1. LABEL INDEXER
   *    Converts "Churn" from "Yes"/"No" to 1.0/0.0
   *    This is the column we're trying to predict (the "label").
   *
   * 2. STRING INDEXERS (one per categorical column)
   *    Converts each categorical string to a number.
   *    Example: "Male" → 0, "Female" → 1
   *    (Like creating an enum: { Male: 0, Female: 1 })
   *
   * 3. ONE-HOT ENCODERS (one per indexed column)
   *    Converts the number to a binary vector.
   *    Example: gender=0 → [1,0], gender=1 → [0,1]
   *    Why? Because "Male"=0, "Female"=1 implies Female > Male,
   *    but that's meaningless. One-hot encoding treats them
   *    as independent categories with no ordering.
   *
   * 4. VECTOR ASSEMBLER
   *    Combines ALL feature columns into ONE single vector.
   *    ML algorithms expect a single "features" column.
   *    Like: { age: 25, income: 50k } → [25, 50000]
   *
   * 5. STANDARD SCALER
   *    Normalizes values so they're on the same scale.
   *    Without scaling, "tenure"=72 and "TotalCharges"=8000
   *    would make the model think TotalCharges is 100x more
   *    important just because the numbers are bigger.
   *
   * @return Array of PipelineStage to be used in a Pipeline
   */
  def buildStages(): Array[org.apache.spark.ml.PipelineStage] = {
    println("\n" + "=" * 60)
    println("STEP 3: Building Preprocessing Pipeline")
    println("=" * 60)

    // --- Stage 1: Label Indexer ---
    // Convert target column Churn ("Yes"/"No") to numeric (1.0/0.0)
    val labelIndexer = new StringIndexer()
      .setInputCol("Churn")
      .setOutputCol("label")
      .setHandleInvalid("keep")  // don't crash on unseen values
    println("  ✅ Label indexer: Churn → label (Yes/No → 1.0/0.0)")

    // --- Stage 2: String Indexers for each categorical column ---
    // Each one creates a new column: "gender" → "gender_idx"
    val stringIndexers = categoricalCols.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_idx")
        .setHandleInvalid("keep")
    }
    println(s"  ✅ String indexers: ${categoricalCols.length} categorical columns")

    // --- Stage 3: One-Hot Encoders ---
    // Each one creates: "gender_idx" → "gender_vec"
    val oneHotEncoders = categoricalCols.map { colName =>
      new OneHotEncoder()
        .setInputCol(s"${colName}_idx")
        .setOutputCol(s"${colName}_vec")
        .setDropLast(true)  // avoid multicollinearity (dummy variable trap)
    }
    println(s"  ✅ One-hot encoders: ${categoricalCols.length} columns")

    // --- Stage 4: Vector Assembler ---
    // Combine all features into a single vector column called "rawFeatures"
    // Input: all one-hot encoded columns + numeric columns
    val featureCols: Array[String] =
      categoricalCols.map(c => s"${c}_vec") ++ numericCols

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("rawFeatures")
      .setHandleInvalid("skip")  // skip rows with nulls
    println(s"  ✅ Vector assembler: ${featureCols.length} columns → rawFeatures")

    // --- Stage 5: Standard Scaler ---
    // Normalize features to mean=0, stddev=1
    val scaler = new StandardScaler()
      .setInputCol("rawFeatures")
      .setOutputCol("features")    // this is what the model reads
      .setWithStd(true)            // divide by standard deviation
      .setWithMean(true)           // subtract the mean
    println("  ✅ Standard scaler: rawFeatures → features (normalized)")

    // Combine all stages into an ordered array
    // Pipeline will execute them in this exact order
    val allStages: Array[org.apache.spark.ml.PipelineStage] =
      Array(labelIndexer) ++ stringIndexers ++ oneHotEncoders ++ Array(assembler, scaler)

    println(s"\n  Pipeline has ${allStages.length} stages total")
    allStages
  }

  /**
   * Get the feature column names (for logging to MLflow).
   */
  def getFeatureColumnNames(): Array[String] = {
    categoricalCols.map(c => s"${c}_vec") ++ numericCols
  }
}
