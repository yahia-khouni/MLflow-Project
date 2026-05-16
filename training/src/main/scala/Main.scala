// ============================================================
// Main.scala — Entry point & orchestrator
// ============================================================
// This is like your index.js or app.ts — it wires everything
// together and runs the full training pipeline:
//   1. Create SparkSession (connect to Spark)
//   2. Load & explore data
//   3. Split into train/test
//   4. Train 3 algorithms with MLflow tracking
//   5. Compare results and identify the champion
//
// Run with: sbt run
// Or with args: sbt "run --data /path/to/data.csv --mlflow http://localhost:5000"
// ============================================================

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification._
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.mlflow.tracking.MlflowClient

object Main {

  def main(args: Array[String]): Unit = {
    println("""
      |╔══════════════════════════════════════════════════════════╗
      |║   ML Production Pipeline — Churn Prediction Training    ║
      |║   Phase 2: Spark MLlib + MLflow Instrumentation         ║
      |╚══════════════════════════════════════════════════════════╝
    """.stripMargin)

    // --- Parse configuration ---
    // Default values work for local development with Docker services
    val dataPath = getArg(args, "--data", "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    val mlflowUri = getArg(args, "--mlflow", "http://localhost:5000")

    println(s"  Data path:     $dataPath")
    println(s"  MLflow URI:    $mlflowUri")

    // --- Set MinIO/S3 credentials for MLflow artifact storage ---
    // These must be set as system properties so the MLflow client
    // can upload artifacts to MinIO. Same values as in .env
    System.setProperty("AWS_ACCESS_KEY_ID",
      sys.env.getOrElse("AWS_ACCESS_KEY_ID", "minioadmin"))
    System.setProperty("AWS_SECRET_ACCESS_KEY",
      sys.env.getOrElse("AWS_SECRET_ACCESS_KEY", "minioadmin_secret_2024"))
    System.setProperty("MLFLOW_S3_ENDPOINT_URL",
      sys.env.getOrElse("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))

    // Set the env vars that the MLflow client reads
    // (system properties alone aren't enough for all code paths)
    val envField = System.getenv().getClass.getDeclaredField("m")
    envField.setAccessible(true)
    val envMap = envField.get(System.getenv()).asInstanceOf[java.util.Map[String, String]]
    envMap.put("AWS_ACCESS_KEY_ID", sys.env.getOrElse("AWS_ACCESS_KEY_ID", "minioadmin"))
    envMap.put("AWS_SECRET_ACCESS_KEY", sys.env.getOrElse("AWS_SECRET_ACCESS_KEY", "minioadmin_secret_2024"))
    envMap.put("MLFLOW_S3_ENDPOINT_URL", sys.env.getOrElse("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))

    // ============================================================
    // STEP 1: Create SparkSession
    // ============================================================
    // SparkSession is your "connection" to Spark — like creating
    // a database connection pool in Express.js:
    //   const pool = new Pool({ connectionString: "..." })
    //
    // .master("local[*]") = use all CPU cores on this machine
    // In production, you'd use "spark://spark-master:7077"
    // ============================================================

    // Set HADOOP_HOME before creating SparkSession to avoid winutils errors
    val hadoopDir = new java.io.File("../hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopDir)

    println("\n" + "=" * 60)
    println("INITIALIZING SPARK")
    println("=" * 60)

    val spark = SparkSession.builder()
      .appName("ChurnPrediction-Training")
      .master("local[*]")  // Use all local cores
      .config("spark.driver.memory", "4g")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .config("spark.ui.enabled", "false")  // disable Spark UI to avoid port conflicts
      .getOrCreate()

    // Reduce Spark's verbose logging (it's VERY chatty by default)
    spark.sparkContext.setLogLevel("WARN")
    println("  ✅ SparkSession created (local mode)")

    // ============================================================
    // STEP 2: Connect to MLflow
    // ============================================================
    println(s"\n  Connecting to MLflow at $mlflowUri...")
    val mlflowClient = new MlflowClient(mlflowUri)
    println("  ✅ MLflow client connected")

    // ============================================================
    // STEP 3: Load & Explore Data
    // ============================================================
    val df = DataLoader.loadAndClean(spark, dataPath)
    val explorationSummary = DataLoader.explore(df)

    // Log the exploration summary as an MLflow artifact
    val exploreExpId = Trainer.getOrCreateExperiment(mlflowClient, "churn-exploration")
    val exploreRun = mlflowClient.createRun(exploreExpId)
    val exploreDir = new java.io.File(s"${Trainer.artifactDir}/exploration")
    exploreDir.mkdirs()
    val summaryFile = new java.io.File(exploreDir, "exploration_summary.txt")
    val pw = new java.io.PrintWriter(summaryFile)
    pw.print(explorationSummary)
    pw.close()
    mlflowClient.logArtifact(exploreRun.getRunId, summaryFile)
    mlflowClient.setTerminated(exploreRun.getRunId,
      org.mlflow.api.proto.Service.RunStatus.FINISHED, System.currentTimeMillis())
    println("  ✅ Exploration summary logged to MLflow (experiment: churn-exploration)")

    // ============================================================
    // STEP 4: Train/Test Split (80/20)
    // ============================================================
    println("\n" + "=" * 60)
    println("SPLITTING DATA: 80% train / 20% test")
    println("=" * 60)

    // seed = 42 ensures reproducibility (same split every time)
    // Like setting Math.random seed in tests
    val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 42)
    println(s"  Training set: ${trainData.count()} rows")
    println(s"  Test set:     ${testData.count()} rows")

    // Cache the data in memory for faster repeated access
    // (each model will read it multiple times during cross-validation)
    trainData.cache()
    testData.cache()

    // ============================================================
    // STEP 5: Get preprocessing stages
    // ============================================================
    val prepStages = Preprocessor.buildStages()
    val featureNames = Preprocessor.getFeatureColumnNames()

    // ============================================================
    // STEP 6: Train all three algorithms
    // ============================================================
    val results = scala.collection.mutable.ArrayBuffer[TrainingResult]()

    // --- Algorithm 1: Logistic Regression ---
    // The simplest ML algorithm. Finds a line (hyperplane) that
    // separates churners from non-churners. Fast to train.
    // Think of it as: probability = sigmoid(w1*x1 + w2*x2 + ... + b)
    println("\n" + "🔵 " * 20)
    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(100)

    val lrParamGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 0.3))         // regularization strength
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))   // L1 vs L2 regularization mix
      .build()

    results += Trainer.trainAndLog(
      "Logistic Regression", "churn-logistic-regression",
      trainData, testData, prepStages, lr, lrParamGrid, mlflowClient, featureNames
    )

    // --- Algorithm 2: Random Forest ---
    // Builds many decision trees (like flowcharts) on random subsets
    // of the data, then takes a vote. More robust than a single tree.
    // Like asking 100 experts and going with the majority opinion.
    println("\n" + "🟢 " * 20)
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(42)

    val rfParamGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(50))           // reduced grid for speed
      .addGrid(rf.maxDepth, Array(5))              // reduced grid for speed
      .build()

    results += Trainer.trainAndLog(
      "Random Forest", "churn-random-forest",
      trainData, testData, prepStages, rf, rfParamGrid, mlflowClient, featureNames
    )

    // --- Algorithm 3: Gradient Boosted Trees ---
    // Builds trees sequentially — each new tree tries to fix the
    // mistakes of the previous ones. Usually the most accurate
    // but slowest to train.
    // Like iteratively improving a draft: v1 → v2 → v3 → ...
    println("\n" + "🟠 " * 20)
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(42)

    val gbtParamGrid = new ParamGridBuilder()
      .addGrid(gbt.maxIter, Array(20, 50))            // number of boosting rounds
      .addGrid(gbt.maxDepth, Array(3, 5))             // tree depth (keep shallow for speed)
      .build()

    results += Trainer.trainAndLog(
      "Gradient Boosted Trees", "churn-gradient-boosted-trees",
      trainData, testData, prepStages, gbt, gbtParamGrid, mlflowClient, featureNames
    )

    // ============================================================
    // STEP 7: Compare results & identify the champion
    // ============================================================
    println("\n" + "=" * 60)
    println("FINAL COMPARISON — All Algorithms")
    println("=" * 60)
    println(f"\n  ${"Algorithm"}%-25s ${"AUC-ROC"}%-10s ${"Accuracy"}%-10s ${"F1"}%-10s ${"Time(s)"}%-10s")
    println("  " + "-" * 65)

    results.foreach { r =>
      println(f"  ${r.algorithmName}%-25s ${r.aucRoc}%.4f     ${r.accuracy}%.4f     ${r.f1Score}%.4f     ${r.durationSeconds}%.1f")
    }

    val champion = results.maxBy(_.aucRoc)
    println(s"\n  🏆 CHAMPION: ${champion.algorithmName}")
    println(f"     AUC-ROC = ${champion.aucRoc}%.4f")
    println(s"     MLflow Run ID: ${champion.runId}")
    println(s"\n  View all experiments at: $mlflowUri")

    // ============================================================
    // STEP 8: Register champion in Model Registry (Phase 3)
    // ============================================================
    val modelName = "churn-predictor"
    val versionNum = ModelRegistry.registerChampion(
      mlflowUri, champion.runId, modelName, champion
    )

    // Validate and promote to Production (AUC > 0.80)
    ModelRegistry.promoteToProduction(
      mlflowUri, modelName, versionNum, champion.aucRoc
    )

    // ============================================================
    // Cleanup
    // ============================================================
    trainData.unpersist()
    testData.unpersist()
    spark.stop()

    println("\n  ✅ Full pipeline completed (Phase 2 + Phase 3)!")
    println(s"  Model '$modelName' v$versionNum is registered and ready for serving.")
  }

  /** Parse command-line arguments (simple key-value pairs) */
  private def getArg(args: Array[String], key: String, default: String): String = {
    val idx = args.indexOf(key)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default
  }
}
