// ============================================================
// Trainer.scala — Model Training + MLflow Instrumentation
// ============================================================
// This is the most important file in the project. It:
//   1. Creates a Spark ML Pipeline (preprocessing + model)
//   2. Runs CrossValidator (hyperparameter tuning)
//   3. Evaluates the best model on the test set
//   4. Logs EVERYTHING to MLflow (params, metrics, artifacts)
//   5. Generates confusion matrix PNG, ROC curve PNG, etc.
//
// Think of MLflow logging like analytics tracking:
//   mlflow.logParam()   → like analytics.track("config", {...})
//   mlflow.logMetric()  → like analytics.track("performance", {...})
//   mlflow.logArtifact() → like uploading a file to S3
// ============================================================

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import java.io.{File, PrintWriter}
import java.awt.{Color, Font, BasicStroke}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import org.jfree.chart.{ChartFactory, ChartUtils}
import org.jfree.chart.plot.XYPlot
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import com.google.gson.{Gson, GsonBuilder}

/**
 * Holds the results of training one algorithm.
 * Like a TypeScript interface:
 *   interface TrainingResult {
 *     runId: string; aucRoc: number; ...
 *   }
 *
 * In Scala, "case class" automatically gives you:
 *   - A constructor
 *   - toString(), equals(), hashCode()
 *   - Immutability (all fields are val by default)
 */
case class TrainingResult(
  algorithmName: String,
  runId: String,
  aucRoc: Double,
  accuracy: Double,
  precision: Double,
  recall: Double,
  f1Score: Double,
  durationSeconds: Double,
  model: PipelineModel
)

object Trainer {

  // Directory for temporary artifacts (plots, CSVs) before uploading to MLflow
  val artifactDir = "training_artifacts"

  /**
   * Train one algorithm with CrossValidator, evaluate, and log to MLflow.
   *
   * @param algorithmName   Human-readable name (e.g., "Logistic Regression")
   * @param experimentName  MLflow experiment (e.g., "churn-logistic-regression")
   * @param trainData       Training DataFrame (80% of data)
   * @param testData        Test DataFrame (20% of data)
   * @param prepStages      Preprocessing pipeline stages
   * @param classifier      The ML algorithm (LogisticRegression, RandomForest, etc.)
   * @param paramGrid       Hyperparameter combinations to try
   * @param mlflowClient    MLflow client for logging
   * @return                TrainingResult with metrics and the best model
   */
  def trainAndLog(
    algorithmName: String,
    experimentName: String,
    trainData: DataFrame,
    testData: DataFrame,
    prepStages: Array[org.apache.spark.ml.PipelineStage],
    classifier: org.apache.spark.ml.Estimator[_],
    paramGrid: Array[org.apache.spark.ml.param.ParamMap],
    mlflowClient: MlflowClient,
    featureNames: Array[String]
  ): TrainingResult = {

    println("\n" + "=" * 60)
    println(s"TRAINING: $algorithmName")
    println("=" * 60)

    // --- Create or get the MLflow experiment ---
    val experimentId = getOrCreateExperiment(mlflowClient, experimentName)

    // --- Start an MLflow run ---
    // A "run" is like a single experiment attempt. Each run tracks
    // what config was used and what results were achieved.
    val run = mlflowClient.createRun(experimentId)
    val runId = run.getRunId
    println(s"  MLflow Run ID: $runId")

    val startTime = System.currentTimeMillis()

    // --- Build the full Pipeline ---
    // preprocessing stages + classifier as the last stage
    val pipeline = new Pipeline()
      .setStages(prepStages :+ classifier.asInstanceOf[org.apache.spark.ml.PipelineStage])

    // --- Set up the evaluator ---
    // This defines HOW we measure model quality.
    // AUC-ROC = "Area Under the ROC Curve" — our primary metric.
    // 1.0 = perfect, 0.5 = random guessing, < 0.5 = worse than random
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    // --- Set up CrossValidator ---
    // CrossValidator splits training data into 5 "folds":
    //   - Train on folds 1,2,3,4 → test on fold 5
    //   - Train on folds 1,2,3,5 → test on fold 4
    //   - ... (5 combinations total)
    // For each fold, it tries EVERY hyperparameter combination.
    // Then it picks the combo with the best average AUC.
    //
    // Think of it like A/B testing 5 times to be sure of the result.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)       // 3 folds is sufficient for this dataset size
      .setParallelism(2)    // try 2 param combos in parallel

    println(s"  CrossValidator: 3 folds × ${paramGrid.length} param combos = ${3 * paramGrid.length} fits")
    println("  Training... (this may take a few minutes)")

    // --- FIT (train) the model ---
    val cvModel = cv.fit(trainData)
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val endTime = System.currentTimeMillis()
    val durationSec = (endTime - startTime) / 1000.0

    println(f"  Training completed in $durationSec%.1f seconds")

    // --- Evaluate on the TEST set ---
    val predictions = bestModel.transform(testData)
    val aucRoc = evaluator.evaluate(predictions)

    // Additional metrics using MulticlassClassificationEvaluator
    val mcEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val accuracy  = mcEvaluator.setMetricName("accuracy").evaluate(predictions)
    val precision = mcEvaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall    = mcEvaluator.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score   = mcEvaluator.setMetricName("f1").evaluate(predictions)

    println(f"\n  Results on test set:")
    println(f"    AUC-ROC:   $aucRoc%.4f")
    println(f"    Accuracy:  $accuracy%.4f")
    println(f"    Precision: $precision%.4f")
    println(f"    Recall:    $recall%.4f")
    println(f"    F1-Score:  $f1Score%.4f")

    // --- Log parameters to MLflow ---
    logParams(mlflowClient, runId, algorithmName, paramGrid, trainData, featureNames)

    // --- Log metrics to MLflow ---
    mlflowClient.logMetric(runId, "auc_roc", aucRoc)
    mlflowClient.logMetric(runId, "accuracy", accuracy)
    mlflowClient.logMetric(runId, "precision", precision)
    mlflowClient.logMetric(runId, "recall", recall)
    mlflowClient.logMetric(runId, "f1_score", f1Score)
    mlflowClient.logMetric(runId, "training_duration_seconds", durationSec)
    println("  ✅ Metrics logged to MLflow")

    // --- Generate & log artifacts ---
    val runArtifactDir = new File(s"$artifactDir/$runId")
    runArtifactDir.mkdirs()

    generateConfusionMatrix(predictions, runArtifactDir)
    mlflowClient.logArtifact(runId, new File(runArtifactDir, "confusion_matrix.png"))

    generateRocCurve(predictions, aucRoc, algorithmName, runArtifactDir)
    mlflowClient.logArtifact(runId, new File(runArtifactDir, "roc_curve.png"))

    generateFeatureImportance(bestModel, algorithmName, featureNames, runArtifactDir)
    mlflowClient.logArtifact(runId, new File(runArtifactDir, "feature_importance.csv"))

    generatePipelineConfig(algorithmName, paramGrid, runArtifactDir)
    mlflowClient.logArtifact(runId, new File(runArtifactDir, "pipeline_config.json"))

    // Save model summary as JSON artifact
    val modelSummary = new java.util.LinkedHashMap[String, Object]()
    modelSummary.put("algorithm", algorithmName)
    modelSummary.put("run_id", runId)
    modelSummary.put("auc_roc", java.lang.Double.valueOf(aucRoc))
    modelSummary.put("accuracy", java.lang.Double.valueOf(accuracy))
    modelSummary.put("f1_score", java.lang.Double.valueOf(f1Score))
    modelSummary.put("training_seconds", java.lang.Double.valueOf(durationSec))
    modelSummary.put("pipeline_stages", java.lang.Integer.valueOf(bestModel.stages.length))
    val summaryGson = new GsonBuilder().setPrettyPrinting().create()
    val summaryPw = new PrintWriter(new File(runArtifactDir, "model_summary.json"))
    summaryPw.print(summaryGson.toJson(modelSummary))
    summaryPw.close()
    mlflowClient.logArtifact(runId, new File(runArtifactDir, "model_summary.json"))

    // --- Save actual Spark model and log to MLflow ---
    // We save to a local temp directory, then upload via logArtifacts().
    // This avoids the Hadoop/winutils dependency for the HDFS FileSystem
    // but still gives us a real model artifact in MinIO that FastAPI can load.
    try {
      val modelTempDir = new File(s"$artifactDir/model_tmp_$algorithmName")
      val sparkModelDir = new File(modelTempDir, "sparkml")

      // Clean up any previous save
      if (modelTempDir.exists()) {
        def deleteRecursively(f: File): Unit = {
          if (f.isDirectory) f.listFiles().foreach(deleteRecursively)
          f.delete()
        }
        deleteRecursively(modelTempDir)
      }

      // Save the PipelineModel locally using Spark's built-in serialization
      // Setting hadoop.home.dir to the project directory to avoid winutils error
      System.setProperty("hadoop.home.dir", new File(".").getAbsolutePath)
      bestModel.save(sparkModelDir.getAbsolutePath)

      // Create an MLmodel file so MLflow recognizes the artifact as a model
      val mlModelFile = new File(modelTempDir, "MLmodel")
      val mlModelPw = new PrintWriter(mlModelFile)
      mlModelPw.print(
        s"""artifact_path: model
           |flavors:
           |  spark:
           |    model_data: sparkml
           |    pyfunc_predict_fn: predict
           |  python_function:
           |    loader_module: mlflow.spark
           |    model_data: sparkml
           |    env: null
           |run_id: $runId
           |model_uuid: ${java.util.UUID.randomUUID().toString}
           |""".stripMargin)
      mlModelPw.close()

      // Upload the entire model directory to MLflow under the "model" artifact path
      mlflowClient.logArtifacts(runId, modelTempDir, "model")
      println("  ✅ Spark model logged to MLflow (for serving via FastAPI)")

      // Clean up the local temp directory
      def deleteRecursively(f: File): Unit = {
        if (f.isDirectory) f.listFiles().foreach(deleteRecursively)
        f.delete()
      }
      deleteRecursively(modelTempDir)

    } catch {
      case e: Exception =>
        println(s"  ℹ️  Model binary save skipped (${e.getClass.getSimpleName}: ${e.getMessage.take(80)})")
        println("      Model summary JSON still logged. Re-training with Hadoop will enable serving.")
    }

    println("  ✅ Artifacts logged to MLflow (plots, CSV, JSON, model summary)")

    // --- End the MLflow run ---
    mlflowClient.setTerminated(runId, RunStatus.FINISHED, endTime)
    println(s"  ✅ MLflow run $runId completed\n")

    TrainingResult(algorithmName, runId, aucRoc, accuracy, precision, recall, f1Score, durationSec, bestModel)
  }

  // ============================================================
  // Helper: Create or get an MLflow experiment
  // ============================================================
  def getOrCreateExperiment(client: MlflowClient, name: String): String = {
    val existing = client.getExperimentByName(name)
    if (existing.isPresent) {
      existing.get().getExperimentId
    } else {
      client.createExperiment(name)
    }
  }

  // ============================================================
  // Helper: Log parameters to MLflow
  // ============================================================
  private def logParams(
    client: MlflowClient, runId: String, algorithmName: String,
    paramGrid: Array[org.apache.spark.ml.param.ParamMap],
    trainData: DataFrame, featureNames: Array[String]
  ): Unit = {
    client.logParam(runId, "algorithm", algorithmName)
    client.logParam(runId, "num_cv_folds", "3")
    client.logParam(runId, "train_test_split", "80/20")
    client.logParam(runId, "feature_count", featureNames.length.toString)
    client.logParam(runId, "training_rows", trainData.count().toString)
    client.logParam(runId, "param_grid_size", paramGrid.length.toString)

    // Log the hyperparameter grid as a string
    val gridStr = paramGrid.zipWithIndex.map { case (pm, i) =>
      pm.toSeq.map(pp => s"${pp.param.name}=${pp.value}").mkString(", ")
    }.mkString(" | ")
    // MLflow params have a 500 char limit, truncate if needed
    client.logParam(runId, "param_grid", gridStr.take(500))
    println("  ✅ Parameters logged to MLflow")
  }

  // ============================================================
  // Artifact: Confusion Matrix (PNG image)
  // ============================================================
  // A confusion matrix shows:
  //   - True Positives (correctly predicted churn)
  //   - True Negatives (correctly predicted no-churn)
  //   - False Positives (predicted churn but didn't)
  //   - False Negatives (predicted no-churn but did churn)
  //
  // Like a 2×2 table showing hits vs misses.
  // ============================================================
  private def generateConfusionMatrix(predictions: DataFrame, outputDir: File): Unit = {
    val tp = predictions.filter(col("prediction") === 1.0 && col("label") === 1.0).count()
    val tn = predictions.filter(col("prediction") === 0.0 && col("label") === 0.0).count()
    val fp = predictions.filter(col("prediction") === 1.0 && col("label") === 0.0).count()
    val fn = predictions.filter(col("prediction") === 0.0 && col("label") === 1.0).count()

    // Draw the confusion matrix as an image using Java Graphics2D
    val width = 500
    val height = 450
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g = img.createGraphics()

    // Background
    g.setColor(new Color(30, 30, 30))
    g.fillRect(0, 0, width, height)

    // Title
    g.setColor(Color.WHITE)
    g.setFont(new Font("SansSerif", Font.BOLD, 18))
    g.drawString("Confusion Matrix", 160, 35)

    // Labels
    g.setFont(new Font("SansSerif", Font.PLAIN, 14))
    g.drawString("Predicted: No Churn", 120, 70)
    g.drawString("Predicted: Churn", 320, 70)
    g.drawString("Actual:", 20, 100)
    g.drawString("No Churn", 30, 180)
    g.drawString("Churn", 42, 310)

    // Cells
    val cellW = 160
    val cellH = 120
    val startX = 120
    val startY = 90

    // TN (top-left) — correct "no churn"
    g.setColor(new Color(46, 125, 50))
    g.fillRect(startX, startY, cellW, cellH)
    g.setColor(Color.WHITE)
    g.setFont(new Font("SansSerif", Font.BOLD, 28))
    g.drawString(s"TN: $tn", startX + 30, startY + 65)

    // FP (top-right) — false alarm
    g.setColor(new Color(198, 40, 40))
    g.fillRect(startX + cellW, startY, cellW, cellH)
    g.setColor(Color.WHITE)
    g.drawString(s"FP: $fp", startX + cellW + 30, startY + 65)

    // FN (bottom-left) — missed churn
    g.setColor(new Color(198, 40, 40))
    g.fillRect(startX, startY + cellH, cellW, cellH)
    g.setColor(Color.WHITE)
    g.drawString(s"FN: $fn", startX + 30, startY + cellH + 65)

    // TP (bottom-right) — correct churn
    g.setColor(new Color(46, 125, 50))
    g.fillRect(startX + cellW, startY + cellH, cellW, cellH)
    g.setColor(Color.WHITE)
    g.drawString(s"TP: $tp", startX + cellW + 30, startY + cellH + 65)

    // Summary text
    g.setFont(new Font("SansSerif", Font.PLAIN, 13))
    val total = tp + tn + fp + fn
    val acc = (tp + tn).toDouble / total * 100
    g.drawString(f"Total: $total  |  Accuracy: $acc%.1f%%", 140, height - 30)

    g.dispose()
    ImageIO.write(img, "png", new File(outputDir, "confusion_matrix.png"))
  }

  // ============================================================
  // Artifact: ROC Curve (PNG image)
  // ============================================================
  // The ROC curve plots True Positive Rate vs False Positive Rate
  // at different classification thresholds. A perfect model hugs
  // the top-left corner (TPR=1, FPR=0). The diagonal line is
  // random guessing. AUC = area under this curve.
  // ============================================================
  private def generateRocCurve(
    predictions: DataFrame, aucRoc: Double,
    algorithmName: String, outputDir: File
  ): Unit = {
    // Compute ROC points from predictions
    val predAndLabels = predictions.select("probability", "label")
      .collect()
      .map { row =>
        val prob = row.getAs[org.apache.spark.ml.linalg.Vector](0)(1) // P(churn)
        val label = row.getDouble(1)
        (prob, label)
      }
      .sortBy(-_._1) // sort by probability descending

    val totalPositives = predAndLabels.count(_._2 == 1.0).toDouble
    val totalNegatives = predAndLabels.count(_._2 == 0.0).toDouble

    // Calculate ROC points
    val rocSeries = new XYSeries(s"$algorithmName (AUC = ${f"$aucRoc%.3f"})")
    rocSeries.add(0.0, 0.0)

    var tpCount = 0.0
    var fpCount = 0.0
    for ((_, label) <- predAndLabels) {
      if (label == 1.0) tpCount += 1 else fpCount += 1
      val tpr = if (totalPositives > 0) tpCount / totalPositives else 0.0
      val fpr = if (totalNegatives > 0) fpCount / totalNegatives else 0.0
      rocSeries.add(fpr, tpr)
    }

    // Random baseline
    val baselineSeries = new XYSeries("Random (AUC = 0.500)")
    baselineSeries.add(0.0, 0.0)
    baselineSeries.add(1.0, 1.0)

    val dataset = new XYSeriesCollection()
    dataset.addSeries(rocSeries)
    dataset.addSeries(baselineSeries)

    val chart = ChartFactory.createXYLineChart(
      s"ROC Curve — $algorithmName",
      "False Positive Rate",
      "True Positive Rate",
      dataset
    )

    // Style the chart
    val plot = chart.getXYPlot
    plot.setBackgroundPaint(new Color(240, 240, 240))
    plot.getRenderer.setSeriesStroke(0, new BasicStroke(2.5f))
    plot.getRenderer.setSeriesPaint(0, new Color(41, 98, 255))
    plot.getRenderer.setSeriesStroke(1, new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, Array(6.0f), 0.0f))
    plot.getRenderer.setSeriesPaint(1, Color.GRAY)

    ChartUtils.saveChartAsPNG(new File(outputDir, "roc_curve.png"), chart, 800, 600)
  }

  // ============================================================
  // Artifact: Feature Importance (CSV)
  // ============================================================
  // Tree-based models (Random Forest, GBT) tell us which features
  // matter most for predictions. Logistic Regression uses
  // coefficient magnitude instead.
  // ============================================================
  private def generateFeatureImportance(
    model: PipelineModel, algorithmName: String,
    featureNames: Array[String], outputDir: File
  ): Unit = {
    val pw = new PrintWriter(new File(outputDir, "feature_importance.csv"))
    pw.println("feature,importance")

    // Extract the last stage (the classifier) from the pipeline
    val lastStage = model.stages.last

    val importances: Array[Double] = lastStage match {
      case rf: RandomForestClassificationModel =>
        rf.featureImportances.toArray
      case gbt: GBTClassificationModel =>
        gbt.featureImportances.toArray
      case lr: LogisticRegressionModel =>
        // For logistic regression, use absolute coefficient values
        lr.coefficients.toArray.map(math.abs)
      case _ =>
        // Fallback: uniform importance
        Array.fill(featureNames.length)(1.0 / featureNames.length)
    }

    // Pair feature names with importances and sort descending
    val paired = featureNames.zip(
      if (importances.length >= featureNames.length) importances.take(featureNames.length)
      else importances ++ Array.fill(featureNames.length - importances.length)(0.0)
    ).sortBy(-_._2)

    paired.foreach { case (name, importance) =>
      pw.println(f"$name,$importance%.6f")
    }
    pw.close()
  }

  // ============================================================
  // Artifact: Pipeline Configuration (JSON)
  // ============================================================
  private def generatePipelineConfig(
    algorithmName: String,
    paramGrid: Array[org.apache.spark.ml.param.ParamMap],
    outputDir: File
  ): Unit = {
    val gson = new GsonBuilder().setPrettyPrinting().create()
    val config = new java.util.LinkedHashMap[String, Object]()
    config.put("algorithm", algorithmName)
    config.put("num_cv_folds", Integer.valueOf(5))
    config.put("train_test_split", "80/20")

    val gridList = new java.util.ArrayList[java.util.Map[String, String]]()
    paramGrid.foreach { pm =>
      val entry = new java.util.LinkedHashMap[String, String]()
      pm.toSeq.foreach { pp =>
        entry.put(pp.param.name, pp.value.toString)
      }
      gridList.add(entry)
    }
    config.put("param_grid", gridList)

    val pw = new PrintWriter(new File(outputDir, "pipeline_config.json"))
    pw.print(gson.toJson(config))
    pw.close()
  }
}
