// ============================================================
// ModelRegistry.scala — Register & manage models in MLflow
// ============================================================
// This module handles Phase 3: taking the best model from
// training and formally registering it in MLflow's Model
// Registry with proper descriptions, tags, and lifecycle
// stage transitions.
//
// Think of this as the "npm publish" step — after you've built
// and tested your package, you publish it to a registry so
// others (like our FastAPI service) can use it by name instead
// of by file path.
//
// MLflow Model Registry concepts:
//   Registered Model = the "package" (e.g., "churn-predictor")
//   Model Version    = a specific version (e.g., Version 1)
//   Stage            = lifecycle state (None → Staging → Production)
//   Tags             = metadata labels (team, use_case, etc.)
// ============================================================

import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.ModelVersionStatus

object ModelRegistry {

  /**
   * Register the champion model in the MLflow Model Registry.
   *
   * @param mlflowClient  MLflow client instance
   * @param runId         The MLflow run ID of the best training run
   * @param modelName     Registry name (e.g., "churn-predictor")
   * @param result        The TrainingResult from the champion model
   * @return              The registered model version number
   */
  def registerChampion(
    mlflowClient: MlflowClient,
    runId: String,
    modelName: String,
    result: TrainingResult
  ): String = {
    println("\n" + "=" * 60)
    println("PHASE 3: Model Registry")
    println("=" * 60)

    // --- Step 1: Create the registered model (if it doesn't exist) ---
    // This is like `npm init` — creates the package in the registry
    try {
      mlflowClient.createRegisteredModel(modelName)
      println(s"  ✅ Created registered model: $modelName")
    } catch {
      case _: org.mlflow.tracking.MlflowClientException =>
        println(s"  ℹ️  Registered model '$modelName' already exists")
    }

    // --- Step 2: Register the model version ---
    // This is like `npm publish` — uploads the model as a new version
    // The source is the artifact path from the training run
    val source = s"runs:/$runId/model"
    val modelVersion = mlflowClient.createModelVersion(modelName, source, runId)
    val versionNum = modelVersion.getVersion

    println(s"  ✅ Model version $versionNum registered from run $runId")

    // --- Step 3: Add description ---
    val description = s"""
      |Champion model for customer churn prediction.
      |
      |Algorithm: ${result.algorithmName}
      |AUC-ROC: ${f"${result.aucRoc}%.4f"}
      |Accuracy: ${f"${result.accuracy}%.4f"}
      |F1-Score: ${f"${result.f1Score}%.4f"}
      |Training Duration: ${f"${result.durationSeconds}%.1f"}s
      |Training Date: ${java.time.LocalDateTime.now().toString}
      |Dataset: IBM Telco Customer Churn (~7,000 rows)
    """.stripMargin.trim

    mlflowClient.updateModelVersion(
      modelName, versionNum, description
    )
    println("  ✅ Model description added")

    // --- Step 4: Add tags ---
    mlflowClient.setModelVersionTag(modelName, versionNum,
      "team", "group-X")
    mlflowClient.setModelVersionTag(modelName, versionNum,
      "use_case", "churn_prediction")
    mlflowClient.setModelVersionTag(modelName, versionNum,
      "framework", "spark_mllib")
    mlflowClient.setModelVersionTag(modelName, versionNum,
      "algorithm", result.algorithmName)
    mlflowClient.setModelVersionTag(modelName, versionNum,
      "auc_roc", f"${result.aucRoc}%.4f")
    println("  ✅ Tags added: team, use_case, framework, algorithm, auc_roc")

    // --- Step 5: Transition to Staging ---
    mlflowClient.transitionModelVersionStage(
      modelName, versionNum, "Staging"
    )
    println(s"  ✅ Version $versionNum transitioned: None → Staging")

    versionNum
  }

  /**
   * Promote a model version from Staging to Production.
   * Called after validation confirms AUC > 0.80.
   *
   * @param mlflowClient  MLflow client instance
   * @param modelName     Registry name (e.g., "churn-predictor")
   * @param version       Version number to promote
   * @param aucRoc        The validated AUC-ROC score
   */
  def promoteToProduction(
    mlflowClient: MlflowClient,
    modelName: String,
    version: String,
    aucRoc: Double
  ): Unit = {
    val threshold = 0.80
    println(s"\n  Validation: AUC-ROC = ${f"$aucRoc%.4f"} (threshold: $threshold)")

    if (aucRoc >= threshold) {
      mlflowClient.transitionModelVersionStage(
        modelName, version, "Production"
      )
      println(s"  ✅ Version $version promoted: Staging → Production")
      println(s"  🎉 Model '$modelName' v$version is now serving in Production!")
    } else {
      println(s"  ❌ AUC-ROC ${f"$aucRoc%.4f"} is below threshold $threshold")
      println(s"  Model NOT promoted to Production. Retrain with better hyperparameters.")
    }
  }
}
