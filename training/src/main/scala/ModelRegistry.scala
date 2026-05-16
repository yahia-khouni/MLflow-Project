// ============================================================
// ModelRegistry.scala — Register & manage models via REST API
// ============================================================
// The MLflow Java client doesn't expose Model Registry methods
// in all versions, so we call the MLflow REST API directly.
// This is like using fetch() instead of an SDK — more verbose
// but works with any MLflow server version.
//
// MLflow REST API docs:
//   https://mlflow.org/docs/latest/rest-api.html
// ============================================================

import java.io.{BufferedReader, InputStreamReader, OutputStream}
import java.net.{HttpURLConnection, URL}
import com.google.gson.{Gson, JsonObject, JsonParser}

object ModelRegistry {

  private val gson = new Gson()

  /**
   * Register the champion model in the MLflow Model Registry.
   *
   * @param mlflowUri   MLflow tracking server URL (e.g., "http://localhost:5000")
   * @param runId       The MLflow run ID of the best training run
   * @param modelName   Registry name (e.g., "churn-predictor")
   * @param result      The TrainingResult from the champion model
   * @return            The registered model version number
   */
  def registerChampion(
    mlflowUri: String,
    runId: String,
    modelName: String,
    result: TrainingResult
  ): String = {
    println("\n" + "=" * 60)
    println("PHASE 3: Model Registry")
    println("=" * 60)

    // --- Step 1: Create the registered model (if it doesn't exist) ---
    try {
      val createBody = new JsonObject()
      createBody.addProperty("name", modelName)
      postJson(s"$mlflowUri/api/2.0/mlflow/registered-models/create", createBody.toString)
      println(s"  ✅ Created registered model: $modelName")
    } catch {
      case e: Exception =>
        if (e.getMessage != null && e.getMessage.contains("RESOURCE_ALREADY_EXISTS")) {
          println(s"  ℹ️  Registered model '$modelName' already exists")
        } else {
          println(s"  ℹ️  Registered model '$modelName' may already exist: ${e.getMessage}")
        }
    }

    // --- Step 2: Register the model version ---
    val source = s"runs:/$runId/model"
    val versionBody = new JsonObject()
    versionBody.addProperty("name", modelName)
    versionBody.addProperty("source", source)
    versionBody.addProperty("run_id", runId)

    val versionResponse = postJson(
      s"$mlflowUri/api/2.0/mlflow/model-versions/create",
      versionBody.toString
    )
    val versionJson = JsonParser.parseString(versionResponse).getAsJsonObject
    val versionNum = versionJson
      .getAsJsonObject("model_version")
      .get("version").getAsString

    println(s"  ✅ Model version $versionNum registered from run $runId")

    // --- Step 3: Add description (best-effort, PATCH may not be supported) ---
    try {
      val description = s"Champion model: ${result.algorithmName} | " +
        f"AUC-ROC: ${result.aucRoc}%.4f | " +
        f"Accuracy: ${result.accuracy}%.4f | " +
        f"F1: ${result.f1Score}%.4f | " +
        f"Trained in ${result.durationSeconds}%.1fs"

      val updateBody = new JsonObject()
      updateBody.addProperty("name", modelName)
      updateBody.addProperty("version", versionNum)
      updateBody.addProperty("description", description)
      patchJson(s"$mlflowUri/api/2.0/mlflow/model-versions/update", updateBody.toString)
      println("  ✅ Model description added")
    } catch {
      case _: Exception =>
        println("  ℹ️  Description update skipped (PATCH not supported by server)")
    }

    // --- Step 4: Add tags ---
    val tags = Map(
      "team" -> "group-X",
      "use_case" -> "churn_prediction",
      "framework" -> "spark_mllib",
      "algorithm" -> result.algorithmName,
      "auc_roc" -> f"${result.aucRoc}%.4f"
    )
    tags.foreach { case (key, value) =>
      val tagBody = new JsonObject()
      tagBody.addProperty("name", modelName)
      tagBody.addProperty("version", versionNum)
      tagBody.addProperty("key", key)
      tagBody.addProperty("value", value)
      postJson(s"$mlflowUri/api/2.0/mlflow/model-versions/set-tag", tagBody.toString)
    }
    println("  ✅ Tags added: " + tags.keys.mkString(", "))

    // --- Step 5: Transition to Staging ---
    val stageBody = new JsonObject()
    stageBody.addProperty("name", modelName)
    stageBody.addProperty("version", versionNum)
    stageBody.addProperty("stage", "Staging")
    postJson(
      s"$mlflowUri/api/2.0/mlflow/model-versions/transition-stage",
      stageBody.toString
    )
    println(s"  ✅ Version $versionNum transitioned: None → Staging")

    versionNum
  }

  /**
   * Promote a model version from Staging to Production.
   * Called after validation confirms AUC > 0.80.
   */
  def promoteToProduction(
    mlflowUri: String,
    modelName: String,
    version: String,
    aucRoc: Double
  ): Unit = {
    val threshold = 0.80
    println(s"\n  Validation: AUC-ROC = ${f"$aucRoc%.4f"} (threshold: $threshold)")

    if (aucRoc >= threshold) {
      val stageBody = new JsonObject()
      stageBody.addProperty("name", modelName)
      stageBody.addProperty("version", version)
      stageBody.addProperty("stage", "Production")
      postJson(
        s"$mlflowUri/api/2.0/mlflow/model-versions/transition-stage",
        stageBody.toString
      )
      println(s"  ✅ Version $version promoted: Staging → Production")
      println(s"  🎉 Model '$modelName' v$version is now serving in Production!")
    } else {
      println(s"  ❌ AUC-ROC ${f"$aucRoc%.4f"} is below threshold $threshold")
      println(s"  Model NOT promoted to Production. Retrain with better hyperparameters.")
    }
  }

  // ============================================================
  // HTTP Helpers — like fetch() in JavaScript
  // ============================================================
  private def postJson(url: String, body: String): String = {
    sendJson(url, body, "POST")
  }

  private def patchJson(url: String, body: String): String = {
    sendJson(url, body, "PATCH")
  }

  private def sendJson(url: String, body: String, method: String): String = {
    val conn = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    // Java's HttpURLConnection doesn't support PATCH directly.
    // Use POST with X-HTTP-Method-Override header instead.
    if (method == "PATCH") {
      conn.setRequestMethod("POST")
      conn.setRequestProperty("X-HTTP-Method-Override", "PATCH")
    } else {
      conn.setRequestMethod(method)
    }
    conn.setRequestProperty("Content-Type", "application/json")
    conn.setDoOutput(true)

    val os: OutputStream = conn.getOutputStream
    os.write(body.getBytes("UTF-8"))
    os.flush()
    os.close()

    val code = conn.getResponseCode
    val stream = if (code >= 200 && code < 300) conn.getInputStream else conn.getErrorStream
    val reader = new BufferedReader(new InputStreamReader(stream))
    val response = new StringBuilder
    var line: String = reader.readLine()
    while (line != null) {
      response.append(line)
      line = reader.readLine()
    }
    reader.close()

    if (code >= 400) {
      throw new RuntimeException(s"HTTP $code from $method $url: ${response.toString}")
    }

    response.toString
  }
}
