// ============================================================
// build.sbt — The "package.json" of our Scala project
// ============================================================
// This file tells SBT:
//   - What Scala version to use
//   - What libraries (dependencies) to download
//   - How to build the project
//
// The "%%" means "append the Scala version to the artifact name"
// (Scala libraries are compiled per-version, unlike npm packages).
// The "%" with a single percent is for Java libraries.
// ============================================================

name := "churn-training"
version := "1.0.0"
scalaVersion := "2.12.18"

// Spark 3.4.x requires Scala 2.12
val sparkVersion = "3.4.4"

libraryDependencies ++= Seq(
  // --- Apache Spark (core framework) ---
  // "provided" means: don't bundle these in the fat JAR because
  // Spark already has them when we run via spark-submit.
  // For local `sbt run`, we override this below.
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql"  % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,

  // --- Log4j2 override ---
  // Spark 3.4.4 ships Log4j 2.19.0 which uses SecurityManager (removed
  // in recent JDK 11 patches). Force upgrade to 2.23.1 which fixes this.
  "org.apache.logging.log4j" % "log4j-api"        % "2.23.1",
  "org.apache.logging.log4j" % "log4j-core"       % "2.23.1",
  "org.apache.logging.log4j" % "log4j-slf4j2-impl" % "2.23.1",
  "org.apache.logging.log4j" % "log4j-1.2-api"    % "2.23.1",

  // --- MLflow Java/Scala client ---
  // This lets us call MLflow's API from Scala to log
  // parameters, metrics, and artifacts.
  "org.mlflow" % "mlflow-client" % "2.16.2",

  // --- JFreeChart (plotting library) ---
  // Used to generate confusion matrix and ROC curve PNGs.
  // Think of it as Chart.js but for Java/Scala.
  "org.jfree" % "jfreechart" % "1.5.5",

  // --- Gson (JSON library by Google) ---
  // For serializing pipeline config to JSON.
  // Like JSON.stringify() but in Java.
  "com.google.code.gson" % "gson" % "2.11.0"
)

// ============================================================
// Assembly settings (for building the fat JAR)
// ============================================================
// When multiple JARs contain the same file (like META-INF),
// we need a "merge strategy" to resolve conflicts.
// This is like resolving npm peer dependency conflicts.
// ============================================================
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf"              => MergeStrategy.concat
  case x                             => MergeStrategy.first
}

// Name of the output fat JAR
assembly / assemblyJarName := "churn-training.jar"

// Don't run tests during assembly
assembly / test := {}

// Fork the JVM when running with `sbt run` to avoid
// classloader issues with Spark
fork := true

// JVM options for Spark local mode
// These --add-opens flags allow Spark/Log4j2 to access internal JDK modules
javaOptions ++= Seq(
  "-Xmx4g",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
  "--add-opens=java.base/java.io=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.net=ALL-UNNAMED",
  "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
  "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
  "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
  s"-Dhadoop.home.dir=${(ThisBuild / baseDirectory).value.getParentFile / "hadoop"}"
)
