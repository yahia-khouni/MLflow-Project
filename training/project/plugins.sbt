// ============================================================
// SBT Plugins — like devDependencies in package.json
// ============================================================

// sbt-assembly: creates a "fat JAR" — a single .jar file that
// contains ALL dependencies bundled together. Think of it like
// webpack bundling all your node_modules into one file.
// We need this to submit the training job to Spark.
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.1.5")
