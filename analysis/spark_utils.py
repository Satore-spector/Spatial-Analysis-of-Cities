"""Starting Spark for the --engine spark options (steps 2 and 4).

Spark runs inside your own computer here ("local mode", one worker per CPU
core); the same code would run unchanged on a cluster by changing the master.

Only Spark SQL and Spark MLlib are used, never Python functions run inside
Spark (UDFs). That keeps it working on Windows without extra setup, where
Python workers launched by Spark are a common source of errors.

Needs Java 17 or newer (check with `java -version`) and `pip install pyspark`.
"""

from __future__ import annotations

import os
import sys
import warnings

from .common import say

_SESSION = None

JAVA_HELP = """
Spark could not start. The usual cause is Java:
  - Spark needs Java 17 or newer. Check with:  java -version
  - Windows: install the "Temurin 21 JDK" from https://adoptium.net and tick
    "Set JAVA_HOME variable" in the installer, then restart VS Code.
  - Mac: brew install openjdk@21   (or the Temurin installer above)
Also check PySpark is installed:  pip install pyspark
Or run without Spark:  --engine pandas
"""


def get_spark(app: str = "spatial-analysis-of-cities", driver_memory: str = "4g"):
    """One shared local Spark session (created on first use)."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise SystemExit("PySpark is not installed. Run: pip install pyspark\n" + JAVA_HELP)
    # PySpark warns about new pandas versions; the features used here work fine.
    warnings.filterwarnings("ignore", message=".*PySpark does not yet fully support pandas.*")
    # Spark must start Python with the same interpreter as this script.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    try:
        spark = (SparkSession.builder.appName(app).master("local[*]")
                 .config("spark.driver.memory", driver_memory)
                 .config("spark.sql.shuffle.partitions", "8")   # default 200 is for clusters
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                 .config("spark.ui.enabled", "false")
                 .config("spark.ui.showConsoleProgress", "false")
                 .config("spark.log.level", "ERROR")
                 .config("spark.sql.session.timeZone", "UTC")
                 .getOrCreate())
    except Exception as e:  # Java missing / wrong version / JAVA_HOME
        raise SystemExit(f"{e}\n{JAVA_HELP}")
    spark.sparkContext.setLogLevel("ERROR")
    say(f"  Spark {spark.version} started (local mode, {spark.sparkContext.defaultParallelism} workers)")
    _SESSION = spark
    return spark


def stop_spark():
    global _SESSION
    if _SESSION is not None:
        _SESSION.stop()
        _SESSION = None
