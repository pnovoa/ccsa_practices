from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Paso 1: Inicializar SparkSession
spark = SparkSession.builder \
    .appName("RandomForestClassifierExample") \
    .getOrCreate()

# Paso 2: Leer el dataset en formato CSV
df = spark.read.csv("/opt/spark-data/extra_small_celestial.csv", header=True, inferSchema=True, sep =';')

# Paso 3: Seleccionar una muestra del dataset y filtrar los atributos numéricos
numeric_features = [col for col, dtype in df.dtypes if dtype == "int" or dtype == "double"]
selected_df = df.select(*numeric_features)

# Paso 4: Imputar valores perdidos con la media
imputer = Imputer(inputCols=selected_df.columns, outputCols=selected_df.columns)
imputed_df = imputer.fit(selected_df).transform(selected_df)

# Paso 5: Normalizar los atributos
assembler = VectorAssembler(inputCols=imputed_df.columns, outputCol="features")
assembled_df = assembler.transform(imputed_df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

# Paso 6: Convertir el atributo clase a tipo double
# label_df = scaled_df.withColumn("label", scaled_df["type"].cast("double")).drop("clase")

indexer = StringIndexer(inputCol="type", outputCol="label")
label_df = indexer.fit(scaled_df).transform(df)

# Paso 7: Hacer un split de 80/20 para entrenar y clasificar el modelo de RandomForest
train_df, test_df = label_df.randomSplit([0.8, 0.2], seed=123)

rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")
rf_model = rf.fit(train_df)

# Paso 8: Evaluar la calidad del modelo en términos del AUC
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
results = rf_model.transform(test_df)
auc = evaluator.evaluate(results)
print("AUC:", auc)

# Paso 9: Guardar los resultados
results.write.mode("overwrite").csv("/opt/spark-data/rfmodel-results")

# Paso 10: Guardar el modelo
rf_model.save("/opt/spark-data/rfmodel")

# Paso 11: Cerrar la sesión de Spark
spark.stop()
