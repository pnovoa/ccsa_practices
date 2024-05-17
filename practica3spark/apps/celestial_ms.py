from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

import json
import argparse

# Para leer parámetros que personalizan a nuestro script, en este caso
# la ruta del dataset (input) y la ruta de la carpeta de salida.
parser = argparse.ArgumentParser(description='Script para ejecutar con Spark')
parser.add_argument('--input', type=str, required=True, help='Ruta del dataset')
parser.add_argument('--output', type=str, required=True, help='Ruta de la carpeta de salida')
args = parser.parse_args()


# Creamos la sesión
spark = SparkSession.builder \
    .appName("CelestialModelSelection") \
    .getOrCreate()

# Tomamos las rutas del objeto args
input_folder = args.input
output_folder = args.output

# Leemos el dataset
data = spark.read.csv(input_folder, sep=";", header=True, inferSchema=True)

# Detectamos los atributos que son numéricos para transformarlos (normalizarlos)
numeric_features = [col for col, dtype in data.dtypes if dtype == "int" or dtype == "double"]

# Etiquetamos a los atributos numéricos como "features"
assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")

# Los normalizamos y los etiquetamos como "scaled_features"
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Indexamos al atributo clase y lo etiquetamos como "indexed_label"
labelIndexer = StringIndexer(inputCol="type", outputCol="indexed_label")

# Creamos un pipeline para realizar todas estas transformaciones en orden
pipeline = Pipeline(stages=[labelIndexer, assembler, scaler])

# Compilamos el pipeline con los datos
prep_model = pipeline.fit(data)

prep_model.save(output_folder + "/pipeline")

# Transformamos los datos con el pipeline compilado y 
# los guardamos en una nueva variable "prep_data"
prep_data = prep_model.transform(data)


# Hacemos un split de los datos 80% para entrenar, y 20% para evaluar...
train, test = prep_data.randomSplit([0.8, 0.2], seed=12345)


# Definimos un Árbol de Decisión 
dt = DecisionTreeClassifier(labelCol="indexed_label", featuresCol="scaled_features")

# Con ayuda de ParamGridBuilder construimos una malla (grid) con los hiperparámetros
# del modelo.
dt_param_grid = ParamGridBuilder()\
    .addGrid(dt.maxDepth, [3, 10]) \
    .addGrid(dt.maxBins, [10, 30])\
    .build()

# Aquí definimos otro modelo, en esta ocasión, un Random Forest.
rf = RandomForestClassifier(labelCol="indexed_label", featuresCol="scaled_features")

# Y esta es la malla correspondiente al Random Forest
rf_param_grid = ParamGridBuilder()\
    .addGrid(rf.maxDepth, [3, 10]) \
    .addGrid(rf.maxBins, [10, 30])\
    .build()

# Creamos un diccionario con los modelos del experimento
to_run = {"dt": (dt, dt_param_grid), "rf": (rf, rf_param_grid)}

# Definimos un evaluador para definir las medidas de rendimiento
# En este caso, hemos definido solo el área bajo la curva ROC.
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="indexed_label")

# Iteramos por los modelos incluidos en to_run

for mid, settings in to_run.items():

    # Obtenemos el modelo y su malla de hiperparámetros
    estimator, parset = settings

    # Construimos un validador que explorará todas las combinaciones de hiperparámetros
    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=parset,
                           evaluator=evaluator,
                           # 80% serán usados para entrenar, 20% para validación interna.
                           trainRatio=0.8)

    # Ejecutamos a TrainValidationSplit.
    tvs_model = tvs.fit(train)

    # Elegimos el mejor modelo ajustado por tvs
    best_model = tvs_model.bestModel

    # Definimos la carpeta de salida del modelo en específico
    inner_out_folder = output_folder + "/" + mid

    # Guardamos al mejor modelo encontrado por tvs
    best_model.save(inner_out_folder)
    
    # Extraemos sus hiperparámetros (todos, incluidos los de interés)
    params = best_model.extractParamMap()

    # Hacemos predicciones sobre el conjunto de prueba "test"
    predictions = best_model.transform(test)

    # Evaluamos el rendimiento sobre "test"
    auc = evaluator.evaluate(predictions)

    # Creamos un diccionario de los hiperparámetros
    final_dict = {param.name: value for param, value in zip(params.keys(), params.values())}

    # Añadimos al diccionario el rendimiento del modelo
    final_dict["AUC"] = auc

    # Guardamos en un archivo json los hiperparámetros y el rendimiento
    with open("{}/params.json".format(inner_out_folder), "w") as json_file:
        json.dump(final_dict, json_file, indent=4)

spark.stop()