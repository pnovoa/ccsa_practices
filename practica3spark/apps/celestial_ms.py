from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

import json

spark = SparkSession.builder \
    .appName("CelestialModelSelection") \
    .getOrCreate()

input_folder = "/opt/spark-data/extra_small_celestial.csv"
output_folder = "/opt/spark-data/app"

# Prepare training and test data.
data = spark.read.csv(input_folder, sep=";", header=True, inferSchema=True)

numeric_features = [col for col, dtype in data.dtypes if dtype == "int" or dtype == "double"]
labelIndexer = StringIndexer(inputCol="type", outputCol="indexedLabel")
assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

pipeline = Pipeline(stages=[labelIndexer, assembler, scaler])

prep_model = pipeline.fit(data)

prep_data = prep_model.transform(data)


# Prepare training and test data.
train, test = prep_data.randomSplit([0.8, 0.2], seed=12345)



dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="scaled_features")

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
dt_param_grid = ParamGridBuilder()\
    .addGrid(dt.maxDepth, [3, 10]) \
    .addGrid(dt.maxBins, [10, 30])\
    .build()

# nb = NaiveBayes(labelCol="indexedLabel", featuresCol="scaled_features")

# nb_param_grid = ParamGridBuilder()\
#     .addGrid(nb.smoothing, [0.0, 1.0]) \
#     .addGrid(nb.modelType, ['multinomial', 'bernoulli'])\
#     .build()

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="scaled_features")

rf_param_grid = ParamGridBuilder()\
    .addGrid(rf.maxDepth, [3, 10]) \
    .addGrid(rf.maxBins, [10, 30])\
    .build()

to_run = {"dt": (dt, dt_param_grid), "rf": (rf, rf_param_grid)}

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="indexedLabel")

for mid, settings in to_run.items():

    estimator, parset = settings

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=parset,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

    # Run TrainValidationSplit, and choose the best set of parameters.
    tvs_model = tvs.fit(train)

    best_model = tvs_model.bestModel

    inner_out_folder = output_folder + "/" + mid

    best_model.save(inner_out_folder)
    
    params = best_model.extractParamMap()

    predictions = best_model.transform(test)

    auc = evaluator.evaluate(predictions)

    final_dict = {param.name: value for param, value in zip(params.keys(), params.values())}

    final_dict["AUC"] = auc

    with open("{}/params.json".format(inner_out_folder), "w") as json_file:
        json.dump(final_dict, json_file, indent=4)
