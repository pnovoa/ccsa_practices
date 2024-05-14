#from pyspark.ml.evaluation import ClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
#from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CelestialDecisionTree") \
    .getOrCreate()


# Prepare training and test data.
data = spark.read.csv("/opt/spark-data/extra_small_celestial.csv", sep=";", header=True, inferSchema=True)

numeric_features = [col for col, dtype in data.dtypes if dtype == "int" or dtype == "double"]

labelIndexer = StringIndexer(inputCol="type", outputCol="indexedLabel")

assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=123)

dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="scaled_features")

pipeline = Pipeline(stages=[labelIndexer, assembler, scaler, dt])

model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="indexedLabel")
auc = evaluator.evaluate(predictions)
print("AUC:", auc)



# # We use a ParamGridBuilder to construct a grid of parameters to search over.
# # TrainValidationSplit will try all combinations of values and determine best model using
# # the evaluator.
# paramGrid = ParamGridBuilder()\
#     .addGrid(lr.regParam, [0.1, 0.01]) \
#     .addGrid(lr.fitIntercept, [False, True])\
#     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
#     .build()

# # In this case the estimator is simply the linear regression.
# # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# tvs = TrainValidationSplit(estimator=lr,
#                            estimatorParamMaps=paramGrid,
#                            evaluator=RegressionEvaluator(),
#                            # 80% of the data will be used for training, 20% for validation.
#                            trainRatio=0.8)

# # Run TrainValidationSplit, and choose the best set of parameters.
# model = tvs.fit(train)

# # Make predictions on test data. model is the model with combination of parameters
# # that performed best.
# model.transform(test)\
#     .select("features", "label", "prediction")\
#     .show()