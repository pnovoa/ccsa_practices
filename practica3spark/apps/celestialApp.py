from flask import Flask, request, jsonify
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType

# Crear una sesión de Spark
spark = SparkSession.builder.appName('MLModelPrediction').getOrCreate()

# Ruta al modelo MLlib
MODEL_PATH = "/opt/spark-data/out/rf"
PIPELINE_PATH = "/opt/spark-data/out/pipeline"

# Leemos el pipeline 
pre_pipeline = PipelineModel.load(PIPELINE_PATH)

# Cargar el modelo MLlib
model = RandomForestClassificationModel.load(MODEL_PATH)

# Crear la aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    spark = SparkSession.builder.appName("example").getOrCreate()
    return "Spark session created successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del request
        data = request.get_json()
        
        # Definir el esquema de los datos esperados
        schema = StructType([
            StructField("expAB_z", FloatType(), True),
            StructField("i", FloatType(), True),
            StructField("q_r", FloatType(), True),
            StructField("modelFlux_r", FloatType(), True),
            StructField("expAB_i", FloatType(), True),
            StructField("expRad_u", FloatType(), True),
            StructField("q_g", FloatType(), True),
            StructField("psfMag_z", FloatType(), True),
            StructField("dec", FloatType(), True),
            StructField("psfMag_r", FloatType(), True)
        ])
        
        # Crear un DataFrame de Spark con los datos recibidos
        input_data = spark.createDataFrame([data], schema)
        
        prep_data = pre_pipeline.transform(input_data)

        # Hacer la predicción
        predictions = model.transform(prep_data)
        
        # Seleccionar solo la columna de predicción
        prediction = predictions.select("prediction").collect()[0][0]
        
        # Devolver la predicción como JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)