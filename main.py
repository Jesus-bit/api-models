import joblib
import logging
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import numpy as np
import xgboost as xgb

# Define las opciones válidas para el tipo de predicción
class PredictionType(str, Enum):
    TEMPERATURE = "temperature"
    RAIN = "rain"
    SUNNY = "sunny"

# Define el modelo de entrada
class PredictionInput(BaseModel):
    prediction_type: PredictionType
    values: list

# Configurar el logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI(title="Weather Prediction API")

# Cache para los modelos y scalers
model_cache = {}
scaler_cache = {}
features_cache = {}

# Mapeo de tipos de predicción a modelos
MODEL_MAPPING = {
    "temperature": "temperature_model_xgb",
    "rain": "rain_model_tree",
    "sunny": "sunny_model_KNN"
}

def load_features():
    """Carga las características de entrada desde el archivo JSON"""
    try:
        with open('models/model_features.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar el archivo de características: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading features: {str(e)}")

def load_scaler(prediction_type: str):
    """Carga el scaler para el tipo de predicción específico"""
    try:
        logger.info(f"Cargando scaler para: {prediction_type}")
        scaler_path = f"models/scaler_{prediction_type}.joblib"
        return joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Error al cargar el scaler para {prediction_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading scaler: {str(e)}")

def load_model(prediction_type: str):
    try:
        model_name = MODEL_MAPPING[prediction_type]
        print(model_name)
        logger.info(f"Intentando cargar el modelo: {model_name}")
        print(f"Intentando cargar el modelo: {model_name}")
        model_path = f"models/{model_name}.joblib"
        print(model_path)
        model = joblib.load(model_path)
        print(model)
        if not hasattr(model, 'predict'):
            raise ValueError(f"El modelo cargado no tiene el método 'predict': {model_name}")
        
        logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def calculate_confidence(model, prediction, prediction_type: str):
    """Calcula el porcentaje de confiabilidad según el tipo de modelo"""
    try:
        if prediction_type == "temperature":
            # Para XGBoost, podemos usar predict_proba si está configurado para clasificación
            # o usar el margen de predicción para regresión
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([prediction])[0]
                return float(np.max(proba) * 100)
            else:
                # Para regresión, podemos usar una medida basada en el rango de predicciones
                return 95.0  # Para regresión, podrías implementar tu propia métrica
                
        elif prediction_type == "rain":
            # Para Decision Tree, podemos usar la probabilidad de la clase predicha
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([prediction])[0]
                return float(np.max(proba) * 100)
            else:
                return 90.0
                
        elif prediction_type == "sunny":
            # Para KNN, podemos usar la proporción de vecinos de la misma clase
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([prediction])[0]
                return float(np.max(proba) * 100)
            else:
                return 85.0
                
    except Exception as e:
        logger.warning(f"No se pudo calcular la confiabilidad: {e}")
        return None

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        print(input_data)
        prediction_type = input_data.prediction_type.value
        logger.info(f"Solicitud de predicción recibida para tipo: {prediction_type}")
        logger.info(f"Valores de entrada: {input_data.values}")
        
        # Cargar features si no están en caché
        if not features_cache:
            features_cache.update(load_features())
        
        # Obtener las características esperadas según el tipo de predicción
        feature_key = f"{prediction_type}_model_features".lower()  # Convertir a mayúsculas
        print(feature_key)
        
        if feature_key not in features_cache:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de predicción no válido: {prediction_type}"
            )
        expected_features = features_cache[feature_key]
        
        # Verificar que la cantidad de valores coincide con las características esperadas
        if len(input_data.values) != len(expected_features):
            raise HTTPException(
                status_code=400,
                detail=f"Número incorrecto de valores. Se esperaban {len(expected_features)} valores para {prediction_type}. Features requeridos: {expected_features}"
            )
        
        # Cargar modelo y scaler si no están en caché
        if prediction_type not in model_cache:
            model_cache[prediction_type] = load_model(prediction_type)
            scaler_cache[prediction_type] = load_scaler(prediction_type)
        
        model = model_cache[prediction_type]
        scaler = scaler_cache[prediction_type]
        
        # Transformar valores de entrada
        values_scaled = scaler.transform([input_data.values])
        
        # Realizar predicción
        prediction = model.predict(input_data.values)
        
        # Invertir transformación de la predicción si es necesario
        # if hasattr(scaler, 'inverse_transform'):
        #    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        
        # Calcular confiabilidad
        confidence = calculate_confidence(model, input_data.values, prediction_type)
        
        result = {
            "prediction_type": prediction_type,
            "prediction": prediction.flatten().tolist(),
            "model_used": MODEL_MAPPING[prediction_type],
            "features_used": dict(zip(expected_features, input_data.values))
        }
        
        if confidence is not None:
            result["confidence"] = confidence
        
        logger.info(f"Predicción realizada: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error en la predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prediction-types")
async def get_prediction_types():
    """Devuelve los tipos de predicción disponibles y sus características requeridas"""
    if not features_cache:
        features_cache.update(load_features())
    
    return {
        "available_prediction_types": [type.value for type in PredictionType],
        "required_features": features_cache
    }