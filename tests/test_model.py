# tests/test_model.py
"""
Pruebas para el entrenamiento y predicción de modelos.
"""
import pytest
import pandas as pd
from pathlib import Path
from main_pipeline import TradingPipeline

OPTIMIZED_CONFIG_PATH = Path(__file__).parent.parent / "config/config_optimizado.yaml"

@pytest.fixture(scope="module")
def pipeline_with_data():
    """
    Crea una instancia del pipeline y carga/procesa los datos.
    """
    if not OPTIMIZED_CONFIG_PATH.exists():
        pytest.skip(f"El archivo de configuración optimizado no existe: {OPTIMIZED_CONFIG_PATH}")
    
    pipeline = TradingPipeline(config_path=str(OPTIMIZED_CONFIG_PATH))
    df_raw = pipeline._load_data()
    df_clean = pipeline._clean_data(df_raw)
    df_features = pipeline._generate_features(df_clean)
    return pipeline, df_features

def test_train_and_predict_with_best_model(pipeline_with_data):
    """
    Prueba el método _train_and_predict usando el mejor modelo de la config optimizada.
    """
    pipeline, df_features = pipeline_with_data
    
    # --- MEJORA: Buscar el modelo marcado como el mejor ---
    best_model_config = None
    for model in pipeline.config.get("models", []):
        if model.get("is_best"):
            best_model_config = model
            break
    assert best_model_config is not None, "No se encontró ningún modelo marcado con 'is_best: true' en config_optimizado.yaml."
    
    model_name = best_model_config["name"]
    params = best_model_config.get("params", {})
    
    # Usar una pequeña porción de datos para una prueba rápida
    y_train_sample = df_features[pipeline.config["backtest"]["target"]].head(100)
    X_train_sample = df_features.head(100)
    X_test_sample = df_features.head(101).tail(1) # Predecir el siguiente paso
    
    prediction = pipeline._train_and_predict(model_name, params, X_train_sample, y_train_sample, X_test_sample)
    
    assert prediction is not None, f"La predicción del modelo {model_name} no debería ser nula."
    assert isinstance(prediction, list) and len(prediction) > 0, "La predicción debería ser una lista no vacía."
    print(f"Predicción del modelo '{model_name}' generada exitosamente: {prediction}")