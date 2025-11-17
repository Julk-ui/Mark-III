# tests/test_data.py
"""
Pruebas para la carga y validación de datos.
"""
import pytest
import pandas as pd
from pathlib import Path
from main_pipeline import TradingPipeline

# Usamos la configuración optimizada para las pruebas de datos y modelos
OPTIMIZED_CONFIG_PATH = Path(__file__).parent.parent / "config/config_optimizado.yaml"

@pytest.fixture(scope="module")
def pipeline_instance():
    """
    Crea una instancia del pipeline usando la configuración optimizada.
    Salta las pruebas si el archivo de configuración no existe.
    """
    if not OPTIMIZED_CONFIG_PATH.exists():
        pytest.skip(f"El archivo de configuración optimizado no existe en: {OPTIMIZED_CONFIG_PATH}. "
                    "Ejecuta el pipeline en modo 'train' o 'backtest' primero.")
    
    return TradingPipeline(config_path=str(OPTIMIZED_CONFIG_PATH))

def test_data_loading(pipeline_instance: TradingPipeline):
    """Prueba que el método _load_data devuelve un DataFrame no vacío."""
    df = pipeline_instance._load_data()
    assert isinstance(df, pd.DataFrame), "La carga de datos debería devolver un DataFrame."
    assert not df.empty, "El DataFrame cargado no debería estar vacío."
    print(f"Datos cargados exitosamente: {df.shape[0]} filas.")

def test_data_cleaning_and_feature_generation(pipeline_instance: TradingPipeline):
    """Prueba que los pasos de limpieza y generación de features se ejecutan."""
    df_raw = pipeline_instance._load_data()
    df_clean = pipeline_instance._clean_data(df_raw)
    df_features = pipeline_instance._generate_features(df_clean)
    
    assert not df_features.empty, "El DataFrame con features no debería estar vacío."
    assert df_features.shape[0] <= df_raw.shape[0], "La limpieza no debería añadir filas."
    print(f"Limpieza y generación de features completadas. Columnas finales: {df_features.shape[1]}.")