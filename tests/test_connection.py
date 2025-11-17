# tests/test_connection.py
"""
Pruebas para la conexión con MetaTrader 5.
"""
import pytest
import yaml
from pathlib import Path
from data.data_loader import DataLoader

CONFIG_PATH = Path(__file__).parent.parent / "config/config.yaml"

@pytest.fixture(scope="module")
def mt5_config():
    """Carga la configuración de MT5 desde el archivo principal."""
    if not CONFIG_PATH.exists():
        pytest.skip(f"Archivo de configuración no encontrado en {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("mt5", {})

def test_mt5_connection_and_shutdown(mt5_config):
    """
    Prueba si el DataLoader puede inicializar y cerrar la conexión a MT5.
    """
    if not mt5_config:
        pytest.skip("Configuración de MT5 no encontrada en config.yaml")
    
    loader = DataLoader(mt5_config=mt5_config)
    assert loader.is_connected(), "La conexión con MT5 debería haberse establecido en la inicialización."
    
    # La conexión se cierra automáticamente al finalizar el contexto del loader.
    print("Conexión con MT5 establecida y cerrada correctamente.")