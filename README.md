# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar config.yaml con tus credenciales MT5

# 3. Ejecutar análisis exploratorio
python main_pipeline.py --mode eda

# 4. Ejecutar backtesting
python main_pipeline.py --mode backtest --config config/config.yaml

# 5. Ejecutar en produccion o testeo
python main_pipeline.py --mode production --config config/config_optimizado.yaml

# 6. Ejecutar test
python main_pipeline.py --mode test --config config/config_optimizado.yaml

