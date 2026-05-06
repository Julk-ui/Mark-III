# Mark III

Sistema de trading algoritmico con pipeline modular para:

- carga de datos desde MetaTrader 5
- limpieza y generacion de features
- analisis exploratorio de datos
- backtesting walk-forward con comparacion de modelos
- seleccion de modelo campeon
- publicacion de una release activa para produccion
- generacion de senales y ejecucion opcional de ordenes reales
- scheduler para automatizar backtest, produccion y reconciliacion de trades

## 1. Objetivo del proyecto

Mark III busca un flujo continuo:

1. descargar y preparar datos historicos
2. comparar varios modelos con backtest
3. elegir el mejor run por modelo y el campeon global
4. entrenar y guardar los mejores modelos
5. publicar una release estable
6. usar esa release en produccion para generar senales y, si esta habilitado, ejecutar ordenes en MT5
7. mantener reportes de senales, ciclo de vida y cierres

## 2. Estructura del repositorio

```text
Mark-III/
|-- main_pipeline.py
|-- scheduler_automation.py
|-- generate_executive_report.py
|-- requirements.txt
|-- README.md
|-- config/
|   |-- config.yaml
|   |-- config_backtest_ajustado.yaml
|   |-- config_profile_aggressive.yaml
|   |-- config_profile_balanced.yaml
|   |-- config_profile_conservative.yaml
|   |-- config_optimizado.yaml
|   |-- config_optimizado_<run_id>.yaml
|   |-- config_optimizado_<profile>.yaml
|   |-- config_scheduler_runtime_profiles.yaml
|   |-- active_release.json
|   `-- active_release_<profile>.json
|-- conexion/
|   `-- easy_Trading.py
|-- data/
|   |-- data_loader.py
|   `-- data_cleaner.py
|-- eda/
|   `-- exploratory_analysis.py
|-- models/
|   `-- modelos predictivos y wrappers de entrenamiento
|-- utils/
|   `-- metricas, riesgo y señales
|-- outputs/
|   |-- backtest/
|   |-- eda/
|   |-- models/
|   |   `-- releases/<run_id>/
|   |-- production/
|   |-- reportes/
|   `-- validation/
`-- logs/
```

## 3. Archivos principales y para que sirve cada uno

| Archivo | Rol |
|---|---|
| `main_pipeline.py` | Orquesta todos los modos del sistema: `eda`, `train`, `backtest`, `test`, `production`, `sync_trades`, `clear_cache`. |
| `scheduler_automation.py` | Scheduler continuo basado en APScheduler. Lanza jobs de backtest, produccion y sincronizacion. |
| `generate_executive_report.py` | Genera un reporte ejecutivo en Word usando resultados de backtest y produccion. |
| `config/config.yaml` | Config base o generica. |
| `config/config_backtest_ajustado.yaml` | Config de trabajo para backtesting, scheduler y tuning. Es el archivo maestro recomendado. |
| `config/config_profile_aggressive.yaml` | Perfil agresivo: horizonte corto, umbrales mas bajos, mas frecuencia. |
| `config/config_profile_balanced.yaml` | Perfil balanceado: punto medio entre frecuencia y calidad de senal. |
| `config/config_profile_conservative.yaml` | Perfil conservador: umbral mas alto y riesgo mas bajo por trade. |
| `config/config_profile_aggressive_light.yaml` | Variante ligera del perfil agresivo para validar flujo y publicar una release rapida. |
| `config/config_profile_balanced_light.yaml` | Variante ligera del perfil balanceado para pruebas funcionales y scheduler rapido. |
| `config/config_profile_conservative_light.yaml` | Variante ligera del perfil conservador para pruebas funcionales y comparacion rapida. |
| `config/config_profile_aggressive_medium.yaml` | Variante intermedia del perfil agresivo para estudios utiles sin el costo del perfil completo. |
| `config/config_profile_balanced_medium.yaml` | Variante intermedia del perfil balanceado para estudios utiles sin el costo del perfil completo. |
| `config/config_profile_conservative_medium.yaml` | Variante intermedia del perfil conservador para estudios utiles sin el costo del perfil completo. |
| `config/config_optimizado.yaml` | Config estable para produccion. Se actualiza con el resultado del backtest. |
| `config/config_optimizado_<run_id>.yaml` | Version congelada de la release optimizada de un backtest especifico. |
| `config/config_optimizado_<profile>.yaml` | Alias estable de produccion para un perfil concreto. Solo existe despues del primer backtest exitoso de ese perfil. |
| `config/config_scheduler_runtime_profiles.yaml` | Scheduler operativo para correr uno o varios perfiles en produccion sin lanzar backtests automaticos. |
| `config/config_scheduler_runtime_profiles_light.yaml` | Scheduler operativo para perfiles `light`, pensado para validacion funcional mientras corren backtests mas largos. |
| `config/config_scheduler_runtime_profiles_medium.yaml` | Scheduler operativo para perfiles canonicos alimentados por releases publicadas desde YAML `medium`. |
| `config/active_release.json` | Manifiesto de la release activa que debe usar produccion. |
| `config/active_release_<profile>.json` | Manifiesto de la release activa de un perfil concreto. |
| `conexion/easy_Trading.py` | Integracion directa con MT5 para datos, cuenta, ticks, ordenes y protecciones. |
| `data/data_loader.py` | Carga datos desde MT5 y maneja cache local. |
| `data/data_cleaner.py` | Limpia OHLCV y genera features tecnicos. |
| `eda/exploratory_analysis.py` | Ejecuta estadistica, tests y graficos del modo EDA. |

## 4. Requisitos

## Software

- Python 3.11 o compatible con el entorno del proyecto
- MetaTrader 5 instalado
- Cuenta MT5 configurada y conectada

## Instalacion

```powershell
pip install -r requirements.txt
```

Si usas entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5. Configuraciones importantes

## `config/config_backtest_ajustado.yaml`

Es el archivo maestro para:

- parametros de carga de datos
- features
- modelos y grillas
- backtest
- riesgo
- scheduler

Tambien es el archivo que normalmente pasas a:

```powershell
python main_pipeline.py --mode backtest --config config/config_backtest_ajustado.yaml
python scheduler_automation.py --config config/config_backtest_ajustado.yaml
```

## `config/config_optimizado.yaml`

Es el archivo operativo de produccion. Contiene:

- lista de modelos ganadores
- `params` del mejor run por modelo
- `is_best: true` en un solo modelo campeon

Importante:

- se sobrescribe cuando un nuevo backtest publica una release
- no es el mejor lugar para documentar cambios manuales permanentes
x 
## `config/active_release.json`

Es el manifiesto que conecta backtest con produccion. Contiene, por ejemplo:

- `release_id`
- `champion_model`
- `config_path`
- `models_dir`
- `summary_csv_path`
- `summary_xlsx_path`

Produccion y `sync_trades` usan esta referencia para operar sobre una release estable, sin depender de artefactos a medio escribir.

### Configs multi-perfil

Cuando trabajas con estrategias separadas, el sistema puede publicar una release distinta por perfil:

- `config/config_profile_aggressive.yaml`
- `config/config_profile_balanced.yaml`
- `config/config_profile_conservative.yaml`
- `config/config_profile_aggressive_light.yaml`
- `config/config_profile_balanced_light.yaml`
- `config/config_profile_conservative_light.yaml`
- `config/config_profile_aggressive_medium.yaml`
- `config/config_profile_balanced_medium.yaml`
- `config/config_profile_conservative_medium.yaml`
- `config/active_release_aggressive.json`
- `config/active_release_balanced.json`
- `config/active_release_conservative.json`
- `config/config_optimizado_aggressive.yaml`
- `config/config_optimizado_balanced.yaml`
- `config/config_optimizado_conservative.yaml`

Idea practica:

- cada perfil se backtestea por separado
- cada perfil publica su propia release activa
- produccion puede correr uno, varios o todos los perfiles ya publicados
- los alias `config_optimizado_<profile>.yaml` y `active_release_<profile>.json` se crean solo cuando ese perfil publica su primera release exitosa

### Familias de perfiles

Hay tres familias de YAML por perfil:

- canonicos: `config_profile_aggressive.yaml`, `config_profile_balanced.yaml`, `config_profile_conservative.yaml`
- `light`: validacion funcional y releases rapidas
- `medium`: estudio util con menos costo que el perfil serio completo, pero publicando sobre el perfil canonico

Regla practica:

- usa `light` cuando quieres comprobar flujo, scheduler, logs y ordenes sin esperar mucho
- usa `medium` cuando quieres actualizar la release seria de un perfil sin pagar el costo del perfil completo
- usa el perfil canonico cuando quieres el estudio mas completo disponible para ese perfil

## 5.1 Regla practica para decidir que archivo cambiar

Usa esta regla simple:

- cambia `config/config_backtest_ajustado.yaml` cuando quieras modificar el comportamiento base del sistema
- cambia `config/config_optimizado.yaml` solo si necesitas tocar produccion de forma puntual e inmediata
- no edites `config/active_release.json` manualmente

En la practica:

- frecuencia del scheduler: `config_backtest_ajustado.yaml`
- grillas de modelos: `config_backtest_ajustado.yaml`
- timeframe del proximo backtest: `config_backtest_ajustado.yaml`
- campeon y parametros operativos actuales: `config_optimizado.yaml`

## 5.2 Perfiles de backtest por nivel de complejidad

Estos perfiles no son obligatorios. Sirven como punto de partida.

### Perfil `smoke`

Objetivo:

- validar que el flujo completo funcione
- obtener resultados rapido
- confirmar generacion de artefactos y release activa

Recomendado para:

- pruebas despues de cambios de codigo
- primer arranque de una estrategia nueva
- validar scheduler y produccion

Bloque sugerido:

```yaml
validation:
  mode: last_n
  n: 250

data:
  n_bars: 3000

backtest:
  initial_train: 1200
  step: 20

logging:
  level: "INFO"
```

Tradeoff:

- muy rapido
- menos fino para comparar modelos

### Perfil `intermedio`

Objetivo:

- comparar varias familias de modelos
- mantener runtime razonable

Recomendado para:

- iteracion diaria
- ajuste de grillas sin costo excesivo

Bloque sugerido:

```yaml
validation:
  mode: last_n
  n: 400

data:
  n_bars: 5000

backtest:
  initial_train: 1800
  step: 10

logging:
  level: "INFO"
```

Tradeoff:

- buen equilibrio entre costo y comparabilidad
- puede tardar bastante si ARIMA y PROPHET tienen varias combinaciones

### Perfil `completo`

Objetivo:

- backtest mas serio y mas cercano a una corrida de investigacion
- mayor estabilidad estadistica

Recomendado para:

- corridas finales antes de promover una estrategia
- comparacion amplia entre modelos y parametros

Bloque sugerido:

```yaml
validation:
  mode: last_n
  n: 1000

data:
  n_bars: 10000

backtest:
  initial_train: 3000
  step: 5

logging:
  level: "WARNING"
```

Tradeoff:

- mejor cobertura temporal
- mucho mas costo computacional

## 5.3 Perfiles por timeframe

La idea clave es esta:

- si el backtest se hizo en `H1`, produccion coherente es cada `60` minutos
- si el backtest se hizo en `M15`, produccion coherente es cada `15` minutos
- si el backtest se hizo en `M5`, produccion coherente es cada `5` minutos

### Perfil `H1` conservador

Uso sugerido:

- swing intradia tranquilo
- menos ruido
- menos carga computacional

```yaml
data:
  timeframe: "H1"
  n_bars: 3000
  use_cache: false
  cache_expiry_hours: 1

validation:
  mode: last_n
  n: 100

backtest:
  initial_train: 1000
  step: 1
  threshold_pips: 5
  periods_per_year: 6048

trading:
  min_pips_signal: 5.0

scheduler:
  backtest_interval_minutes: 2880
  production_interval_minutes: 60
  sync_trades_interval_minutes: 5
```

### Perfil `M15` balanceado

Uso sugerido:

- mas frecuencia que `H1`
- costo moderado
- buen punto medio entre reactividad y estabilidad

```yaml
data:
  timeframe: "M15"
  n_bars: 8000
  use_cache: false
  cache_expiry_hours: 1

validation:
  mode: last_n
  n: 500

backtest:
  initial_train: 2500
  step: 5
  threshold_pips: 3
  periods_per_year: 24192

trading:
  min_pips_signal: 3.0

scheduler:
  backtest_interval_minutes: 2880
  production_interval_minutes: 15
  sync_trades_interval_minutes: 2
  sync_trades_offset_seconds: 30
```

### Perfil `M5` rapido

Uso sugerido:

- pruebas funcionales
- validacion del flujo end-to-end
- produccion frecuente

```yaml
data:
  timeframe: "M5"
  n_bars: 3000
  use_cache: false
  cache_expiry_hours: 1

validation:
  mode: last_n
  n: 250

backtest:
  initial_train: 1200
  step: 20
  threshold_pips: 2
  periods_per_year: 72576

trading:
  min_pips_signal: 2.0

scheduler:
  backtest_interval_minutes: 2880
  production_interval_minutes: 5
  sync_trades_interval_minutes: 1
  sync_trades_offset_seconds: 30
```

### Perfil `M5` intermedio

Uso sugerido:

- comparar varios modelos sin irse al extremo de tiempo
- perfil recomendado para trabajo normal en `M5`

```yaml
data:
  timeframe: "M5"
  n_bars: 5000
  use_cache: false
  cache_expiry_hours: 1

validation:
  mode: last_n
  n: 400

backtest:
  initial_train: 1800
  step: 10
  threshold_pips: 2
  periods_per_year: 72576

trading:
  min_pips_signal: 2.0

scheduler:
  backtest_interval_minutes: 2880
  production_interval_minutes: 5
  sync_trades_interval_minutes: 1
  sync_trades_offset_seconds: 30
```

### Perfil `M5` completo

Uso sugerido:

- corridas finales en alta frecuencia
- investigacion mas exigente

```yaml
data:
  timeframe: "M5"
  n_bars: 10000
  use_cache: false
  cache_expiry_hours: 1

validation:
  mode: last_n
  n: 1000

backtest:
  initial_train: 3000
  step: 5
  threshold_pips: 2
  periods_per_year: 72576

trading:
  min_pips_signal: 2.0

scheduler:
  backtest_interval_minutes: 2880
  production_interval_minutes: 5
  sync_trades_interval_minutes: 1
  sync_trades_offset_seconds: 30
```

## 5.4 Perfiles del scheduler

### Scheduler minimo

Solo backtest manual y produccion manual:

```yaml
scheduler:
  enabled: false
```

Uso:

- ideal para depuracion
- no entra en loop

### Scheduler simple por intervalos

Es el modo mas facil de entender:

```yaml
scheduler:
  enabled: true
  backtest_interval_minutes: 2880
  production_interval_minutes: 5
  sync_trades_interval_minutes: 1
  sync_trades_offset_seconds: 30
  production_use_optimized_config: true
```

Uso:

- recomendable si no quieres usar cron
- corre cada `N` minutos desde que arrancas el scheduler

### Scheduler exacto por cron

Para horarios fijos:

```yaml
scheduler:
  enabled: true
  backtest_cron: "0 7 * * 1"
  production_cron: "5 * * * 1-5"
  sync_trades_cron: "*/1 * * * 1-5"
  production_use_optimized_config: true
```

Uso:

- mejor si quieres ejecuciones en horarios exactos
- requiere mas cuidado en configuracion

## 5.5 Recomendaciones de cache de datos

La configuracion del cache afecta mucho el comportamiento percibido de produccion.

### Para backtest

```yaml
data:
  use_cache: false
```

Motivo:

- evita mezclar corridas con datos viejos

### Para produccion frecuente

```yaml
data:
  use_cache: false
  cache_expiry_hours: 1
```

Motivo:

- si produces senales cada 5 minutos o 15 minutos, no conviene usar cache largo

### Para pruebas livianas

```yaml
data:
  use_cache: true
  cache_expiry_hours: 24
```

Motivo:

- acelera pruebas locales
- no es lo mejor para produccion intradia

## 5.6 Como elegir un perfil rapido

Si tu prioridad principal es velocidad:

- `M5` rapido
- `smoke`
- pocas combinaciones por modelo

Si tu prioridad principal es comparabilidad:

- `M5` intermedio o `M15` balanceado
- `intermedio`

Si tu prioridad principal es rigor:

- `H1` conservador para menos ruido
- o `M5` completo si aceptas el costo de CPU

## 6. Modos del pipeline

El CLI principal expone estos modos:

```powershell
python main_pipeline.py --mode <modo> --config <archivo_yaml>
```

Modos disponibles:

- `eda`
- `train`
- `backtest`
- `test`
- `production`
- `sync_trades`
- `clear_cache`

## 6.1 Modo `eda`

Comando:

```powershell
python main_pipeline.py --mode eda --config config/config_backtest_ajustado.yaml
```

Que hace:

1. carga datos con `DataLoader`
2. limpia con `DataCleaner`
3. genera features con `FeatureEngineer`
4. ejecuta `ExploratoryAnalysis`
5. guarda datos procesados y reportes

Archivos que usa:

- `main_pipeline.py`
- `data/data_loader.py`
- `data/data_cleaner.py`
- `eda/exploratory_analysis.py`

Salidas tipicas:

- `outputs/eda/<SYMBOL>_01_price_series.png`
- `outputs/eda/<SYMBOL>_02_returns_distribution.png`
- `outputs/eda/<SYMBOL>_03_qq_plot.png`
- `outputs/eda/<SYMBOL>_04_acf.png`
- `outputs/eda/<SYMBOL>_05_pacf.png`
- `outputs/eda/<SYMBOL>_06_rolling_volatility.png`
- `outputs/eda/<SYMBOL>_07_decomposition.png`
- `outputs/eda/<SYMBOL>_08_rolling_sharpe.png`
- `outputs/eda/<SYMBOL>_09_drawdown_curve.png`
- `outputs/eda/<SYMBOL>_EDA_report.xlsx`
- `outputs/processed_data.csv`
- `outputs/trading_data_analysis.xlsx`

## 6.2 Modo `train`

Comando:

```powershell
python main_pipeline.py --mode train --config config/config_backtest_ajustado.yaml
```

Que hace:

1. carga, limpia y genera features
2. divide train/test usando `validation.test_size`
3. corre la busqueda de hiperparametros sobre train
4. carga la release optimizada activa
5. valida los modelos finales sobre el hold-out

Uso practico:

- util cuando quieres separar tuning y validacion final
- menos usado que `backtest` + `production`, pero sigue disponible

## 6.3 Modo `backtest`

Comando:

```powershell
python main_pipeline.py --mode backtest --config config/config_backtest_ajustado.yaml
```

Que hace:

1. carga datos historicos
2. limpia y genera features
3. reserva hold-out final si `validation.mode: last_n`
4. corre backtest walk-forward sobre la porcion in-sample
5. calcula metricas por combinacion
6. genera reportes por modelo
7. selecciona el mejor run por modelo
8. selecciona el campeon global
9. reentrena los mejores modelos
10. guarda una release versionada
11. actualiza `config_optimizado.yaml` y `active_release.json`

Archivos principales que genera:

- `outputs/backtest/report_<Modelo>.csv`
- `outputs/backtest/<Modelo>_<params>_series.csv`
- `outputs/backtest/<Modelo>_best_backtest_detail.csv`
- `outputs/backtest/<Modelo>_best_backtest_detail.xlsx`
- `outputs/backtest/summary_best_runs.csv`
- `outputs/backtest/summary_best_runs.xlsx`
- `outputs/backtest/*_<run_id>.csv`
- `outputs/backtest/*_<run_id>.xlsx`
- `config/config_optimizado.yaml`
- `config/config_optimizado_<run_id>.yaml`
- `outputs/models/releases/<run_id>/`
- `config/active_release.json`

## 6.4 Modo `test`

Comando:

```powershell
python main_pipeline.py --mode test --config config/config_optimizado.yaml
```

Que hace:

- toma la release optimizada
- carga y procesa datos
- valida los mejores modelos sobre un hold-out final
- sirve como validacion posterior al backtest

## 6.5 Modo `production`

Comando:

```powershell
python main_pipeline.py --mode production --config config/config_optimizado.yaml
```

Que hace:

1. resuelve la release activa
2. carga datos recientes
3. limpia y genera features
4. carga los modelos guardados en disco
5. predice el siguiente retorno
6. traduce la prediccion a `BUY`, `SELL` o `HOLD`
7. calcula `entry`, `SL`, `TP`, lotaje y riesgo
8. guarda la señal
9. si `auto_execute_orders: true`, envia la orden a MT5
10. actualiza los reportes de trades

Archivos de salida:

- `outputs/production/production_signals.csv`
- `outputs/production/trade_lifecycle_report.csv`
- `outputs/production/closed_trades_report.csv`
- `outputs/production/daily_trade_report.csv`

## 6.6 Modo `sync_trades`

Comando:

```powershell
python main_pipeline.py --mode sync_trades --config config/config_optimizado.yaml
```

Que hace:

- consulta MT5
- revisa posiciones abiertas y deals cerrados
- actualiza el estado local del ciclo de vida de trades
- reconstruye el reporte diario
- reaplica proteccion si detecta posiciones sin `SL/TP`

Este modo no genera señales nuevas. Su objetivo es reconciliacion operativa.

## 6.7 Modo `clear_cache`

Comando:

```powershell
python main_pipeline.py --mode clear_cache --config config/config_backtest_ajustado.yaml
```

Que hace:

- elimina cache de datos del simbolo configurado
- util cuando quieres forzar recarga desde MT5

## 7. Como funciona el backtest

## 7.1 Flujo

El backtest sigue este orden:

1. `main_pipeline.py::_load_data`
2. `main_pipeline.py::_clean_data`
3. `main_pipeline.py::_generate_features`
4. limpieza de `NaN` en target y features
5. aplicacion opcional de hold-out final con `validation.mode`
6. `main_pipeline.py::_run_hyperparameter_tuning`
7. loop por modelo y por combinacion de `ParameterGrid`
8. `main_pipeline.py::_run_walk_forward_for_params`
9. `main_pipeline.py::_train_and_predict`
10. `main_pipeline.py::_calculate_metrics`
11. guardado de reportes por modelo y series del mejor run
12. `main_pipeline.py::_save_consolidated_summary`
13. `main_pipeline.py::_find_and_save_best_params`
14. publicacion de release activa

## 7.2 Walk-forward

El walk-forward usa:

- `backtest.initial_train`
- `backtest.step`
- `backtest.horizon`

Interpretacion:

- `initial_train`: tamano de entrenamiento inicial
- `step`: cada cuantas observaciones avanza la ventana
- `horizon`: horizonte del target, normalmente `1`

Si `step=1`, el backtest es mas fino pero mas costoso.

### Horizonte economico real

En este proyecto, el horizonte economico real lo define principalmente `backtest.target`, no `backtest.horizon`.

Ejemplos:

- `ReturnFwd_1` en `M5`: retorno desde `t` hasta `t+1`, aproximadamente 5 minutos
- `ReturnFwd_2` en `M5`: retorno desde `t` hasta `t+2`, aproximadamente 10 minutos
- `ReturnFwd_3` en `M5`: retorno desde `t` hasta `t+3`, aproximadamente 15 minutos
- `ReturnFwd_4` en `M5`: retorno desde `t` hasta `t+4`, aproximadamente 20 minutos

Esto importa porque:

- si quieres capturar micro-movimientos, `ReturnFwd_1` o `ReturnFwd_2` suele ser mas coherente
- si quieres un `TP` esperado mayor, normalmente necesitas `ReturnFwd_3` o `ReturnFwd_4`

### Cambio metodologico importante

El pipeline ya no usa columnas `ReturnFwd_*` como features del modelo.

Esto evita leakage de informacion futura. Antes de este ajuste, un modelo podia ver objetivos futuros que no deberia conocer en entrenamiento o prediccion. Ahora:

- `ReturnFwd_*` solo se usa como target
- las features validas son retornos pasados, lags, indicadores tecnicos y variables contemporaneas

### Indicadores opcionales y no obligatorios

El sistema ahora soporta estos indicadores adicionales:

- `ROC_6`
- `TickVolume_ROC_3`
- `TickVolume_ZScore_20`
- `MFI_14`
- `ADX_14`

Puntos a tener en cuenta:

- `ROC_6` es momentum explicito y es el filtro mas util para una primera confirmacion
- `TickVolume_*` usa tick volume de MT5, no volumen centralizado de mercado
- `MFI_14` mezcla precio y volumen para medir presion compradora/vendedora
- `ADX_14` sirve mas como filtro de regimen que como gatillo de entrada
- ninguno de estos filtros queda forzado por defecto

### Que hace cada modelo de backtesting

`Momentum`

- modelo simple de continuidad o sesgo de muy corto plazo
- sirve como baseline rapido
- util para detectar si una estrategia compleja realmente mejora algo

`ARIMA`

- modelo lineal autoregresivo con integracion y media movil
- intenta capturar estructura temporal lineal
- suele ser mas lento y sensible a la configuracion

`PROPHET`

- modelo aditivo de tendencia y estacionalidad
- puede ser util si la serie tiene componentes suaves o repetitivos
- en intradia corto no siempre domina frente a modelos de arboles

`RandomForestRegressor`

- ensamble de arboles que captura no linealidades
- robusto con features tecnicas y lags
- normalmente mas interpretable operacionalmente que una red profunda

`HistGradientBoosting`

- boosting de arboles, rapido y potente con features tabulares
- suele ser uno de los mejores compromisos entre calidad y tiempo de entrenamiento
- en este proyecto ha sido una familia fuerte en varias corridas

`LSTM`

- red recurrente para secuencias
- puede modelar dependencias temporales complejas
- tiene mayor costo computacional y mayor sensibilidad a ruido, por eso suele dejarse desactivado al inicio

### Puntos criticos de consistencia

- `backtest.threshold_pips` y `trading.min_pips_signal` deben quedar alineados si quieres que el numero de trades del backtest sea comparable con produccion
- si agregas nuevos indicadores y quieres que el modelo aprenda de ellos, debes correr un backtest nuevo; no basta con tocar el YAML de produccion
- los filtros de confirmacion pueden mejorar calidad, pero tambien reducir mucho `n_trades`; no los actives todos a la vez sin medir impacto
- en Forex via MT5, `Volume` es normalmente tick volume, no volumen consolidado de mercado

## 7.3 Seleccion del mejor modelo

La seleccion se controla con:

```yaml
model_selection:
  primary_metric: "sharpe"
  primary_greater_is_better: true
  secondary_metric: "profit_factor"
  secondary_greater_is_better: true
  min_trades: 5
  min_test_points: 40
```

La logica actual:

- elige el mejor run por modelo con `primary_metric` y `secondary_metric`
- luego elige el campeon global con la misma logica
- marca un solo modelo con `is_best: true`

## 7.4 Metricas del backtest

El backtest usa `utils.metrics_v2` como fuente principal. Entre las metricas disponibles:

- `mae`
- `rmse`
- `hit_rate`
- `dm_stat`
- `dm_pvalue`
- `sharpe`
- `sortino`
- `calmar`
- `max_drawdown`
- `profit_factor`
- `win_rate`
- `payoff_ratio`
- `consistency_ratio`
- `avg_trade_return`
- `n_test_points`
- `n_trades`

Notas:

- `hit_rate` conserva semantica legacy: excluye `HOLD`
- `dm_stat` conserva el signo historico del proyecto

## 7.5 Artefactos versionados del backtest

Cada corrida genera:

- archivos estables, por ejemplo `summary_best_runs.csv`
- copias archivadas con `run_id`, por ejemplo `summary_best_runs_20260429_063853.csv`

Ademas publica:

- `config/config_optimizado_<run_id>.yaml`
- `outputs/models/releases/<run_id>/`
- `config/active_release.json`

Si el backtest corre con `strategy_profile.name`, tambien puede publicar:

- `config/active_release_<profile>.json`
- `config/config_optimizado_<profile>.yaml`

Esto permite que produccion use una release estable aunque un backtest nuevo este corriendo en paralelo.

## 8. Como funciona produccion

## 8.1 Flujo

1. resuelve la release activa del perfil o del modo por defecto
2. carga datos y features recientes
3. filtra modelos habilitados
4. opcionalmente opera solo el campeon si `execute_best_model_only: true`
5. carga metricas del `summary_best_runs` de la release activa
6. carga el modelo desde `outputs/models/releases/<run_id>/`
7. predice retorno
8. convierte retorno a senal con `build_signal_from_prediction`
9. opcionalmente aplica confirmacion por momentum, volumen y/o regimen
10. calcula niveles planificados y niveles reales de mercado
11. calcula lotaje con `risk_per_trade_pct`
12. guarda la fila de senal
13. ejecuta orden si `auto_execute_orders: true`
14. reconcilia reportes con `_sync_live_trade_report`

## 8.2 Como se decide una señal

Produccion transforma la prediccion en:

- `BUY`
- `SELL`
- `HOLD`

segun:

- magnitud en pips
- `trading.min_pips_signal`
- opcionalmente `confidence`
- opcionalmente confirmacion de features

Si la senal es `HOLD`, no se abre orden.

Secuencia real:

1. el modelo produce `pred_return`
2. `pred_return` se convierte a pips usando `pip_size`
3. si no supera `trading.min_pips_signal`, la salida es `HOLD`
4. si `enable_confidence_filter: true`, tambien debe superar `min_confidence`
5. si `trading.signal_confirmation.enabled: true`, la senal propuesta debe pasar la capa opcional de confirmacion

### Confirmacion hibrida opcional

El sistema soporta un filtro posterior al modelo:

```yaml
trading:
  signal_confirmation:
    enabled: false
    require_momentum_alignment: true
    momentum_column: "ROC_6"
    require_volume_confirmation: false
    volume_column: "TickVolume_ZScore_20"
    require_regime_confirmation: false
    regime_column: "ADX_14"
```

Interpretacion:

- `enabled: false`: la senal depende solo del modelo, umbral y confianza
- `require_momentum_alignment`: si se activa la capa, este es el filtro principal recomendado
- `require_volume_confirmation`: opcional; sirve para exigir que la actividad acompanhe la senal
- `require_regime_confirmation`: opcional; sirve para exigir una estructura de mercado mas tendencial

Recomendacion practica:

- empieza con momentum solamente
- deja volumen y regimen apagados al inicio
- activa filtros adicionales solo si el backtest demuestra mejora real

### Produccion por perfil

Si corres produccion por perfil, la release activa puede venir de:

- `active_release_aggressive.json`
- `active_release_balanced.json`
- `active_release_conservative.json`

Cada perfil:

- carga su propia config optimizada
- carga sus propios modelos guardados
- registra `strategy_profile` en `production_signals.csv`

## 8.3 Stop loss, take profit y lotaje

La logica actual usa:

- `risk.sl_mode`
- `risk.fixed_sl_pips`
- `risk.atr_sl_multiplier`
- `risk.tp_rr_ratio`
- `risk.entry_mode`

Puntos importantes:

- se calculan niveles planificados y niveles reales
- la ejecucion real usa el precio de mercado disponible en ese momento
- se normalizan precios segun `digits`, `point`, `stops_level` y `freeze_level`
- si la posicion queda sin proteccion, el sistema intenta reaplicar `SL/TP`
- el lote depende del riesgo monetario y de la distancia al `SL`
- el sistema ya descuenta el riesgo abierto antes de asignar un nuevo lote

## 8.4 Reportes de produccion

### `production_signals.csv`

Contiene, entre otros:

- timestamp
- release_id
- symbol
- timeframe
- model
- pred_return
- signal
- confidence
- entry_price
- live_entry_price
- sl_price
- tp_price
- live_sl_price
- live_tp_price
- volume_lots
- metricas historicas del modelo en backtest

### Chequeo rapido de salud operativa

Para confirmar que produccion esta sana, revisa primero estos 3 campos en cada nueva fila de `production_signals.csv`:

1. `timestamp`

- debe avanzar con cada nueva vela del timeframe
- si se repite durante muchos ciclos seguidos, probablemente produccion esta leyendo datos viejos o cache desactualizada

2. `signal`, `pips`, `confidence`

- `BUY` o `SELL` solo deberian aparecer cuando la magnitud esperada supera `min_pips_signal`
- si el filtro de confianza esta activo, la `confidence` debe quedar por encima de `min_confidence`
- si aparece `HOLD`, normalmente significa que la señal no paso el umbral de pips, de confianza o una confirmacion adicional

3. `volume_lots`, `live_sl_price`, `live_tp_price`

- si hay `BUY` o `SELL`, deberias ver `volume_lots > 0` y niveles `live_sl_price` / `live_tp_price` validos
- si `volume_lots = 0`, la señal fue bloqueada por riesgo disponible, lote minimo o validaciones previas a la orden
- si los niveles `live_*` estan vacios, la señal no quedo lista para ejecucion real

Lectura rapida:

- `timestamp` nuevo + `signal` nuevo + `volume_lots > 0` = pipeline sano y listo para operar
- `timestamp` repetido + mismo `signal` = probablemente no llego una vela nueva o se esta reusando cache
- `signal=BUY/SELL` pero `volume_lots=0` = el modelo ve oportunidad, pero la capa de riesgo la esta bloqueando

### Duplicados de señal

El pipeline evita reabrir la misma señal usando un `signal_id` compuesto por:

- `symbol`
- `timeframe`
- `model`
- `signal`
- `timestamp`

Por eso, si aparece en el log:

- `Señal ya ejecutada anteriormente, se omite`

normalmente significa que la señal pertenece a la misma vela y ya fue ejecutada antes. No necesariamente es un error.

### `trade_lifecycle_report.csv`

Registro consolidado del ciclo de vida de cada trade:

- señal
- ticket
- estado
- precios de entrada
- niveles de proteccion
- resultado cuando cierre

### `closed_trades_report.csv`

Vista de trades cerrados reconciliados desde MT5.

### `daily_trade_report.csv`

Resumen diario derivado del lifecycle:

- entradas del dia
- trades cerrados
- PnL
- win/loss por fecha

## 9. Como funciona el scheduler

Archivo:

```text
scheduler_automation.py
```

Comando base:

```powershell
python scheduler_automation.py --config config/config_backtest_ajustado.yaml
```

## 9.1 Que automatiza

Puede lanzar estos jobs:

- `backtest`
- `production`
- `sync_trades`

## 9.2 Como decide que config usar

El scheduler toma como base el YAML pasado en `--config`.

Pero si:

```yaml
scheduler:
  production_use_optimized_config: true
```

entonces `production` y `sync_trades` intentan usar:

1. la release activa de `active_release.json`
2. si no existe, `config/config_optimizado.yaml`

Si el job de `production` corre con perfil, intenta usar en este orden:

1. `active_release_<profile>.json`
2. `config/config_optimizado_<profile>.yaml`
3. fallback a la release por defecto si no existe release perfilada

### Produccion multi-perfil

El archivo recomendado para operar varios perfiles es:

```text
config/config_scheduler_runtime_profiles.yaml
```

Ejemplo:

```yaml
scheduler:
  production_profiles:
    - aggressive
    - balanced
  production_profile_spacing_seconds: 20
```

Interpretacion:

- el scheduler lanza un job de `production` por cada perfil
- cada job usa su propia release activa
- `production_profile_spacing_seconds` separa los arranques para que no choquen entre si

## 9.3 Locks y paralelismo

La implementacion actual separa locks:

- `backtest` usa un lock propio
- `production` y `sync_trades` comparten un lock de runtime

Esto permite:

- correr `backtest` en paralelo con produccion
- evitar que `production` y `sync_trades` se pisen entre si

Ademas existe soporte para `offset` en segundos, util para que `sync_trades` no choque con `production`.

### Como activar o desactivar perfiles

Caso 1: operar solo `balanced`

```yaml
scheduler:
  production_profiles:
    - balanced
```

Caso 2: operar `aggressive` y `conservative` a la vez

```yaml
scheduler:
  production_profiles:
    - aggressive
    - conservative
```

Caso 3: desactivar solo `aggressive`

- quitando `aggressive` de `production_profiles`

Caso 4: desactivar `aggressive` y `conservative`, dejando solo `balanced`

```yaml
scheduler:
  production_profiles:
    - balanced
```

Caso 5: desactivar toda la produccion automatica

No dejes `production_profiles` vacio pensando que eso la apaga. Si la lista queda vacia, el scheduler vuelve al modo de produccion por defecto.

Para apagar de verdad la produccion automatica:

- detiene el scheduler, o
- deja vacios `production_interval_minutes` y `production_cron`

## 9.4 Cron vs intervalos

El scheduler soporta dos estilos:

### Cron

Formato:

```text
minuto hora dia_del_mes mes dia_de_semana
```

Ejemplo:

```yaml
production_cron: "5 * * * 1-5"
```

### Intervalos simples

Ejemplo:

```yaml
backtest_interval_minutes: 2880
production_interval_minutes: 5
sync_trades_interval_minutes: 1
sync_trades_offset_seconds: 30
```

Interpretacion:

- backtest cada 2 dias
- produccion cada 5 minutos
- sync cada 1 minuto
- sync se desplaza 30 segundos para evitar colision

## 9.5 Argumentos del scheduler

CLI disponible:

```powershell
python scheduler_automation.py ^
  --config config/config_backtest_ajustado.yaml ^
  --backtest-interval-minutes 2880 ^
  --production-interval-minutes 5 ^
  --sync-interval-minutes 1 ^
  --sync-offset-seconds 30
```

Flags utiles:

- `--run-backtest-now`
- `--run-production-now`
- `--run-sync-now`

Ejemplo:

```powershell
python scheduler_automation.py --config config/config_backtest_ajustado.yaml --run-production-now
```

## 9.6 Logs del scheduler

Por defecto escribe en:

- `logs/automation_scheduler.log`

Rotacion:

- diaria por medianoche
- guarda historico en archivos tipo `automation_scheduler.log.YYYY-MM-DD`

## 10. Que informacion genera el modo EDA

El modo EDA produce cuatro grupos de informacion:

## 10.1 Estadistica descriptiva

- conteo
- media
- desviacion estandar
- minimos y maximos
- percentiles
- rango
- coeficiente de variacion

## 10.2 Analisis de retornos

- retornos simples
- log-retornos
- skewness
- kurtosis
- Sharpe anualizado
- Jarque-Bera
- VaR 95%
- CVaR 95%

## 10.3 Tests de estacionariedad y autocorrelacion

- ADF
- KPSS
- Ljung-Box
- ACF
- PACF

## 10.4 Graficos

- serie de precios
- volumen
- distribucion de retornos
- QQ plot
- ACF
- PACF
- volatilidad movil
- descomposicion estacional
- Sharpe movil
- curva de drawdown

## 10.5 Archivos del EDA

- `outputs/eda/<SYMBOL>_EDA_report.xlsx`
- `outputs/eda/*.png`
- `outputs/processed_data.csv`
- `outputs/trading_data_analysis.xlsx`

## 11. Archivos de salida mas importantes

## Backtest

- `outputs/backtest/report_<Modelo>.csv`
- `outputs/backtest/<Modelo>_<params>_series.csv`
- `outputs/backtest/<Modelo>_best_backtest_detail.csv`
- `outputs/backtest/summary_best_runs.csv`
- `outputs/backtest/summary_best_runs.xlsx`

## Modelos

- `outputs/models/releases/<run_id>/<modelo>_best.pkl`
- `outputs/models/releases/<run_id>/<modelo>_best.keras`

## Produccion

- `outputs/production/production_signals.csv`
- `outputs/production/trade_lifecycle_report.csv`
- `outputs/production/closed_trades_report.csv`
- `outputs/production/daily_trade_report.csv`

## Config y release

- `config/config_optimizado.yaml`
- `config/config_optimizado_<run_id>.yaml`
- `config/active_release.json`

## Reporte ejecutivo

- `outputs/reportes/reporte_ejecutivo_modelos.docx`

## 12. Comandos mas usados

## EDA

```powershell
python main_pipeline.py --mode eda --config config/config_backtest_ajustado.yaml
```

## Backtest

```powershell
python main_pipeline.py --mode backtest --config config/config_backtest_ajustado.yaml
```

## Backtest por perfil agresivo

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_aggressive.yaml
```

## Backtest por perfil balanceado

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_balanced.yaml
```

## Backtest por perfil conservador

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_conservative.yaml
```

Notas:

- cada backtest publica su propia release activa por perfil
- despues del primer backtest de un perfil, quedan disponibles `active_release_<profile>.json` y `config_optimizado_<profile>.yaml`
- si el alias estable de un perfil no existe todavia, necesitas correr al menos un backtest exitoso de ese perfil

## Backtest `light` por perfil

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_light.yaml
python main_pipeline.py --mode backtest --config config/config_profile_balanced_light.yaml
python main_pipeline.py --mode backtest --config config/config_profile_conservative_light.yaml
```

Uso recomendado:

- validar flujo completo rapido
- poblar releases `light`
- operar temporalmente con scheduler `light`

## Backtest `medium` por perfil

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_medium.yaml
python main_pipeline.py --mode backtest --config config/config_profile_balanced_medium.yaml
python main_pipeline.py --mode backtest --config config/config_profile_conservative_medium.yaml
```

Uso recomendado:

- actualizar releases serias de `aggressive`, `balanced` o `conservative`
- estudiar un perfil con mas rigor que `light`
- mantener costo mas bajo que el perfil canonico completo

## Test

```powershell
python main_pipeline.py --mode test --config config/config_optimizado.yaml
```

## Produccion manual

```powershell
python main_pipeline.py --mode production --config config/config_optimizado.yaml
```

## Produccion manual por perfil

```powershell
python main_pipeline.py --mode production --config config/config_optimizado_aggressive.yaml
python main_pipeline.py --mode production --config config/config_optimizado_balanced.yaml
python main_pipeline.py --mode production --config config/config_optimizado_conservative.yaml
```

Nota:

- esos alias estables solo existen si ese perfil ya publico al menos una release exitosa

## Produccion manual por perfil `light`

```powershell
python main_pipeline.py --mode production --config config/config_optimizado_aggressive_light.yaml
python main_pipeline.py --mode production --config config/config_optimizado_balanced_light.yaml
python main_pipeline.py --mode production --config config/config_optimizado_conservative_light.yaml
```

## Sincronizacion manual

```powershell
python main_pipeline.py --mode sync_trades --config config/config_optimizado.yaml
```

## Limpiar cache

```powershell
python main_pipeline.py --mode clear_cache --config config/config_backtest_ajustado.yaml
```

## Scheduler continuo

```powershell
python scheduler_automation.py --config config/config_backtest_ajustado.yaml
```

## Scheduler multi-perfil

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles.yaml
```

Uso:

- sirve para correr uno o varios perfiles en produccion
- deja el backtest para ejecucion manual

## Scheduler `light`

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles_light.yaml
```

Uso:

- operar temporalmente con perfiles `light`
- dejar corriendo produccion mientras se ejecutan backtests largos o `medium`
- ideal para validacion funcional

## Scheduler `medium`

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles_medium.yaml
```

Uso:

- operar con releases canonicas publicadas por los YAML `medium`
- reemplazar el scheduler `light` cuando un perfil serio ya tiene campeon confiable
- empezar normalmente con `balanced` y luego agregar `aggressive` o `conservative`

## Scheduler con disparo inmediato

```powershell
python scheduler_automation.py --config config/config_backtest_ajustado.yaml --run-production-now
```

## Scheduler multi-perfil con disparo inmediato

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles.yaml --run-production-now
```

## Generar reporte ejecutivo

```powershell
python generate_executive_report.py
```

## 12.1 Recetas operativas

Estas recetas asumen que ya existe al menos una release activa por perfil. Si no existe, primero corre el backtest del perfil correspondiente.

### Receta 1: operar solo `balanced`

Archivo a editar:

- `config/config_scheduler_runtime_profiles.yaml`

Bloque:

```yaml
scheduler:
  production_profiles:
    - balanced
```

Comando:

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles.yaml
```

Uso sugerido:

- modo base recomendado
- mejor equilibrio entre frecuencia y control de ruido

### Receta 2: operar `aggressive` + `balanced`

Bloque:

```yaml
scheduler:
  production_profiles:
    - aggressive
    - balanced
```

Comando:

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles.yaml
```

Uso sugerido:

- `aggressive` busca entradas mas frecuentes
- `balanced` actua como capa media de confirmacion operativa

Punto a vigilar:

- puede aumentar bastante el numero total de trades
- conviene revisar el riesgo total abierto y la convivencia de posiciones

### Receta 3: operar `balanced` + `conservative`

Bloque:

```yaml
scheduler:
  production_profiles:
    - balanced
    - conservative
```

Uso sugerido:

- `balanced` mantiene actividad
- `conservative` filtra mejor y suele operar menos

Es una combinacion razonable si quieres diversidad sin ir al extremo agresivo.

### Receta 4: apagar `aggressive` y dejar solo perfiles lentos

Bloque:

```yaml
scheduler:
  production_profiles:
    - balanced
    - conservative
```

Si quieres apagar tambien `conservative`, deja:

```yaml
scheduler:
  production_profiles:
    - balanced
```

### Receta 5: apagar toda la produccion automatica

Opcion directa:

- detener el scheduler

Opcion por config:

```yaml
scheduler:
  production_interval_minutes:
  production_cron:
```

Nota:

- no dejes `production_profiles` vacio esperando apagar produccion; eso fuerza fallback al modo por defecto

### Receta 6: validar solo momentum

Archivo sugerido:

- `config/config_profile_balanced.yaml`

Bloque:

```yaml
trading:
  signal_confirmation:
    enabled: true
    require_momentum_alignment: true
    momentum_column: "ROC_6"
    require_volume_confirmation: false
    require_regime_confirmation: false
```

Secuencia recomendada:

1. ajustar el YAML del perfil
2. correr un backtest nuevo del perfil
3. revisar `n_trades`, `profit_factor`, `sharpe` y `hit_rate`
4. si el resultado es aceptable, activar produccion para ese perfil

Comando de backtest:

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_balanced.yaml
```

### Receta 7: validar momentum en `aggressive`

Archivo:

- `config/config_profile_aggressive.yaml`

Bloque:

```yaml
trading:
  signal_confirmation:
    enabled: true
    require_momentum_alignment: true
    momentum_column: "ROC_6"
    require_volume_confirmation: false
    require_regime_confirmation: false
```

Comando:

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_aggressive.yaml
```

Uso sugerido:

- sirve para depurar si el filtro de momentum reduce demasiado ruido de `ReturnFwd_1`

### Receta 8: activar volumen despues de momentum

No actives volumen antes de validar momentum solo.

Bloque:

```yaml
trading:
  signal_confirmation:
    enabled: true
    require_momentum_alignment: true
    require_volume_confirmation: true
    volume_column: "TickVolume_ZScore_20"
    volume_min_strength: 0.0
    require_regime_confirmation: false
```

Interpretacion:

- la senal debe ir en la direccion del momentum
- ademas debe venir con actividad no debil en tick volume

### Receta 9: activacion completa y conservadora

Archivo sugerido:

- `config/config_profile_conservative.yaml`

Bloque:

```yaml
trading:
  signal_confirmation:
    enabled: true
    require_momentum_alignment: true
    require_volume_confirmation: true
    volume_column: "TickVolume_ZScore_20"
    require_regime_confirmation: true
    regime_column: "ADX_14"
    regime_min_strength: 20.0
```

Uso sugerido:

- solo despues de demostrar en backtest que el recorte de trades sigue dejando muestra suficiente

### Receta 10: secuencia minima para levantar un perfil nuevo

Ejemplo con `conservative`:

1. ajustar `config/config_profile_conservative.yaml`
2. correr:

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_conservative.yaml
```

3. confirmar que existan:

- `config/active_release_conservative.json`
- `config/config_optimizado_conservative.yaml`

4. agregar el perfil al scheduler:

```yaml
scheduler:
  production_profiles:
    - conservative
```

5. arrancar:

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles.yaml
```

## 13. Mapa de funciones por modulo

Esta seccion sirve como referencia rapida del codigo. Incluye las funciones operativas y auxiliares principales.

## 13.1 `main_pipeline.py`

### Clase `TradingPipeline`

| Funcion | Descripcion |
|---|---|
| `__init__` | Carga config, logging, directorios y estado inicial del pipeline. |
| `_get_model_selection_settings` | Normaliza la config usada para elegir mejores runs y campeon. |
| `_select_best_run` | Selecciona el mejor registro segun `model_selection`. |
| `_start_backtest_run` | Crea o reinicia el `run_id` actual del backtest. |
| `_ensure_backtest_run_label` | Garantiza que exista un `run_id` vigente. |
| `_get_backtest_output_dir` | Devuelve `outputs/backtest`. |
| `_build_backtest_archive_path` | Construye la ruta versionada con fecha para un artefacto. |
| `_archive_backtest_artifact` | Hace copia archivada de un archivo de backtest. |
| `_get_config_dir` | Devuelve `config/`. |
| `_get_models_output_dir` | Devuelve `outputs/models`. |
| `_get_release_models_dir` | Devuelve la carpeta de modelos para una release concreta. |
| `_get_active_release_manifest_path` | Devuelve la ruta de `active_release.json`. |
| `_write_yaml_atomic` | Escribe YAML de forma atomica. |
| `_write_json_atomic` | Escribe JSON de forma atomica. |
| `_copy_file_atomic` | Copia archivos de forma segura. |
| `_load_active_release_manifest` | Carga el manifiesto de release activa. |
| `_resolve_manifest_path` | Convierte rutas del manifiesto a `Path`. |
| `_resolve_active_release_assets` | Resuelve config, modelos y resumen activos. |
| `_resolve_active_release_config_path` | Obtiene el YAML activo para test o produccion. |
| `_build_dated_log_path` | Genera el nombre diario del log del pipeline. |
| `_publish_active_release` | Publica la release activa al finalizar un backtest. |
| `_ensure_mt5_client` | Garantiza cliente MT5 listo para operar. |
| `_get_live_trading_settings` | Lee la seccion `trading` relevante para produccion. |
| `_get_model_feature_columns` | Selecciona solo features validas y excluye `ReturnFwd_*` para evitar leakage. |
| `_get_signal_confirmation_settings` | Normaliza la configuracion de confirmacion opcional de senal. |
| `_evaluate_signal_confirmation` | Evalua momentum, volumen y regimen antes de autorizar una senal operable. |
| `_get_production_output_paths` | Devuelve rutas de CSV de produccion. |
| `_append_rows_to_csv` | Agrega filas a un CSV con reintentos y fallback si hay lock. |
| `_build_signal_id` | Construye un id unico por señal. |
| `_build_order_comment` | Construye comentario corto para orden MT5. |
| `_save_daily_trade_report` | Resume el lifecycle a nivel diario. |
| `_sync_live_trade_report` | Reconcilia posiciones y cierres reales con los reportes locales. |
| `_execute_live_orders` | Filtra y ejecuta ordenes reales desde filas de `production_signals`. |
| `_save_backtest_detail` | Guarda el detalle del mejor backtest por modelo. |
| `_load_config` | Carga el YAML de configuracion. |
| `_setup_logging` | Configura logger a consola y archivo. |
| `_setup_directories` | Crea directorios de salida necesarios. |
| `run` | Punto de entrada general del pipeline. |
| `_run_eda_mode` | Ejecuta el flujo EDA completo. |
| `_run_train_mode` | Ejecuta entrenamiento con split train/test. |
| `_run_backtest_mode` | Ejecuta backtest y reserva hold-out si aplica. |
| `_run_hyperparameter_tuning` | Orquesta la comparacion de modelos y combinaciones. |
| `_run_test_mode` | Valida la release optimizada sobre un set final. |
| `_find_and_save_best_params` | Elige mejores parametros, reentrena modelos y publica release. |
| `_save_model_report` | Guarda `report_<Modelo>.csv`. |
| `_save_backtest_series` | Guarda `*_series.csv` del mejor run por modelo. |
| `_save_consolidated_summary` | Construye `summary_best_runs.csv/.xlsx`. |
| `_run_walk_forward_for_params` | Ejecuta el loop walk-forward para una combinacion concreta. |
| `_train_and_predict` | Instancia el modelo, entrena y predice para una ventana. |
| `_calculate_metrics` | Calcula las metricas activas del backtest. |
| `_generate_backtest_plots_for_model` | Genera plots del mejor run por modelo. |
| `_plot_predictions_series` | Grafica observado vs predicho en backtest. |
| `_validate_model_on_test` | Evalua un modelo final sobre el hold-out. |
| `_run_production_mode` | Genera senales y ejecuta ordenes en vivo. |
| `_run_sync_trades_mode` | Reconciliacion de trades y protecciones. |
| `_get_best_model_from_config` | Lee el modelo campeon desde config. |
| `_run_clear_cache_mode` | Limpia cache de datos. |
| `_load_data` | Paso 1 del pipeline. |
| `_clean_data` | Paso 2 del pipeline. |
| `_generate_features` | Paso 3 del pipeline. |
| `_perform_eda` | Paso 4 del flujo EDA. |
| `_save_processed_data` | Guarda CSV de datos procesados. |
| `_save_dataframes_to_excel` | Guarda multiples DataFrames en un Excel consolidado. |

## 13.2 `data/data_loader.py`

### Clase `DataLoader`

| Funcion | Descripcion |
|---|---|
| `__init__` | Inicializa config MT5 y directorio de cache. |
| `is_connected` | Indica si la conexion MT5 esta activa. |
| `connect` | Conecta a MetaTrader 5. |
| `disconnect` | Cierra conexion MT5. |
| `load_data` | Carga un simbolo/timeframe desde cache o MT5. |
| `load_multiple_symbols` | Carga varios simbolos y devuelve un dict de DataFrames. |
| `_get_cache_path` | Construye el nombre del archivo cache. |
| `_is_cache_valid` | Determina si un cache sigue vigente. |
| `_load_from_cache` | Lee cache de disco. |
| `_save_to_cache` | Guarda cache de disco. |
| `get_symbol_info` | Consulta metadatos del simbolo en MT5. |
| `clear_cache` | Elimina cache total o por simbolo. |
| `validate_dataframe` | Valida estructura y consistencia del DataFrame cargado. |
| `detect_outliers` | Detecta outliers basicos en columnas numericas. |

## 13.3 `data/data_cleaner.py`

### Clase `DataCleaner`

| Funcion | Descripcion |
|---|---|
| `__init__` | Recibe configuracion de limpieza. |
| `_get_default_config` | Devuelve configuracion por defecto. |
| `clean` | Ejecuta todo el pipeline de limpieza. |
| `_ensure_sorted_index` | Ordena cronologicamente el indice. |
| `_remove_duplicates` | Elimina timestamps duplicados. |
| `_validate_ohlc` | Corrige inconsistencias `Open/High/Low/Close`. |
| `_handle_missing_values` | Maneja faltantes segun estrategia configurada. |
| `_handle_outliers` | Aplica tratamiento sobre outliers detectados. |
| `_detect_outliers` | Detecta outliers usando IQR, z-score u otro metodo. |
| `get_report` | Devuelve un resumen textual de la limpieza. |

### Clase `FeatureEngineer`

| Funcion | Descripcion |
|---|---|
| `add_returns` | Agrega retornos simples, log-retornos y targets `ReturnFwd_*`. |
| `add_technical_indicators` | Agrega SMA, EMA, RSI, MACD, Bollinger, ATR y los indicadores opcionales ROC, tick volume, MFI y ADX. |
| `add_lag_features` | Agrega rezagos por columna. |

## 13.4 `eda/exploratory_analysis.py`

### Clase `ExploratoryAnalysis`

| Funcion | Descripcion |
|---|---|
| `__init__` | Prepara directorio de salida del EDA. |
| `analyze` | Ejecuta todo el analisis exploratorio. |
| `_descriptive_statistics` | Calcula estadistica descriptiva del precio. |
| `_returns_analysis` | Calcula distribucion y riesgo de retornos. |
| `_stationarity_tests` | Ejecuta ADF y KPSS. |
| `_autocorrelation_analysis` | Ejecuta ACF y Ljung-Box. |
| `_seasonal_decomposition` | Descompone la serie en tendencia/estacionalidad/residuo. |
| `_generate_plots` | Orquesta todos los graficos del EDA. |
| `_plot_price_series` | Grafica precio y volumen. |
| `_plot_returns_distribution` | Grafica histogramas y KDE de retornos. |
| `_plot_qq` | Grafica QQ plot. |
| `_plot_acf_pacf` | Genera ACF y PACF. |
| `_plot_rolling_volatility` | Grafica volatilidad movil. |
| `_plot_decomposition` | Grafica descomposicion estacional. |
| `plot_predicted_vs_observed_returns` | Grafica observado vs pronosticado para un modelo. |
| `_plot_rolling_sharpe` | Grafica Sharpe movil. |
| `_plot_drawdown_curve` | Grafica equity y drawdown. |
| `_print_summary` | Imprime resumen ejecutivo del EDA. |
| `_save_excel_report` | Guarda el reporte Excel del EDA. |

## 13.5 `conexion/easy_Trading.py`

### Clase `easy_Trading`

| Funcion | Descripcion |
|---|---|
| `_connect` | Inicializa la conexion MT5. |
| `get_data_for_bt` | Descarga barras recientes para backtest. |
| `get_data_from_dates` | Descarga barras por rango de fechas. |
| `modify_orders` | Modifica ordenes existentes. |
| `open_operations` | API legacy para apertura de operaciones. |
| `get_account_info` | Devuelve balance y datos de cuenta. |
| `get_symbol_tick` | Consulta `bid/ask` actual. |
| `get_symbol_spec` | Consulta especificacion del simbolo. |
| `_normalize_price` | Normaliza precio a `digits` del broker. |
| `_sanitize_protection_levels` | Ajusta `SL/TP` a restricciones del broker. |
| `get_position_by_ticket` | Busca una posicion por ticket. |
| `ensure_position_protection` | Garantiza que una posicion tenga `SL/TP`. |
| `open_market_order` | Abre orden de mercado con protecciones. |
| `obtener_ordenes_pendientes` | Lista ordenes pendientes. |
| `remover_operacion_pendiente` | Cancela orden pendiente. |
| `close_all_open_operations` | Cierra multiples posiciones. |
| `get_opened_positions` | Consulta posiciones abiertas por simbolo. |
| `get_all_positions` | Consulta todas las posiciones. |
| `get_history_deals` | Consulta historial de deals cerrados. |
| `send_to_breakeven` | Mueve proteccion a break-even. |
| `calculate_position_size` | Calcula lotaje desde riesgo y `SL`. |
| `get_today_calendar` | Consulta calendario economico del dia. |

## 13.6 `scheduler_automation.py`

| Funcion | Descripcion |
|---|---|
| `parse_cron` | Convierte expresion cron de 5 campos a kwargs de APScheduler. |
| `resolve_interval_minutes` | Resuelve intervalos desde CLI o YAML. |
| `resolve_offset_seconds` | Resuelve offsets desde CLI o YAML. |
| `build_trigger` | Construye trigger cron o interval. |
| `load_yaml_config` | Carga config del scheduler. |
| `load_active_release_manifest` | Lee `active_release.json`. |
| `setup_logger` | Configura log del scheduler con rotacion diaria. |
| `resolve_scheduler_setting` | Aplica prioridad CLI > YAML > default. |
| `resolve_mode_config` | Decide que config usar por modo. |
| `acquire_lock` | Intenta tomar lock de ejecucion. |
| `release_lock` | Libera lock. |
| `resolve_lock_path` | Separa lock de `backtest` y lock de `runtime`. |
| `run_pipeline_job` | Lanza `main_pipeline.py` como subproceso. |
| `main` | Punto de entrada del scheduler. |

## 13.7 `generate_executive_report.py`

| Funcion | Descripcion |
|---|---|
| `load_table` | Carga CSV/XLSX desde una lista de rutas candidatas. |
| `format_float` | Formatea valores para el documento final. |
| `add_title` | Inserta titulo en Word. |
| `add_heading` | Inserta subtitulo en Word. |
| `add_paragraph` | Inserta parrafos con formato simple. |
| `create_summary_table` | Crea tabla resumen desde `summary_best_runs`. |
| `get_champion_row` | Elige al campeon usando `is_best` si existe. |
| `generate_executive_report` | Construye y guarda el `.docx`. |

## 14. Flujo recomendado de uso

## Opcion 1: analisis manual

1. ajustar `config/config_backtest_ajustado.yaml`
2. correr backtest
3. revisar `summary_best_runs`
4. revisar `config_optimizado.yaml` y `active_release.json`
5. correr `production` manual
6. revisar `production_signals.csv`

## Opcion 2: automatizacion completa

1. ajustar `config/config_backtest_ajustado.yaml`
2. correr un backtest manual inicial
3. verificar que exista `config/active_release.json`
4. activar scheduler
5. monitorear logs y CSV de produccion

## 15. Problemas comunes

## No aparecen señales en produccion

Posibles causas:

- `production` no ha corrido, solo `sync_trades`
- la señal fue `HOLD`
- `volume_lots` resulto `0.0`
- no existe release activa valida

Revisa:

- `logs/automation_scheduler.log`
- `logs/trading_YYYY-MM-DD.log`
- `outputs/production/production_signals.csv`

## `sync_trades` corre pero `production` no

Revisa:

- `sync_trades_offset_seconds`
- locks del scheduler
- que el job de produccion no este siendo omitido por colision

## Produccion usa datos viejos

Revisa:

- `data.use_cache`
- `data.cache_expiry_hours`
- `data.runtime_use_cache`
- `data.runtime_cache_expiry_minutes`
- timeframe configurado

Nota:

- `backtest` puede seguir usando cache normalmente
- `production` y `sync_trades` ahora usan una politica de runtime aparte
- si `runtime_use_cache` no se define, el pipeline opera por defecto sin cache en runtime para evitar reciclar la misma vela

## No se ejecutan ordenes

Revisa:

- `trading.auto_execute_orders: true`
- `allow_multiple_positions`
- conexion MT5
- señal distinta de `HOLD`
- lotaje mayor a cero

## El backtest tarda demasiado

Revisa:

- `data.n_bars`
- `validation.n`
- `backtest.initial_train`
- `backtest.step`
- cantidad de modelos habilitados
- tamano de `param_grid`

## 16. Observaciones finales

- usa `config/config_backtest_ajustado.yaml` como archivo maestro
- deja `config/config_optimizado.yaml` como archivo derivado de produccion
- no edites manualmente `active_release.json`
- si quieres operar en vivo, valida primero el flujo completo con `auto_execute_orders: false`
