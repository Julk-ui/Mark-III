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

### 1.1 Explicacion simple para no expertos

Si nunca has trabajado con un robot de trading, piensa en Mark III como un sistema con 5 decisiones seguidas:

1. **Elegir una tesis**
- decide si en ese momento la idea es `BUY`, `SELL` o `HOLD`
- eso sale del modelo principal y del filtro del bundle

2. **Decidir si la entrada vale la pena**
- aunque la direccion pueda ser correcta, el robot revisa si el punto de entrada es malo
- por ejemplo: entrar demasiado arriba en una compra, demasiado abajo en una venta, o sobre una mecha de rechazo

3. **Elegir como entrar**
- puede entrar de una vez a mercado
- puede dividir la entrada en una parte inmediata y otra pendiente
- puede esperar un mejor retroceso antes de ejecutar
- puede no entrar si el punto se volvio malo

4. **Gestionar la posicion si ya entro**
- mueve el `SL` a `break-even`
- hace parciales antes del `TP`
- cierra antes si detecta debilidad, reversal o una senal fuerte opuesta

5. **Medir si se esta degradando**
- el scheduler vigila el rendimiento reciente
- si un perfil entra en drift, deja trazas para recomendar nuevo backtest o nueva release

En lenguaje simple:

- `backtest` busca que combinacion de reglas y modelos sirve mejor con historico
- `release` es la configuracion publicada que quedo aprobada para operar
- `production` genera la idea de trade
- `sync_trades` gestiona lo que ya esta abierto
- `monitor_runtime` protege cuando el mercado cambia rapido

### 1.2 Que significa que una senal no abra una orden

No toda fila `BUY` o `SELL` termina en una orden real.

Puede pasar una de estas cosas:

- la tesis existe, pero la entrada se degrada a `retrace_only`
- la idea queda `staged`, esperando mejor precio
- `M1` no confirma el timing y la candidata sigue viva sin ejecutar
- el sistema detecta que la idea es valida, pero el punto ya no lo es
- el lote calculado queda demasiado pequeno y la pierna no puede abrirse

Eso no siempre es un error. Muchas veces significa que la direccion era razonable, pero el robot decidio no perseguir precio.

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

Ejemplo importante:

- si corres un backtest de `aggressive` y luego uno de `balanced_medium`, no se pisan
- `aggressive` actualiza `active_release_aggressive.json`
- `balanced_medium` publica sobre el perfil canonico `balanced`, asi que actualiza `active_release_balanced.json`
- si luego el scheduler opera ambos perfiles, `aggressive` usa su propia release activa y `balanced` usa la suya

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

Notas operativas nuevas:

- una fila de `production` ya no implica siempre una orden inmediata
- la seÃ±al puede quedar retenida como candidata staged para esperar retroceso de la vela
- `production` puede crear o actualizar candidatas en `staged_signal_report.csv`
- `sync_trades` no activa candidatas staged; solo gestiona posiciones ya abiertas

Archivos de salida:

- `outputs/production/production_signals.csv`
- `outputs/production/staged_signal_report.csv`
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
- `CloseLocationValue`
- `DirectionalVolumeProxy`
- `DirectionalVolumeProxy_ZScore_20`
- `MFI_14`
- `ADX_14`

Puntos a tener en cuenta:

- `ROC_6` es momentum explicito y es el filtro mas util para una primera confirmacion
- `TickVolume_*` usa tick volume de MT5, no volumen centralizado de mercado
- `DirectionalVolumeProxy*` es un proxy direccional construido con tick volume y la posicion del cierre dentro del rango de la vela; no equivale a order flow real
- `MFI_14` mezcla precio y volumen para medir presion compradora/vendedora
- `ADX_14` sirve mas como filtro de regimen que como gatillo de entrada
- ninguno de estos filtros queda forzado por defecto

### Que hace cada modelo de backtesting

Antes de ver modelos concretos, hay dos modos de aprendizaje distintos en el proyecto:

#### `return_regression`

El objetivo tipico es algo como:

- `ReturnFwd_1`
- `ReturnFwd_2`
- `ReturnFwd_n`

Interpretacion:

- el modelo no predice el precio exacto futuro
- predice un retorno futuro `close-to-close`
- luego el pipeline convierte ese retorno a pips
- con eso decide `BUY`, `SELL` o `HOLD`

#### `barrier_event`

El objetivo responde a una pregunta operativa:

- en las proximas `N` velas, toca primero `+X` pips o `-X` pips

Interpretacion:

- el modelo no predice un retorno continuo
- predice probabilidades de evento
- el pipeline construye la senal desde `prob_up`, `prob_hold` y `prob_down`

### Modelos de `return_regression`

| Modelo | Usa como entrada | Que predice | Cuando conviene |
| --- | --- | --- | --- |
| `Momentum` | Solo historial reciente del target `y_train` | Promedio de los ultimos targets observados | Como `baseline` rapido y fuerte para verificar si un modelo complejo realmente agrega valor |
| `RandomWalk` | Historial muy simple del target | Referencia trivial de persistencia o cambio nulo segun implementacion | Solo como referencia minima |
| `ARIMA` | La serie temporal del target y, en este proyecto, una capa de residuos con features tabulares | Retorno futuro continuo | Cuando hay dependencia temporal lineal de corto plazo y se quiere una senal mas reactiva que Prophet |
| `PROPHET` | Serie temporal del target y regresores laggeados opcionales | Retorno futuro continuo, normalmente suave | Cuando hay tendencia o estacionalidad mas estable; en intradia `M5` suele quedar demasiado amortiguado |
| `Ridge` | Features tabulares del instante `t` | Retorno futuro continuo | Baseline lineal robusta y barata para medir si los modelos complejos realmente agregan no linealidad util |
| `RandomForestRegressor` | Features tabulares: retornos, lags, indicadores, volumen | Retorno futuro continuo | Cuando se quiere capturar no linealidades e interacciones entre indicadores |
| `HistGradientBoostingRegressor` | Features tabulares | Retorno futuro continuo | Uno de los mejores compromisos CPU/calidad para tabular financiero |
| `LSTM` | Secuencias de observaciones, no solo una fila tabular | Retorno futuro continuo | Solo si se justifica el costo; suele ser mas pesado y fragil que los arboles en CPU |

### Modelos de `barrier_event`

| Modelo | Usa como entrada | Que predice | Cuando conviene |
| --- | --- | --- | --- |
| `LogisticRegressionClassifier` | Features tabulares del instante `t` | `prob_up`, `prob_hold`, `prob_down` | Como baseline probabilistico serio y muy barato |
| `RandomForestClassifier` | Features tabulares | Probabilidades por clase de evento | Cuando hay relaciones no lineales, pero sin costo excesivo |
| `ExtraTreesClassifier` | Features tabulares | Probabilidades por clase de evento | Variante muy barata y agresiva frente a `RandomForestClassifier`, util para explorar diversidad sin gran costo |
| `HistGradientBoostingClassifier` | Features tabulares | Probabilidades por clase de evento | Candidato principal cuando se busca equilibrio entre tiempo y calidad |

### Lectura cuantitativa rapida de cada familia

`Momentum`

- no usa indicadores complejos
- usa solo el comportamiento reciente del target
- por eso puede salir sorprendentemente fuerte
- en este proyecto debe tratarse como `baseline`, no como campeon final por defecto

`ARIMA`

- modela estructura temporal lineal inmediata
- en esta implementacion puede apoyarse en residuos con features
- suele ser el modelo temporal clasico mas util en intradia corto
- si gana muy justo y con pocos trades, la ventaja puede no ser robusta

`PROPHET`

- si predice, pero suele hacerlo con amplitud pequena
- en `M5 + ReturnFwd_1` muchas veces genera valores demasiado cercanos a cero
- eso produce muchos `HOLD`, no porque falle el modelo, sino porque no cruza el umbral operativo
- suele tener mas sentido en horizontes mas suaves o mas largos

`RandomForestRegressor`

- divide el espacio de features con reglas no lineales
- robusto y razonable en CPU
- muchas veces predice valores amortiguados cerca de cero si el target es muy ruidoso

`HistGradientBoostingRegressor`

- corrige errores de arboles previos de forma secuencial
- suele superar a `RandomForestRegressor` en tabular corto
- es una de las mejores opciones cuando no se quiere usar redes

`LogisticRegressionClassifier`

- aprende fronteras lineales entre `up`, `hold` y `down`
- es muy barata y facil de interpretar
- si queda en `0` trades, normalmente significa que el problema es demasiado no lineal o que el target esta desbalanceado

`RandomForestClassifier`

- ensamble de arboles de clasificacion
- puede rescatar no linealidades sin disparar el costo
- aun asi puede sesgarse demasiado a `hold` si el target esta mal calibrado

`ExtraTreesClassifier`

- similar a `RandomForestClassifier`, pero con mayor aleatoriedad en los cortes
- muy barato en CPU
- util para comprobar si un poco mas de diversidad mejora cobertura o estabilidad
- no siempre gana, pero suele ser una buena familia adicional para experimentar

`HistGradientBoostingClassifier`

- hoy es la familia mas fuerte del modo `barrier_event`
- suele producir la mejor senal entre clasificadores tabulares en CPU
- es la opcion mas balanceada para este proyecto si se quiere una senal operativa interpretable

### Como interpretar una prediccion en produccion

En `return_regression`:

- el modelo produce `y_pred`
- `y_pred` representa un retorno futuro
- el pipeline lo convierte a `predicted_pips`
- luego aplica filtro de `pips`, confianza y confirmacion

En `barrier_event`:

- el modelo produce `prob_up`, `prob_hold`, `prob_down`
- el pipeline decide la senal segun umbral probabilistico y margen entre clases
- ademas calcula `expected_move_pips` y niveles objetivo de `TP` y `SL` de la senal

### Recomendacion practica para este proyecto

- `Momentum`: baseline
- `ARIMA`: mejor modelo temporal clasico de corto plazo
- `PROPHET`: util solo si el horizonte o la estructura temporal le favorecen
- `HistGradientBoostingRegressor`: mejor regresor tabular generalista
- `HistGradientBoostingClassifier`: mejor candidato actual para el enfoque de barrera
- `LSTM`: dejarla fuera al inicio salvo que haya una razon fuerte para asumir ese costo

### Puntos criticos de consistencia

- `backtest.threshold_pips` y `trading.min_pips_signal` deben quedar alineados si quieres que el numero de trades del backtest sea comparable con produccion
- si agregas nuevos indicadores y quieres que el modelo aprenda de ellos, debes correr un backtest nuevo; no basta con tocar el YAML de produccion
- los filtros de confirmacion pueden mejorar calidad, pero tambien reducir mucho `n_trades`; no los actives todos a la vez sin medir impacto
- en Forex via MT5, `Volume` es normalmente tick volume, no volumen consolidado de mercado
- si ningun modelo candidato cumple los minimos de `model_selection`, el pipeline ya no deberia publicar una nueva release activa por fallback

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
  publish_requires_candidate_thresholds: true
```

La logica actual:

- elige el mejor run por modelo con `primary_metric` y `secondary_metric`
- luego elige el campeon global con la misma logica
- marca un solo modelo con `is_best: true`
- si `publish_requires_candidate_thresholds: true` y ningun candidato cumple `min_trades` y `min_test_points`, no publica una nueva `active_release_<profile>.json`

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

### Entradas escalonadas y staging

La apertura en produccion ya no es solo "senal valida -> orden inmediata". Ahora existen dos capas:

- `entry_management`
- `entry_staging`

`entry_management` divide una entrada ya autorizada en:

- una pierna `market`
- una pierna `limit` de mejora

Configuracion tipica:

```yaml
entry_management:
  enabled: true
  mode: split_retrace_limit
  initial_market_fraction: 0.5
  pending_fraction: 0.5
  retrace_fraction_of_stop: 0.55
  cancel_pending_after_bars: 2
  disable_pending_when_filter_hold: true
```

Semantica:

- la `limit` se coloca a una fraccion del camino entre la entrada y el `SL`
- la `limit` calcula su propio `SL` y `TP` desde su propio precio de entrada
- si `disable_pending_when_filter_hold: true`, una senal con `filter_signal = HOLD` entra sin segunda pierna `limit`

Lectura simple:

- `market`: la parte que entra ya
- `pending_limit`: la parte que espera mejor precio en MT5
- `staged`: idea retenida internamente; todavia no es una orden real
- `retrace_only`: la tesis sigue viva, pero el robot solo quiere entrar si el precio mejora
- `skip`: la idea se descarta

Para una evolucion a varias patas, ver [docs/entry_grid_v1_design.md](docs/entry_grid_v1_design.md). Esa propuesta define una `v1` de `3` patas con riesgo total fijo, cierre de patas peores en positivo y una pata final `runner`.

`entry_staging` retiene senales antes de abrir una orden real:

```yaml
entry_staging:
  enabled: true
  mode: candidate_retrace
  max_stage_bars: 2
  convert_direct_filter_hold_to_staged: true
  convert_direct_confirmed_to_staged: true
  pilot_entry_enabled: true
  pilot_convert_to_staged: true
```

En `tp5` esto significa:

- las senales directas confirmadas ya no entran necesariamente al cierre de la vela
- las senales con `filter_signal = HOLD` se retienen para esperar un mejor retroceso
- las `pilot` tambien se convierten a staging en vez de entrar `market_only`

Para estas activaciones staged:

- el trigger se calcula desde el retroceso de la vela de senal
- el `SL` puede anclarse al extremo de la vela de senal
- se puede usar buffer por pips y por `ATR`
- `dynamic_stop_min_pips` fuerza un stop minimo aunque la vela sea demasiado pequena

Importante para leer el live:

- una candidata `staged` puede existir sin que veas ninguna orden pendiente en MT5
- eso es correcto cuando la idea sigue en observacion y el robot todavia no quiere enviar una `BUY_LIMIT` o `SELL_LIMIT`
- cuando el flujo dice `market + pending`, la pierna pendiente ya deberia verse como orden real

### Volumen direccional en activacion staged

El perfil `aggressive_hybrid_v1_3_tp5_sl3` puede exigir que una candidata staged solo se active si el volumen acompana el lado del trade.

Configuracion:

```yaml
entry_staging:
  require_directional_volume_activation: true
  directional_volume_column: "DirectionalVolumeProxy_ZScore_20"
  directional_volume_buy_min: 0.1
  directional_volume_sell_max: -0.1
```

Interpretacion:

- `BUY` staged: el proxy direccional debe ser suficientemente positivo
- `SELL` staged: el proxy direccional debe ser suficientemente negativo
- si el precio llega al trigger pero el volumen no acompana, la candidata sigue viva como `WAITING_VOLUME_CONFIRMATION`

### Control contextual y calidad de entrada

La capa de ejecucion ya no depende solo de la direccion del modelo. Antes de decidir si entra `market`, `retrace_only` o `skip`, el pipeline evalua:

- contradiccion suave o dura del contexto (`CloseLocationValue`, `DirectionalVolumeProxy_ZScore_20`, cuerpo de vela)
- si la entrada inmediata queda demasiado arriba para `BUY` o demasiado abajo para `SELL`
- si la vela ya muestra rechazo contrario
- si el precio esta estirado frente a `EMA_20` o `SessionVWAP`
- si hay alineacion estructural minima con:
  - `ROC_3`
  - `ROC_6`
  - pendiente de `EMA_20`
  - pendiente de `SessionVWAP`
  - ruptura reciente (`BreakAboveRecentHigh3` / `BreakBelowRecentLow3`)

Features nuevas de ejecucion:

- `EMA_20`
- `EMA20SlopePips`
- `SessionVWAP`
- `SessionVWAPSlopePips`
- `SignedDistanceToEMA20Pips`
- `SignedDistanceToVWAPPips`
- `EMA20StretchVsAvgRange`
- `VWAPStretchVsAvgRange`

Campos nuevos que quedan publicados en `production_signals.csv`:

- `entry_context_quality_score`
- `entry_context_quality_decision`
- `entry_context_quality_alignment_hits`
- `entry_context_quality_alignment_total`
- `entry_context_quality_signed_distance_to_ema20_pips`
- `entry_context_quality_signed_distance_to_vwap_pips`
- `entry_context_quality_ema20_stretch_vs_avg_range`
- `entry_context_quality_vwap_stretch_vs_avg_range`
- `entry_context_quality_range_vs_avg_value`
- `entry_context_quality_stretched_from_ema20`
- `entry_context_quality_stretched_from_vwap`
- `entry_context_quality_stretched_entry`

Semantica operativa:

- `entry_quality_score >= entry_quality_min_score_for_market`: puede mantener `market` o `split_retrace_limit`
- `entry_quality_min_score_for_retrace <= score < entry_quality_min_score_for_market`: fuerza `entry_quality_retrace_only`
- `score < entry_quality_min_score_for_retrace`: bloquea la idea con `entry_quality_low_skip`

En la practica:

- la direccion puede seguir siendo correcta aunque el punto de entrada sea malo
- si el punto es malo, el sistema ya no deberia perseguir precio por defecto
- el score de entrada convive con `context_guard`, `entry_staging`, `cluster_guard` y el filtro del bundle

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
- staged_candidate_id
- staged_status
- staged_action
- staged_reason
- staged_trigger_price
- staged_expires_at
- staged_activation_reason
- staged_directional_volume_column
- staged_directional_volume_value
- staged_directional_volume_passed
- staged_directional_volume_reason
- metricas historicas del modelo en backtest

### `staged_signal_report.csv`

Registro de candidatas retenidas antes de abrir una orden real.

Campos importantes:

- `candidate_id`
- `candidate_mode`
- `side`
- `reference_price`
- `trigger_price`
- `custom_stop_price`
- `candidate_volume_scale`
- `expires_at`
- `status`
- `status_reason`
- `activation_reason`
- `last_directional_volume_column`
- `last_directional_volume_value`
- `last_directional_volume_passed`
- `last_directional_volume_reason`

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

4. `staged_status`, `staged_action`

- `ACTIVE` + `WAITING` = existe candidata, pero aun no llego al trigger
- `ACTIVE` + `WAITING_VOLUME_CONFIRMATION` = el precio ya estaba en zona, pero el volumen direccional no acompano
- `ACTIVATED` = la candidata ya se convirtio en trade real
- `CANCELLED` = vencio, fue contradicha o aparecio una senal opuesta

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

Importante:

- `production` puede dejar una fila `HOLD` y aun asi crear una candidata activa en `staged_signal_report.csv`
- `sync_trades` no activa candidatas staged; solo gestiona posiciones ya abiertas
- la activacion o cancelacion de candidatas staged ocurre dentro del siguiente ciclo de `production`

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

Interpretacion practica:

- no existe una sola "ultima release global" cuando operas por perfil
- cada perfil mantiene su propio puntero a la ultima release valida
- por eso `aggressive`, `balanced` y `conservative` pueden convivir con releases distintas al mismo tiempo

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

Ejemplo concreto:

- `aggressive` puede estar usando la release `20260501_075630`
- `balanced` puede estar usando la release `20260506_051115`
- si `production_profiles` contiene ambos, el mismo scheduler lanzara dos jobs:
  - uno para `aggressive` con `active_release_aggressive.json`
  - otro para `balanced` con `active_release_balanced.json`
- no se mezclan modelos ni configs entre perfiles

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

Si quieres operar `aggressive + balanced` a la vez, deja en `config/config_scheduler_runtime_profiles_medium.yaml`:

```yaml
scheduler:
  production_profiles:
    - aggressive
    - balanced
  sync_trades_profile: balanced
```

Con esa configuracion:

- `aggressive` tomara la release apuntada por `config/active_release_aggressive.json`
- `balanced` tomara la release apuntada por `config/active_release_balanced.json`
- `sync_trades` seguira usando `balanced` como perfil base de reconciliacion

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

## 17. Target de barrera y perfiles nuevos

Ahora existe un modo adicional de aprendizaje:

- `backtest.target_mode: barrier_event`

Ese modo deja de predecir retorno `close-to-close` como objetivo principal y pasa a responder:

- si en las proximas `N` velas toca primero `+X` pips o `-X` pips

Columnas nuevas que puede generar el pipeline:

- `BarrierDir_<Xp>_<Nb>`
- `BarrierReturn_<Xp>_<Nb>`
- `BarrierMovePips_<Xp>_<Nb>`
- `BarrierBarsToTouch_<Xp>_<Nb>`
- `BarrierAmbiguous_<Xp>_<Nb>`
- `MFEPips_<Xp>_<Nb>`
- `MAEPips_<Xp>_<Nb>`

Perfiles de barrera agregados:

- `config/config_profile_aggressive_barrier.yaml`
- `config/config_profile_aggressive_barrier_v2.yaml`
- `config/config_profile_aggressive_barrier_v3.yaml`
- `config/config_profile_aggressive_barrier_v4.yaml`
- `config/config_profile_balanced_barrier.yaml`

Comandos:

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_barrier.yaml
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_barrier_v2.yaml
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_barrier_v3.yaml
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_barrier_v4.yaml
python main_pipeline.py --mode backtest --config config/config_profile_balanced_barrier.yaml
```

`aggressive_barrier_v3` mantiene la estructura de `v2` y anade `ExtraTreesClassifier` al torneo de modelos de barrera sin pisar la release activa de `aggressive_barrier_v2`.

`aggressive_barrier_v4` anade features de microestructura / price action, suaviza ligeramente los umbrales de probabilidad (`0.52 / 0.03 / 0.52`) y da mas profundidad a `RandomForestClassifier` / `ExtraTreesClassifier` para intentar que participen realmente en el torneo sin disparar demasiado el costo computacional.

## 18. Auditoria visual de trades en backtest

El mejor run de cada modelo ahora puede generar una auditoria visual adicional, pensada para revisar:

- entrada
- `TP` objetivo de la senal
- `SL` objetivo de la senal
- salida real
- si el trade termino en `WIN`, `LOSS`, `TIMEOUT` o `AMBIGUOUS`

Archivos nuevos del backtest:

- `outputs/backtest/<Modelo>_<params>_trade_audit.csv`
- `outputs/backtest/<Modelo>_<params>_trade_audit_summary.csv`
- `outputs/backtest/<Modelo>_<params>_monthly_stability.csv`
- `outputs/backtest/plots/<Modelo>_<params>_trade_audit.png`

El grafico `trade_audit.png` muestra en una linea:

- precio real
- punto de entrada
- `TP` y `SL` objetivo de la senal
- punto de salida

## 19. Perfiles dedicados a ARIMA y PROPHET

Tambien existen perfiles de costo contenido para comparar `ARIMA` y `PROPHET` sin mezclar el resto del stack:

- `config/config_profile_aggressive_arima_prophet_v1.yaml`
- `config/config_profile_balanced_arima_prophet_v1.yaml`

Comandos:

```powershell
python main_pipeline.py --mode backtest --config config/config_profile_aggressive_arima_prophet_v1.yaml
python main_pipeline.py --mode backtest --config config/config_profile_balanced_arima_prophet_v1.yaml
```

Estos perfiles:

- dejan `Momentum` como `baseline`
- compiten solo `ARIMA` y `PROPHET`
- publican release activa propia por perfil
- generan tambien auditoria visual de trades

## 20. Scheduler para `aggressive_barrier_v2`

Existe un scheduler dedicado a la release activa del perfil `aggressive_barrier_v2`:

- `config/config_scheduler_runtime_profiles_barrier_v2.yaml`

Comando:

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles_barrier_v2.yaml
```

Disparo inmediato:

```powershell
python scheduler_automation.py --config config/config_scheduler_runtime_profiles_barrier_v2.yaml --run-production-now --run-sync-now
```

Ese scheduler resuelve automaticamente la ultima release activa de:

- `active_release_aggressive_barrier_v2.json`

## 21. Gestion de posiciones abiertas en `sync_trades`

`sync_trades` ya no solo reconcilia cierres y protecciones. Ahora aplica tres capas:

- gestion progresiva por avance hacia `TP`
- proteccion y salida contextual desde `runtime_monitor`
- cierres condicionados por senal opuesta del mismo perfil

### 21.1 Gestion progresiva tipo grilla

Configuracion actual por perfil live:

- perfil viejo `aggressive_hybrid_v1_3_tp5_sl3`
  - empieza a gestionar desde `20%` del camino al `TP`
  - descarga cada `5%`
  - cierra `15%` del volumen remanente por etapa
  - mueve `SL` a `break-even` desde `20%`
  - puede ejecutar hasta `4` acciones por ciclo

- perfil nuevo `aggressive_hybrid_v1_6_m15_signal_exec_m5_fast`
  - empieza a gestionar desde `20%` del camino al `TP`
  - descarga cada `5%`
  - cierra `25%` del volumen remanente por etapa
  - mueve `SL` a `break-even` desde `20%`
  - puede ejecutar hasta `4` acciones por ciclo

Regla adicional importante:

- si el trade ya supera `95%` del camino al `TP`
- y el parcial siguiente quedaria por debajo del `lot_step` minimo del broker
- el sistema cierra el remanente completo en vez de dejar una cola muy pequena viva

Semantica:

- el avance se mide como porcentaje del recorrido hacia el `TP`
- cuanto antes llega a verde, antes empieza a proteger
- los lotes pequenos pueden no permitir parciales finos; por eso existe el cierre total cerca del `TP`
- esta capa busca cobrar antes, no esperar siempre el `TP` completo

Todavia se conservan y se registran tambien los nombres clasicos de etapas, por ejemplo:

- `progress_50_close_30`
- `progress_70_close_50`
- `progress_85_close_80`
- `grid_progress_30_close_10`
- `grid_progress_35_close_10`

### 21.2 Runtime monitor: proteccion y cierres contextuales

`runtime_monitor` ahora vigila, entre otros:

- `reversal`
- `lateralization`
- `no_followthrough`
- `compact_oscillation`
- `shock_reversal`
- senal opuesta del mismo perfil

Comportamiento importante:

- si detecta debilidad y la posicion ya esta protegible, puede mover a `break-even`
- si la debilidad aparece con progreso suficiente, puede hacer parcial o cierre total
- `shock_reversal` cierra completo solo si el trade ya esta no negativo; no realiza perdida
- si aparece una senal fuerte opuesta del mismo perfil, la posicion vieja puede quedar armada como `first-to-exit` y cerrar en cuanto vuelva a `break-even`
- si una pierna `market` ya va suficientemente bien, la `pending_limit` sobrante puede cancelarse temprano para no seguir apilando riesgo

Regla operativa actual:

- la cancelacion temprana de `pending` por progreso arranca desde `20%` del camino al `TP`

### 21.3 Cierre por senal opuesta

La regla actual es intencionalmente conservadora:

- no cierra por senal opuesta si la posicion sigue en rojo
- si esta en `break-even` o positiva, puede cerrar completa
- si todavia no puede cerrar, queda armada para salir en el primer `BE`

Esto evita liquidar contradicciones a perdida solo porque aparecio una tesis nueva.

### 21.4 Campos relevantes en `trade_lifecycle_report.csv`

- `managed_stage_ids`
- `break_even_applied`
- `break_even_applied_time`
- `break_even_sl_price`
- `last_management_time`
- `last_management_action`
- `last_partial_close_volume`
- `partial_close_total_volume`
- `remaining_volume_lots_estimate`
- `management_progress_to_tp`
- `trade_management_comment`
- `runtime_monitor_action`
- `runtime_monitor_reason`

Nota operativa:

- toda esta gestion ocurre dentro de `sync_trades` y `monitor_runtime`
- no reemplaza la logica de `production`; solo actua despues de que existe una posicion abierta
- si el lote es demasiado pequeno para un parcial sin violar `volume_min`, la etapa se registra y el parcial se omite

## 22. Stack híbrido `primary + filter`

El pipeline ahora soporta un modo híbrido para que backtest y producción usen la misma lógica final de señal:

- un modelo principal predice dirección y magnitud (`pred_return`, `predicted_pips`)
- un modelo filtro predice probabilidad operable (`prob_up`, `prob_hold`, `prob_down`)
- la señal final solo entra si ambos están alineados según las reglas del perfil

Configuración base:

```yaml
prediction_stack:
  mode: "hybrid_primary_plus_filter"
  primary_models: [ARIMA, Ridge, HistGradientBoosting]
  filter_models: [HistGradientBoostingClassifier, LogisticRegressionClassifier, ExtraTreesClassifier]
  require_alignment: true
  top_k_primary_for_bundle_eval: 2
  top_k_filter_for_bundle_eval: 2
```

Compuertas soportadas:

- `filter_gate_mode: "full_signal"`
  El filtro debe convertirse por si mismo en `BUY` o `SELL`.
- `filter_gate_mode: "direction_support"`
  El filtro no necesita emitir una señal completa; basta con que apoye la
  dirección del primario con suficiente probabilidad y margen.
- `filter_gate_mode: "support_score"`
  El filtro actúa como veto suave. Se calcula:
  - `support_score = prob_lado_primario - prob_lado_opuesto`
  - si el `support_score` supera `support_score_min`, el bundle puede pasar
  - si el lado opuesto domina por más de `contradiction_margin`, el filtro contradice la idea del primario

En este modo:

- `backtest.target` sigue siendo el target de retorno del modelo principal
- `backtest.filter_target` define el target de barrera del filtro probabilístico
- el campeón publicado ya no es solo un modelo, sino un `decision_bundle`

La release optimizada guarda:

- `models`: mejores parámetros por modelo individual
- `decision_bundle`: pareja ganadora `primary_model + filter_model`

Producción usa ese mismo `decision_bundle` para:

- generar `BUY / SELL / HOLD`
- conservar `pred_return`, `pips`, `confidence`
- guardar `prob_up`, `prob_hold`, `prob_down`
- publicar `signal_target_tp_price` y `signal_target_sl_price`

En la version `tp5` mas reciente, la capa de ejecucion alrededor del bundle funciona asi:

- `filter_signal = HOLD`: la idea puede pasar a `entry_staging` y esperar retroceso de la vela
- `direct_confirmed`: tambien puede convertirse a staging para evitar entrar justo en el cierre
- `pilot`: puede retenerse como `pilot_candle_retrace` en vez de entrar a mercado inmediatamente
- la activacion staged puede exigir volumen direccional compatible antes de abrir la orden

Perfil inicial listo para correr:

- `config/config_profile_aggressive_hybrid_v1.yaml`
- `config/config_profile_aggressive_hybrid_v1_1.yaml`
- `config/config_profile_aggressive_hybrid_v1_2.yaml`
- `config/config_profile_aggressive_hybrid_v1_3_tp4_sl2_5.yaml`
- `config/config_profile_aggressive_hybrid_v1_3_tp5_sl3.yaml`

Comando:

```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1.yaml
```

Versión menos rígida del filtro:

```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1_1.yaml
```

Versión con `support_score` y filtro de barrera más permisivo:

```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1_2.yaml
```

VersiÃ³n orientada a `TP 4 / SL 2.5` con primario `ReturnFwd_3` y filtro `BarrierReturn_4p_4b`:

```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1_3_tp4_sl2_5.yaml
```

VersiÃ³n orientada a `TP 5 / SL 3` con primario `ReturnFwd_4` y filtro `BarrierReturn_5p_5b`:

```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1_3_tp5_sl3.yaml
```

## 23. Estado operativo actual del live

Esta seccion resume el comportamiento vigente del sistema en produccion para no depender solo de notas dispersas de cambios.

### 23.1 Perfiles live actuales

Perfil viejo principal:

- `aggressive_hybrid_v1_3_tp5_sl3`
- timeframe operativo: `M5`
- `magic_number: 202357`
- `order_comment_prefix: MarkIII_AggH13B`
- riesgo por trade:
  - `risk_per_trade_pct: 0.01`
  - `max_total_open_risk_pct: 0.03`

Perfil nuevo en paralelo:

- `aggressive_hybrid_v1_6_m15_signal_exec_m5_fast`
- señal en `M15`, ejecucion y reevaluacion frecuente
- `magic_number: 202364`
- `order_comment_prefix: MarkIII_M15Sig`
- riesgo por trade:
  - `risk_per_trade_pct: 0.0035`
  - `max_total_open_risk_pct: 0.012`

Scheduler dual recomendado:

- `config/config_scheduler_runtime_profiles_live_vs_m15_signal_exec_m5_fast.yaml`

Comando:

```powershell
Set-Location "C:\Users\USER\Documentos\Maestria\Mark-III"
Remove-Item logs\automation_scheduler*.lock -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scheduler_automation.py --config config/config_scheduler_runtime_profiles_live_vs_m15_signal_exec_m5_fast.yaml --run-production-now --run-sync-now --run-monitor-now
```

### 23.2 Como interpretar el scheduler en multi-perfil

El resumen del scheduler se emite por perfil, no como total global de MT5.

Ejemplo:

- si el viejo reporta `open=2`
- y el nuevo reporta `open=1`
- el total visible en MT5 puede ser `3`

Ademas:

- una `pending_limit` ya activada deja de contar como `pending` y pasa a `open`
- los logs duales deben leerse por perfil y por `magic_number`
- `production_profile_spacing_seconds` del scheduler dual actual esta en `75` para darle tiempo al perfil `M15`

### 23.3 Logica actual de entrada

La entrada ya no es solo "modelo dice BUY/SELL y se ejecuta".

Traduccion a lenguaje simple:

- primero el sistema decide si la idea general es `BUY`, `SELL` o `HOLD`
- luego decide si el punto concreto de entrada es bueno o malo
- despues decide si conviene:
  - entrar ya
  - entrar poco y dejar una mejora pendiente
  - esperar mejor precio
  - o no entrar

La meta no es ser siempre mas conservador. La meta es evitar dos extremos:

- entrar demasiado pronto por impulso falso
- entrar demasiado tarde cuando el movimiento ya esta agotado

Capas activas:

1. `decision_bundle`
- decide direccion y conviccion final (`BUY`, `SELL`, `HOLD`)

2. `entry_staging`
- puede retener ideas para mejor retroceso
- convierte varias rutas directas en candidatas staged

3. `entry_management`
- puede dividir una entrada en `market + limit`
- o degradarla a `retrace_only`

4. `context_guard + entry_quality_score`
- decide si el punto de entrada es defendible
- puede forzar:
  - `market`
  - `entry_quality_retrace_only`
  - `entry_quality_low_skip`

5. `cluster_guard`
- evita seguir apilando demasiadas piernas del mismo lado cuando el cluster ya avanzo o ya esta cargado

6. `execution_confirmation_m1`
- la tesis sigue saliendo del timeframe principal del perfil (`M5` en el viejo, `M15` en el nuevo)
- `M1` no cambia la direccion; solo confirma el timing de ejecucion
- se usa para dos momentos:
  - habilitar o frenar una `micro market` cerca del trigger
  - habilitar o retener la activacion final de una candidata staged
- si `M1` no acompana, la candidata queda `ACTIVE` y espera nueva confirmacion; no fuerza cancelacion inmediata
- chequea breakout corto, aceleracion (`ROC_1` y `ROC_3`), alineacion de pendiente (`EMA20` y `VWAP`), ubicacion del cierre, wick contrario y stretch intrabar
- por diseno esta capa busca afinar la entrada, no endurecer la tesis principal

7. `mature_non_aligned_filter`
- esta es una proteccion simetrica nueva
- aplica tanto a `BUY` como a `SELL`
- si el impulso ya esta `mature` y el filtro no acompana, la idea puede bajar a `retrace_only`
- no es un veto absoluto: depende de calidad y `dirvol`

### 23.3.1 Como estan balanceados hoy compras y ventas

La intencion actual del sistema es:

- no castigar `BUY` por defecto solo por ser `BUY`
- no dejar pasar `SELL` maduras malas solo por ser `SELL`
- evaluar ambos lados con la misma idea:
  - si el impulso esta naciendo y el punto es defendible, puede entrar
  - si el impulso ya esta maduro y el filtro no acompana, debe esperar mejor precio

En el perfil viejo:

- `filter_hold_small_market_only` ya no esta sesgado contra `BUY`
- ahora `BUY` y `SELL` pueden existir en esa ruta
- pero ambos lados se degradan si el impulso ya viene maduro

En el perfil nuevo:

- ya no hay una proteccion dura solo para `SELL`
- la proteccion se reemplazo por una regla de "impulso maduro + filtro no alineado" que sirve para ambos lados

Esto busca que el robot no quede:

- demasiado agresivo en ventas maduras
- ni demasiado conservador bloqueando compras por regla fija

Comentarios de entrada frecuentes:

- `split_retrace_limit`
- `split_retrace_filter_opposite_retrace_only`
- `filter_hold_context_retrace_only`
- `candle_context_retrace_only`
- `entry_quality_retrace_only`
- `candle_context_market_only`
- `filter_hold_small_market_only`
- `entry_context_hard_contradiction`
- `entry_quality_low_skip`
- `m1_execution_confirmed`
- `WAITING_M1_CONFIRMATION`

### 23.4 Que intenta evitar la logica nueva

Patrones que ahora se degradan o bloquean:

- vender demasiado abajo o comprar demasiado arriba dentro de la vela de senal
- entrar al mercado sobre una mecha fuerte o rechazo contrario
- perseguir precio cuando el trade ya esta muy estirado frente a `EMA_20` o `SessionVWAP`
- mantener `market` cuando el filtro va opuesto y la estructura parece spike/news move
- ejecutar una entrada madura con filtro no alineado solo porque el lado principal sigue fuerte
- bloquear una compra o una venta solo por el lado, en vez de por la calidad del contexto

En esos casos, el sistema intenta:

- degradar a `retrace_only`
- pasar a staging
- o bloquear la idea si la calidad de entrada cae demasiado

Lectura simple:

- `conservador bueno`: espera mejor punto sin matar la tesis
- `conservador malo`: cancela demasiado y deja pasar movimientos limpios
- `agresivo bueno`: entra temprano y protege rapido
- `agresivo malo`: persigue impulsos cortos o agotados

La logica actual intenta quedarse en el medio:

- tesis viva
- timing exigente
- gestion rapida una vez que el trade ya entro

### 23.5 Gestion actual de riesgo y salida

Una vez abierta la posicion, las capas activas son:

- `trade_management` progresivo por avance a `TP`
- `runtime_monitor` para reversals, lateralizacion y perdida de follow-through
- cierre por senal opuesta solo si la posicion ya esta no negativa
- armado de salida a `BE` para la tesis vieja cuando el mismo perfil emite una senal fuerte contraria

Eso significa:

- el sistema intenta realizar ganancias parciales antes del `TP`
- no deberia cerrar por contradiccion nueva si la posicion sigue en rojo
- si un spike o reversal ocurre cuando el trade ya esta en verde, puede cerrar completo mas rapido

Casos practicos:

- si ves una posicion con poco lote restante y muy cerca del `TP`, puede cerrar completa por regla de remanente minimo
- si ves una `pending_limit` desaparecer mientras la market sigue viva, normalmente fue cancelacion por progreso
- si ves una tesis nueva opuesta y la vieja sigue abierta, revisa si quedo armada para salir en `BE`

### 23.6 Pausa automatica por mercado cerrado o feed invalido

El pipeline puede pausar `production`, `sync_trades` y `monitor_runtime` si:

- no hay tick
- el tick no tiene timestamp util
- `bid/ask` son invalidos
- el ultimo tick esta demasiado viejo

La pausa se quita sola cuando vuelven ticks frescos.

### 23.7 Como auditar una senal o trade rapidamente

Para una senal nueva, revisar:

- `outputs/production/production_signals.csv`
- columnas:
  - `signal`
  - `primary_signal`
  - `primary_confidence`
  - `filter_signal`
  - `filter_confidence`
  - `entry_management_comment`
  - `entry_context_reason`
  - `entry_context_quality_score`
  - `entry_context_quality_decision`
  - `entry_execution_confirmation_tf`
  - `entry_execution_confirmation_score`
  - `entry_execution_confirmation_passed`
  - `entry_execution_confirmation_reason`
  - `live_entry_price`
  - `pending_order_price`

Para una candidata retenida:

- `outputs/production/staged_signal_report.csv`
- columnas:
  - `candidate_mode`
  - `status`
  - `status_reason`
  - `trigger_price`
  - `expires_at`
  - `refresh_action`
  - `refresh_reason`
  - `last_execution_confirmation_timeframe`
  - `last_execution_confirmation_score`
  - `last_execution_confirmation_passed`
  - `last_execution_confirmation_reason`
  - `cancel_reason`

Lectura practica del staging:

- `status = ACTIVE` + `staged_action = WAITING_M1_CONFIRMATION` significa que la tesis sigue viva pero `M1` no confirmo el timing todavia
- `entry_execution_confirmation_passed = true` indica que la capa `M1` acompano la entrada o la activacion
- `entry_execution_confirmation_reason` resume por que `M1` confirmo o retuvo la ejecucion

Para una posicion ya abierta:

- `outputs/production/trade_lifecycle_report.csv`
- columnas:
  - `status`
  - `entry_leg`
  - `execution_price`
  - `sl_price`
  - `tp_price`
  - `managed_stage_ids`
  - `break_even_applied`
  - `management_progress_to_tp`
  - `last_management_action`
  - `runtime_monitor_action`
  - `runtime_monitor_reason`

Para ver si un perfil ya esta entrando en drift:

- `outputs/production/drift_gate_status.json`
- campos importantes:
  - `status`
  - `rerun_recommended`
  - `reasons`
  - `profit_factor_recent`
  - `net_pnl_recent`
  - `dominant_loss_route`

### 23.8 Lectura practica de desempeno

Si ves muchas ganancias y muchas perdidas al mismo tiempo, normalmente el problema esta en una de estas capas:

- la direccion fue correcta, pero la entrada fue mala:
  - revisar `entry_context_quality_score`
  - revisar `entry_management_comment`

- la entrada fue razonable, pero la gestion no descargo a tiempo:
  - revisar `management_progress_to_tp`
  - revisar `managed_stage_ids`
  - revisar `runtime_monitor_action`

- la tesis cambio y el sistema quedo con lados opuestos abiertos:
  - revisar `production_signals.csv` por perfil
  - confirmar si la posicion vieja quedo armada para salir en `BE`

En general:

- el perfil viejo suele reaccionar antes en `M5`
- el perfil nuevo `M15` tiende a entrar con menos lotaje y mas contexto
- la comparacion correcta no es solo `PnL`, sino:
  - calidad de entrada
  - velocidad de deteccion del giro
  - si protege al llegar a verde

### 23.9 Como leer el drift gate

El `drift gate` no cierra trades ni cambia la entrada. Es una alarma operacional.

Su funcion es responder:

- "este perfil sigue comportandose parecido al backtest reciente?"
- "o ya entro en una racha donde conviene revisar release, config o regimen?"

Senales importantes:

- `status = healthy`
  - el comportamiento reciente no muestra deterioro fuerte

- `status = warning`
  - hay sintomas de degradacion, pero todavia no necesariamente amerita apagar

- `status = critical`
  - el perfil reciente ya no se parece a su comportamiento esperado
  - normalmente conviene revisar:
    - ruta de perdida dominante
    - sesiones recientes
    - si hace falta nuevo backtest o nueva release

`rerun_recommended = true` significa:

- no que debas parar todo de inmediato
- sino que ya hay evidencia suficiente para justificar un backtest o una reevaluacion seria

### 23.10 Glosario corto para operar el live

- `tesis`: la idea principal del trade (`BUY`, `SELL`, `HOLD`)
- `market`: orden inmediata al precio disponible
- `pending_limit`: orden real en MT5 esperando mejor precio
- `staged`: idea guardada internamente, sin orden real todavia
- `retrace_only`: solo entra si mejora el punto
- `split`: entrada dividida en dos piernas
- `BE`: `break-even`; salida sin perdida
- `follow-through`: continuidad del movimiento despues de entrar
- `impulse_birth`: impulso naciendo
- `impulse_mature`: impulso ya avanzado
- `impulse_exhausted`: impulso agotado o demasiado estirado
