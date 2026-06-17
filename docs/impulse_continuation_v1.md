# Impulse + Continuation V1

Objetivo:
- capturar antes el arranque del movimiento
- separar `impulso temprano` de `continuacion/pullback`
- dejar listo un paquete v1 usable en el repo sin tocar el live actual

## 1. Problema que corrige

El live actual (`ReturnFwd_4 + BarrierReturn_5p_5b`) suele:
- detectar tarde el primer tramo de un impulso
- confundir algunas caidas lentas con ruido
- llegar a staging cuando el mejor punto ya paso

La idea de `v1` no es "un modelo mas grande", sino:
- target mas corto
- estructura M5 explicita
- dos perfiles de backtest separados

## 2. Features estructurales nuevas implementadas

Se agregaron en [data_cleaner.py](</c:/Users/USER/Documentos/Maestria/Mark-III/data/data_cleaner.py>) dentro de `price_action`.

### Ruptura / estructura inmediata

- `BreakAbovePrevHigh`
- `BreakBelowPrevLow`
- `BreakAboveRecentHigh3`
- `BreakBelowRecentLow3`
- `BreakAboveRecentHigh6`
- `BreakBelowRecentLow6`

### Margen respecto a estructura reciente

- `BreakoutMarginHigh3Pips`
- `BreakoutMarginLow3Pips`
- `BreakoutMarginHigh6Pips`
- `BreakoutMarginLow6Pips`

### Estructura de maximos/minimos

- `HigherHighFlag`
- `HigherLowFlag`
- `LowerHighFlag`
- `LowerLowFlag`
- `StructureScore3`
- `StructureScore6`

### Expansion / pullback

- `RangeVsAvg6`
- `PullbackFromRecentHigh6OverATR`
- `BounceFromRecentLow6OverATR`
- `EMA1226SpreadOverATR`

Estas se suman a las ya existentes:
- `BodyPips`
- `UpperWickPips`
- `LowerWickPips`
- `RangePips`
- `BodyOverRange`
- `CloseLocationInBar`
- `RangeOverATR`
- `BodyOverATR`
- `RealizedRange3Pips`
- `RealizedRange6Pips`
- `HourSin`
- `HourCos`
- `SessionLondon`
- `SessionNewYork`
- `SessionOverlap`

## 3. Targets nuevos de v1

### Impulse Detector

Objetivo:
- detectar el arranque del movimiento en 1 vela

Config:
- target principal: `ReturnFwd_1`
- target filtro: `BarrierReturn_3p_2b`

Interpretacion:
- el primario estima el movimiento inmediato
- el filtro responde a "en 2 velas, toca primero +3 pips o -3 pips"

### Continuation Model

Objetivo:
- distinguir si el movimiento tiene seguimiento despues del impulso inicial

Config:
- target principal: `ReturnFwd_2`
- target filtro: `BarrierReturn_4p_4b`

Interpretacion:
- el primario mide continuidad a 2 velas
- el filtro valida si el movimiento logra completar una barrera un poco mas amplia

## 4. Perfiles nuevos

### Impulse Detector

Archivo:
- [config_profile_aggressive_hybrid_v1_5_impulse_detector.yaml](</c:/Users/USER/Documentos/Maestria/Mark-III/config/config_profile_aggressive_hybrid_v1_5_impulse_detector.yaml>)

Uso:
```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1_5_impulse_detector.yaml
```

### Continuation Model

Archivo:
- [config_profile_aggressive_hybrid_v1_5_continuation_model.yaml](</c:/Users/USER/Documentos/Maestria/Mark-III/config/config_profile_aggressive_hybrid_v1_5_continuation_model.yaml>)

Uso:
```powershell
.\.venv\Scripts\python.exe main_pipeline.py --mode backtest --config config/config_profile_aggressive_hybrid_v1_5_continuation_model.yaml
```

## 5. Como leer resultados

Mirar no solo:
- `profit_factor`
- `win_rate`

Sino tambien:
- recall de compras
- recall de ventas
- cuantas velas tarda en detectar el movimiento
- cuantas senales buenas se degradan por llegada tardia
- cuantas operaciones nacen tarde y se revierten rapido

Comparacion recomendada:
- `v1_4_short_r1_3p2b`
- `v1_4_short_r2_3p2b`
- `v1_5_impulse_detector`
- `v1_5_continuation_model`
- `v1_4_mid_r3_4p6b`

## 6. Fase 2: multi-timeframe regime

No se implemento todavia en codigo en esta fase para no mezclar dos cambios grandes.

Diseno ya documentado en:
- [multitimeframe_regime_design.md](</c:/Users/USER/Documentos/Maestria/Mark-III/docs/multitimeframe_regime_design.md>)

La idea correcta sigue siendo:
- `M5` detecta oportunidad
- `M15/H1` modulan agresividad
- no usar `M15/H1` como veto duro

## 7. Recomendacion practica

Orden sugerido:
1. correr `impulse_detector`
2. correr `continuation_model`
3. comparar `buy/sell recall` contra el perfil live actual
4. solo despues decidir si:
   - reemplazas el primario
   - agregas un modelo auxiliar de impulso
   - o mezclas impulso + continuacion en ejecucion live
