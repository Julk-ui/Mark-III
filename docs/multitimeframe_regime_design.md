# Multi-Timeframe Regime Design

Objetivo:
- añadir contexto de `M15` y `H1` para mejorar entradas en `M5`
- evitar compras débiles dentro de contextos bajistas y viceversa
- usar el contexto como ponderación de ejecución y gestión, no como veto duro

## 1. Features a agregar

Para cada timeframe auxiliar (`M15`, `H1`), calcular y anexar al dataframe base de `M5`:

- `mtf_m15_close`
- `mtf_m15_ema20`
- `mtf_m15_ema50`
- `mtf_m15_ema_spread`
- `mtf_m15_roc_6`
- `mtf_m15_adx_14`
- `mtf_m15_atr_14`
- `mtf_m15_close_location_value`
- `mtf_m15_break_recent_high`
- `mtf_m15_break_recent_low`

- `mtf_h1_close`
- `mtf_h1_ema20`
- `mtf_h1_ema50`
- `mtf_h1_ema_spread`
- `mtf_h1_roc_6`
- `mtf_h1_adx_14`
- `mtf_h1_atr_14`
- `mtf_h1_close_location_value`
- `mtf_h1_break_recent_high`
- `mtf_h1_break_recent_low`

Derivadas útiles:

- `mtf_m15_bias`
- `mtf_h1_bias`
- `mtf_alignment_buy_score`
- `mtf_alignment_sell_score`
- `mtf_regime_score`

## 2. Cálculo del regime_score

Sesgo por timeframe:

- `+1` si:
  - `ema20 > ema50`
  - `roc_6 > 0`
  - `close > ema20`
- `-1` si:
  - `ema20 < ema50`
  - `roc_6 < 0`
  - `close < ema20`
- `0` si está mixto

Fuerza por timeframe:

- `trend_strength = min(1.0, adx_14 / 35.0)`

Score sugerido:

```text
mtf_regime_score =
  0.4 * mtf_m15_bias * mtf_m15_trend_strength +
  0.6 * mtf_h1_bias  * mtf_h1_trend_strength
```

Interpretación:

- `>= +0.45`: contexto alcista
- `<= -0.45`: contexto bajista
- entre esos valores: mixto/lateral

## 3. Uso en ejecución

No usar como bloqueo absoluto.

Aplicación sugerida:

- si `M5 BUY` y `mtf_regime_score >= +0.45`
  - retrace más corto
  - breakout parcial permitido
  - más paciencia para dejar correr la posición

- si `M5 SELL` y `mtf_regime_score <= -0.45`
  - retrace más corto
  - breakout parcial permitido
  - menos presión por tomar salida temprana

- si `M5 BUY` contra contexto claramente bajista
  - retrace más profundo
  - menor tamaño
  - TP más conservador

- si `M5 SELL` contra contexto claramente alcista
  - retrace más profundo
  - menor tamaño
  - TP más conservador

## 4. Uso en gestión

Cuando la operación ya está abierta:

- trade a favor del contexto mayor:
  - permitir más recorrido antes de apretar SL
  - menos probabilidad de salida discrecional temprana

- trade contra contexto mayor:
  - break-even más temprano cuando entre en ganancia
  - parciales antes
  - menos tolerancia a lateralización

## 5. Implementación sugerida

Fase 1:
- construir features `M15/H1` en el pipeline
- guardar `mtf_regime_score` en `production_signals.csv`
- usarlo solo en `entry_staging` y `runtime_monitor`

Fase 2:
- incluir features MTF en el entrenamiento del modelo
- medir mejora con recall separado de `BUY` y `SELL`

## 6. Regla práctica

La regla importante es:

- `M5` decide la oportunidad
- `M15/H1` modulan agresividad y calidad de entrada

No reemplazar el modelo de `M5` con una regla dura multi-timeframe.
