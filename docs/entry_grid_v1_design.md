# Entry Grid V1 Design

Objetivo:
- mejorar el punto medio de entrada sin depender de una sola ejecucion
- repartir el riesgo total del trade en varias patas con contexto direccional
- cerrar primero las patas peores cuando la cesta vaya a favor
- dejar una pata final como `runner` para continuidad

Importante:
- esto **no** es una grid martingale
- esto **no** aumenta el riesgo total despues de entrar
- esto **no** promedia indefinidamente

La regla base es:
- un solo presupuesto de riesgo por trade
- varias patas dentro de ese mismo presupuesto
- invalidez estructural comun

## 1. Alcance de V1

No empezar con `5-6` patas live.

V1 recomendada:
- `3` patas
- `1` pata inmediata opcional
- `2` patas `limit`
- una sola logica de `SL`
- gestion por cesta y por pata

Motivo:
- el repo hoy soporta bien `1 market + 1 pending`
- pasar directo a `6` patas vuelve mucho mas complejos:
  - lifecycle
  - sync de pendientes
  - adopcion de fills en cuentas hedging
  - gestion por cesta

## 2. Casos de uso

### BUY

- `leg_1`
  - entrada inmediata parcial si el setup es fuerte
- `leg_2`
  - `BUY LIMIT` en retroceso medio
- `leg_3`
  - `BUY LIMIT` mas cerca del extremo de la vela/zona

### SELL

- `leg_1`
  - entrada inmediata parcial si el setup es fuerte
- `leg_2`
  - `SELL LIMIT` en retroceso medio
- `leg_3`
  - `SELL LIMIT` mas arriba, cerca del extremo

## 3. Regla de riesgo

La pata no define el riesgo. La **cesta completa** define el riesgo.

Regla:
- `risk_total_trade <= allocated_risk_budget`

Entonces:
- cada leg tiene un peso de volumen
- todas comparten un `SL` estructural coherente
- la suma del riesgo de todas las legs no puede superar el presupuesto del trade

## 4. Configuracion propuesta

```yaml
entry_grid:
  enabled: false
  mode: risk_based_ladder
  apply_to_profiles:
    - strong_trend
    - normal_trend
  require_confirmed_bundle: true
  allow_filter_hold_variant: false

  legs:
    - leg_id: leg_1
      entry_type: market_or_breakout
      volume_weight: 0.25
      spacing_fraction_of_stop: 0.00
      expiry_bars: 0

    - leg_id: leg_2
      entry_type: limit
      volume_weight: 0.35
      spacing_fraction_of_stop: 0.25
      expiry_bars: 1

    - leg_id: leg_3
      entry_type: limit
      volume_weight: 0.40
      spacing_fraction_of_stop: 0.55
      expiry_bars: 2

  stop:
    mode: shared_structural_dynamic
    anchor: signal_candle_extreme
    buffer_pips: 0.3
    atr_fraction: 0.25
    min_pips: 5.0
    max_pips: 15.0

  take_profit:
    mode: shared_dynamic_rr
    rr_min: 1.2
    rr_target: 1.5
    use_predicted_move_cap: true
    max_pips: 20.0

  basket_management:
    enabled: true
    runner_legs: 1
    close_worst_legs_first: true
    only_manage_when_net_positive: true
    move_to_break_even_on_net_progress: 0.25
    close_one_leg_on_net_progress: 0.40
    close_second_leg_on_net_progress: 0.65
    let_runner_continue_after_progress: 0.75
    cancel_unfilled_after_break_even: true
```

## 5. Interpretacion operativa

Con esa configuracion:
- `25%` del lote total entra de inmediato o por breakout
- `35%` espera retroceso medio
- `40%` espera retroceso mas profundo

Si la cesta entra en ganancia:
- primero se protege la cesta
- luego se cierran las patas peores
- se deja viva la mejor pata como `runner`

Para `BUY`:
- las patas peores son las mas altas
- la mejor pata suele ser la mas baja

Para `SELL`:
- las patas peores son las mas bajas
- la mejor pata suele ser la mas alta

## 6. Como calcular los precios de cada leg

Usar un `reference_stop_distance` dinamico desde la vela de señal.

Ejemplo:
- `entry_reference = live_entry_price`
- `stop_distance = abs(entry_reference - shared_sl_price)`

Entonces:

### BUY

- `leg_1_price = market / breakout`
- `leg_2_price = entry_reference - 0.25 * stop_distance`
- `leg_3_price = entry_reference - 0.55 * stop_distance`

### SELL

- `leg_1_price = market / breakout`
- `leg_2_price = entry_reference + 0.25 * stop_distance`
- `leg_3_price = entry_reference + 0.55 * stop_distance`

Cada leg usa:
- mismo `SL` estructural compartido
- mismo esquema de `TP` compartido o derivado del mismo `RR`

## 7. Gestion de ganancias

La gestion correcta no es por orden de apertura. Es por **calidad del precio**.

### Regla de cierre

Cuando la cesta vaya en positivo:
- cerrar primero las patas peores
- dejar la mejor pata

### Secuencia recomendada

1. `net_progress >= 25%`
- mover la cesta a break-even si ya no esta en rojo

2. `net_progress >= 40%`
- cerrar `leg_1` o la pata peor abierta

3. `net_progress >= 65%`
- cerrar `leg_2` o la siguiente peor

4. `net_progress >= 75%`
- dejar `leg_3` como runner
- trailing o gestion de continuidad

## 8. Gestion de invalidez

Antes de llenarse todas las patas:
- si cambia el bundle
- o se invalida el setup
- cancelar las pendientes no ejecutadas

Despues de llenarse una parte:
- no cerrar en perdida por simple debilidad
- respetar el `SL` estructural
- usar gestion discrecional solo cuando la cesta ya no este en rojo

## 9. Estructura de datos propuesta

Hoy el sistema usa:
- una fila padre en `production_signals.csv`
- una pending principal
- filas hijas limitadas en `trade_lifecycle_report.csv`

Para grid V1 se necesita una tabla hija por leg.

### Nuevo archivo

`outputs/production/entry_grid_legs_report.csv`

Columnas minimas:
- `grid_parent_signal_id`
- `grid_group_id`
- `leg_id`
- `leg_rank`
- `side`
- `entry_type`
- `volume_weight`
- `planned_volume_lots`
- `planned_entry_price`
- `planned_sl_price`
- `planned_tp_price`
- `expiry_time`
- `status`
- `status_reason`
- `mt5_order_ticket`
- `mt5_position_id`
- `execution_price`
- `entry_quality_rank`
- `is_runner_candidate`

### Reuso del lifecycle actual

En `trade_lifecycle_report.csv`:
- mantener una fila por posicion real
- agregar:
  - `grid_parent_signal_id`
  - `grid_group_id`
  - `grid_leg_id`
  - `grid_leg_rank`
  - `grid_entry_type`
  - `grid_quality_rank`
  - `grid_runner_candidate`

Eso permite que `sync_trades` y `monitor_runtime` gestionen cada pata sin mezclar tickets.

## 10. Integracion con el pipeline actual

### `production`

En vez de un solo `entry_management_plan`, construir:
- `entry_grid_plan`

Salida agregada al `production_signals.csv`:
- `entry_grid_enabled`
- `entry_grid_group_id`
- `entry_grid_leg_count`
- `entry_grid_market_legs`
- `entry_grid_pending_legs`
- `entry_grid_total_volume_lots`
- `entry_grid_runner_legs`

### `easy_Trading.py`

Hoy ya existe:
- `open_market_order`
- `open_pending_limit_order`
- `cancel_pending_order`

V1 solo necesita:
- llamar eso varias veces
- no requiere un nuevo primitive de broker

### `sync_trades`

Debe:
- sincronizar todas las legs activas
- cancelar las pendientes vencidas o invalidadas
- promover fills de pending a posiciones reales
- recalcular estado de cesta

### `monitor_runtime`

Debe agregar:
- `net_open_pnl_grid`
- `net_progress_to_tp_grid`
- `best_leg_price`
- `worst_leg_price`
- `positive_legs_count`

Y aplicar la gestion por cesta solo si:
- `net_open_pnl_grid >= 0`

## 11. Alcance de implementacion por fases

### Fase 1

- nuevo `entry_grid_legs_report.csv`
- plan de `3` patas
- envio de `1 market + 2 limits`
- sync de fills/cancelaciones

### Fase 2

- gestion por cesta
- cierre de patas peores
- dejar runner

### Fase 3

- ampliar a `5-6` patas
- usar `mtf_regime_score` para cambiar:
  - pesos
  - espaciamiento
  - runner count

## 12. Regla de decision para BUY y SELL sin volverlo rigido

No usar grid siempre.

Usarla solo si:
- setup confirmado
- riesgo disponible
- perfil `strong_trend` o `normal_trend`

No usarla en:
- `weak_or_mixed`
- `filter_contradicted` fuerte
- señales pequenas o sin follow-through

## 13. Recomendacion concreta para este repo

V1 exacta recomendada:
- `3` patas
- `0.25 / 0.35 / 0.40`
- `1 market_or_breakout + 2 limits`
- `SL` compartido estructural
- `TP` comun por RR dinamico
- `1` runner final
- gestion solo cuando la cesta ya no este en rojo

Esa version es suficiente para validar:
- si la grid mejora el punto medio de entrada
- si reduce quedarse fuera de impulsos
- si las patas peores se pueden descargar antes

Y todavia cabe bien dentro de la arquitectura actual con un refactor controlado.
