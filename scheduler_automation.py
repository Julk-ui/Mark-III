from __future__ import annotations

"""Automation scheduler for backtest, production, and trade sync."""

import argparse
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


_DRIFT_GATE_LOG_CACHE: dict[str, tuple] = {}
_DRIFT_GATE_STATUS_FILE_LOCK = threading.Lock()


def parse_cron(expr: str) -> dict[str, str]:
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError("El cron debe tener 5 campos: minuto hora dia mes dia_semana")
    minute, hour, day, month, day_of_week = parts
    return {
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
    }


def resolve_interval_minutes(args_value, cfg_value):
    if args_value is not None:
        return int(args_value)
    if cfg_value is not None:
        return int(cfg_value)
    return None


def resolve_interval_seconds(args_value, cfg_value):
    if args_value is not None:
        return int(args_value)
    if cfg_value is not None:
        return int(cfg_value)
    return None


def resolve_offset_seconds(args_value, cfg_value, fallback=0):
    if args_value is not None:
        return int(args_value)
    if cfg_value is not None:
        return int(cfg_value)
    return int(fallback or 0)


def build_trigger(
    *,
    cron_expr: str | None,
    interval_minutes: int | None,
    interval_seconds: int | None,
    timezone: str,
    start_offset_seconds: int = 0,
):
    if interval_seconds is not None and interval_seconds > 0:
        trigger_kwargs = {
            "seconds": int(interval_seconds),
            "timezone": timezone,
        }
        schedule_desc = f"cada {int(interval_seconds)} segundo(s)"
        if start_offset_seconds and int(start_offset_seconds) > 0:
            trigger_kwargs["start_date"] = datetime.now() + timedelta(
                seconds=int(start_offset_seconds)
            )
            schedule_desc += f" con offset de {int(start_offset_seconds)}s"
        return IntervalTrigger(**trigger_kwargs), schedule_desc
    if interval_minutes is not None and interval_minutes > 0:
        trigger_kwargs = {
            "minutes": int(interval_minutes),
            "timezone": timezone,
        }
        schedule_desc = f"cada {int(interval_minutes)} minuto(s)"
        if start_offset_seconds and int(start_offset_seconds) > 0:
            trigger_kwargs["start_date"] = datetime.now() + timedelta(
                seconds=int(start_offset_seconds)
            )
            schedule_desc += f" con offset de {int(start_offset_seconds)}s"
        return IntervalTrigger(**trigger_kwargs), schedule_desc
    if cron_expr:
        return CronTrigger(timezone=timezone, **parse_cron(cron_expr)), f"cron={cron_expr}"
    return None, None


def load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def normalize_profile_label(profile_name: str | None) -> str | None:
    if not profile_name:
        return None
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in str(profile_name).strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or None


def load_active_release_manifest(base_config_file: str, profile_name: str | None = None) -> dict | None:
    profile_label = normalize_profile_label(profile_name)
    suffix = f"_{profile_label}" if profile_label else ""
    manifest_path = Path(base_config_file).resolve().with_name(f"active_release{suffix}.json")
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def load_automation_halt_state(config_file: str) -> dict | None:
    cfg = load_yaml_config(config_file)
    output_dir = Path(cfg.get("output", {}).get("dir", "outputs")) / "production"
    halt_path = output_dir / "automation_halt_state.json"
    if not halt_path.exists():
        return None
    try:
        with halt_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _get_production_output_dir(config_file: str) -> Path:
    cfg = load_yaml_config(config_file)
    return Path(cfg.get("output", {}).get("dir", "outputs")) / "production"


def _read_last_csv_row(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    try:
        last_row = None
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row:
                    last_row = row
        return last_row
    except Exception:
        return None


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def log_latest_production_signal(*, logger: logging.Logger, config_file: str, profile_name: str | None) -> None:
    output_dir = _get_production_output_dir(config_file)
    row = _read_last_csv_row(output_dir / "production_signals.csv")
    if not row:
        return
    signal = str(row.get("signal") or "").upper() or "UNKNOWN"
    logger.info(
        "Señal production%s: ts=%s signal=%s primary=%s(%.3f) filter=%s support=%s pips=%s reason=%s entry=%s sl=%s tp=%s",
        f" [{profile_name}]" if profile_name else "",
        row.get("timestamp"),
        signal,
        row.get("primary_signal"),
        _safe_float(row.get("primary_confidence")) or 0.0,
        row.get("filter_signal"),
        row.get("filter_support_score"),
        row.get("pips"),
        row.get("signal_confirmation_reason"),
        row.get("live_entry_price"),
        row.get("live_sl_price"),
        row.get("live_tp_price"),
    )


def log_risk_management_snapshot(*, logger: logging.Logger, config_file: str, profile_name: str | None, source_mode: str) -> None:
    def _normalize_profile_label(value: str | None) -> str:
        if not value:
            return ""
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
        return "_".join(part for part in cleaned.split("_") if part)

    output_dir = _get_production_output_dir(config_file)
    lifecycle_path = output_dir / "trade_lifecycle_report.csv"
    if not lifecycle_path.exists():
        return
    try:
        open_count = 0
        break_even_count = 0
        pending_count = 0
        partial_count = 0
        last_action = None
        last_signal_time = None
        pending_keys: set[str] = set()
        target_profile = _normalize_profile_label(profile_name)
        with lifecycle_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_profile = _normalize_profile_label(
                    row.get("profile_name") or row.get("strategy_profile") or ""
                )
                if target_profile and row_profile != target_profile:
                    continue

                status = str(row.get("status") or "").upper()
                entry_leg = str(row.get("entry_leg") or "").strip().lower()
                pending_status = str(row.get("pending_order_status") or "").upper()
                pending_is_active = pending_status in {"ACTIVE", "PLACED", "PENDING", "OPEN"} and (
                    status == "PENDING_LIMIT" or entry_leg == "pending_limit"
                )
                if pending_is_active:
                    pending_ticket = str(row.get("pending_order_ticket") or "").strip()
                    signal_id = str(row.get("signal_id") or "").strip()
                    pending_keys.add(pending_ticket or signal_id or f"row:{reader.line_num}")

                if status == "OPEN":
                    open_count += 1
                    be_flag = str(row.get("break_even_applied") or "").strip().lower()
                    if be_flag in {"true", "1", "yes", "si"}:
                        break_even_count += 1
                    partial_volume = _safe_float(row.get("partial_close_total_volume")) or 0.0
                    if partial_volume > 0:
                        partial_count += 1
                    signal_time = row.get("signal_time")
                    if signal_time and (last_signal_time is None or str(signal_time) >= str(last_signal_time)):
                        last_signal_time = signal_time
                        last_action = row.get("last_management_action") or row.get("trade_management_comment")
        pending_count = len(pending_keys)
        logger.info(
            "Riesgo %s%s: open=%s break_even=%s partials=%s pending=%s last_signal=%s last_action=%s",
            source_mode,
            f" [{profile_name}]" if profile_name else "",
            open_count,
            break_even_count,
            partial_count,
            pending_count,
            last_signal_time or "-",
            last_action or "-",
        )
    except Exception:
        return


def log_drift_gate_snapshot(
    *,
    logger: logging.Logger,
    config_file: str,
    profile_name: str | None,
    source_mode: str,
    drift_gate_config: dict | None,
) -> None:
    cfg = drift_gate_config or {}
    if not bool(cfg.get("enabled", False)):
        return

    evaluate_modes_raw = cfg.get("evaluate_on_modes") or ["sync_trades"]
    if isinstance(evaluate_modes_raw, str):
        evaluate_modes = {evaluate_modes_raw.strip().lower()}
    else:
        evaluate_modes = {
            str(value).strip().lower()
            for value in evaluate_modes_raw
            if str(value).strip()
        }
    if evaluate_modes and source_mode.lower() not in evaluate_modes:
        return

    recent_closed_trades = max(int(cfg.get("recent_closed_trades", 20) or 20), 1)
    minimum_closed_trades = max(int(cfg.get("minimum_closed_trades", 8) or 8), 1)
    warn_profit_factor_below = float(cfg.get("warn_profit_factor_below", 0.90) or 0.90)
    critical_profit_factor_below = float(cfg.get("critical_profit_factor_below", 0.60) or 0.60)
    warn_net_pnl_below = float(cfg.get("warn_net_pnl_below", -20.0) or -20.0)
    warn_consecutive_losses = max(int(cfg.get("warn_consecutive_losses", 4) or 4), 1)
    warn_dominant_route_loss_share = float(cfg.get("warn_dominant_route_loss_share", 0.55) or 0.55)
    warn_dominant_route_loss_count = max(int(cfg.get("warn_dominant_route_loss_count", 3) or 3), 1)
    recommend_if_conditions_at_least = max(int(cfg.get("recommend_if_conditions_at_least", 2) or 2), 1)
    status_file_name = str(cfg.get("status_file") or "drift_gate_status.json").strip() or "drift_gate_status.json"

    output_dir = _get_production_output_dir(config_file)
    lifecycle_path = output_dir / "trade_lifecycle_report.csv"
    if not lifecycle_path.exists():
        return

    target_profile = normalize_profile_label(profile_name) or ""
    closed_rows: list[dict[str, object]] = []
    try:
        with lifecycle_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_profile = normalize_profile_label(row.get("strategy_profile") or row.get("profile_name") or "")
                if target_profile and row_profile != target_profile:
                    continue
                if str(row.get("status") or "").upper() != "CLOSED":
                    continue
                pnl = _safe_float(row.get("close_profit_net"))
                if pnl is None:
                    continue
                route = (
                    str(row.get("entry_management_comment") or "").strip()
                    or str(row.get("entry_management_mode") or "").strip()
                    or str(row.get("trade_management_comment") or "").strip()
                    or "unknown"
                )
                closed_rows.append(
                    {
                        "close_time": str(row.get("close_time") or row.get("signal_time") or ""),
                        "pnl": pnl,
                        "route": route,
                    }
                )
    except Exception:
        return

    if not closed_rows:
        return

    closed_rows.sort(key=lambda item: str(item.get("close_time") or ""))
    recent_rows = closed_rows[-recent_closed_trades:]
    recent_values = [float(row["pnl"]) for row in recent_rows]
    closed_count = len(recent_rows)
    wins = sum(1 for value in recent_values if value > 0)
    losses = sum(1 for value in recent_values if value < 0)
    flats = closed_count - wins - losses
    net_pnl = sum(recent_values)
    gross_profit = sum(value for value in recent_values if value > 0)
    gross_loss = abs(sum(value for value in recent_values if value < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    consecutive_losses = 0
    for row in reversed(recent_rows):
        pnl = float(row["pnl"])
        if pnl < 0:
            consecutive_losses += 1
            continue
        break

    negative_rows = [row for row in recent_rows if float(row["pnl"]) < 0]
    worst_route = "-"
    worst_route_count = 0
    worst_route_loss_share = 0.0
    worst_route_net = 0.0
    if negative_rows:
        route_stats: dict[str, dict[str, float]] = {}
        for row in negative_rows:
            route = str(row.get("route") or "unknown")
            stats = route_stats.setdefault(route, {"count": 0.0, "net": 0.0})
            stats["count"] += 1.0
            stats["net"] += float(row["pnl"])
        worst_route, worst_stats = min(route_stats.items(), key=lambda item: (item[1]["net"], -item[1]["count"]))
        worst_route_count = int(worst_stats["count"])
        worst_route_net = float(worst_stats["net"])
        worst_route_loss_share = worst_route_count / max(len(negative_rows), 1)

    status = "warmup"
    warning_conditions: list[str] = []
    rerun_recommended = False
    if closed_count >= minimum_closed_trades:
        status = "ok"
        pf_is_critical = profit_factor < critical_profit_factor_below
        if profit_factor < warn_profit_factor_below:
            warning_conditions.append("profit_factor")
        if net_pnl <= warn_net_pnl_below:
            warning_conditions.append("net_pnl")
        if consecutive_losses >= warn_consecutive_losses:
            warning_conditions.append("consecutive_losses")
        if (
            worst_route_count >= warn_dominant_route_loss_count
            and worst_route_loss_share >= warn_dominant_route_loss_share
        ):
            warning_conditions.append("dominant_route")
        if pf_is_critical:
            status = "critical"
        elif warning_conditions:
            status = "warn"
        rerun_recommended = pf_is_critical or len(warning_conditions) >= recommend_if_conditions_at_least

    pf_label = "INF" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    fingerprint = (
        status,
        rerun_recommended,
        closed_count,
        pf_label,
        round(net_pnl, 2),
        wins,
        losses,
        flats,
        consecutive_losses,
        worst_route,
        worst_route_count,
        round(worst_route_loss_share, 2),
        round(worst_route_net, 2),
    )
    cache_key = f"{source_mode}|{target_profile or 'all'}|{lifecycle_path}"
    profile_key = target_profile or "all_profiles"
    drift_status_path = output_dir / status_file_name
    payload = {
        "profile_name": profile_name,
        "profile_key": profile_key,
        "source_mode": source_mode,
        "status": status,
        "rerun_recommended": rerun_recommended,
        "reasons": list(warning_conditions),
        "evaluated_at": datetime.now().isoformat(),
        "recent_closed_trades_window": recent_closed_trades,
        "minimum_closed_trades": minimum_closed_trades,
        "metrics": {
            "closed_count": closed_count,
            "wins": wins,
            "losses": losses,
            "flats": flats,
            "profit_factor": None if profit_factor == float("inf") else round(profit_factor, 6),
            "profit_factor_label": pf_label,
            "net_pnl": round(net_pnl, 6),
            "consecutive_losses": consecutive_losses,
            "worst_route": worst_route,
            "worst_route_loss_count": worst_route_count,
            "worst_route_loss_share": round(worst_route_loss_share, 6),
            "worst_route_net": round(worst_route_net, 6),
        },
        "thresholds": {
            "warn_profit_factor_below": warn_profit_factor_below,
            "critical_profit_factor_below": critical_profit_factor_below,
            "warn_net_pnl_below": warn_net_pnl_below,
            "warn_consecutive_losses": warn_consecutive_losses,
            "warn_dominant_route_loss_share": warn_dominant_route_loss_share,
            "warn_dominant_route_loss_count": warn_dominant_route_loss_count,
            "recommend_if_conditions_at_least": recommend_if_conditions_at_least,
        },
    }
    try:
        drift_status_path.parent.mkdir(parents=True, exist_ok=True)
        with _DRIFT_GATE_STATUS_FILE_LOCK:
            existing_payload: dict[str, object]
            if drift_status_path.exists():
                try:
                    existing_payload = json.loads(drift_status_path.read_text(encoding="utf-8"))
                    if not isinstance(existing_payload, dict):
                        existing_payload = {}
                except Exception:
                    existing_payload = {}
            else:
                existing_payload = {}
            profiles_payload = existing_payload.get("profiles")
            if not isinstance(profiles_payload, dict):
                profiles_payload = {}
            profiles_payload[profile_key] = payload
            existing_payload["profiles"] = profiles_payload
            existing_payload["updated_at"] = payload["evaluated_at"]
            existing_payload["status_file_version"] = 1
            tmp_path = drift_status_path.with_name(f"{drift_status_path.stem}.tmp{drift_status_path.suffix}")
            tmp_path.write_text(json.dumps(existing_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, drift_status_path)
    except Exception:
        pass

    if _DRIFT_GATE_LOG_CACHE.get(cache_key) == fingerprint:
        return
    _DRIFT_GATE_LOG_CACHE[cache_key] = fingerprint

    log_fn = logger.warning if status in {"warn", "critical"} else logger.info
    reasons_label = ",".join(warning_conditions) if warning_conditions else "-"
    rerun_label = "YES" if rerun_recommended else "NO"
    log_fn(
        "Drift %s%s: status=%s closed=%s wins=%s losses=%s flats=%s pf=%s net=%.2f consec_losses=%s worst_route=%s route_losses=%s route_share=%.2f route_net=%.2f rerun=%s reasons=%s",
        source_mode,
        f" [{profile_name}]" if profile_name else "",
        status,
        closed_count,
        wins,
        losses,
        flats,
        pf_label,
        net_pnl,
        consecutive_losses,
        worst_route,
        worst_route_count,
        worst_route_loss_share,
        worst_route_net,
        rerun_label,
        reasons_label,
    )


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scheduler_automation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def resolve_scheduler_setting(args_value, cfg_value, fallback=None):
    if args_value is not None:
        return args_value
    if cfg_value is not None:
        return cfg_value
    return fallback


def resolve_mode_config(
    mode: str,
    base_config_file: str,
    explicit_production_config: str | None,
    use_optimized_config: bool,
    profile_name: str | None = None,
) -> str:
    if mode in {"production", "sync_trades", "monitor_runtime"}:
        if explicit_production_config:
            return explicit_production_config
        if use_optimized_config:
            manifest = load_active_release_manifest(base_config_file, profile_name=profile_name)
            if manifest:
                release_config = manifest.get("config_path")
                if release_config and Path(release_config).exists():
                    return str(Path(release_config))
            profile_label = normalize_profile_label(profile_name)
            if profile_label:
                candidate = Path(base_config_file).with_name(f"config_optimizado_{profile_label}.yaml")
                if candidate.exists():
                    return str(candidate)
            candidate = Path(base_config_file).with_name("config_optimizado.yaml")
            if candidate.exists():
                return str(candidate)
    return base_config_file


def _pid_is_running(pid_value: str | int | None) -> bool:
    try:
        pid = int(str(pid_value).strip())
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _cleanup_stale_lock(lock_file: Path) -> bool:
    if not lock_file.exists():
        return False
    try:
        pid_text = lock_file.read_text(encoding="utf-8").strip()
    except Exception:
        pid_text = ""
    if _pid_is_running(pid_text):
        return False
    try:
        lock_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def acquire_lock(lock_file: Path):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            if attempt == 0 and _cleanup_stale_lock(lock_file):
                continue
            return None


def release_lock(lock_file: Path, fd) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except Exception:
            pass
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def install_console_interrupt_guard(logger: logging.Logger):
    """Ignora interrupciones espurias de la consola anfitriona."""
    registered: list[tuple[int, object]] = []

    def _ignore_handler(signum, _frame):
        logger.warning(
            "Interrupcion de consola ignorada (signal=%s). El scheduler sigue activo.",
            signum,
        )

    for signal_name in ("SIGINT", "SIGBREAK"):
        sig = getattr(signal, signal_name, None)
        if sig is None:
            continue
        try:
            previous = signal.getsignal(sig)
            signal.signal(sig, _ignore_handler)
            registered.append((sig, previous))
        except Exception:
            continue

    def _restore() -> None:
        for sig, previous in registered:
            try:
                signal.signal(sig, previous)
            except Exception:
                pass

    return _restore


def resolve_lock_path(lock_file: str, mode: str, profile_name: str | None = None) -> Path:
    base = Path(lock_file)
    if mode == "backtest":
        group = "backtest"
    elif mode == "production":
        group = "production"
    else:
        if profile_name:
            safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(profile_name).strip())
            group = f"management.{safe_profile or 'default'}"
        else:
            group = "management"
    suffix = base.suffix or ".lock"
    stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    return base.with_name(f"{stem}.{group}{suffix}")


def run_pipeline_job(
    *,
    mode: str,
    python_exe: str,
    pipeline_file: str,
    base_config_file: str,
    production_config_file: str | None,
    use_optimized_config: bool,
    logger: logging.Logger,
    lock_file: str,
    profile_name: str | None = None,
    drift_gate_config: dict | None = None,
) -> None:
    def _clean_stderr(stderr_text: str) -> str:
        cleaned_lines = []
        for raw_line in (stderr_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "oneDNN custom operations are on" in line:
                continue
            cleaned_lines.append(raw_line)
        return "\n".join(cleaned_lines).strip()

    lock_path = resolve_lock_path(lock_file, mode, profile_name=profile_name)
    fd = acquire_lock(lock_path)
    if fd is None:
        if mode == "monitor_runtime":
            logger.debug(
                "Se omite job '%s': hay otra ejecucion activa en el grupo '%s'.",
                mode,
                lock_path.name,
            )
        else:
            logger.warning(
                "Se omite job '%s': hay otra ejecucion activa en el grupo '%s'.",
                mode,
                lock_path.name,
            )
        return

    try:
        config_file = resolve_mode_config(
            mode=mode,
            base_config_file=base_config_file,
            explicit_production_config=production_config_file,
            use_optimized_config=use_optimized_config,
            profile_name=profile_name,
        )
        if mode == "production":
            halt_state = load_automation_halt_state(config_file)
            today_label = datetime.now().date().isoformat()
            if halt_state and halt_state.get("active") and halt_state.get("date") == today_label:
                logger.warning(
                    "Se omite job '%s': kill switch diario activo (perdida %.2f%%, limite %.2f%%).",
                    mode,
                    float(halt_state.get("daily_loss_pct", 0.0)) * 100.0,
                    float(halt_state.get("daily_loss_limit_pct", 0.0)) * 100.0,
                )
                return
        cmd = [python_exe, pipeline_file, "--mode", mode, "--config", config_file]
        if mode != "monitor_runtime":
            if profile_name:
                logger.info("Ejecutando job '%s' [%s]: %s", mode, profile_name, " ".join(cmd))
            else:
                logger.info("Ejecutando job '%s': %s", mode, " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        if mode == "monitor_runtime":
            env["MARKIII_QUIET_RUNTIME_MONITOR"] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags,
        )
        clean_stderr = _clean_stderr(proc.stderr)
        if proc.returncode != 0:
            if proc.stdout:
                logger.info("STDOUT %s:\n%s", mode, proc.stdout[-8000:])
            if clean_stderr:
                logger.warning("STDERR %s:\n%s", mode, clean_stderr[-8000:])
            raise RuntimeError(f"El job '{mode}' termino con codigo {proc.returncode}")
        if mode == "production":
            log_latest_production_signal(
                logger=logger,
                config_file=config_file,
                profile_name=profile_name,
            )
        elif mode in {"sync_trades", "monitor_runtime"}:
            log_risk_management_snapshot(
                logger=logger,
                config_file=config_file,
                profile_name=profile_name,
                source_mode=mode,
            )
            log_drift_gate_snapshot(
                logger=logger,
                config_file=config_file,
                profile_name=profile_name,
                source_mode=mode,
                drift_gate_config=drift_gate_config,
            )
        if clean_stderr and mode != "monitor_runtime":
            logger.warning("STDERR %s:\n%s", mode, clean_stderr[-4000:])
        if mode != "monitor_runtime":
            logger.info("Job '%s' completado.", mode)
    finally:
        release_lock(lock_path, fd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler de automatizacion para Mark III")
    parser.add_argument("--python", default=sys.executable, help="Ruta al ejecutable de Python")
    parser.add_argument("--pipeline", default="main_pipeline.py", help="Ruta al pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config base para backtest")
    parser.add_argument("--production-config", default=None, help="Config explicita para produccion/sync")
    parser.add_argument("--backtest-cron", default=None, help="Cron para backtest")
    parser.add_argument("--production-cron", default=None, help="Cron para produccion")
    parser.add_argument("--sync-cron", default=None, help="Cron para sync_trades")
    parser.add_argument("--backtest-interval-minutes", type=int, default=None, help="Intervalo en minutos para backtest")
    parser.add_argument("--production-interval-minutes", type=int, default=None, help="Intervalo en minutos para produccion")
    parser.add_argument("--sync-interval-minutes", type=int, default=None, help="Intervalo en minutos para sync_trades")
    parser.add_argument("--monitor-interval-seconds", type=int, default=None, help="Intervalo en segundos para monitor_runtime")
    parser.add_argument("--backtest-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para backtest")
    parser.add_argument("--production-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para produccion")
    parser.add_argument("--sync-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para sync_trades")
    parser.add_argument("--monitor-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para monitor_runtime")
    parser.add_argument("--timezone", default=None, help="Zona horaria del scheduler")
    parser.add_argument("--log-file", default=None, help="Archivo log del scheduler")
    parser.add_argument("--lock-file", default="logs/automation_scheduler.lock")
    parser.add_argument("--run-backtest-now", action="store_true")
    parser.add_argument("--run-production-now", action="store_true")
    parser.add_argument("--run-sync-now", action="store_true")
    parser.add_argument("--run-monitor-now", action="store_true")
    parser.add_argument("--production-use-optimized-config", dest="production_use_optimized_config", action="store_true")
    parser.add_argument("--no-production-use-optimized-config", dest="production_use_optimized_config", action="store_false")
    parser.set_defaults(production_use_optimized_config=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    scheduler_cfg = cfg.get("scheduler", {}) or {}
    production_profiles_cfg = scheduler_cfg.get("production_profiles")
    if isinstance(production_profiles_cfg, str):
        production_profiles = [production_profiles_cfg]
    elif isinstance(production_profiles_cfg, list):
        production_profiles = [str(p).strip() for p in production_profiles_cfg if str(p).strip()]
    else:
        production_profiles = []
    sync_profiles_cfg = scheduler_cfg.get("sync_trades_profiles")
    if isinstance(sync_profiles_cfg, str):
        sync_trades_profiles = [sync_profiles_cfg]
    elif isinstance(sync_profiles_cfg, list):
        sync_trades_profiles = [str(p).strip() for p in sync_profiles_cfg if str(p).strip()]
    else:
        sync_profile_raw = scheduler_cfg.get("sync_trades_profile")
        sync_trades_profiles = [str(sync_profile_raw).strip()] if sync_profile_raw else []

    monitor_profiles_cfg = scheduler_cfg.get("monitor_runtime_profiles")
    if isinstance(monitor_profiles_cfg, str):
        monitor_runtime_profiles = [monitor_profiles_cfg]
    elif isinstance(monitor_profiles_cfg, list):
        monitor_runtime_profiles = [str(p).strip() for p in monitor_profiles_cfg if str(p).strip()]
    else:
        monitor_profile_raw = scheduler_cfg.get("monitor_runtime_profile")
        if monitor_profile_raw:
            monitor_runtime_profiles = [str(monitor_profile_raw).strip()]
        else:
            monitor_runtime_profiles = list(sync_trades_profiles)

    production_profile_spacing = int(scheduler_cfg.get("production_profile_spacing_seconds", 20) or 20)
    sync_profile_spacing = int(scheduler_cfg.get("sync_trades_profile_spacing_seconds", 10) or 10)
    monitor_profile_spacing = int(scheduler_cfg.get("monitor_runtime_profile_spacing_seconds", 6) or 6)

    backtest_cron = resolve_scheduler_setting(
        args.backtest_cron,
        scheduler_cfg.get("backtest_cron"),
        scheduler_cfg.get("cron"),
    )
    production_cron = resolve_scheduler_setting(
        args.production_cron,
        scheduler_cfg.get("production_cron"),
    )
    sync_cron = resolve_scheduler_setting(
        args.sync_cron,
        scheduler_cfg.get("sync_trades_cron"),
    )
    backtest_interval = resolve_interval_minutes(
        args.backtest_interval_minutes,
        scheduler_cfg.get("backtest_interval_minutes"),
    )
    production_interval = resolve_interval_minutes(
        args.production_interval_minutes,
        scheduler_cfg.get("production_interval_minutes"),
    )
    sync_interval = resolve_interval_minutes(
        args.sync_interval_minutes,
        scheduler_cfg.get("sync_trades_interval_minutes"),
    )
    monitor_interval_seconds = resolve_interval_seconds(
        args.monitor_interval_seconds,
        scheduler_cfg.get("monitor_runtime_interval_seconds"),
    )
    backtest_offset = resolve_offset_seconds(
        args.backtest_offset_seconds,
        scheduler_cfg.get("backtest_offset_seconds"),
        0,
    )
    production_offset = resolve_offset_seconds(
        args.production_offset_seconds,
        scheduler_cfg.get("production_offset_seconds"),
        0,
    )
    sync_offset_default = 30 if sync_interval else 0
    sync_offset = resolve_offset_seconds(
        args.sync_offset_seconds,
        scheduler_cfg.get("sync_trades_offset_seconds"),
        sync_offset_default,
    )
    monitor_offset = resolve_offset_seconds(
        args.monitor_offset_seconds,
        scheduler_cfg.get("monitor_runtime_offset_seconds"),
        0,
    )
    timezone = resolve_scheduler_setting(
        args.timezone,
        scheduler_cfg.get("timezone"),
        "America/Bogota",
    )
    log_file = resolve_scheduler_setting(
        args.log_file,
        scheduler_cfg.get("log_file"),
        "logs/automation_scheduler.log",
    )
    use_optimized = resolve_scheduler_setting(
        args.production_use_optimized_config,
        scheduler_cfg.get("production_use_optimized_config"),
        True,
    )
    drift_gate_config = scheduler_cfg.get("drift_gate") or {}

    logger = setup_logger(Path(log_file))
    scheduler = BlockingScheduler(timezone=timezone)

    jobs = []
    for mode, cron_expr, interval_minutes, start_offset_seconds in [
        ("backtest", backtest_cron, backtest_interval, backtest_offset),
    ]:
        trigger, schedule_desc = build_trigger(
            cron_expr=cron_expr,
            interval_minutes=interval_minutes,
            interval_seconds=None,
            timezone=timezone,
            start_offset_seconds=start_offset_seconds,
        )
        if trigger is not None:
            jobs.append((mode, None, trigger, schedule_desc))

    trigger, schedule_desc = build_trigger(
        cron_expr=sync_cron,
        interval_minutes=sync_interval,
        interval_seconds=None,
        timezone=timezone,
        start_offset_seconds=sync_offset,
    )
    if trigger is not None:
        profiles = sync_trades_profiles or [None]
        for idx, profile_name in enumerate(profiles):
            profile_trigger, profile_schedule_desc = build_trigger(
                cron_expr=sync_cron,
                interval_minutes=sync_interval,
                interval_seconds=None,
                timezone=timezone,
                start_offset_seconds=sync_offset + (idx * sync_profile_spacing),
            )
            if profile_trigger is not None:
                profile_desc = profile_schedule_desc
                if profile_name:
                    profile_desc = f"{profile_schedule_desc} profile={profile_name}"
                jobs.append(("sync_trades", profile_name, profile_trigger, profile_desc))

    profiles = monitor_runtime_profiles or [None]
    for idx, profile_name in enumerate(profiles):
        trigger, schedule_desc = build_trigger(
            cron_expr=None,
            interval_minutes=None,
            interval_seconds=monitor_interval_seconds,
            timezone=timezone,
            start_offset_seconds=monitor_offset + (idx * monitor_profile_spacing),
        )
        if trigger is not None:
            profile_desc = schedule_desc
            if profile_name:
                profile_desc = f"{schedule_desc} profile={profile_name}"
            jobs.append(("monitor_runtime", profile_name, trigger, profile_desc))

    if production_profiles:
        for idx, profile_name in enumerate(production_profiles):
            trigger, schedule_desc = build_trigger(
                cron_expr=production_cron,
                interval_minutes=production_interval,
                interval_seconds=None,
                timezone=timezone,
                start_offset_seconds=production_offset + (idx * production_profile_spacing),
            )
            if trigger is not None:
                jobs.append(("production", profile_name, trigger, f"{schedule_desc} profile={profile_name}"))
    else:
        trigger, schedule_desc = build_trigger(
            cron_expr=production_cron,
            interval_minutes=production_interval,
            interval_seconds=None,
            timezone=timezone,
            start_offset_seconds=production_offset,
        )
        if trigger is not None:
            jobs.append(("production", None, trigger, schedule_desc))

    if not jobs and not any([args.run_backtest_now, args.run_production_now, args.run_sync_now, args.run_monitor_now]):
        raise ValueError("No se definio ningun job. Usa cron por CLI o en scheduler.* del YAML.")

    for mode, profile_name, trigger, schedule_desc in jobs:
        job_id = f"{mode}_{normalize_profile_label(profile_name)}_job" if profile_name else f"{mode}_job"
        scheduler.add_job(
            run_pipeline_job,
            trigger=trigger,
            kwargs={
                "mode": mode,
                "python_exe": args.python,
                "pipeline_file": args.pipeline,
                "base_config_file": args.config,
                "production_config_file": args.production_config,
                "use_optimized_config": bool(use_optimized),
                "logger": logger,
                "lock_file": args.lock_file,
                "profile_name": profile_name,
                "drift_gate_config": drift_gate_config,
            },
            id=job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        if profile_name:
            logger.info("Job registrado: mode=%s profile=%s schedule=%s timezone=%s", mode, profile_name, schedule_desc, timezone)
        else:
            logger.info("Job registrado: mode=%s schedule=%s timezone=%s", mode, schedule_desc, timezone)

    if args.run_backtest_now:
        run_pipeline_job(
            mode="backtest",
            python_exe=args.python,
            pipeline_file=args.pipeline,
            base_config_file=args.config,
            production_config_file=args.production_config,
            use_optimized_config=bool(use_optimized),
            logger=logger,
            lock_file=args.lock_file,
            drift_gate_config=drift_gate_config,
        )
    if args.run_production_now:
        profiles_to_run = production_profiles or [None]
        for profile_name in profiles_to_run:
            run_pipeline_job(
                mode="production",
                python_exe=args.python,
                pipeline_file=args.pipeline,
                base_config_file=args.config,
                production_config_file=args.production_config,
                use_optimized_config=bool(use_optimized),
                logger=logger,
                lock_file=args.lock_file,
                profile_name=profile_name,
                drift_gate_config=drift_gate_config,
            )
    if args.run_sync_now:
        profiles_to_run = sync_trades_profiles or [None]
        for profile_name in profiles_to_run:
            run_pipeline_job(
                mode="sync_trades",
                python_exe=args.python,
                pipeline_file=args.pipeline,
                base_config_file=args.config,
                production_config_file=args.production_config,
                use_optimized_config=bool(use_optimized),
                logger=logger,
                lock_file=args.lock_file,
                profile_name=profile_name,
                drift_gate_config=drift_gate_config,
            )
    if args.run_monitor_now:
        profiles_to_run = monitor_runtime_profiles or [None]
        for profile_name in profiles_to_run:
            run_pipeline_job(
                mode="monitor_runtime",
                python_exe=args.python,
                pipeline_file=args.pipeline,
                base_config_file=args.config,
                production_config_file=args.production_config,
                use_optimized_config=bool(use_optimized),
                logger=logger,
                lock_file=args.lock_file,
                profile_name=profile_name,
                drift_gate_config=drift_gate_config,
            )

    if jobs:
        logger.info("Scheduler iniciado.")
        restore_interrupt_handlers = install_console_interrupt_guard(logger)
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler detenido por interrupcion externa.")
        finally:
            restore_interrupt_handlers()
            try:
                scheduler.shutdown(wait=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
