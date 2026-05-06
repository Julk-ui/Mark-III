from __future__ import annotations

"""Automation scheduler for backtest, production, and trade sync."""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


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
    timezone: str,
    start_offset_seconds: int = 0,
):
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
    if mode in {"production", "sync_trades"}:
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


def acquire_lock(lock_file: Path):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        return fd
    except FileExistsError:
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


def resolve_lock_path(lock_file: str, mode: str) -> Path:
    base = Path(lock_file)
    group = "backtest" if mode == "backtest" else "runtime"
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
) -> None:
    lock_path = resolve_lock_path(lock_file, mode)
    fd = acquire_lock(lock_path)
    if fd is None:
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
        if profile_name:
            logger.info("Ejecutando job '%s' [%s]: %s", mode, profile_name, " ".join(cmd))
        else:
            logger.info("Ejecutando job '%s': %s", mode, " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if proc.stdout:
            logger.info("STDOUT %s:\n%s", mode, proc.stdout[-8000:])
        if proc.stderr:
            logger.warning("STDERR %s:\n%s", mode, proc.stderr[-8000:])
        if proc.returncode != 0:
            raise RuntimeError(f"El job '{mode}' termino con codigo {proc.returncode}")
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
    parser.add_argument("--backtest-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para backtest")
    parser.add_argument("--production-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para produccion")
    parser.add_argument("--sync-offset-seconds", type=int, default=None, help="Desfase inicial en segundos para sync_trades")
    parser.add_argument("--timezone", default=None, help="Zona horaria del scheduler")
    parser.add_argument("--log-file", default=None, help="Archivo log del scheduler")
    parser.add_argument("--lock-file", default="logs/automation_scheduler.lock")
    parser.add_argument("--run-backtest-now", action="store_true")
    parser.add_argument("--run-production-now", action="store_true")
    parser.add_argument("--run-sync-now", action="store_true")
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
    sync_trades_profile_raw = scheduler_cfg.get("sync_trades_profile")
    sync_trades_profile = str(sync_trades_profile_raw).strip() if sync_trades_profile_raw else None
    production_profile_spacing = int(scheduler_cfg.get("production_profile_spacing_seconds", 20) or 20)

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

    logger = setup_logger(Path(log_file))
    scheduler = BlockingScheduler(timezone=timezone)

    jobs = []
    for mode, cron_expr, interval_minutes, start_offset_seconds in [
        ("backtest", backtest_cron, backtest_interval, backtest_offset),
    ]:
        trigger, schedule_desc = build_trigger(
            cron_expr=cron_expr,
            interval_minutes=interval_minutes,
            timezone=timezone,
            start_offset_seconds=start_offset_seconds,
        )
        if trigger is not None:
            jobs.append((mode, None, trigger, schedule_desc))

    trigger, schedule_desc = build_trigger(
        cron_expr=sync_cron,
        interval_minutes=sync_interval,
        timezone=timezone,
        start_offset_seconds=sync_offset,
    )
    if trigger is not None:
        jobs.append(("sync_trades", sync_trades_profile, trigger, schedule_desc))

    if production_profiles:
        for idx, profile_name in enumerate(production_profiles):
            trigger, schedule_desc = build_trigger(
                cron_expr=production_cron,
                interval_minutes=production_interval,
                timezone=timezone,
                start_offset_seconds=production_offset + (idx * production_profile_spacing),
            )
            if trigger is not None:
                jobs.append(("production", profile_name, trigger, f"{schedule_desc} profile={profile_name}"))
    else:
        trigger, schedule_desc = build_trigger(
            cron_expr=production_cron,
            interval_minutes=production_interval,
            timezone=timezone,
            start_offset_seconds=production_offset,
        )
        if trigger is not None:
            jobs.append(("production", None, trigger, schedule_desc))

    if not jobs and not any([args.run_backtest_now, args.run_production_now, args.run_sync_now]):
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
            )
    if args.run_sync_now:
        run_pipeline_job(
            mode="sync_trades",
            python_exe=args.python,
            pipeline_file=args.pipeline,
            base_config_file=args.config,
            production_config_file=args.production_config,
            use_optimized_config=bool(use_optimized),
            logger=logger,
            lock_file=args.lock_file,
        )

    if jobs:
        logger.info("Scheduler iniciado.")
        scheduler.start()


if __name__ == "__main__":
    main()
