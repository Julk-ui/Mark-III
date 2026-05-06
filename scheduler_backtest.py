from __future__ import annotations

"""Scheduler simple para ejecutar backtesting de forma periódica.

Uso ejemplo:
python scheduler_backtest.py --python .venv/Scripts/python.exe --pipeline main_pipeline.py --config config/config.yaml --cron "0 7 * * 1"

Cron esperado: minuto hora dia mes dia_semana
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def parse_cron(expr: str) -> dict[str, str]:
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError("El cron debe tener 5 campos: minuto hora día mes día_semana")
    minute, hour, day, month, day_of_week = parts
    return {
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
    }


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scheduler_backtest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=14, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def run_pipeline(python_exe: str, pipeline_file: str, config_file: str, logger: logging.Logger) -> None:
    cmd = [python_exe, pipeline_file, "--mode", "backtest", "--config", config_file]
    logger.info("Ejecutando backtest programado: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("STDOUT:\n%s", proc.stdout[-8000:] if proc.stdout else "")
    if proc.stderr:
        logger.warning("STDERR:\n%s", proc.stderr[-8000:])
    if proc.returncode != 0:
        raise RuntimeError(f"El backtest terminó con código {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler periódico para backtesting")
    parser.add_argument("--python", default=sys.executable, help="Ruta al ejecutable de Python")
    parser.add_argument("--pipeline", default="main_pipeline.py", help="Ruta al pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Ruta al YAML de configuración")
    parser.add_argument("--cron", default="0 7 * * 1", help="Expresión cron: m h dom mon dow")
    parser.add_argument("--timezone", default="America/Bogota")
    parser.add_argument("--log-file", default="logs/backtest_scheduler.log")
    args = parser.parse_args()

    logger = setup_logger(Path(args.log_file))
    cron_kwargs = parse_cron(args.cron)

    scheduler = BlockingScheduler(timezone=args.timezone)
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(timezone=args.timezone, **cron_kwargs),
        kwargs={
            "python_exe": args.python,
            "pipeline_file": args.pipeline,
            "config_file": args.config,
            "logger": logger,
        },
        id="weekly_backtest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    logger.info("Scheduler iniciado con cron=%s timezone=%s", args.cron, args.timezone)
    scheduler.start()


if __name__ == "__main__":
    main()
