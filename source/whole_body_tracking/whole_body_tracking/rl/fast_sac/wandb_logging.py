from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_LOGGERS = {"none", "wandb"}
VALID_WANDB_MODES = {"online", "offline", "disabled"}


def parse_wandb_tags(tags: str | None) -> list[str] | None:
    if tags is None:
        return None
    parsed = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return parsed or None


def resolve_wandb_mode(mode: str | None) -> str:
    resolved = mode or os.environ.get("WANDB_MODE") or "offline"
    if resolved not in VALID_WANDB_MODES:
        raise ValueError(
            f"W&B mode must be one of {sorted(VALID_WANDB_MODES)}, got {resolved!r}"
        )
    return resolved


@dataclass
class WandbLoggerConfig:
    logger: str = "none"
    project: str | None = None
    entity: str | None = None
    mode: str | None = None
    group: str | None = None
    tags: str | None = None
    run_id: str | None = None
    resume: str | None = None
    name: str | None = None
    directory: str | Path | None = None
    config: dict[str, Any] | None = None


class NoOpWandbLogger:
    enabled = False

    def log(self, row: dict[str, Any], *, step: int) -> None:
        return None

    def update_summary(self, values: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbRunLogger:
    enabled = True

    def __init__(self, wandb_module: Any, run: Any):
        self._wandb = wandb_module
        self._run = run
        self._finished = False

    def log(self, row: dict[str, Any], *, step: int) -> None:
        try:
            self._wandb.log(row, step=step)
        except Exception as exc:
            warnings.warn(f"W&B log failed; continuing without this row: {exc}", RuntimeWarning, stacklevel=2)

    def update_summary(self, values: dict[str, Any]) -> None:
        summary = getattr(self._run, "summary", None)
        if summary is None:
            run = getattr(self._wandb, "run", None)
            summary = getattr(run, "summary", None)
        if summary is not None:
            try:
                summary.update(values)
            except Exception as exc:
                warnings.warn(f"W&B summary update failed; continuing: {exc}", RuntimeWarning, stacklevel=2)

    def finish(self) -> None:
        if self._finished:
            return None
        self._finished = True
        try:
            self._wandb.finish()
        except Exception as exc:
            warnings.warn(f"W&B finish failed; continuing: {exc}", RuntimeWarning, stacklevel=2)


def create_wandb_logger(cfg: WandbLoggerConfig):
    logger = cfg.logger.strip().lower()
    if logger not in VALID_LOGGERS:
        raise ValueError(f"logger must be one of {sorted(VALID_LOGGERS)}, got {cfg.logger!r}")
    if logger == "none":
        return NoOpWandbLogger()

    mode = resolve_wandb_mode(cfg.mode)
    if mode == "disabled":
        return NoOpWandbLogger()

    try:
        wandb = importlib.import_module("wandb")
    except Exception as exc:
        warnings.warn(f"W&B import failed; continuing with local curve logging only: {exc}", RuntimeWarning, stacklevel=2)
        return NoOpWandbLogger()
    try:
        run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            mode=mode,
            group=cfg.group,
            tags=parse_wandb_tags(cfg.tags),
            id=cfg.run_id,
            resume=cfg.resume,
            name=cfg.name,
            dir=str(cfg.directory) if cfg.directory is not None else None,
            config=cfg.config,
        )
    except Exception as exc:
        warnings.warn(f"W&B init failed; continuing with local curve logging only: {exc}", RuntimeWarning, stacklevel=2)
        return NoOpWandbLogger()
    return WandbRunLogger(wandb, run)
