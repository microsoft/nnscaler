#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This logger implementation is inspired by Fairseq https://github.com/facebookresearch/fairseq/blob/main/fairseq/logging/progress_bar.py

from typing import Dict, Optional
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

from nnscaler.utils import rank_zero_only

from .logger_base import LoggerBase


class WandbLogger(LoggerBase):
    def __init__(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        dir: Optional[str] = None,
        **kwargs
    ) -> None:
        if wandb is None:
            raise RuntimeError(
                "wandb not found, please install with: pip install wandb"
            )

        super().__init__()

        self._name = name
        self._project = project
        self._entity = entity
        self._dir = dir
        self._kwargs = kwargs

    @rank_zero_only
    def setup(self, config: Dict) -> None:
        if self._dir is not None:
            self._dir = Path(self._dir).expanduser().resolve()
            self._dir.mkdir(parents=True, exist_ok=True)

        # reinit=False to ensure if wandb.init() is called multiple times
        # within one process it still references the same run
        wandb.init(name=self._name, project=self._project,
            entity=self._entity,
            reinit=False,
            dir=self._dir,
            config=config,
            **self._kwargs
        )

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: int, *, tag: Optional[str] = None) -> None:
        prefix = "" if tag is None else tag + "/"
        metrics = {prefix + k: v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    @rank_zero_only
    def finalize(self) -> None:
        wandb.finish()
