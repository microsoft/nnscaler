#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This logger implementation is inspired by Fairseq https://github.com/facebookresearch/fairseq/blob/main/fairseq/logging/progress_bar.py

import atexit
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import yaml
import torch
try:
    _tensorboard_writers = {}
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from nnscaler.utils import rank_zero_only
from .logger_base import LoggerBase


class TensorBoardLogger(LoggerBase):
    def __init__(
        self,
        name: str,
        root_dir: str,
        **kwargs,
    ):
        if SummaryWriter is None:
            raise RuntimeError(
                "tensorboard not found, please install with: pip install tensorboard"
            )

        super().__init__()
        self._name = name
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._kwargs = kwargs
        self._yaml_config = None  # will be set in `setup`

    @property
    def log_dir(self) -> Path:
        """
        Root directory to save logging output, which is `_log_dir/_name`.
        """
        sub_path = [s for s in [self._name] if s]
        ld = self._root_dir.joinpath(*sub_path)
        ld.mkdir(parents=True, exist_ok=True)
        return ld

    @rank_zero_only
    def setup(self, config: Dict) -> None:
        self._yaml_config = yaml.dump(config)

    def _get_or_create_writer(self, tag: Optional[str] = None):
        tag = tag or ''
        if tag not in _tensorboard_writers:
            _tensorboard_writers[tag] = SummaryWriter(log_dir=self.log_dir / tag, **self._kwargs)
            _tensorboard_writers[tag].add_text("config", self._yaml_config)
        return _tensorboard_writers[tag]

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: int, *, tag: Optional[str] = None) -> None:
        summary_writer = self._get_or_create_writer(tag)

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                summary_writer.add_scalars(k, v, step)
            else:
                summary_writer.add_scalar(k, v, step)

    @rank_zero_only
    def finalize(self) -> None:
        # will do nothing, as the writers will be closed on exit
        pass


def _close_writers():
    for w in _tensorboard_writers.values():
        w.close()

# Close all writers on exit
atexit.register(_close_writers)
