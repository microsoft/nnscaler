#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Optional, Dict


class LoggerBase(ABC):
    """
    Base class for experiment loggers.
    """

    @abstractmethod
    def setup(self, config: Dict) -> None:
        """
        Setup logger with trainer args. This is useful for saving hyperparameters.
        Will be called once before `log_metrics`
        """
        ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int, *, tag: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def finalize(self) -> None:
        ...
