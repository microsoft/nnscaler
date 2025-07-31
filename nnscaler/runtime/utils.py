#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""Runtime Utilities"""

from typing import Any, List
import logging

_logger = logging.getLogger(__name__)


class MicroBatchDataLoader:
    """
    MicroBatchDataLoader is used for scenarios of gradient accumulation,
    where a training iteration will have multiple data samples and perform
    multiple forward and backward on each sample (i.e., each refers to 
    as a micro-batch).

    To support more flexible training patterns, e.g., pipeline parallelism,
    MicroBatchDataLoader supports wrapping all data samples of a training iteration
    into a light dataloader and passed as input for compilation.

    e.g.,

    ```python
    # compilation phase
    dataloader = MicroBatchDataLoader([(input1,),]) # only need one micro-batch
    
    @nnscaler.compile(model, dataloader, ...)
    def train_iter(model, dataloader):
        input1 = next(dataloader)
        loss = model(input1)
        loss.backward()
        return loss

    ...

    # runtime phase
    
    for mini_batch_samples in iter(dataloader):
        # mini_batch_samples are sample list for 
        # all micro-batches in one iteration.
        dl = MicroBatchDataLoader(mini_batch_samples)
        loss =train_iter(model, dl)
        ...
    ```
    """

    def __init__(self, samples: List[Any], cycle: bool = False):
        """Create a micro-batch data loader for a mini-batch.

        Args:
            samples (List[Any]): a list of micro-batch samples. Each element
                in the list is a micro-batch sample.
            cycle (bool): whether to cycle the micro-batch samples. If True,
                the micro-batch samples will be cycled infinitely. Note this
                is only needed when the number of micro-batch samples is less
                than expected micro-batch number during runtime.
        """

        if not isinstance(samples, (tuple, list)):
            raise TypeError("Samples must be a tuple or list of samples.")
        self.samples = samples
        self.nmicros = len(samples)
        self.cycle = cycle
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self
    
    def __next__(self):
        if self._idx == self.nmicros:
            raise StopIteration
        batch = self.samples[self._idx]
        self._idx += 1
        if self.cycle:
            self._idx = self._idx % self.nmicros
        return batch
    
    def __len__(self):
        return self.nmicros
    
    def get_micro_batch(self, idx: int):
        idx = idx % self.nmicros if self.cycle else idx
        return self.samples[idx]


def microbatches(samples: List[Any], cycle: bool = False) -> MicroBatchDataLoader:
    """Create a micro-batch data loader for a mini-batch.

    This is for gradient accumulation scenarios. More details refer to
    documents of MicroBatchDataLoader.

    Args:
        samples (List[Any]): a list of micro-batch samples. Each element
            in the list is a micro-batch sample.
        cycle (bool): whether to cycle the micro-batch samples. If True,
            the micro-batch samples will be cycled infinitely. Note this
            is only needed when the number of micro-batch samples is less
            than expected micro-batch number during runtime.

    Returns:
        MicroBatchDataLoader: a micro-batch data loader.
    """
    return MicroBatchDataLoader(samples, cycle=cycle)
