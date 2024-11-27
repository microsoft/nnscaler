#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset


from lightning.pytorch.utilities.types import STEP_OUTPUT


class ClassificationModel(LightningModule):
    def __init__(self, num_features=32, num_classes=3, batch_size=10, lr=0.01):
        super().__init__()

        self.lr = lr
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        for i in range(3):
            setattr(self, f"layer_{i}", nn.Linear(num_features, num_features))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(num_features, 3))

        acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc = acc.clone()
        self.valid_acc = acc.clone()
        self.test_acc = acc.clone()
        self.dummy_forward_args_fn = lambda batch: {"x": batch[0]}
        self.update_history = []
        self.loss_history = []
        self.val_loss_history = []

    # @property
    # def dummy_forward_args(self):
    #     return {'x': torch.randn(self.batch_size, self.num_features)}

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        assert self.training
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True, sync_dist=True)
        return {'loss': loss}


    def test_step(self, batch, batch_idx):
        assert not self.training
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        assert not self.training
        x, _ = batch
        return self.forward(x)

    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val, gradient_clip_algorithm) -> None:
        def _fix_name(name):
            prefix = 'nnscaler_pmodule.'
            if name.startswith(prefix):
                return name[len(prefix):]
            return name
        grads = {_fix_name(n): p.grad.cpu() for n, p in self.named_parameters()}
        weights = {_fix_name(n): p.data.cpu() for n, p in self.named_parameters()}
        self.update_history.append((grads, weights))
        return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.loss_history.append(outputs["loss"].item())

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ):
        self.val_loss_history.append(outputs["loss"].item())


class ClassificationModelWithLRScheduler(ClassificationModel):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]


class RandomDictDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self) -> int:
        return self.len


class RandomDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class RandomIterableDataset(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self) -> int:
        return self.count


class BoringModel(LightningModule):
    """Testing PL Module.

    Use as follows:
    - subclass
    - modify the behavior for what you want

    .. warning::  This is meant for testing/debugging and is experimental.

    Example::

        class TestModel(BoringModel):
            def training_step(self, ...):
                ...  # do your own thing

    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    @property
    def dummy_forward_args(self):
        return {'x': torch.randn(32)}

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def loss(self, preds: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if labels is None:
            labels = torch.ones_like(preds)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(preds, labels)

    def step(self, batch: Any) -> Tensor:
        output = self(batch)
        return self.loss(output)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        assert self.training
        return {"loss": self.step(batch)}

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        assert not self.training
        return {"x": self.step(batch)}

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        assert not self.training
        return {"y": self.step(batch)}

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[LRScheduler]]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))
