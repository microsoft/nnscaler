import random
import string

import torch
import torch.utils.data as data


class VDataset(data.Dataset):
    """
    What has been changed:
        Generates random data for training and evaluation.
    """
    def __init__(
        self,
        size=256,
        is_train=True,
        evaluate_all=True,
        data_type="two",
        **kwargs,
    ) -> None:
        super().__init__()
        self.size = size
        self.is_train = is_train
        self.evaluate_all = evaluate_all
        self.data_type = data_type
        self.len = 128
        torch.manual_seed(42)
        self.sources = torch.normal(0.0, 1.0, size=(self.len, 3, 256, 256))
        self.driving = torch.normal(1.1, 3.0, size=(self.len, 3, 256, 256))
        self.video = torch.normal(2.1, 5.1, size=(self.len, 3, 256, 256))


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.is_train:
            source = self.sources[index]
            driving = self.driving[index]
            data_sample = {
                "source": source,
                "driving": driving,
            }
            return data_sample

        else:
            video = self.video[index]
            if "norm" in self.data_type:
                video = video * 2.0 - 1.0

            out_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".mp4"
            data_sample = {
                "video": video,
                "out_name": out_name,
            }
            return data_sample
