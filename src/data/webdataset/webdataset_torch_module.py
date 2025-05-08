import random
from pathlib import Path

import lightning as L
import webdataset as wds
from torch.utils.data import DataLoader

from .webdataset_torch import WebAudioDataset


class WebDatasetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        data_loader: DataLoader,
        batch_size: int,
        urls: str | None = None,
        validation_split: float = 0.2,
        seed=42,
        **data_loader_settings: dict,
    ):
        super().__init__()
        self.urls = urls
        self.data_loader = data_loader
        self.validation_split = validation_split
        self.seed = seed
        self.data_path = data_path

        self.data_loader_settings = data_loader_settings
        self.batch_size = batch_size

    def prepare_data(self):
        self.urls = WebAudioDataset.prepare_data(
            data_path=self.data_path,
            data_loader=self.data_loader,
        )

        self.dataset = WebAudioDataset(
            urls=self.urls,
            data_length=len(self.data_loader.dataset),
        )

    def is_validation(self, sample):
        random.seed(self.seed + hash(sample.get("__key__", "")))
        return random.random() < self.validation_split

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.dataset.get_data_pipeline(
                data_length=int(
                    len(self.data_loader.dataset) * (1 - self.validation_split)
                ),
                selector=lambda x: not self.is_validation(x),
            )

            self.val_dataset = self.dataset.get_data_pipeline(
                data_length=int(len(self.data_loader.dataset) * self.validation_split),
                selector=self.is_validation,
            )
        elif stage == "test":
            assert NotImplementedError("Not implemented yet")
        elif stage == "predict":
            assert NotImplementedError("Not implemented yet")
        else:
            assert ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        loader = wds.WebLoader(
            self.train_dataset, batch_size=None, **self.data_loader_settings
        )
        return loader.unbatched().shuffle(True).batched(self.batch_size)

    def val_dataloader(self):
        loader = wds.WebLoader(
            self.val_dataset, batch_size=None, **self.data_loader_settings
        )
        return loader.unbatched().shuffle(False).batched(self.batch_size)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
