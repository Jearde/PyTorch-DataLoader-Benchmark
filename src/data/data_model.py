from pathlib import Path

import lightning as L
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_loader_class: type,
        data_loader_settings: dict,
        data_path: Path,
        labels_keys: list[str],
        files: list[str] | None = None,
        labels: list[list[int]] | None = None,
        dataset: torch.utils.data.Dataset | None = None,
        validation_split: float = 0.2,
    ):
        super().__init__()
        self.files = files
        self.labels = labels
        self.dataset = dataset
        self.validation_split = validation_split
        self.data_path = data_path
        self.labels_keys = labels_keys

        self.data_loader_class = data_loader_class
        self.data_loader_settings = data_loader_settings

        if files is not None:
            idcs = list(range(len(files)))
        elif dataset is not None:
            idcs = list(range(len(dataset)))
        else:
            raise ValueError("Either dataset or files must be provided.")

        if dataset is not None:
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.validation_split, self.validation_split]
            )
        else:
            self.train_dataset, self.val_dataset = None, None

        self.train_idcs, self.val_idcs = train_test_split(
            idcs,
            test_size=self.validation_split,
        )

        self.train_files = (
            None if files is None else [files[i] for i in self.train_idcs]
        )
        self.val_files = None if files is None else [files[i] for i in self.val_idcs]

        if labels is not None:
            self.train_labels = []
            for label in labels:
                self.train_labels.append([label[i] for i in self.train_idcs])
            self.val_labels = []
            for label in labels:
                self.val_labels.append([label[i] for i in self.val_idcs])
        else:
            self.train_labels = None
            self.val_labels = None

        pass

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            assert NotImplementedError("Not implemented yet")
        elif stage == "test":
            assert NotImplementedError("Not implemented yet")
        elif stage == "predict":
            assert NotImplementedError("Not implemented yet")
        else:
            assert ValueError(f"Unknown stage: {stage}")

    def get_dataloader(self, dataset, files, labels):
        if dataset is not None:
            return self.data_loader_class(dataset=dataset, **self.data_loader_settings)
        elif files is not None and labels is not None:
            return self.data_loader_class(
                files=files, labels=labels, **self.data_loader_settings
            )
        else:
            raise ValueError("Either dataset or files and labels must be provided.")

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_dataset, self.train_files, self.train_labels
        )

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, self.val_files, self.val_labels)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
