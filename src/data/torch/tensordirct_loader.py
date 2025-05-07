import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyMemmapStorage, ReplayBuffer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def save_memmap_stream(
    chunk_iter,
    memmap_path,
    total_rows=None,
    device="cpu",
):
    storage = LazyMemmapStorage(
        max_size=total_rows,
        scratch_dir=memmap_path,
        device=device,
        existsok=True,
    )
    buffer = ReplayBuffer(storage=storage, device=device)

    rows_written = 0
    for batch in tqdm(chunk_iter, desc="Writing memmap", unit="batch"):
        td = TensorDict(
            {
                "x": batch[0],
            },
            batch_size=batch[0].shape[0],
        )
        for i, y in enumerate(batch[1]):
            td[f"label_{i}"] = y

        buffer.extend(td)
        rows_written += len(batch[0])

    storage.dump()

    return memmap_path


def load_memmap(pickle_path=None):
    logger.info(f"Loading features from memmap: {pickle_path}")
    with open(pickle_path, "rb") as f:
        meta_data = pickle.load(f)

    memmap_path = Path(meta_data["mm_path"])
    data_shape = meta_data["mm_shape"]
    data_dtype = meta_data["mm_dtype"]

    features_memmap = np.memmap(
        memmap_path, dtype=data_dtype, mode="r", shape=data_shape
    )

    return features_memmap, meta_data


class TensorDictMemmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        memmap_path: Path,
    ):
        self.memmap_path = memmap_path

        if self.memmap_path.exists():
            logger.info(f"Memmap already exists: {self.memmap_path}")
        else:
            logger.info(f"Memmap does not exist: {self.memmap_path}")
            raise FileNotFoundError(
                "Memmap file not found. Please run the prepare_data method first."
            )

        self.data, self.meta = load_memmap(pickle_path=self.memmap_path)
        self.labels_keys = [key for key in self.meta.keys() if key.startswith("label_")]

    @classmethod
    def prepare_data(cls, data_path: Path, data_loader: DataLoader):
        memmap_path = data_path.parents[0] / (data_path.name + "_memmap") / "tensordict"

        logger.info(f"Creating memmap: {memmap_path}")
        save_memmap_stream(
            chunk_iter=data_loader,
            memmap_path=memmap_path,
            total_rows=len(data_loader.dataset),
            device="cpu",
        )
        return memmap_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx].copy()).to(torch.float32)
        labels = [self.meta[key][idx] for key in self.labels_keys]

        return feature, labels
