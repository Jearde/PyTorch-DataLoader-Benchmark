import logging
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyMemmapStorage
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
    # buffer = ReplayBuffer(storage=storage)

    rows_written = 0
    for batch in tqdm(chunk_iter, desc="Writing TensorDict memmap", unit="batch"):
        td = TensorDict(
            {
                "x": batch[0],
            },
            batch_size=batch[0].shape[0],
        )
        for i, y in enumerate(batch[1]):
            td[f"label_{i}"] = y

        # buffer.extend(td)
        storage.set(range(rows_written, rows_written + len(batch[0])), td)
        rows_written += len(batch[0])

    storage.dump(memmap_path)
    return memmap_path


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

        self.storage = LazyMemmapStorage(
            max_size=0, scratch_dir=memmap_path, existsok=True
        )
        self.storage.load(memmap_path)

        self.labels_keys = [
            key for key in self.storage[0].keys() if key.startswith("label_")
        ]

    @classmethod
    def prepare_data(cls, data_path: Path, data_loader: DataLoader):
        memmap_path = data_path.parents[0] / (data_path.name + "_memmap") / "tensordict"

        if memmap_path.exists():
            logger.info(f"Memmap already exists: {memmap_path}")
            return memmap_path

        logger.info(f"Creating memmap: {memmap_path}")
        save_memmap_stream(
            chunk_iter=data_loader,
            memmap_path=memmap_path,
            total_rows=len(data_loader.dataset),
            device="cpu",
        )
        return memmap_path

    def __len__(self) -> int:
        return len(self.storage)

    def __getitem__(self, idx):
        feature = self.storage[idx]["x"]
        labels = [self.storage[idx][key] for key in self.labels_keys]

        return feature, labels
