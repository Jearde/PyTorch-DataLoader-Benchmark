import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def save_memmap(data, meta_data, pickle_path=None):
    logger.info(f"Saving features to memmap: {pickle_path}")
    memmap_path = Path(pickle_path).with_suffix(".memmap")

    meta_data["mm_path"] = str(memmap_path)
    meta_data["mm_shape"] = data.shape
    meta_data["mm_dtype"] = data.dtype

    features_memmap = np.memmap(
        memmap_path, dtype=data.dtype, mode="w+", shape=data.shape
    )
    features_memmap[:] = data
    features_memmap.flush()

    with open(pickle_path, "wb") as f:
        pickle.dump(meta_data, f)

    return pickle_path


def save_memmap_stream(
    chunk_iter,
    pickle_path,
    dtype,
    feature_dim,
    total_rows=None,
    meta=None,
    grow_block=10_000,
):
    """
    Write batches from `chunk_iter` to a memory‑mapped array.

    Parameters
    ----------
    chunk_iter   : iterable yielding 2‑D arrays shaped (n_batch, feature_dim)
    pickle_path  : str | Path, path for the side‑car pickle with metadata
    dtype        : np.dtype of the features
    feature_dim  : int, width of each sample
    total_rows   : int | None.  If given, file is pre‑allocated.  If None,
                   the file is grown with memmap.resize().
    meta         : dict for extra metadata to pickle
    grow_block   : int, rows to add when resizing (ignored if total_rows given)
    """
    meta = {} if meta is None else dict(meta)
    mm_path = Path(pickle_path).with_suffix(".memmap")

    # --- create the memmap -----------------------------------------------
    if total_rows is not None:  # fixed‑size path
        mm = np.memmap(
            mm_path, dtype=dtype, mode="w+", shape=(total_rows, *feature_dim)
        )
    else:  # start tiny and grow
        mm = np.memmap(mm_path, dtype=dtype, mode="w+", shape=(0, *feature_dim))

    # --- stream batches ---------------------------------------------------
    cursor = 0
    for batch in tqdm(chunk_iter, desc="Writing memmap", unit="batch"):
        labels = batch[1]  # labels are not stored in memmap
        batch = batch[0]
        batch = np.asarray(batch, dtype=dtype)
        n = len(batch)

        # Enlarge if needed
        if cursor + n > mm.shape[0]:
            if total_rows is not None:
                raise RuntimeError("More data than total_rows")
            # grow to next multiple of grow_block or just enough for this batch
            new_rows = max(cursor + n, mm.shape[0] + grow_block)
            mm.resize((new_rows, feature_dim))  # in‑place grow

        mm[cursor : cursor + n] = batch  # copy this slice
        cursor += n

        # Add to metadata
        for i, label in enumerate(labels):
            if f"label_{i}" not in meta:
                meta[f"label_{i}"] = []
            meta[f"label_{i}"].extend(label)

    # Trim excess in dynamic mode
    if total_rows is None and cursor < mm.shape[0]:
        mm.resize((cursor, feature_dim))

    mm.flush()  # persist to disk

    # --- write side‑car metadata -----------------------------------------
    meta.update(mm_path=str(mm_path), mm_shape=mm.shape, mm_dtype=mm.dtype)
    with open(pickle_path, "wb") as f:
        pickle.dump(meta, f)

    logger.info("Saved memmap of shape %s to %s", mm.shape, mm_path)
    return pickle_path


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


class NumpyMemmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pickle_path: Path,
    ):
        self.pickle_path = pickle_path

        if self.pickle_path.exists():
            logger.info(f"Memmap already exists: {self.pickle_path}")
        else:
            logger.info(f"Memmap does not exist: {self.pickle_path}")
            raise FileNotFoundError(
                "Memmap file not found. Please run the prepare_data method first."
            )

        self.data, self.meta = load_memmap(pickle_path=self.pickle_path)
        self.labels_keys = [key for key in self.meta.keys() if key.startswith("label_")]

    @classmethod
    def prepare_data(cls, data_path: Path, data_loader: DataLoader):
        pickle_path = (
            data_path.parents[0] / (data_path.name + "_memmap") / "numpy_memmap.pkl"
        )

        if pickle_path.exists():
            logger.info(f"Memmap already exists: {pickle_path}")
            return pickle_path

        logger.info(f"Creating memmap: {pickle_path}")
        save_memmap_stream(
            chunk_iter=data_loader,
            pickle_path=pickle_path,
            dtype=np.float32,
            feature_dim=(1, 160000),
            total_rows=len(data_loader.dataset),
        )
        return pickle_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx].copy()).to(torch.float32)
        labels = [self.meta[key][idx] for key in self.labels_keys]

        return feature, labels
