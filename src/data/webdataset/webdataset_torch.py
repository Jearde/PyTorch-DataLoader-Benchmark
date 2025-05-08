import json
import logging
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def write_wds(chunk_iter, out_dir, maxcount=10_000, compression=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = (out_dir / "data-%06d.tar").as_posix()

    with wds.ShardWriter(pattern, maxcount=maxcount, compress=compression) as sink:
        sid = 0
        for features, labels in tqdm(chunk_iter, desc="wds-write"):
            for i in range(len(features)):
                key = f"sample_{sid:09d}"

                label_dict = {
                    f"label_{j}": labels[j][i].numpy().astype(float).tolist()
                    for j in range(len(labels))
                }

                sink.write(
                    {
                        "__key__": key,
                        "x.npy": features[i].numpy(),
                        "x.pyd": torch.as_tensor(features[i]),
                        "y.cls": int(labels[0][i]),
                        "ylists.json": json.dumps(label_dict).encode("utf-8"),
                    }
                )
                sid += 1

    # glob get all the shards
    urls = f"data-{{{0:06d}..{sid:06d}}}.tar"

    return urls


from typing import Callable, Sequence


def dict_to_tensor_list(sample):
    """
    x : feature tensor
    y : scalar class (already int from .cls)
    z : dict like {'label_0': 413.0, 'label_1': [0.0, ...]}
    """
    if isinstance(sample, dict):
        keys = ("label_0", "label_1")  # or sorted(z) if you like
        tensors = [torch.as_tensor(sample[k]) for k in keys]
        return tensors
    elif isinstance(sample, np.ndarray):
        return torch.as_tensor(sample)
    elif isinstance(sample, list):
        return [torch.as_tensor(x) for x in sample]
    elif isinstance(sample, tuple):
        return tuple(torch.as_tensor(x) for x in sample)
    elif isinstance(sample, (int, float, complex)):
        return torch.as_tensor(sample)
    else:
        return sample


class WebAudioDataset:
    def __init__(
        self,
        urls: str | list[str],
        batch_size: int = 16,
        shuffle_shards: int = 100,
        shuffle_buffer: int = 10_000,
        tuple_fields: Sequence[str] = ("x.npy", "ylists.json"),
        preprocess: Callable | None = None,
        data_length: int = None,
    ):
        super().__init__()  # important for IterableDataset
        self.urls = str(urls)
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.tuple_fields = tuple_fields
        self.preprocess = lambda x: x if preprocess is None else preprocess(x)
        self.total_length = data_length

    def get_data_pipeline(self, data_length: int = None, selector=None):
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.urls),
            # at this point we have an iterator over all the shards
            # this shuffles the shards
            wds.shuffle(self.shuffle_shards),
            wds.split_by_node,  # if you are using multiple nodes
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            wds.select(selector),
            # this shuffles the samples in memory
            wds.shuffle(self.shuffle_buffer),
            # this decodes the images and json
            wds.decode(),
            wds.to_tuple(*self.tuple_fields),
            # wds.map(self.preprocess),
            wds.map_tuple(dict_to_tensor_list, dict_to_tensor_list),
            wds.batched(self.batch_size),
        )

        if data_length is not None:
            self.dataset = pipeline.with_length(data_length)

        return pipeline

    @classmethod
    def prepare_data(cls, data_path: Path, data_loader: DataLoader):
        shard_path = data_path.parents[0] / (data_path.name + "_webdataset")

        if shard_path.exists():
            sid = len(list(shard_path.glob("data-*.tar"))) - 1

            if sid >= 0:
                logger.info(f"{sid + 1} WebDataset Shards already exists: {shard_path}")
                return shard_path / f"data-{{{0:06d}..{sid:06d}}}.tar"

        logger.info(f"Creating WebDataset Shards: {shard_path}")
        urls = write_wds(
            chunk_iter=data_loader,
            out_dir=shard_path,
            maxcount=10_000,
            compression=None,
        )
        return urls
