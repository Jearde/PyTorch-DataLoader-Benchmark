from pathlib import Path
from typing import List

import lightning as pl
import numpy as np
import nvidia.dali.fn as fn
import pandas as pd
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tqdm.auto import tqdm


class PyTorchIterator(DALIGenericIterator):
    def __init__(self, labels: list[list[int]], *kargs, **kvargs):
        """Overloading of DALIGenericIterator for Multi-Label

        This is for supporting multiple labels per file. The index of the file is used to get the value in the list.

        Args:
            labels (List[List[int]]): _description_
        """
        super().__init__(*kargs, **kvargs)
        # self.labels = labels
        # Convert numpy elements to normal data types
        self.labels = [torch.tensor(label).squeeze(-1) for label in labels]

        # self.labels = torch.tensor(labels) # Should this be moved to gpu already?
        # biggest_dim = np.argmax(list(self.labels.shape)) # Biggest dimension should be number of files to be first
        # self.labels = self.labels.movedim(biggest_dim, 0)

    def __next__(self):
        out = super().__next__()

        # Use only the output of the first pipeline (not suitable for multiple pipelines in parallel)
        out = out[0]

        # Convert int to string and then to int
        idcs = [int("".join(map(chr, row))) for row in out["label"]]

        labels = [
            self.labels[i][idcs].to(out["audio"].device)
            for i in range(len(self.labels))
        ]

        # Get the label based on the index
        return out["audio"], labels


@pipeline_def
def numpy_data_pipeline(
    files: List[str],
    filename_len: int,
    target_sr: int = 16000,
    target_length: int = 10,
    shuffle: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    device: str = "cpu",
    direct_store: bool = False,
    mono: bool = True,
    rnd_crop_size: float = None,
    start_sec: float = None,
):
    """Load audio files and return audio and label

    Args:
        files (List[str]): List of file paths
        target_sr (int, optional): Target sample rate. Defaults to 16000.
        target_length (int, optional): Target length in seconds. Defaults to 10.
        shuffle (bool, optional): Shuffle the dataset. Defaults to False.
        device_id (int, optional): Device ID of the GPU. Also known as local rank in DDP. Defaults to -1.
        shard_id (int, optional): Shard ID. Also known as global rank in DDP. Defaults to 0.
        num_shards (int, optional): Number of shards. Also known as world size in DDP. Defaults to 1.
        py_num_workers(int, optional): Number of Python workers for data loading via fn.external_source(). Defaults to 1.
    """
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.html

    if rnd_crop_size is not None:
        rnd_choice_list = list(
            range(int(target_length * target_sr) - int(rnd_crop_size * target_sr))
        )
        if len(rnd_choice_list) == 0:
            rnd_choice_list = [0]
        start = fn.random.choice(rnd_choice_list, shape=[1])
        end = start + int(rnd_crop_size * target_sr)
    elif start_sec is not None:
        start = int(start_sec * target_sr)
        end = start + int(target_length * target_sr)
    else:
        start = 0
        end = target_length * target_sr

    # Load audio
    audio = fn.readers.numpy(
        files=files,
        cache_header_information=True,
        out_of_bounds_policy="pad",
        fill_value=0.0,
        # roi_shape=[target_length * target_sr],
        roi_start=start,
        roi_end=end,
        roi_axes=[1],
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        device=device if direct_store else "cpu",
        seed=42,
        name="Reader",
    )

    label = fn.get_property(audio, key="source_info", name="Label")

    # Get only the name of the file. It is important that all file names have the same length
    label = label[-filename_len:-4]

    # If not using downmix for reducing the dimension, you can use mean and keep the dimension
    if mono:
        audio = fn.reductions.mean(audio, axes=[-2], keep_dims=True)

    return audio, label


def DaliNumpyPipeline(
    files: List[str],
    labels: List[List[int]],
    batch_size: int,
    target_sr: int = 16000,
    target_length: int = 10,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    mono: bool = True,
    random_crop_size: float = None,
    start_sec: float = None,
    direct_store: bool = False,
    **kwargs,
):
    num_threads = num_threads if num_threads > 0 else torch.multiprocessing.cpu_count()
    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = "gpu" if device_id >= 0 else "cpu"

    filename_len = len(files[0].split("/")[-1])

    # Map the labels to the file index
    # labels_dict = {}
    # for idx, file in enumerate(files):
    #     labels_dict[int(file[-filename_len:-4])] = [
    #         torch.tensor(labels[i][idx]).squeeze(-1) for i in range(len(labels))
    #     ]

    pipeline = numpy_data_pipeline(
        files=files,
        filename_len=filename_len,
        # label_lists=[external_data_source(label_lists[0]), external_data_source(label_lists[1])],
        target_sr=target_sr,
        target_length=target_length,
        batch_size=batch_size,
        num_threads=num_threads,
        shuffle=shuffle,
        device=device,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        direct_store=direct_store,
        mono=mono,
        rnd_crop_size=random_crop_size,
        start_sec=start_sec,
        # prefetch_factor=prefetch_factor,
        **kwargs,
    )
    pipeline.build()

    # TODO How to get labels?
    # img , labels = pipeline.run()
    # print(img)
    # print(labels)
    # ascii_array = labels.at(0)
    # decoded_string = ''.join(chr(value) for value in ascii_array)
    # labels_array = labels.as_array()

    return PyTorchIterator(
        pipelines=[pipeline],
        labels=labels,
        output_map=["audio", "label"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        reader_name="Reader",
        prepare_first_batch=True,
    )


# PyTorch Lightning DataModule to use this dataset
class NumpyDataModule(pl.LightningDataModule):
    def __init__(self, file_paths: List[str], labels: List[str], batch_size: int = 4):
        super(NumpyDataModule, self).__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.device_id = device_id
        self.num_threads = num_threads
        self.labels = [labels, labels]

    def train_dataloader(self):
        dali_iterator = DaliNumpyPipeline(
            files=self.file_paths,
            labels=self.labels,
            batch_size=self.batch_size,
            target_sr=16000,
            target_length=10,
            num_threads=4,
            shuffle=True,
            local_rank=0,
            global_rank=0,
            world_size=1,
        )

        return dali_iterator


def preprocess_wav(
    data_loader: torch.utils.data.DataLoader, data_path: Path, labels_keys: List[str]
):
    path_numbered = data_path.parents[0] / (data_path.name + "_numpy")
    path_numbered.mkdir(parents=True, exist_ok=True)
    meta_numpy_file = path_numbered / "meta.pkl"

    num_files = len(data_loader) * data_loader.batch_size

    if meta_numpy_file.exists():
        # meta_numpy = pd.read_csv(meta_numpy_file)
        meta_numpy = pd.read_pickle(meta_numpy_file)
        files_numpy = meta_numpy["file"].tolist()
    else:
        digits_files = len(str(num_files))
        files_numpy = []
        labels_numpy = []

        for i, (audio, label) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Converting data into numpy",
        ):
            for j, audio_j in enumerate(audio):
                file_name = f"{i * data_loader.batch_size + j}".zfill(digits_files)
                np.save(path_numbered / f"{file_name}.npy", audio_j.cpu().numpy())
                files_numpy.append(str(path_numbered / f"{file_name}.npy"))
                labels_numpy.append([l[j].cpu().numpy() for l in label])

        assert len(files_numpy) == len(labels_numpy), (
            f"Number of files and labels do not match: {len(files_numpy)} != {len(labels_numpy)}"
        )

        meta_numpy = pd.DataFrame(
            {
                "file": files_numpy,
            }
        )
        for i, label in enumerate(labels_keys):
            # meta_numpy[label] = [l[i] for l in labels_numpy]
            if isinstance(labels_numpy[0][i], list):
                meta_numpy[label] = [int(l[i]) for l in labels_numpy]
            elif isinstance(labels_numpy[0][i], np.ndarray):
                meta_numpy[label] = [l[i].astype(int).tolist() for l in labels_numpy]
            else:
                meta_numpy[label] = [int(l[i]) for l in labels_numpy]

        meta_numpy.to_pickle(meta_numpy_file)

    return meta_numpy
