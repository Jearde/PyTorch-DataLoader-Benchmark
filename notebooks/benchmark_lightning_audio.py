# %%
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parents[1] / "src"))
from data.audioset.audioset_dataset import AudiosetDataset
from data.dali.dali_numpy_loader import DaliNumpyPipeline, preprocess_wav
from data.dali.dali_wav_loader import DaliAudioPipeline
from data.data_model import AudioDataModule
from data.torch.torchaudio_wav_loader import TorchAudioDataset
from utils.logger import get_data_size

# %%
data_path = Path("/mnt/data/data/audioset/eval/")
meta_path = Path("/mnt/data/data/audioset/meta/")

# %%
audioset_dataset = AudiosetDataset(
    data_dir=data_path,
    meta_dir=meta_path,
    perform_checks=False,
)

# %%
files = audioset_dataset.wav_dataset.wavs
meta = audioset_dataset.meta
labels_keys = ["class", "class_logits"]

epochs: int = 10
batch_size: int = 32
num_workers: int = -1
prefetch_factor: int = 2
target_sr: int = 16000
target_audio_length: int = 10
audio_slice_length: int | None = None
mono: bool = True

local_rank: int = 0
global_rank: int = 0
world_size: int = 1


num_workers = (
    num_workers
    if num_workers is not None and num_workers != -1
    else torch.multiprocessing.cpu_count() - 1
)
prefetch_factor = prefetch_factor if num_workers > 0 else None
persistent_workers = True if num_workers > 0 else False

data_loader_settings_dali = {
    "batch_size": batch_size,
    "num_threads": num_workers,
    "prefetch_factor": prefetch_factor,
    "shuffle": False,
    "local_rank": local_rank,
    "global_rank": global_rank,
    "world_size": world_size,
    "target_sr": target_sr,
    "target_length": target_audio_length,
    "mono": mono,
    "random_crop_size": audio_slice_length,
}

data_loader_settings_pytorch = {
    "batch_size": batch_size,
    "num_workers": num_workers,
    "prefetch_factor": prefetch_factor,
    "shuffle": False,
    "pin_memory": True,
    "persistent_workers": persistent_workers,
}

dataset_settings = {
    "target_sr": target_sr,
    "target_length": target_audio_length,
    "mono": mono,
    "random_slice": False,
    "audio_slice_length": audio_slice_length,
    "audio_slice_overlap": None,
    "check_silence": False,
    "window_size": None,
    "overlap": 0.5,
}

# %% Get storage size of all files combined
get_data_size(files)

# %% Get Lightning Module
import lightning as L

from model.dummy_model import DummyModel

model = DummyModel(input_shape=[1, 160000], num_classes=2)

devices = 1  # -1 for using all available GPUs with DDP

trainer = L.Trainer(
    max_epochs=epochs,
    accelerator="auto",
    devices=devices,  # -1 for using all available GPUs with DDP
    num_nodes=1,
    precision=32,
    log_every_n_steps=50,
    deterministic=False,
    benchmark=None,
    fast_dev_run=False,
    plugins=None,
    enable_checkpointing=False,
    enable_progress_bar=True,
    profiler="simple",  # e.g. "simple", "pytorch", "advanced"
    callbacks=[
        L.pytorch.callbacks.RichProgressBar(refresh_rate=10),
        L.pytorch.callbacks.RichModelSummary(max_depth=-1),
    ],
    logger=False,
    # logger=[tb_logger, mlflow_logger],
    strategy="ddp_find_unused_parameters_true"
    if devices > 1 or devices == -1
    else "auto",
)

# %% TorchAudioDataset
torch_audio_dataset = TorchAudioDataset(
    files=files,
    meta=meta,
    labels=labels_keys,
    **dataset_settings,
)

data_module = AudioDataModule(
    dataset=torch_audio_dataset,
    data_loader_class=DataLoader,
    data_loader_settings=data_loader_settings_pytorch,
)
data_module.setup("fit")

# %% DaliAudioPipeline
labels = [meta[label].values.tolist() for label in labels_keys]

data_module = AudioDataModule(
    files=files,
    labels=labels,
    data_loader_class=DaliAudioPipeline,
    data_loader_settings=data_loader_settings_dali,
)
data_module.setup("fit")

# %% DaliNumpyPipeline
torch_audio_loader = DataLoader(
    torch_audio_dataset,
    **data_loader_settings_pytorch,
)
meta_numpy = preprocess_wav(
    data_loader=torch_audio_loader, data_path=data_path, labels_keys=labels_keys
)
files_numpy = meta_numpy["file"].values.tolist()
labels_numpy = [meta_numpy[key].values.tolist() for key in labels_keys]

data_module = AudioDataModule(
    files=files_numpy,
    labels=labels_numpy,
    data_loader_class=DaliNumpyPipeline,
    data_loader_settings=data_loader_settings_dali | {"direct_store": True},
)
data_module.setup("fit")

# %%
trainer.fit(
    model,
    datamodule=data_module,
)

# %%
pass
