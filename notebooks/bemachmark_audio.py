# %%
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parents[1] / "src"))
from data.audioset.audioset_dataset import AudiosetDataset
from data.dali.dali_numpy_loader import DaliNumpyPipeline, preprocess_wav
from data.dali.dali_wav_loader import DaliAudioPipeline
from data.data_model import AudioDataModule


# %%
def test_data_loader(data_loader, epochs: int):
    for epoch in range(epochs):
        for i, (audio, label) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Loading data",
        ):
            # Check the shape of the audio tensor
            # assert audio.shape[0] == batch_size, (
            #     f"Batch size mismatch: {audio.shape[0]}"
            # )
            assert audio.shape[1] == 1 if mono else 2, (
                f"Audio channel mismatch: {audio.shape[1]}"
            )
            assert audio.shape[2] == target_audio_length * target_sr, (
                f"Audio length mismatch: {audio.shape[2]}"
            )

            # Check the metadata
            # assert len(label[0]) == batch_size, f"Label size mismatch: {len(label)}"


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

epochs: int = 3
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
total_size = 0
for file in files:
    total_size += Path(file).stat().st_size
print(f"Total size of all files: {total_size / (1024**3):.2f} GB")

# %% TorchAudioDataset
from data.torch.torchaudio_wav_loader import TorchAudioDataset

torch_audio_dataset = TorchAudioDataset(
    files=files,
    meta=meta,
    labels=labels_keys,
    **dataset_settings,
)

torch_audio_loader = DataLoader(
    torch_audio_dataset,
    **data_loader_settings_pytorch,
)

# %% Test the DataLoader
test_data_loader(torch_audio_loader, epochs)

# %% Init DALI WAV
labels = [meta[label].values.tolist() for label in labels_keys]

dali_wav_loader = DaliAudioPipeline(
    files=files,
    labels=labels,
    **data_loader_settings_dali,
)

# %% Test the DataLoader
test_data_loader(dali_wav_loader, epochs)

# %% Pre-Process for numpy

meta_numpy = preprocess_wav(
    data_loader=torch_audio_loader, data_path=data_path, labels_keys=labels_keys
)
files_numpy = meta_numpy["file"].values.tolist()
labels_numpy = [meta_numpy[key].values.tolist() for key in labels_keys]

dali_numpy_loader = DaliNumpyPipeline(
    files=files_numpy,
    labels=labels_numpy,
    **data_loader_settings_dali,
)

# %% Test the DataLoader
test_data_loader(dali_numpy_loader, epochs)

# %%
data_module = AudioDataModule(
    files=files,
    labels=labels,
    data_loader_class=DaliAudioPipeline,
    data_loader_settings=data_loader_settings_dali,
)

# %% Test the DataModule
data_module.setup("fit")
train_loader = data_module.train_dataloader()
test_data_loader(train_loader, epochs)

# %%
pass
