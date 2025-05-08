# %%
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parents[1] / "src"))
from config.configs import get_settings
from data.audioset.audioset_dataset import AudiosetDataset
from data.dali.dali_external_numpy_loader import DaliNumpyExternalPipeline
from data.dali.dali_numpy_loader import DaliNumpyPipeline, preprocess_wav
from data.dali.dali_wav_loader import DaliAudioPipeline
from data.data_model import AudioDataModule
from data.torch.torchaudio_wav_loader import TorchAudioDataset
from data.webdataset.webdataset_torch_module import WebDatasetDataModule
from utils.logger import get_data_size, print_stats

# %%
print_stats()

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
epochs: int = 1

data_loader_settings_dali, data_loader_settings_pytorch, dataset_settings = (
    get_settings()
)

labels = [meta[label].values.tolist() for label in labels_keys]


# %% Get storage size of all files combined
get_data_size(files)

# %% Get Lightning Module
from utils.benchmark import DataLoaderBenchmark

dataloader_benchmark = DataLoaderBenchmark(epochs=epochs, devices=1)

# %% TorchAudioDataset
torch_audio_dataset = TorchAudioDataset(
    files=files,
    labels=labels,
    **dataset_settings,
)

torch_audio_loader = DataLoader(
    torch_audio_dataset,
    **data_loader_settings_pytorch,
)

data_module = AudioDataModule(
    dataset=torch_audio_dataset,
    data_loader_class=DataLoader,
    data_loader_settings=data_loader_settings_pytorch,
    data_path=data_path,
    labels_keys=labels_keys,
)

time_torchaudio = dataloader_benchmark.run(
    data_module=data_module,
    name="TorchAudioDataset",
)

# %% WebDatasetPyTorch
data_module = WebDatasetDataModule(
    data_path=Path("/mnt/data/data/audioset/eval/"),
    data_loader=torch_audio_loader,
    urls=None,
    validation_split=0.2,
    seed=42,
    **data_loader_settings_pytorch,
)
data_module.prepare_data()

# %%
time_webdataset = dataloader_benchmark.run(
    data_module=data_module,
    name="WebDataset",
)
# %% MemmapDataset
from data.torch.tensordirct_loader import TensorDictMemmapDataset

memmap_path = TensorDictMemmapDataset.prepare_data(
    data_path=data_path,
    data_loader=torch_audio_loader,
)

memmap_dataset = TensorDictMemmapDataset(
    memmap_path=memmap_path,
)
data_module = AudioDataModule(
    dataset=memmap_dataset,
    data_loader_class=DataLoader,
    data_loader_settings=data_loader_settings_pytorch,
    data_path=data_path,
    labels_keys=labels_keys,
)

# %%
time_tensordict_memmap = dataloader_benchmark.run(
    data_module=data_module,
    name="TensorDictMemmapDataset",
)

# %% MemmapDataset
from data.numpy.numpy_memmap_loader import NumpyMemmapDataset

pickle_path = NumpyMemmapDataset.prepare_data(
    data_path=data_path,
    data_loader=torch_audio_loader,
)

memmap_dataset = NumpyMemmapDataset(
    pickle_path=pickle_path,
)
data_module = AudioDataModule(
    dataset=memmap_dataset,
    data_loader_class=DataLoader,
    data_loader_settings=data_loader_settings_pytorch,
    data_path=data_path,
    labels_keys=labels_keys,
)

# %%
time_numpy_memmap = dataloader_benchmark.run(
    data_module=data_module,
    name="NumpyMemmapDataset",
)


# %% DaliAudioPipeline
data_module = AudioDataModule(
    files=files,
    labels=labels,
    data_loader_class=DaliAudioPipeline,
    data_loader_settings=data_loader_settings_dali,
    data_path=data_path,
    labels_keys=labels_keys,
)

# %%
time_dali_wav = dataloader_benchmark.run(
    data_module=data_module,
    name="DaliAudioPipeline",
)

# %% DaliNumpyPipeline
meta_numpy = preprocess_wav(
    data_loader=torch_audio_loader, data_path=data_path, labels_keys=labels_keys
)
files_numpy = meta_numpy["file"].values.tolist()
labels_numpy = [meta_numpy[key].values.tolist() for key in labels_keys]

data_module = AudioDataModule(
    files=files_numpy,
    labels=labels_numpy,
    data_loader_class=DaliNumpyPipeline,
    data_loader_settings=data_loader_settings_dali | {"direct_store": False},
    data_path=data_path,
    labels_keys=labels_keys,
)

# %%
time_numpy = dataloader_benchmark.run(
    data_module=data_module,
    name="DaliNumpyPipeline",
)

# %% DaliNumpyExternalPipeline
data_module = AudioDataModule(
    files=files_numpy,
    labels=labels_numpy,
    data_loader_class=DaliNumpyExternalPipeline,
    data_loader_settings=data_loader_settings_dali,
    data_path=data_path,
    labels_keys=labels_keys,
)

# %%
time_dali_numpy_external = dataloader_benchmark.run(
    data_module=data_module,
    name="DaliNumpyExternalPipeline",
)

# %% Print results
results_df = pd.DataFrame(
    {
        "Dataset": [
            "TorchAudioDataset",
            "TensorDictMemmapDataset",
            "NumpyMemmapDataset",
            "DaliAudioPipeline",
            "DaliNumpyPipeline",
            "WebDataset",
            "DaliNumpyExternalPipeline",
        ],
        "Time (s)": [
            time_torchaudio,
            time_tensordict_memmap,
            time_numpy_memmap,
            time_dali_wav,
            time_numpy,
            time_webdataset,
            time_dali_numpy_external,
        ],
    }
)
results_df["Speed (samples/s)"] = (
    (results_df["Time (s)"].sum() / results_df["Time (s)"])
    * len(files)
    / results_df["Time (s)"]
)
results_df["Speed (samples/s)"] = results_df["Speed (samples/s)"].round(2)
results_df["Time (s)"] = results_df["Time (s)"].round(2)
results_df = results_df.sort_values(by="Speed (samples/s)", ascending=False)
print(results_df)

# %% Save results
results_df.to_csv("logs/benchmark_results.csv", index=True)

# %%
pass
