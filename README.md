# PyTorch-DataLoader-Benchmark
This repository contains a benchmark for PyTorch DataLoader using different data loading strategies and configurations. The goal is to compare the performance of various data loading methods, including the following:
- PyTorch DataLoader with Torchaudio
- NVIDIA DALI with WAV files
- NVIDIA DALI with numbered Numpy files
- NVIDIA DALI with Numpy files using external source
- Numpy Memmap
- TorchRL TensorDict LazyMemmapStorage
- WebDataset with Numpy files

For testing, PyTorch Lightning with a 2.1M parameter model (linear autoencoder) is used to simulate a real-world scenario without putting the bottleneck on preprocessing or GPU training.
The benchmark is designed to be extensible, allowing for easy addition of new data loading strategies and configurations.

Slides of the [PDSC3K](https://skunkforce.org/pdsc3k) Talk can be found here: [How fast can you go? - PyTorch (Audio) DataLoader](https://docs.google.com/presentation/d/1ZE19MCjcEgdmuSpU9kxYvuXNF0LT1XroJ_p15hlz4Vw/edit?usp=sharing)

## Getting Started
### Prerequisites
It's recommended to use this repository with the provided Development Container. This ensures that all dependencies are installed and configured correctly.
To use the Development Container, follow these steps:
1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Clone this repository to your local machine.
3. Open the repository in Visual Studio Code.
 - You might want to change some mount paths in the `.devcontainer/devcontainer.json` file to match your local setup. For example, if you want to mount a different path for the `data` directory, you can change the `mounts` section in the `devcontainer.json` file.
4. Press `F1` to open the Command Palette, then type and select `Rebuiild and Reopen in Container`.
5. Wait for the container to build and start. This may take a few minutes, depending on your internet connection and system performance.

### Running the Benchmark
To run the benchmark, you can use the provided scripts. The main script is `notebooks/benchmark_lightning_audio`, which allows you to test different data loading strategies and configurations.
All files located in the `notebooks` can be either run as a script or as a Jupyter Notebook
0. Download Google AudioSet evaluation set
 - Add your Google Username and password to the `src/data/audioset/yt-dlp.conf` for authentication when downloading the dataset from YouTube. Otherwise your IP may be blocked.
```bash
python notebooks/download_audioset.py
```
1. To run as a script, use the command:
 - It will generate tensorboard logs in the `logs` directory.
```bash
python notebooks/benchmark_lightning_audio.py
```
2. To visualize the results, you can use TensorBoard. Run the following command in your terminal:
```bash
tensorboard --logdir logs/tensorboard
```
Then, open your web browser and go to `http://localhost:6006` to view the TensorBoard dashboard.
```
3. Or you can create plots from the logs using the `notebooks/plot.py` script:
```bash
python notebooks/plot.py
```

## Good Resources

### WebDataset
- [WebDataset with Large Datasets (YouTube)](https://youtube.com/playlist?list=PL0dsKxFNMcX59napupGk3cNVzbpx3yU0_&si=sM8V_FqCzD89FHsh)
- [AIStore with WebDataset](https://aiatscale.org/blog/2023/05/05/aisio-transforms-with-webdataset-pt-1)

### AIStore
- [Local Filesystems Optimization](https://aiatscale.org/docs/performance)

## TODOs
- [ ] WebDataset (Tune and correct the train/validation split)
- [ ] WebDataset using DALI
- [ ] HDF5
- [ ] AIStore
