import logging
import time

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def test_data_loader(data_loader, epochs: int):
    for epoch in range(epochs):
        for i, (audio, label) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Loading data",
        ):
            audio.shape


import lightning as L
import nvidia_smi
import torch
from lightning.pytorch.callbacks import Callback

from model.dummy_model import DummyModel


class GPUUtilizationCallback(Callback):
    def __init__(self, log_frequency=1):
        super().__init__()
        self.log_frequency = log_frequency
        nvidia_smi.nvmlInit()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_frequency == 0:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            # Log metrics to Lightning
            pl_module.log("gpu_utilization", res.gpu)
            pl_module.log("gpu_memory_utilization", res.memory)
            pl_module.log("gpu_memory_free_gb", mem.free / (1024**3))
            pl_module.log("gpu_memory_used_gb", mem.used / (1024**3))


class DataLoaderBenchmark:
    def __init__(self, epochs: int, devices: int = 1):
        self.epochs = epochs
        self.devices = devices

        torch.set_float32_matmul_precision("high")

    def run(self, data_module: L.LightningDataModule, name: str):
        logger.info(f"Running benchmark for {name}...")

        data_module.prepare_data()
        data_module.setup(stage="fit")

        tb_logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir="logs/tensorboard/", max_queue=1000, name=name
        )

        model = DummyModel(input_shape=[1, 160000], num_classes=2)
        model = torch.compile(model)

        trainer = L.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices=self.devices,  # -1 for using all available GPUs with DDP
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
                # L.pytorch.callbacks.RichModelSummary(max_depth=-1),
                # L.pytorch.callbacks.DeviceStatsMonitor(),
                GPUUtilizationCallback(log_frequency=1),
            ],
            enable_model_summary=False,
            # logger=True,
            logger=[tb_logger],
            strategy="ddp_find_unused_parameters_true"
            if self.devices > 1 or self.devices == -1
            else "auto",
        )

        trainer.fit_loop.epoch_progress.reset()
        start = time.time()
        trainer.fit(
            model,
            datamodule=data_module,
        )
        end = time.time()
        time_delta = end - start

        return time_delta
