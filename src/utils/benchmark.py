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
import torch

from model.dummy_model import DummyModel


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
                L.pytorch.callbacks.DeviceStatsMonitor(),
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
