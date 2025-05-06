import logging
from pathlib import Path

import lightning as L
import torch
import torchaudio
import torchmetrics
import torchvision

logger = logging.getLogger(__name__)


def print_stats():
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Lightning Version: {L.__version__}")
    logger.info(f"Torchvision Version: {torchvision.__version__}")
    logger.info(f"Torchaudio Version: {torchaudio.__version__}")
    logger.info(f"PyTorch Metrics Version: {torchmetrics.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")


def get_data_size(files: list[str | Path]):
    total_size = 0
    for file in files:
        total_size += Path(file).stat().st_size
    print(f"Total size of all files: {total_size / (1024**3):.2f} GB")
