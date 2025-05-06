import logging

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
