from pathlib import Path

import torch

from data.wav_dataset import WAVDataset


class TorchAudioDataset(WAVDataset):
    def __init__(
        self,
        files: list[str | Path],
        labels: list[list[int]],
        **kwargs,
    ):
        """
        Args:
        files (list[str | Path]): List of file paths to the audio files.
        meta (pd.DataFrame): DataFrame containing metadata for the audio files.
        labels (list[str], optional): List of labels to be used. Defaults to ["class"].
        **kwargs: Additional keyword arguments.
        """
        super().__init__(wav_files=files, **kwargs)
        self.files = files
        self.labels = [torch.tensor(label).squeeze(-1) for label in labels]

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
        idx (int): Index of the item to retrieve.

        Returns:
        tuple: A tuple containing the audio waveform, sample rate, and metadata.
        """
        # Get the audio waveform and sample rate
        audio = super().__getitem__(idx)

        labels = [self.labels[i][idx].to(audio.device) for i in range(len(self.labels))]

        return audio, labels
