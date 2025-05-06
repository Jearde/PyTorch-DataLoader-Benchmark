import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

from data.wav_dataset import WAVDataset

# logger = logging.getLogger(__name__)
logger = logging.getLogger("lightning.pytorch")


def parse_label(labels_df, label, hot=False, first_class=True, normalize_class=False):
    label = label.replace('"', "")
    label = label.replace(" ", "")
    labels = label.split(";")
    labels = [label.split(",") for label in labels]
    # Flatten list
    labels = [item for sublist in labels for item in sublist]

    for idx, label in enumerate(labels):
        if label[0] != "/":
            labels[idx] = "/" + label
    labels_num = []
    for _, row in labels_df.iterrows():
        if row["mid"] in labels:
            labels_num.append(row["index"])
    # labels_num = [row['index'] for _, row in labels_df.iterrows() if row['mid'] in labels]

    if hot:
        hot_labels = torch.zeros(len(labels_df))
        if first_class:
            labels_num = [labels_num[0]]
        hot_labels[labels_num] = 1
        if normalize_class:
            hot_labels /= hot_labels.sum()
        return hot_labels
    else:
        if first_class:
            labels_num = labels_num[0]
        return labels_num


class AudiosetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path = Path("/mnt/data/audioset/eval"),
        meta_dir: Path = Path("/mnt/data/audioset/meta"),
        perform_checks: bool = True,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.meta_dir = Path(meta_dir)
        meta_data_file = self.meta_dir / f"{data_dir.name}_meta.csv"
        self.perform_checks = perform_checks

        len_seconds = kwargs.get("len_seconds", 5)

        self.transforms = None

        # Look if class_labels exits
        class_labels_indices = self.meta_dir / "class_labels_indices.csv"
        if not (class_labels_indices).exists():
            # Download the class labels
            url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
            class_labels_indices.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, class_labels_indices)

        self.labels_df = pd.read_csv(self.meta_dir / "class_labels_indices.csv")

        self.data = []
        file_list = []
        self.label_dict = {}
        self.meta = pd.DataFrame(columns=["class", "basename"])

        # create a label dictionary
        labels = [label.name for label in self.data_dir.iterdir() if label.is_dir()]
        labels.sort()
        # labels_df = pd.read_csv(os.path.join(self.meta_dir, 'class_labels_indices.csv'))
        for _, row in self.labels_df.iterrows():
            # Mid: cryptic label, display_name: human readable label
            self.label_dict[row["display_name"]] = row["mid"]

        # Load the file where signals are associated with labels
        indicator = (
            self.data_dir.name
        )  # Not all labels are present in eval_segments.csv
        # indicator = 'unbalanced_train'
        segments_df = pd.read_csv(
            self.data_dir / (indicator + "_segments.csv"),
            comment="#",
            sep=",",
            quotechar='"',
            skipinitialspace=True,
            encoding="utf-8",
        )

        segments_dict = {}
        for _, row in segments_df.iterrows():
            segments_dict[row["YTID"]] = row["positive_labels"]

        # Check if meta_data_file exists and load it
        if meta_data_file.exists():
            self.meta = pd.read_csv(meta_data_file)
            self.meta["class"] = self.meta["class"].astype(int)
            self.meta["class_logits"] = self.meta["class_logits"].apply(
                lambda x: list(map(float, x.strip("[]").split()))
            )
            self.meta["basename"] = self.meta["basename"].astype(str)
            self.meta["file"] = self.meta["file"].astype(str)
            file_list = self.meta["file"].tolist()

            for file in file_list:
                file = Path(file)
                try:
                    label = segments_dict["_".join(file.stem.split("_")[:-1])]
                except:
                    try:
                        label = segments_dict[file.stem]
                    except:
                        continue

                self.data.append((file, label))
        else:
            files_wav = [str(f) for f in self.data_dir.rglob("*.wav")]
            files_npy = [str(f) for f in self.data_dir.rglob("*.npy")]

            if len(files_wav) == 0 and len(files_npy) == 0:
                logger.warning(f"No files found in {self.data_dir}")
            elif len(files_npy) > len(files_wav):
                logger.info(f"Found {len(files_npy)} npy files")
                files = files_npy
            else:
                logger.info(f"Found {len(files_wav)} wav files")
                files = files_wav

            for file in files:
                file_path = Path(file)
                try:
                    self.data.append(
                        (
                            file_path,
                            segments_dict["_".join(file_path.stem.split("_")[:-1])],
                        )
                    )
                except:
                    try:
                        self.data.append(
                            (file_path, segments_dict[file_path.parents[1].name])
                        )
                    except:
                        # print(f"Could not find {file.stem} in segments_dict")
                        continue

            # TODO: Save this as a csv file to reduce initialization time
            if self.perform_checks:
                logger.info(
                    f"Checking files for length of {len_seconds} seconds and if they can be loaded"
                )

            meta_dict = {
                "class": [],
                "class_logits": [],
                "basename": [],
                "file": [],
            }

            for i, (file, str_label) in tqdm(
                enumerate(self.data), total=len(self.data), desc="Checking files"
            ):
                if self.perform_checks:
                    if file.suffix == ".npy":
                        try:
                            vec = np.load(file)
                            if vec.shape[1] / 16000 < len_seconds:
                                continue
                        except:
                            continue
                    elif file.suffix == ".wav":
                        try:
                            audio_info = torchaudio.info(file)
                        except:
                            continue

                        if audio_info.num_frames / audio_info.sample_rate < len_seconds:
                            continue

                        try:
                            # Load the audio file
                            audio, sr = torchaudio.load(file, normalize=True)
                        except:
                            continue

                label = parse_label(
                    self.labels_df, str_label, hot=False, first_class=True
                )
                label_logits = parse_label(
                    self.labels_df,
                    str_label,
                    hot=True,
                    first_class=False,
                    normalize_class=False,
                ).numpy()

                files = [file]

                file_list.extend([str(file) for file in files])

                meta_dict["class"].extend([label for f in files])
                meta_dict["class_logits"].extend([label_logits for f in files])
                meta_dict["basename"].extend([str(f.name) for f in files])
                meta_dict["file"].extend([str(f) for f in files])

            logger.info(
                f"Found {len(self.meta)} suitable files out of {len(self.data)} ({len(self.meta) / len(self.data) * 100:.2f} %)"
            )

            self.meta = pd.DataFrame(meta_dict)

            # Save self.meta
            self.meta.to_csv(meta_data_file, index=False)

        self.wav_dataset = WAVDataset(
            wav_files=file_list,
            **kwargs,
        )

        if len(self.wav_dataset) > len(file_list):
            # Extend meta to match the length of the dataset
            self.meta = self.meta.iloc[
                np.repeat(np.arange(len(self.meta)), self.wav_dataset.ratio_lists)
            ]

    @classmethod
    def download(cls, data_dir: str, train: bool = True, **kwargs):
        AssertionError("Not implemented yet")

    def get_sample_data(self, idx):
        feature, label, meta = self.__getitem__(idx)

        basename = meta["basename"]

        return feature, feature, label, basename

    def __len__(self):
        assert len(self.wav_dataset) == len(self.meta)

        return len(self.wav_dataset)

    def __getitem__(self, idx):
        return self.wav_dataset[idx], torch.tensor(
            [self.meta.iloc[idx]["y_true"], self.meta.iloc[idx]["class"]]
        )
