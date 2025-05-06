import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def check_audioset(dataset_path: str, size_thres: float = 0.2):
    sub_dirs = [d for d in os.listdir(dataset_path)]

    # Remove files smaller than 200 bytes
    size_thres = size_thres  # in KB
    success_counter = 0
    fail_counter = 0
    file_counter = 0
    for sub_dir in tqdm(sub_dirs):
        sub_dir_path = os.path.join(dataset_path, sub_dir)
        files = os.listdir(sub_dir_path)
        files = [os.path.join(sub_dir_path, f) for f in files]

        for file in tqdm(files, leave=False):
            file_counter += 1
            if os.path.getsize(file) / 1024 < size_thres:
                os.remove(file)
                fail_counter += 1

            else:
                success_counter += 1

    print("Success", success_counter, "/", file_counter)
    print("Fails", fail_counter, "/", file_counter)

    for sub_dir in tqdm(sub_dirs):
        sub_dir_path = os.path.join(dataset_path, sub_dir)
        if len(os.listdir(sub_dir_path)) == 0:
            os.rmdir(sub_dir_path)

    print("FIN")


def download_audioset(
    root_dir: str = "/mnt/nvme_nfs/home/rglitza/datasets/audioset",
    labels: list[str] = None,
    n_jobs: int = 12,
    download_type: str = "unbalanced_train",  # 'eval' 'unbalanced_train'
    copy_and_replicate: bool = True,
    verbose_output: str = "",
    file_format: str = "wav",  # 'wav' 'npy'
):
    dataset_path = Path(root_dir) / download_type

    d = Downloader(
        root_path=dataset_path,
        labels=labels,
        n_jobs=n_jobs,
        download_type=download_type,
        copy_and_replicate=copy_and_replicate,
        verbose_output=verbose_output,
    )
    d.download(
        file_format=file_format,
        quality=0,
    )

    return dataset_path


class Downloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """

    def __init__(
        self,
        root_path: str,
        labels: list = None,  # None to download all the dataset
        n_jobs: int = 1,
        download_type: str = "unbalanced_train",
        copy_and_replicate: bool = True,
        verbose_output: str = "> /dev/null 2>&1",
        mono: bool = False,
        check_files: bool = True,
        min_length_sec: float = 2.0,
        export_format: str = "wav",
    ):
        """
        This method initializes the class.
        :param root_path: root path of the dataset
        :param labels: list of labels to download
        :param n_jobs: number of parallel jobs
        :param download_type: type of download (unbalanced_train, balanced_train, eval)
        :param copy_and_replicate: if True, the audio file is copied and replicated for each label.
                                    If False, the audio file is stored only once in the folder corresponding to the first label.
        """
        # Set the parameters
        self.root_path = root_path
        self.labels = labels
        self.n_jobs = n_jobs
        self.download_type = download_type
        self.copy_and_replicate = copy_and_replicate
        self.verbose_output = verbose_output
        self.mono = mono
        self.perform_checks = check_files
        self.min_length_sec = min_length_sec
        self.export_format = export_format

        # Create the path
        os.makedirs(self.root_path, exist_ok=True)
        self.read_class_mapping()

    def read_class_mapping(self):
        """
        This method reads the class mapping.
        :return: class mapping
        """

        class_df = pd.read_csv(
            "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
            sep=",",
        )

        self.display_to_machine_mapping = dict(
            zip(class_df["display_name"], class_df["mid"])
        )
        self.machine_to_display_mapping = dict(
            zip(class_df["mid"], class_df["display_name"])
        )
        return

    def download(
        self,
        file_format: str = "wav",
        quality: int = 10,
        parallel_verbose: int = 0,
    ):
        """
        This method downloads the dataset using the provided parameters.
        :param format: format of the audio file (vorbis, mp3, m4a, wav), default is vorbis
        :param quality: quality of the audio file (0: best, 10: worst), default is 5
        """
        self.format_numpy = False
        self.file_format = file_format
        if isinstance(file_format, list):
            self.file_format = file_format[0]
            self.format_numpy = True
            self.export_format = file_format
        elif file_format == "npy":
            self.file_format = "wav"
            self.format_numpy = True
            self.export_format = "npy"
        elif file_format not in ["vorbis", "mp3", "m4a", "wav"]:
            raise ValueError(
                f"Format {format} is not supported. Please use vorbis, mp3, m4a, wav, or npy."
            )

        self.quality = quality

        # Load the metadata
        if "strong" in self.download_type:
            raise ValueError("The strong labels are not supported yet.")
            metadata = pd.read_csv(
                f"http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset{self.download_type}.tsv",
                sep="\t",
                skiprows=3,
                header=None,
                names=[
                    "YTID",
                    "start_seconds",
                    "end_seconds",
                    "MID",
                    "positive_labels",
                ],
                engine="python",
            )
        else:
            metadata = pd.read_csv(
                f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.download_type}_segments.csv",
                sep=", ",
                skiprows=3,
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
        if self.labels is not None:
            self.real_labels = [
                self.display_to_machine_mapping[label] for label in self.labels
            ]
            metadata = metadata[
                metadata["positive_labels"].apply(
                    lambda x: any([label in x for label in self.real_labels])
                )
            ]
            # remove " in the labels
        metadata["positive_labels"] = metadata["positive_labels"].apply(
            lambda x: x.replace('"', "")
        )
        metadata = metadata.reset_index(drop=True)

        inv_display_to_machine_mapping = {
            v: k for k, v in self.display_to_machine_mapping.items()
        }

        # Add self.display_to_machine_mapping[label] to the metadata
        metadata["positive_human_labels"] = metadata["positive_labels"].apply(
            lambda x: ",".join(
                [self.machine_to_display_mapping[label] for label in x.split(",")]
            )
        )

        # Save the metadata as a CSV file
        metadata.to_csv(
            os.path.join(self.root_path, f"{self.download_type}_segments.csv"),
            index=False,
        )

        logger.info(f"Downloading to {self.root_path}")
        logger.info(f"Downloading {len(metadata)} files...")

        self.digits_files = len(str(len(metadata)))

        # Download the dataset
        outputs = list(
            tqdm(
                joblib.Parallel(
                    return_as="generator", n_jobs=self.n_jobs, verbose=parallel_verbose
                )(
                    joblib.delayed(self.download_file)(
                        i,
                        metadata.loc[i, "YTID"],
                        metadata.loc[i, "start_seconds"],
                        metadata.loc[i, "end_seconds"],
                        metadata.loc[i, "positive_labels"],
                        self.verbose_output,
                        self.format_numpy,
                    )
                    for i in range(len(metadata))
                ),
                total=len(metadata),
                desc="Downloading files with yt-dlp",
            )
        )

        logger.info("Done.")
        logger.info(f"{outputs.count(0)}/{len(outputs)} files downloaded successfully.")

    def download_file(
        self,
        file_idx: int,
        ytid: str,
        start_seconds: float,
        end_seconds: float,
        positive_labels: str,
        output_options: str = "",
        convert_to_numpy: bool = False,
    ):
        """
        This method downloads a single file. It only download the audio file at 16kHz.
        If a file is associated to multiple labels, it will be stored multiple times.
        :param ytid: YouTube ID.
        :param start_seconds: start time of the audio clip.
        :param end_seconds: end time of the audio clip.
        :param positive_labels: labels associated with the audio clip.
        """

        # Create the path for each label that is associated with the file
        if self.copy_and_replicate:
            for label in positive_labels.split(","):
                display_label = self.machine_to_display_mapping[label]
                os.makedirs(os.path.join(self.root_path, display_label), exist_ok=True)
        else:
            display_label = self.machine_to_display_mapping[
                positive_labels.split(",")[0]
            ]
            os.makedirs(os.path.join(self.root_path, display_label), exist_ok=True)

        # Download the file using yt-dlp
        # store in the folder of the first label
        first_display_label = self.machine_to_display_mapping[
            positive_labels.split(",")[0]
        ]
        time_interval = f"*{time.strftime('%H:%M:%S', time.gmtime(start_seconds))}-{time.strftime('%H:%M:%S', time.gmtime(end_seconds))}"
        self.quality = 5
        return_code = os.system(
            f'''
            yt-dlp
            -x
            --audio-format {self.file_format}
            --audio-quality {self.quality}
            --format bestaudio/best
            --postprocessor-args "ExtractAudio:-ar 16000 -ac 2"
            --output "{os.path.join(self.root_path, first_display_label, ytid)}_{start_seconds}-{end_seconds}.%(ext)s"
            --download-sections "{time_interval}"
            https://www.youtube.com/watch?v={ytid}
            {output_options}
            '''.replace("\n", "")
        )

        if return_code != 0:
            return return_code

        export_format = (
            self.export_format[0]
            if isinstance(self.export_format, list)
            else self.export_format
        )

        source_file = f"{os.path.join(self.root_path, first_display_label, ytid)}_{start_seconds}-{end_seconds}.{self.file_format}"
        target_file = f"{os.path.join(self.root_path, first_display_label, ytid)}_{start_seconds}-{end_seconds}.{export_format}"

        if convert_to_numpy:
            audio, sr = torchaudio.load(source_file, normalize=True)
            if self.perform_checks:
                try:
                    audio_info = torchaudio.info(source_file)
                except:
                    logger.warning(f"Error with file {source_file}")
                    os.remove(source_file)
                    return -1

                if (
                    self.min_length_sec
                    and audio_info.num_frames / audio_info.sample_rate
                    < self.min_length_sec
                ):
                    os.remove(source_file)
                    return -1

            # To mono
            audio = torch.mean(audio, dim=0, keepdim=True) if self.mono else audio

            file_name = f"{file_idx}".zfill(self.digits_files)
            target_file = os.path.join(
                self.root_path,
                first_display_label,
                ytid,
                f"{start_seconds}-{end_seconds}",
                f"{file_name}.npy",
            )

            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            np.save(target_file, audio.numpy())

            # Remove the audio file
            if self.file_format not in self.export_format:
                os.remove(source_file)

        elif self.perform_checks:
            try:
                audio_info = torchaudio.info(source_file)
                audio, sr = torchaudio.load(source_file, normalize=True)
                # torchaudio.save(source_file, audio, sr)
            except:
                logger.warning(f"Error with file {source_file}")
                os.remove(source_file)

            if (
                self.min_length_sec
                and audio_info.num_frames / audio_info.sample_rate < self.min_length_sec
            ):
                os.remove(source_file)

        if self.copy_and_replicate:
            # copy the file in the other folders
            for label in positive_labels.split(",")[1:]:
                display_label = self.machine_to_display_mapping[label]
                if isinstance(self.export_format, list):
                    for exp_format in self.export_format:
                        os.system(
                            f'cp "{target_file.replace("npy", exp_format)}" "{target_file.replace(first_display_label, display_label).replace("npy", exp_format)}" {output_options}'
                        )
                else:
                    os.system(
                        f'cp "{target_file}" "{target_file.replace(first_display_label, display_label)}" {output_options}'
                    )

        return return_code
