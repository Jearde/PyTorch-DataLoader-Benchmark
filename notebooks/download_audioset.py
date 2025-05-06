# %%
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from src.data.audioset.audioset_downloader import download_audioset

# %%
debug = True

if debug:
    verbose_output = ""
    n_jobs = 1
else:
    verbose_output = "> /dev/null 2>&1"
    n_jobs = int(os.cpu_count() / 1.5)

# Download Numpy and WAV files
root_dir = "/mnt/nfs/datasets/audioset"
root_dir = "/mnt/data/datasets/audioset"
file_format = ["wav", "npy"]

# %% [markdown]
# ## Add credentials for YouTube Downloader
# In `yt-dlp.conf` add your username and password. It will be stored in a cookie for later use
# ```bash
# --username <user>@gmail.com
# --password <password>
# ```

# %%
download_audioset(
    root_dir=root_dir,
    download_type="eval",
    n_jobs=n_jobs,
    copy_and_replicate=True,
    verbose_output=verbose_output,
    file_format=file_format,
)

# %%
download_audioset(
    root_dir=root_dir,
    download_type="balanced_train",
    n_jobs=n_jobs,
    copy_and_replicate=False,
    file_format=file_format,
    verbose_output=verbose_output,
)

# %%
download_audioset(
    root_dir=root_dir,
    download_type="unbalanced_train",
    n_jobs=n_jobs,
    copy_and_replicate=False,
    file_format=file_format,
    verbose_output=verbose_output,
)
