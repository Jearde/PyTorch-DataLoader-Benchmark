import torch


def get_settings(
    batch_size: int = 32,
    num_workers: int = -1,
    prefetch_factor: int = 2,
    target_sr: int = 16000,
    target_audio_length: int = 10,
    audio_slice_length: int | None = None,
    mono: bool = True,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
):
    num_workers = (
        num_workers
        if num_workers is not None and num_workers != -1
        else torch.multiprocessing.cpu_count() - 1
    )
    prefetch_factor = prefetch_factor if num_workers > 0 else None
    persistent_workers = True if num_workers > 0 else False

    data_loader_settings_dali = {
        "batch_size": batch_size,
        "num_threads": num_workers,
        "prefetch_factor": prefetch_factor,
        "shuffle": False,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
        "target_sr": target_sr,
        "target_length": target_audio_length,
        "mono": mono,
        "random_crop_size": audio_slice_length,
    }

    data_loader_settings_pytorch = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "shuffle": False,
        "pin_memory": True,
        "persistent_workers": persistent_workers,
    }

    dataset_settings = {
        "target_sr": target_sr,
        "target_length": target_audio_length,
        "mono": mono,
        "random_slice": False,
        "audio_slice_length": audio_slice_length,
        "audio_slice_overlap": None,
        "check_silence": False,
        "window_size": None,
        "overlap": 0.5,
    }

    return (
        data_loader_settings_dali,
        data_loader_settings_pytorch,
        dataset_settings,
    )
