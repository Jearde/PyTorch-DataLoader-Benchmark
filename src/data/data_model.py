import lightning as L


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        files: list[str],
        labels: list[list[int]],
        data_loader_class: type,
        data_loader_settings: dict,
    ):
        super().__init__()
        self.files = files
        self.labels = labels
        self.data_loader_class = data_loader_class
        self.data_loader_settings = data_loader_settings

    def setup(self, stage: str):
        if stage == "fit":
            assert NotImplementedError("Not implemented yet")
        elif stage == "test":
            assert NotImplementedError("Not implemented yet")
        elif stage == "predict":
            assert NotImplementedError("Not implemented yet")
        else:
            assert ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return self.data_loader_class(
            files=self.files, labels=self.labels, **self.data_loader_settings
        )

    def val_dataloader(self):
        return self.data_loader_class(
            files=self.files, labels=self.labels, **self.data_loader_settings
        )

    def test_dataloader(self):
        return self.data_loader_class(
            files=self.files, labels=self.labels, **self.data_loader_settings
        )

    def predict_dataloader(self):
        return self.data_loader_class(
            files=self.files, labels=self.labels, **self.data_loader_settings
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
