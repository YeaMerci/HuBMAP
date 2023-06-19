import os
import yaml
from ..imagewriter import ImageWriter


class DataModuleImage(ImageWriter):
    def __init__(self):
        self._filename = "sample_datamodule_image.yaml"

        self.main_struct = {
            "train_size": 0.85,
            "batch_size": 4,
            "num_workers": 4,
            "random_state": None,
            "target_path": "some/path/to/annotations",
            "data_path": "some/path/to/images",
            "config_path": "some/path/to/config.yaml"
        }

    def generate_pattern(self) -> dict:
        return self.main_struct.copy()
