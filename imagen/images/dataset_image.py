import os
import yaml
from ..imagewriter import ImageWriter


class DatasetImage(ImageWriter):
    def __init__(self):
        self._filename = "sample_dataset_image.yaml"

        self.main_struct: dict = {
            "head": {},
            "body": {}
        }

        self.body_keys: tuple = (
            "background", "blood_vessel",
            "glomerulus", "unsure", "border"
        )
        self.body_values: dict = {
            "apply": True,
            "label": 0,
            "rgb": (0, 0, 0),
            "loss_weight": None
        }

        self.head = {
            "instance": False,
            "multiborder": False,
            "thickness": 1
        }

    def generate_pattern(self) -> dict:
        stem = self.main_struct.copy()
        stem["head"] = self.head
        stem["body"] = {}.fromkeys(
            self.body_keys, self.body_values
        )
        return stem
