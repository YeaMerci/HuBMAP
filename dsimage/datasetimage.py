import os
import yaml


class DatasetImage:
    def __init__(self):
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


class ImageWriter(DatasetImage):
    def __init__(self,
                 dirpath: str = None,
                 filename: str = "sample_image.yaml"
                 ):

        super().__init__()
        self.__filename = filename
        self.__dirpath = dirpath if dirpath else os.getcwd()
        self.__path = os.path.join(
            self.__dirpath, self.__filename
        )

    def write_config(self, image: dict[dict, ...]) -> None:
        with open(self.__path, mode="w") as f:
            yaml.safe_dump(stream=f, data=image)

    def generate_sample(self):
        sample_image = self.generate_pattern()
        self.write_config(sample_image)
