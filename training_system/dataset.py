from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as T
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import yaml
from typing import Any
import random


class AttributeValidator:
    def __init__(self,
                 annotation_path: str,
                 image_path: str,
                 config_path: str,
                 train_size: float,
                 stage: str,
                 shuffle: bool,
                 random_state: int):

        self.__type_checking(
            annotation_path,
            image_path,
            config_path,
            train_size,
            stage,
            shuffle,
            random_state
        )

        self.__split_checking(
            train_size
        )

        self.__path_checking(
            annotation_path,
            image_path,
            config_path
        )

        self.__stage_checking(
            stage
        )

    @staticmethod
    def __type_checking(annotation_path: str,
                        image_path: str,
                        config_path: str,
                        train_size: float,
                        stage: str,
                        shuffle: bool,
                        random_state: int) -> None:

        assert isinstance(image_path, str)
        assert isinstance(annotation_path, str)
        assert isinstance(config_path, str)
        assert isinstance(train_size, float)
        assert isinstance(stage, str)
        assert isinstance(shuffle, bool)
        assert isinstance(random_state, (int, type(None)))

    @staticmethod
    def __split_checking(train_size: float) -> None:
        assert train_size <= 1 or train_size >= 0

    @staticmethod
    def __path_checking(
            annotation_path: str,
            image_path: str,
            config_path: str) -> None:

        assert os.path.isdir(image_path)
        assert os.path.isfile(annotation_path)
        assert os.path.isfile(config_path)

    @staticmethod
    def __stage_checking(stage: str) -> None:
        assert stage in ["train", "val"]


class DatasetConfiguration:
    def __init__(self):
        self.__root_dir = os.path.join(os.getcwd(), "config/dataset")
        self.__find_root()

        self.__keys = (
            "background", "blood_vessel",
            "glomerulus", "unsure"
        )

        self.__values = {
            "apply_mask": True,
            "label": 0,
            "rgb": (0, 0, 0),
            "loss_weight": None
        }
        self.sample_config = {}.fromkeys(self.__keys, self.__values)

    def __find_root(self):
        if not os.path.exists(self.__root_dir):
            os.makedirs(self.__root_dir, exist_ok=True)

    def load_config(self, filename: str) -> dict:
        path = self.__get_path(filename)
        if os.path.isfile(filename):
            path = filename
        with open(path, mode="r") as f:
            data = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return data

    def __get_path(self, filename: str) -> str:
        path = os.path.join(self.__root_dir, filename)
        return path

    def get_configs(self) -> list:
        return os.listdir(self.__root_dir)

    def write_config(self, data: dict[dict, ...], filename: str) -> None:
        path = self.__get_path(filename)
        with open(path, mode="w") as f:
            yaml.safe_dump(stream=f, data=data)


class HuBMAPDataset(Dataset):
    def __init__(self,
                 stage: str,
                 annotation_path: str,
                 image_path: str,
                 config_path: str = None,
                 transforms: Any = None,
                 train_size: float = 0.85,
                 shuffle: bool = True,
                 random_state: int = None
                 ):

        AttributeValidator(
            annotation_path,
            image_path,
            config_path,
            train_size,
            stage,
            shuffle,
            random_state
        )

        self.configurator = DatasetConfiguration()
        self.__image_path = image_path
        self.__samples = self.__parse_jsonl(annotation_path)
        self.__config = self.configurator.load_config(config_path)
        self.transforms = transforms
        self.total_length = None

        self.__identifiers = self.__split_identifiers(
            stage,
            train_size,
            shuffle,
            random_state
        )

    def __len__(self) -> int:
        return len(self.__identifiers)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx, identifier = self.__get_sample(idx)
        image = self.__get_image(identifier)
        mask = self.__get_mask(idx)
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask

    def __get_sample(self, idx: int) -> tuple[int, str]:
        idx = self.__identifiers[idx]
        identifier = self.__samples[idx]["id"]
        return idx, identifier

    @staticmethod
    def __parse_jsonl(path: str) -> list[dict, ...]:
        with open(path, 'r') as json_file:
            jsonl_labels = [
                json.loads(line)
                for line in tqdm(json_file)
            ]
        return jsonl_labels

    def __get_image_path(self, identifier: str) -> str:
        path = os.path.join(
            self.__image_path, f"{identifier}.tif"
        )
        return path

    def __get_image(self, identifier: str) -> np.ndarray:
        image_path = self.__get_image_path(identifier)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        return image.transpose(2, 0, 1)

    def __get_mask(self, idx: int) -> np.ndarray:
        mask = np.zeros((512, 512), dtype=np.uint8)
        annotations = self.__samples[idx]["annotations"]

        for vessel in annotations:
            vessel_type = vessel["type"]
            config = self.__config[vessel_type]
            apply_mask = config["apply_mask"]
            label = config["label"]

            if apply_mask:
                coordinates = np.array(vessel["coordinates"])
                mask = cv2.fillPoly(
                    mask, pts=coordinates,
                    color=(label, label, label)
                )
        return mask

    def __get_identifiers(self) -> np.ndarray:
        return np.array(
            [identifier["id"]
             for identifier in self.__samples]
        )

    def __split_identifiers(self, stage, train_size, shuffle, random_state) -> list:
        identifiers = self.__get_identifiers()
        self.total_length = len(identifiers)
        indices = np.arange(self.total_length)

        if random_state:
            np.random.seed(random_state)
        if shuffle:
            np.random.shuffle(indices)

        train_length = round(self.total_length * train_size)
        val_length = self.total_length - train_length

        assert self.total_length == train_length + val_length

        val_indices = indices[0:val_length]
        train_indices = indices[val_length:train_length]

        stage_indices = {
            "train": train_indices,
            "val": val_indices
        }
        return stage_indices[stage]
