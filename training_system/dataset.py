import random

import albumentations as A
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import os
from typing import Any


class HuBMAPDataset(Dataset):
    def __init__(self,
                 stage: str,
                 target_path: str,
                 data_path: str,
                 transforms: Any,
                 train_size: float = 0.80,
                 shuffle: bool = True):

        self.target_path = target_path
        self.data_path = data_path
        self.transforms = transforms
        self.test_size = test_size
        self.train_size = train_size
        self.val_size = val_size
        self.stage = stage
        self.shuffle = shuffle
        self.random_state = random_state
        self.total_len = None
        self._X, self._Y = self.__create_dataset()

    def __create_dataset(self) -> dict:
        dict_paths = {
            "image": [],
            "mask": []
        }

        for dir_name, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                name = filename.split('.')[0]
                dict_paths["image"].append(f"{self.data_path}/{name}.jpg")
                dict_paths["mask"].append(f"{self.target_path}/{name}.png")

        dataframe = pd.DataFrame(
            data=dict_paths,
            index=np.arange(0, len(dict_paths["image"]))
        )

        self.total_len = len(dataframe)
        x_data = dataframe["image"].values
        y_data = dataframe["mask"].values
        data_dict = self.__split_data(x_data=x_data, y_data=y_data)
        return data_dict[self.stage]

    def __split_data(self, x_data: np.array,
                     y_data: np.array) -> dict:

        total_eval_size = self.test_size + self.val_size
        test_size = 1 / (total_eval_size / self.test_size)
        val_size = 1 / (total_eval_size / self.val_size)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            train_size=self.train_size,
                                                            test_size=total_eval_size,
                                                            random_state=self.random_state,
                                                            shuffle=self.shuffle)

        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                        train_size=val_size,
                                                        test_size=test_size,
                                                        random_state=self.random_state,
                                                        shuffle=self.shuffle)
        return {"train": (x_train, y_train),
                "val": (x_val, y_val)}

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx) -> tuple:
        data_path, target_path = self._X[idx], self._Y[idx]
        image = cv2.imread(data_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask


class AttributeValidator:
    def __init__(self,
                 annotation_path: str,
                 image_path: str,
                 config_path: str,
                 train_size: float,
                 stage: str,
                 shuffle: bool):

        self.__type_checking(
            annotation_path,
            image_path,
            config_path,
            train_size,
            stage,
            shuffle
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
                        shuffle: bool) -> None:

        assert isinstance(image_path, str)
        assert isinstance(annotation_path, str)
        assert isinstance(config_path, str)
        assert isinstance(train_size, float)
        assert isinstance(stage, str)
        assert isinstance(shuffle, bool)

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

    def write_config(self, data: dict[dict, ...], filename: str) -> None:
        path = self.__get_path(filename)
        with open(path, mode="w") as f:
            yaml.safe_dump(stream=f, data=data)


class PolygonsAnnotation:
    def __init__(self,
                 stage: str,
                 annotation_path: str,
                 image_path: str,
                 config_path: str = None,
                 transforms: Any = None,
                 train_size: float = 0.85,
                 shuffle: bool = True,
                 ):

        AttributeValidator(
            annotation_path,
            image_path,
            config_path,
            train_size,
            stage,
            shuffle
        )

        self.configurator = DatasetConfiguration()
        self.__image_path = image_path
        self.__samples = self.__parse_jsonl(annotation_path)
        self.__config = self.configurator.load_config(config_path)
        self.transforms = transforms
        self.total_length = None
        self.__identifiers = self.__split_identifiers(stage, train_size, shuffle)

    def __len__(self) -> int:
        return len(self.__identifiers)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = self.__get_image(idx)
        mask = self.__get_mask(idx)
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask

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

    def __get_image(self, idx: int) -> np.ndarray:
        identifier = self.__samples[idx]["id"]
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

    def __get_identifier(self):
        return [identifier for identifier in self.__samples[idx]["id"]]

    def __split_identifiers(self, stage, train_size, shuffle):
        identifiers = self.__get_identifier()
        self.total_length = len(identifiers)
        if shuffle:
            random.shuffle(identifiers)

        train_length = (self.total_length // 100) * train_size
        val_length = self.total_length - train_length
        train_sample = identifiers[0:val_length]
        val_sample = identifiers[val_length:train_length]

        data_dict = {
            "train": train_sample,
            "val": val_sample
        }
        return data_dict[stage]
