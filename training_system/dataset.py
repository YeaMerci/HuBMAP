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
                 test_size: float = 0.05,
                 train_size: float = 0.80,
                 val_size: float = 0.15,
                 shuffle: bool = True,
                 random_state: int = 42):

        self.__attribute_checking(target_path, data_path, test_size,
                                  train_size, val_size,
                                  stage, shuffle, random_state)

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

    @staticmethod
    def __type_checking(target_path: str, data_path: str,
                        stage: str, shuffle: bool,
                        test_size: float, train_size: float,
                        val_size: float, random_state: int) -> None:

        assert isinstance(target_path, str)
        assert isinstance(data_path, str)
        assert isinstance(test_size, float)
        assert isinstance(train_size, float)
        assert isinstance(val_size, float)
        assert isinstance(stage, str)
        assert isinstance(shuffle, bool)
        assert isinstance(random_state, int)

    @staticmethod
    def __split_checking(train_size: float, test_size: float, val_size: float) -> None:
        total_size = train_size + test_size + val_size
        assert total_size == 1

    @staticmethod
    def __path_checking(target_path: str, data_path: str) -> None:
        assert os.path.isdir(data_path)
        assert os.path.isdir(target_path)

    @staticmethod
    def __stage_checking(stage: str) -> None:
        assert stage in ["train", "test", "val"]

    @classmethod
    def __attribute_checking(cls, target_path: str,
                             data_path: str,
                             test_size: float,
                             train_size: float,
                             val_size: float,
                             stage: str,
                             shuffle: bool,
                             random_state: int) -> None:

        cls.__type_checking(target_path=target_path,
                            data_path=data_path,
                            train_size=test_size,
                            test_size=test_size,
                            val_size=val_size,
                            stage=stage,
                            shuffle=shuffle,
                            random_state=random_state)

        cls.__split_checking(train_size=train_size,
                             test_size=test_size,
                             val_size=val_size)

        cls.__path_checking(target_path=target_path,
                            data_path=data_path)

        cls.__stage_checking(stage=stage)

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
                "test": (x_test, y_test),
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


class PolygonsAnnotation:
    def __init__(self,
                 stage: str,
                 annotation_path: str,
                 image_path: str,
                 config_path: str,
                 transforms: Any = None,
                 train_size: float = 0.85,
                 val_size: float = 0.15,
                 shuffle: bool = True,
                 ):

        self.__attribute_checking(
            annotation_path, image_path, config_path,
            train_size, val_size, stage, shuffle
        )

        self.__image_path = image_path
        self.__samples = self.__parse_jsonl(annotation_path)
        self.__config = pd.read_csv(config_path)
        self.transforms = transforms
        self.train_size = train_size
        self.val_size = val_size
        self.stage = stage
        self.shuffle = shuffle
        self.total_len = None
        self._X, self._Y = self.__create_dataset()

    def __len__(self) -> int:
        return len(self.__samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = self.__get_image(idx)
        mask = self.__get_mask(idx)
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask

    @staticmethod
    def __type_checking(annotation_path: str,
                        image_path: str,
                        config_path: str,
                        train_size: float,
                        val_size: float,
                        stage: str,
                        shuffle: bool
                        ) -> None:

        assert isinstance(image_path, str)
        assert isinstance(annotation_path, str)
        assert isinstance(config_path, str)
        assert isinstance(train_size, float)
        assert isinstance(val_size, float)
        assert isinstance(stage, str)
        assert isinstance(shuffle, bool)

    @staticmethod
    def __split_checking(train_size: float, val_size: float) -> None:
        total_size = train_size + val_size
        assert total_size == 1

    @staticmethod
    def __path_checking(annotation_path: str,
                        image_path: str,
                        config_path: str) -> None:
        assert os.path.isdir(image_path)
        assert os.path.isfile(annotation_path)
        assert os.path.isfile(config_path)

    @staticmethod
    def __stage_checking(stage: str) -> None:
        assert stage in ["train", "val"]

    @classmethod
    def __attribute_checking(cls,
                             annotation_path: str,
                             image_path: str,
                             config_path: str,
                             train_size: float,
                             val_size: float,
                             stage: str,
                             shuffle: bool
                             ) -> None:

        cls.__type_checking(
            annotation_path, image_path,
            config_path, train_size,
            val_size, stage, shuffle
        )

        cls.__split_checking(
            train_size, val_size
        )

        cls.__path_checking(
            annotation_path,
            image_path,
            config_path
        )

        cls.__stage_checking(stage)

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

    @staticmethod
    def load_config(path: str) -> dict:
        with open(path, mode="r") as f:
            data = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return data

    @staticmethod
    def write_config(data: dict[dict, ...], path: str) -> None:
        with open(path, mode="w") as f:
            yaml.safe_dump(stream=f, data=data)

    def __get_mask(self, idx: int) -> np.ndarray:
        mask = np.zeros((512, 512), dtype=np.uint8)
        annotations = self.__samples[idx]["annotations"]

        for vessel in annotations:
            vessel_type = vessel["type"]
            apply_mask, label, rgb, loss_weight = self.__config[vessel_type]

            if apply_mask:
                coordinates = np.array(vessel["coordinates"])
                mask = cv2.fillPoly(
                    mask, pts=coordinates,
                    color=(label, label, label)
                )
        return mask
