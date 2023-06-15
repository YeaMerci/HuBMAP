__all__ = ["HuBMAPDataset", "DebugDataset"]

import torch
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as T
import cv2
import numpy as np
import os
import yaml
from typing import Any
import random
from tqdm import tqdm
import json


class DatasetValidator:
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

        self.__path_checking(
            annotation_path,
            image_path,
            config_path
        )

        self.__split_checking(train_size)
        self.__stage_checking(stage)

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
    def __path_checking(annotation_path: str,
                        image_path: str,
                        config_path: str) -> None:

        assert os.path.isdir(image_path)
        assert os.path.isfile(annotation_path)
        assert os.path.isfile(config_path)

    @staticmethod
    def __stage_checking(stage: str) -> None:
        assert stage in ["train", "val"]


class DatasetBuilder(DatasetValidator):
    def __init__(self,
                 root_path: str,
                 annotation_path: str,
                 image_path: str,
                 config_path: str,
                 train_size: float,
                 stage: str,
                 shuffle: bool,
                 random_state: int):

        super().__init__(
            annotation_path, image_path,
            config_path, train_size, stage,
            shuffle, random_state
        )

        self.__root_dirpath = root_path if root_path else os.getcwd()
        self.__config_dirpath = os.path.join(self.__root_dirpath, "config/dataset")
        self.__runs_dirpath = os.path.join(self.__config_dirpath, "runs")
        self.__image_dirpath = os.path.join(self.__config_dirpath, "images")

        self.__dir_struct = (
            self.__config_dirpath,
            self.__runs_dirpath,
            self.__image_dirpath
        )

        self.__build_struct()
        self._config = self.load_config(config_path)

    def __build_struct(self) -> None:
        if os.path.exists(self.__root_dir):
            for dirpath in self.__dir_struct:
                os.makedirs(dirpath, exist_ok=True)
        else:
            raise ValueError("Could not find root directory!")

    def load_config(self, filename: str) -> dict:
        path = self.__get_path(filename)
        with open(path, mode="r") as f:
            data = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return data

    def __get_path(self, filename: str) -> str:
        if os.path.isfile(filename):
            print("Warning! The configuration file "
                  "is not located in the root directory "
                  "and may not be secure.")
            return filename

        elif len(filename.split("/")) == 2:
            return os.path.join(self.__image_dirpath, filename)

        else:
            raise ValueError(
                f"Filename was expected to be a path"
                f" or file name, but obtained: {filename}"
            )

    def get_config_images(self) -> list[str, ...]:
        return os.listdir(self.__image_dirpath)

    def get_config_runs(self) -> list[str, ...]:
        return os.listdir(self.__runs_dirpath)

    def write_config(self, data: dict[dict, ...], filename: str) -> None:
        path = self.__get_path(filename)
        with open(path, mode="w") as f:
            yaml.safe_dump(stream=f, data=data)


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
            "apply_mask": True,
            "label": 0,
            "rgb": (0, 0, 0),
            "loss_weight": None
        }

        self.head_keys: tuple = (
            "background", "blood_vessel",
            "glomerulus", "unsure", "border"
        )

        self.head_values: dict = {
            "apply_mask": True,
            "label": 0,
            "rgb": (0, 0, 0),
            "loss_weight": None
        }

    def generate_pattern(self):
        self.main_struct["head"] = {}.fromkeys(
            self.head_keys, self.head_values
        )
        self.main_struct["body"] = {}.fromkeys(
            self.body_keys, self.body_values
        )


class HuBMAPDataset(
    DatasetBuilder,
    Dataset
):
    def __init__(self,
                 stage: str,
                 annotation_path: str,
                 image_path: str,
                 config_path: str,
                 root_path: str = None,
                 transforms: Any = None,
                 train_size: float = 0.85,
                 shuffle: bool = True,
                 random_state: int = None
                 ):

        super().__init__(
            root_path, annotation_path,
            image_path, config_path,
            train_size, stage,
            shuffle, random_state
        )

        self.__image_path = image_path
        self.__samples = self.__parse_jsonl(annotation_path)
        self.transforms = transforms
        self.total_length = None

        self.__identifiers = self.__split_identifiers(
            stage, train_size,
            shuffle, random_state
        )

    def __len__(self) -> int:
        return len(self.__identifiers)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            config = self._config[vessel_type]
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

    def __split_identifiers(self, stage, train_size, shuffle, random_state) -> np.ndarray:
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


class PostProcessing:
    def __init__(self,):
        self.__colormap = {
            0: [0, 0, 0],
            1: [255, 8, 8],
            2: [8, 12, 255]
        }

    def __decode_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        elif mask.ndim != 3:
            raise ValueError(
                f"Input matrix must have a shape of 1xHxW, "
                f"but expected shape {mask.shape}"
            )

        mask = np.apply_along_axis(
            func1d=lambda index: self.__colormap[int(index)],
            axis=0,
            arr=mask,
        )
        return mask.transpose(1, 2, 0)

    @staticmethod
    def __get_alpha_channel(mask: np.ndarray, alpha: float = 1) -> np.ndarray:
        mask = np.where(mask != 0, round(255*alpha), 0)
        return np.expand_dims(mask, axis=0)

    @staticmethod
    def __apply_alpha_channel(mask: np.ndarray, alpha_channel: np.ndarray) -> np.ndarray:
        mask = mask.transpose(2, 0, 1)
        red_channel, green_channel, blue_channel = np.array_split(mask, 3, axis=0)
        return np.concatenate(
            [red_channel,
             green_channel,
             blue_channel,
             alpha_channel], axis=0
        )

    def __call__(self, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        decoded = self.__decode_mask(mask)
        alpha_channel = self.__get_alpha_channel(mask, alpha)
        image = self.__apply_alpha_channel(decoded, alpha_channel)
        return image.transpose(1, 2, 0)


class DebugDataset(HuBMAPDataset):
    def __init__(self, spins: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert spins > self.__len__() or spins < self.__len__()
        self.__spins = spins

    @staticmethod
    def show(image,
             mask,
             original: bool = True,
             alpha: float = 0.5,
             timeout: int = 2
             ) -> None:

        description = "original" if original else "with mask"
        winname = f"Image {description}"

        print(
            f"{Fore.CYAN}|| Image shape: {Fore.BLUE}"
            f"{np.asarray(image).shape}{Fore.CYAN} ||"
        )

        print(
            f"|| Mask shape: {Fore.BLUE}"
            f"{np.asarray(mask).shape}{Fore.CYAN} ||"
        )

        if not original:
            image = cv2.addWeighted(image, 1-alpha, mask, alpha, 0)
        cv2.imshow(winname, image)
        cv2.waitKey(timeout)

    def roll_transformations(self,
                             start_roll: int,
                             end_roll: int,
                             original: bool,
                             timeout: int = 0,
                             alpha: float = 0.5
                             ) -> None:

        for i in range(self.__len__()):
            image, mask = self.__getitem__(i)
            self.show(image, mask, original, alpha, timeout)
        cv2.destroyAllWindows()

        print(
            f"\n{Fore.CYAN}|| {Fore.BLUE}Run-in complete!\n"
            f"{Fore.CYAN}|| Rolled out images: {Fore.BLUE}{end_roll-start_roll}\n"
            f"{Fore.CYAN}|| Scroll mode: {Fore.BLUE}{'original' if original else 'with mask'}\n"
            f"{Fore.CYAN}|| Alpha channel: {Fore.BLUE}{alpha}{Fore.WHITE}"
        )
