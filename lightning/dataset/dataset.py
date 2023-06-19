from ..lightbuilder import LightBuilder
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
    @classmethod
    def validate(cls,
                 stage,
                 annotation_path,
                 image_path,
                 template_path,
                 train_size,
                 shuffle,
                 random_state
                 ):

        cls.__type_checking(
            annotation_path,
            image_path,
            train_size,
            stage,
            shuffle,
            random_state
        )

        cls.__path_checking(
            annotation_path,
            image_path,
            template_path
        )

        cls.__split_checking(train_size)
        cls.__stage_checking(stage)

    @staticmethod
    def __type_checking(annotation_path: str,
                        image_path: str,
                        train_size: float,
                        stage: str,
                        shuffle: bool,
                        random_state: int
                        ) -> None:

        assert isinstance(image_path, str)
        assert isinstance(annotation_path, str)
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
                        template_path: str
                        ) -> None:

        assert os.path.isdir(image_path)
        assert os.path.isfile(annotation_path)
        assert os.path.isfile(template_path)

    @staticmethod
    def __stage_checking(stage: str) -> None:
        assert stage in ["train", "val"]


class HuBMAPDataset(
    DatasetValidator,
    LightBuilder,
    Dataset
):
    def __init__(self,
                 stage: str,
                 annotation_path: str,
                 image_path: str,
                 template_path: str,
                 transforms: Any = None,
                 train_size: float = 0.85,
                 shuffle: bool = True,
                 random_state: int = None,
                 *args, **kwargs
                 ):

        super().validate(
            stage, annotation_path,
            image_path, template_path,
            train_size, shuffle, random_state
        )

        super().__init__(template_path)

        self._head_config = self._config["head"]
        self._classes_config = self._config["body"]

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
        idx, identifier = self._get_sample(idx)
        image = self._get_image(identifier)
        target = self._get_target(idx)
        image, target = self.__gate_transforms(image, target)
        return image.transpose(2, 0, 1), target

    def __gate_transforms(self, image, target):
        if type(target) == list:
            return self.__instance_transforms(image, target)
        elif type(target) == np.ndarray:
            return self.__semantic_transforms(image, target)
        else:
            raise TypeError("Unsupported type!")

    def __instance_transforms(self, image, masks):
        image_copies = [image]*len(masks)
        for index, (image, mask) in enumerate(zip(image_copies, masks)):
            image, mask = self.__semantic_transforms(image, mask)
            masks[index] = mask
        return image, masks

    def _semantic_transforms(self, image, mask):
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        return image, mask

    def _get_sample(self, idx: int) -> tuple[int, str]:
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

    def _get_image(self, identifier: str) -> np.ndarray:
        image_path = self.__get_image_path(identifier)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        return image

    def _get_target(self, idx: int) -> np.ndarray:
        annotations = self.__samples[idx]["annotations"]
        mask = np.zeros((512, 512), dtype=np.uint8)
        instance = self._head_config["instance"]
        masks = [] if instance else None

        for vessel in annotations:
            vessel_type = vessel["type"]
            class_config = self._classes_config[vessel_type]
            is_apply = class_config["apply"]

            if is_apply:
                label = class_config["label"]
                coordinates = np.array(vessel["coordinates"])
                mask = self.__get_mask(mask, label, coordinates)

            if instance:
                masks.append(mask)
                mask = np.zeros((512, 512), dtype=np.uint8)
        return masks if masks else mask

    def __get_mask(self,
                   mask: np.ndarray,
                   label: int,
                   coordinates: np.ndarray
                   ) -> np.ndarray:

        multiborder = self._head_config["multiborder"]
        border_config = self._classes_config["border"]
        is_apply = border_config["apply"]

        mask = cv2.fillPoly(
            mask, pts=coordinates,
            color=(label, label, label)
        )

        if is_apply:
            label = label+1 if multiborder else border_config["label"]
            thickness = self._head_config["thickness"]
            mask = cv2.polylines(
                mask, coordinates,
                isClosed=True, color=label,
                thickness=thickness
            )
        return mask

    def __get_identifiers(self) -> np.ndarray:
        return np.array(
            [identifier["id"]
             for identifier in self.__samples]
        )

    def __split_identifiers(self,
                            stage: str,
                            train_size: float,
                            shuffle: bool,
                            random_state: int
                            ) -> np.ndarray:

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
