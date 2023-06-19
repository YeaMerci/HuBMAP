from ..dataset import HuBMAPDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
import torchvision.transforms as T
import albumentations.pytorch as pytorch
from typing import Union, Any


class HuBMAPDataModule(pl.LightningDataModule):
    def __init__(self,
                 target_path: str,
                 data_path: str,
                 config_path: str,
                 transform: Union[T.Compose, A.Compose, Any],
                 train_size: float = 0.85,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 random_state: int = None
                 ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_path = target_path
        self.data_path = data_path
        self.config_path = config_path
        self.train_size = train_size
        self.random_state = random_state

        self.data_train = None
        self.data_val = None

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = transform
        self.eval_transform = A.Compose([
            A.Resize(height=transform.height,
                     width=transform.width),
            A.Normalize(mean=self.mean,
                        std=self.std),
            pytorch.ToTensorV2()
        ])

        self.__data_source = "https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data"

    def prepare_data(self):
        print(
            f"Warning! "
            f"This function cannot load and process data. "
            f"You have to download the data yourself. "
            f"Data source: {self.__data_source}"
        )

    def setup(self, stage: str = None) -> None:
        self.data_train = HuBMAPDataset(
            annotation_path=self.target_path,
            image_path=self.data_path,
            template_path=self.config_path,
            transforms=self.train_transform,
            stage="train",
            train_size=self.train_size,
            shuffle=True,
            random_state=self.random_state
            )

        self.data_val = HuBMAPDataset(
            annotation_path=self.target_path,
            image_path=self.data_path,
            template_path=self.config_path,
            transforms=self.eval_transform,
            stage="val",
            train_size=self.train_size,
            shuffle=True,
            random_state=self.random_state
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
