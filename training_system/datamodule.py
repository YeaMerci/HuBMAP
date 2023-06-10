import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
import torchvision.transforms as T
import albumentations.pytorch as pytorch
from typing import Union


class HuBMAPDataModule(pl.LightningDataModule):
    def __init__(self, target_path: str,
                 data_path: str,
                 transform: Union[T.Compose, A.Compose],
                 train_size: float = 0.67,
                 val_size: float = 0.165,
                 test_size: float = 0.165,
                 batch_size: int = 4,
                 num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_path = target_path
        self.data_path = data_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_predict = None
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

    def setup(self, stage: str = None) -> None:
        self.data_train = PhotopicVisionDataset(
            target_path=self.target_path,
            data_path=self.data_path,
            transforms=self.train_transform,
            stage="train",
            train_size=self.train_size,
            test_size=self.test_size,
            val_size=self.val_size,
            shuffle=True,
            random_state=42
            )

        self.data_val = PhotopicVisionDataset(
            target_path=self.target_path,
            data_path=self.data_path,
            transforms=self.eval_transform,
            stage="val",
            train_size=self.train_size,
            test_size=self.test_size,
            val_size=self.val_size,
            shuffle=True,
            random_state=42
            )

        self.data_test = PhotopicVisionDataset(
            target_path=self.target_path,
            data_path=self.data_path,
            transforms=self.eval_transform,
            stage="test",
            train_size=self.train_size,
            test_size=self.test_size,
            val_size=self.val_size,
            shuffle=True,
            random_state=42
            )

        self.data_predict = PhotopicVisionDataset(
            target_path=self.target_path,
            data_path=self.data_path,
            transforms=self.eval_transform,
            stage="test",
            train_size=self.train_size,
            test_size=self.test_size,
            val_size=self.val_size,
            shuffle=True,
            random_state=42
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
