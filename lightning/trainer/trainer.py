from ..datamodule import HuBMAPDataModule
from ..lightmodule import HuBMAPLightningModule
from ..augmentations import AugmentPipeline
from ..lightbuilder import LightBuilder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch.nn as nn
from typing import Any
import os


class HuBMAPTrainer(LightBuilder):
    def __init__(self,
                 job_type: str,
                 project: str,
                 tags: str,
                 num_classes: int,
                 experiment: str,
                 model: nn.Module,
                 datamodule_config_path: dict,
                 augmentations_config_path: dict,
                 trainer_config_path: dict,
                 seed: int = 42,
                 precision: str = "medium"
                 ):
        self.seed_all(seed)
        self.set_precision(precision)

        super().__init__(
            datamodule_config_path,
            augmentations_config_path,
            trainer_config_path,
            job_type, project, tags
        )

        self.experiment = experiment
        self.model = model

        self.callbacks = self.__get_callbacks()
        self.logger = self.__get_loger()

        self.transform = self.__get_transform()
        self.datamodule = self.__get_datamodule()
        self.lightmodule = self.__get_lightmodule()
        self.trainer = self.__get_trainer()

    def __get_datamodule(self):
        return HuBMAPDataModule(
            transform=self.transform,
            **self.configs.datamodule
        )

    def __get_lightmodule(self):
        return HuBMAPLightningModule(
            model=self.model,
            num_classes=self.num_classes,
        )

    @staticmethod
    def __get_transform():
        return AugmentPipeline(
            **self.configs.augmodule
        )

    def __get_trainer(self):
        return pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            **self.configs.trainer
        )

    def __get_loger(self):
        return TensorBoardLogger(
            save_dir="./logs", name=self.experiment
        )

    def __get_callbacks(self):
        return [
            ModelCheckpoint(
                dirpath=f"models/{self.experiment}",
                filename="{epoch}_{val_loss:.2f}_{val_accuracy:.2f}",
                save_top_k=5,
                monitor="val_loss",
                mode="min"
            ),

            EarlyStopping(
                monitor="val_loss",
                min_delta=2e-2,
                patience=5,
                verbose=False,
                mode="min"
            ),

            LearningRateMonitor(
                logging_interval="step"
            )
        ]

    def fit(self):
        self.trainer.fit(
            model=self.lightmodule,
            datamodule=self.datamodule
        )

    @staticmethod
    def set_precision(precision: str = "medium") -> None:
        torch.set_float32_matmul_precision(precision)

    @staticmethod
    def seed_all(seed: int) -> None:
        pl.seed_everything(seed)
