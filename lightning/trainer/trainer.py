from ..datamodule import HuBMAPDataModule
from ..lightmodule import HuBMAPLightningModule
from ..augmentations import AugmentPipeline

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
from typing import Any


class HuBMAPTrainer:
    def __init__(self,
                 num_classes: int,
                 experiment: str,
                 model: Any
                 ):
        self.experiment = experiment
        self.num_classes = num_classes
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
            train_size=0.80,
            target_path=os.environ["ANNOT_PATH"],
            data_path=os.environ["DATA_PATH"],
            config_path=os.environ["DATASET_IMAGE_PATH"],
            batch_size=2,
            num_workers=2,
            random_state=42
        )

    def __get_lightmodule(self):
        return HuBMAPLightningModule(
            model=self.model,
            num_classes=self.num_classes,
            lr=5e-2
        )

    @staticmethod
    def __get_transform():
        return AugmentPipeline(
            height=512, width=512,
            spatial=True
        )

    def __get_trainer(self):
        return pl.Trainer(
            fast_dev_run=False,
            accelerator="auto",
            strategy="auto",
            devices="auto",
            num_nodes=1,
            logger=self.logger,
            callbacks=self.callbacks,
            max_epochs=80,
            min_epochs=15
        )

    def __get_loger(self):
        return TensorBoardLogger(save_dir="./logs", name=self.experiment)

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
