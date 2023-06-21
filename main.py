from setup import SetupPipline
from lightning import HuBMAPTrainer

import segmentation_models_pytorch as smp
from typing import Any
import torch
import os


def main():

    model = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_depth=5,
        in_channels=3,
        classes=3,
        activation=None
    )

    trainer = HuBMAPTrainer(
        job_type="train",
        project="HubMAP",
        tags=["fast_dev_run", "test_run"],
        num_classes=3,
        experiment="experiment",
        model=model,
        datamodule_config_path=os.environ["DATAMODULE_CONFIG"],
        augmentations_config_path=os.environ["AUGMODULE_CONFIG"],
        trainer_config_path=os.environ["TRAINER_CONFIG"]
    )
    trainer.fit()


if __name__ == '__main__':
    SetupPipline().set_variable()
    main()
