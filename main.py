from setup import SetupPipline
from lightning import HuBMAPTrainer
import segmentation_models_pytorch as smp
from typing import Any
import torch


def main(num_classes: int,
         model: Any,
         experiment: str
         ):

    trainer = HuBMAPTrainer(
        num_classes=num_classes,
        model=model,
        experiment=experiment
    )

    trainer.fit()


if __name__ == '__main__':
    SetupPipline().set_variable()
    num_classes = 3
    torch.set_float32_matmul_precision("medium")

    model = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_depth=5,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

    experiment = "UnetPlusPlus"
    main(num_classes, model, experiment)
