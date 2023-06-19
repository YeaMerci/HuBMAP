from lightning import HuBMAPTrainer
import segmentation_models_pytorch as smp
from typing import Any
from setup import SetupPipline


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

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_depth=5,
        in_channels=3,
        classes=4,
        activation=None
    )

    num_classes = 2
    experiment = "test"
    main(num_classes, model, experiment)
