from ..datamodule import HuBMAPDataModule
from ..lightmodule import HuBMAPLightningModule


class HuBMAPTrainer:
    def __init__(self, datamodule_config: str, ):
        self.datamodule =
        self.lightmodule = HuBMAPLightningModule()

    def get_datamodule(self):
        return HuBMAPDataModule()

    def get_lightmodule(self):
        return HuBMAPLightningModule()

    def get_config(self):
        return self.read_config()

    def read_config(self):
        return 1
