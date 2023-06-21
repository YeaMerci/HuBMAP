from dataclasses import dataclass
import os
import torch
import numpy as np
import random


@dataclass
class ConfigPath:
    # Project Paths
    root_dirpath: str = "/home/merci/PycharmProjects/competitions/HuBMAP/"
    annot_path: str = os.path.join(root_dirpath, "data/hubmap-hacking-the-human-vasculature/polygons.jsonl")
    image_dirpath: str = os.path.join(root_dirpath, "data/hubmap-hacking-the-human-vasculature/train")

    # Debug Configs
    debug_dataset_config: str = os.path.join(root_dirpath, "config/debug/debug_dataset_image.yaml")
    debug_augmodule_config: str = os.path.join(root_dirpath, "config/debug/debug_augmodule_image.yaml")

    # Runs Configs
    augmodule_config: str = os.path.join(root_dirpath, "config/runs/augmodule_image.yaml")
    dataset_config: str = os.path.join(root_dirpath, "config/runs/dataset_image.yaml")
    datamodule_config: str = os.path.join(root_dirpath, "config/runs/datamodule_image.yaml")
    trainer_config: str = os.path.join(root_dirpath, "config/runs/trainer_image.yaml")


class SetupPipline(ConfigPath):
    def __init__(self):
        super().__init__()

    @staticmethod
    def pycocotools_setup() -> None:
        if not os.path.exists("/kaggle/working/packages"):
            os.mkdirs("/kaggle/working/packages")
            os.system("cp -r /kaggle/input/pycocotools/* /kaggle/working/packages")
            os.chdir("/kaggle/working/packages/pycocotools-2.0.6/")
            os.system("python setup.py install")
            os.system("pip install . --no-index --find-links /kaggle/working/packages/")
            os.chdir("/kaggle/working")

    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_variable(self):
        # Project Paths
        os.environ["ROOT_DIRPATH"] = self.root_dirpath
        os.environ["ANNOT_PATH"] = self.annot_path
        os.environ["DATA_PATH"] = self.image_dirpath

        # Runs Configs
        os.environ["DATASET_CONFIG"] = self.dataset_config
        os.environ["DATAMODULE_CONFIG"] = self.datamodule_config
        os.environ["AUGMODULE_CONFIG"] = self.augmodule_config
        os.environ["TRAINER_CONFIG"] = self.trainer_config

        # Debug Configs
        os.environ["DEBUG_DATASET_CONFIG"] = self.debug_dataset_config
        os.environ["DEBUG_AUGMODULE_CONFIG"] = self.debug_augmodule_config
