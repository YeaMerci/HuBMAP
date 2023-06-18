import os
import torch
import numpy as np
import random


class SetupPipline:
    def __init__(self):
        self.__root_dirpath = "/home/merci/PycharmProjects/competitions/HuBMAP/"

        self.__annot_path = os.path.join(
            self.__root_dirpath,
            "data/hubmap-hacking-the-human-vasculature/polygons.jsonl"
        )

        self.__image_path = os.path.join(
            self.__root_dirpath,
            "data/hubmap-hacking-the-human-vasculature/train"
        )

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
        os.environ["ANNOT_PATH"] = self.__annot_path
        os.environ["IMAGE_PATH"] = self.__image_path
        os.environ["ROOT_DIRPATH"] = self.__root_dirpath
