import os
import torch
import random


class SetupPipline:
    @staticmethod
    def pycocotools_setup() -> None:
        if not os.path.exists("/kaggle/working/packages"):
            os.mkdir("/kaggle/working/packages")
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

    def __call__(self, seed: int = 42, pycoco: bool = True) -> None:
        if pycoco:
            self.pycocotools_setup()
        if seed:
            self.seed_everything(seed)
