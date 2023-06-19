import albumentations as A
import albumentations.pytorch as pytorch
import numpy as np
import torch
import cv2
from colorama import Fore


class AugmentPipeline:
    def __init__(self,
                 mean: list[float, float, float] = None,
                 std: list[float, float, float] = None,
                 height: int = 512,
                 width: int = 512,
                 spatial: bool = False,
                 debug_mode: bool = False
                 ):

        # Customization options
        self.height = height
        self.width = width
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        self.std = std if std else [0.229, 0.224, 0.225]
        self.debug_mode = debug_mode

        self._spatial = spatial
        self._base_transforms = None
        self._resize = None
        self.__define_pipeline()

    def __define_spatial(self) -> None:
        """
        Spatial transformations.
        Namely: horizontal flip, vertical flip, affine, perspective.
        """
        self._spatial = A.Compose([
            # horizontal flip
            A.HorizontalFlip(p=0.5),

            # vertical flip
            A.VerticalFlip(p=0.5),

            # affine
            A.Affine(
                rotate=[-15, 15], p=0.35,
                mode=cv2.BORDER_REFLECT,
                interpolation=cv2.INTER_LINEAR
            ),

            # perspective
            A.Perspective(
                scale=(0.05, 0.25),
                pad_mode=cv2.BORDER_REFLECT,
                interpolation=cv2.INTER_LINEAR,
                p=1
            )
        ])

    def __define_base(self) -> None:
        """
        A must-have composition.
        Translates the image into tensors.
        Normalizes tensors.
        Resizes.
        """
        self._resize = A.Resize(
            height=self.height,
            width=self.width
        )

        self._base_transforms = A.Compose([
            A.Normalize(
                mean=self.mean,
                std=self.std,
            ),
            pytorch.ToTensorV2()
        ])

    def __define_pipeline(self) -> None:
        """
        Causes transformation pipeline modules to adapt.
        Verifies that modules participate in the pipeline.
        """
        if self._spatial:
            self.__define_spatial()
        self.__define_base()

    def __call__(self,
                 image: np.ndarray,
                 mask: np.ndarray,
                 ) -> dict[str, torch.Tensor]:

        transformed = self._resize(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]

        # Spatial transformation
        if self._spatial:
            transformed = self._spatial(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Mandatory transformations for learning
        if not self.debug_mode:
            transformed = self._base_transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return {"image": image,
                "mask": mask}
