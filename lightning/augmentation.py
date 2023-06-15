__all__ = ["AugmentPipelineModule"]


import albumentations as A
import albumentations.pytorch as pytorch
import numpy as np
import torch
import cv2
from colorama import Fore


class AugmentPipelineModule:
    """
    This high-level API manages the transformation pipeline
    Includes dynamic adjustment of conversion parameters based on the size of the input image.
    The ratio size of the input image to the desired output signal size calculated,
     and then the conversion parameters adjusted accordingly.
    """
    def __init__(self,
                 height: int,
                 width: int):

        # Customization options.
        self.height = height
        self.width = width
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Image tearing.
        self.defocus = defocus
        self.pixel_dropout = pixel_dropout
        self.pieces_dropout = pieces_dropout
        self.horizontal_lines = horizontal_lines
        self.vertical_lines = vertical_lines
        self.__damage = any([self.defocus, self.pieces_dropout,
                             self.pieces_dropout, self.horizontal_lines,
                             self.vertical_lines])



        self.__debug_state = None

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__}: {self.__dict__}"

    def __len__(self) -> int:
        # Returns the number blocks in the pipeline.
        return sum([
            int(value)
            for value in self.__dict__.values()
            if type(value) is bool
        ])

    def __activate_debugging(self, image: np.array,
                             mask: np.array) -> tuple[np.array, np.array]:
        """
        Activates the debugging mode.
        Transforms the image without translation into tensors.
        """

        transform = A.Resize(height=self.height, width=self.width,
                             interpolation=cv2.INTER_LINEAR)
        transformed = transform(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]

        if not self.__debug_state:
            print(Fore.YELLOW+f"|| Debugging mode activated ||"+Fore.WHITE)
            self.__debug_state = True
        return image, mask

    def __weather_adaptation(self) -> None:
        """
        Creates weather effects.
        Namely: rain, sunny, snow, fog.
        """

        # Rain effects.
        if self.rain:
            self._rain_transform = A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=15,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=12,
                brightness_coefficient=0.52,
                rain_type="heavy",
                p=0.25)

        # Sunny effects.
        if self.sunny:
            self._sunny_transform = A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=4,
                num_flare_circles_upper=8,
                src_radius=100,
                src_color=(255, 255, 255),
                p=0.25)

        # Snow effects.
        if self.snow:
            self._snow_transform = A.RandomSnow(
                snow_point_lower=0.1,
                snow_point_upper=0.3,
                brightness_coeff=2.5,
                p=0.25)

        # Fog effects.
        if self.foggy:
            self._foggy_transform = A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.4,
                alpha_coef=0.1,
                p=0.01)

    def __damage_adaptation(self) -> None:
        """
        Creates image corruption effects.
        Namely: defocus, pixel dropout, vertical stripes, horizontal stripes, loss parts of the image.
        """

        # defocus.
        if self.defocus:
            self._defocus_transform = A.Defocus(
                radius=(0.5, 2.5),
                alias_blur=(0.1, 0.5),
                p=0.05)

        # pixel dropout.
        if self.pixel_dropout:
            self._pixel_dropout_transform = A.PixelDropout(dropout_prob=0.01, p=0.05)

        # loss parts of the image.
        if self.pieces_dropout:
            self._pieces_dropout_transform = A.CoarseDropout(
                max_holes=50,
                max_height=512,
                max_width=2,
                min_holes=10,
                min_height=128,
                min_width=1,
                fill_value=0,
                mask_fill_value=None,
                p=0.05)

        # horizontal stripes.
        if self.horizontal_lines:
            self._horizontal_lines_transform = A.CoarseDropout(
                max_holes=50,
                max_height=2,
                max_width=512,
                min_holes=10,
                min_height=1,
                min_width=128,
                fill_value=0,
                mask_fill_value=None,
                p=0.05)

        # vertical stripes.
        if self.vertical_lines:
            self._vertical_lines_transform = A.CoarseDropout(
                max_holes=50,
                max_height=8,
                max_width=8,
                min_holes=10,
                min_height=2,
                min_width=2,
                fill_value=0,
                mask_fill_value=None,
                p=0.05)

    def __cut_adaptation(self) -> None:
        # Includes three blocks: single, double and triple slice.
        # Single cut of the image.
        if self.cut == 1:
            self._cut_transforms = A.Compose([
                A.RandomCrop(width=2667, height=4000, p=0.15)
                ])

        # Double cut the image.
        elif self.cut == 2:
            self._cut_transforms = A.Compose([
                A.RandomCrop(width=2667, height=4000, p=0.10),
                A.RandomCrop(width=1283, height=2000, p=0.05),
            ])

        # Triple cut the image.
        else:
            self._cut_transforms = A.Compose([
                A.RandomCrop(width=2667, height=4000, p=0.15),
                A.RandomCrop(width=1283, height=2000, p=0.10),
                A.RandomCrop(width=855, height=1333, p=0.05),
            ])

    def __spatial_adaptation(self) -> None:
        """
        Spatial transformations.
        Namely: horizontal flip, vertical flip, affine, perspective.
        """

        if self.spatial:
            self._spatial_transforms = A.Compose([
                # horizontal flip.
                A.HorizontalFlip(p=0.5),

                # vertical flip.
                A.VerticalFlip(p=0.5),

                # affine.
                A.Affine(
                    rotate=[-15, 15], p=0.35,
                    mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_LINEAR),

                # perspective.
                A.Perspective(
                    scale=(0.05, 0.25),
                    pad_mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_LINEAR,
                    p=1)
            ])

    def __base_transform_adaptation(self) -> None:
        """
        A must-have composition.
        Translates the image into tensors.
        Normalizes tensors.
        Resizes.
        """

        self._base_transforms = A.Compose([
            A.Resize(height=self.height, width=self.width),
            A.Normalize(
                mean=self.mean,
                std=self.std,
            ),
            pytorch.ToTensorV2()
        ])

    def __pipeline_adaptation(self) -> None:
        """
        Causes transformation pipeline modules to adapt.
        Verifies that modules participate in the pipeline.
        """

        if self.__damage:
            self.__damage_adaptation()
        if self.__weather:
            self.__weather_adaptation()
        if self.cut:
            self.__cut_adaptation()
        if self.spatial:
            self.__spatial_adaptation()
        self.__base_transform_adaptation()

    def __call__(self, image: np.array, mask: np.array,
                 debug_mode: bool = False) -> dict[str, torch.Tensor]:
        self.__pipeline_adaptation()

        # Weather transformations.
        # Rain
        if self._rain_transform:
            transformed = self._rain_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Sunny.
        if self._sunny_transform:
            transformed = self._sunny_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Snow.
        if self._snow_transform:
            transformed = self._snow_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Foggy.
        if self._foggy_transform:
            transformed = self._foggy_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Damage transformation
        # Defocus.
        if self._defocus_transform:
            transformed = self._defocus_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Pixel dropout.
        if self._pixel_dropout_transform:
            transformed = self._pixel_dropout_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Pieces dropout.
        if self._pieces_dropout_transform:
            transformed = self._pieces_dropout_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Horizontal lines.
        if self._horizontal_lines_transform:
            transformed = self._horizontal_lines_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Vertical lines.
        if self._vertical_lines_transform:
            transformed = self._vertical_lines_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Cut transformation.
        if self.cut:
            transformed = self._cut_transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Spatial transformation.
        if self.spatial:
            transformed = self._spatial_transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Mandatory Transformations for Learning
        if not debug_mode:
            transformed = self._base_transforms(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        else:
            image, mask = self.__activate_debugging(image=image, mask=mask)

        return {"image": image,
                "mask": mask}
