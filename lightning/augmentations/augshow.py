import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lightning.dataset import HuBMAPDataset
from ..dataset import HuBMAPDataset


class MaskDecoder:
    def __init__(self, colormap: dict):
        self.__colormap = colormap

    def __decode_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        elif mask.shape[-1] != 1:
            raise ValueError(
                f"Input matrix must have a shape of 1xHxW, "
                f"but expected shape {mask.shape}"
            )

        mask = np.apply_along_axis(
            func1d=lambda index: self.__colormap[int(index)],
            axis=0,
            arr=mask,
        )
        return mask.transpose(1, 2, 0)

    @staticmethod
    def __get_alpha_channel(mask: np.ndarray,
                            alpha: float = 1,
                            background: int = 0
                            ) -> np.ndarray:

        alpha = round(255 * alpha)
        mask = np.where(mask != background, alpha, 0)
        return np.expand_dims(mask, axis=0)

    @staticmethod
    def __apply_alpha_channel(image: np.ndarray,
                              alpha_channel: np.ndarray
                              ) -> np.ndarray:

        image = image.transpose(2, 0, 1)

        red_channel, green_channel, blue_channel = np.array_split(
            image, 3, axis=0
        )

        return np.concatenate(
            [red_channel,
             green_channel,
             blue_channel,
             alpha_channel], axis=0).transpose(1, 2, 0)

    def __call__(self,
                 mask: np.ndarray,
                 alpha: float = 1
                 ) -> np.ndarray:

        decoded = self.__decode_mask(mask)
        alpha_channel = self.__get_alpha_channel(mask, alpha=alpha)
        image = self.__apply_alpha_channel(decoded, alpha_channel)
        return image


class AugmentDataset(HuBMAPDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__check_instance()
        self.decoder = self.__get_decoder()

    def __get_decoder(self) -> MaskDecoder:
        colormap = self.__get_colormap()
        return MaskDecoder(colormap)

    def __check_instance(self) -> None:
        instance = self._head_config["instance"]
        if instance:
            raise ValueError(
                f"Instance must be False when using"
                f" the AugmentDataset, but got {instance}"
            )

    def __get_colormap(self) -> dict:
        colormap = {}
        for key, value in self._classes_config.items():
            rgb = value["rgb"]
            label = value["label"]
            colormap.update({label: rgb})
        return colormap

    def __getitem__(self, idx: int) -> dict:
        idx, identifier = self._get_sample(idx)
        image = self._get_image(identifier)
        mask = self._get_target(idx)
        transformed = self.__transforms(image, mask)
        return transformed

    def __transforms(self, image: np.ndarray, mask: np.ndarray) -> dict:
        mask_before = self.decoder(mask)
        image_after, mask_after = self._semantic_transforms(image, mask)
        mask_after = self.decoder(mask_after)
        return {
            "before": (image, mask_before),
            "after": (image_after, mask_after)
        }


class DisplayAugment(AugmentDataset):
    def __init__(self,
                 figsize: tuple = (10, 10),
                 facecolor: str = "#000000",
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.figsize = figsize
        self.facecolor = facecolor

    def __define_gridspec(self) -> GridSpec:
        figure = plt.figure(figsize=self.figsize)
        figure.set_facecolor(self.facecolor)
        gs = GridSpec(figure=figure, nrows=1, ncols=2)
        return gs

    def __imshow(self,
                 transformed: dict,
                 alpha: float,
                 ) -> None:

        gs = self.__define_gridspec()
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        image, mask = transformed["before"]
        ax1.imshow(image)
        ax1.imshow(mask, alpha=alpha)

        image, mask = transformed["after"]
        ax2.imshow(image)
        ax2.imshow(mask, alpha=alpha)

        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.show()

    def scroll(self,
               scrolls: int = 5,
               alpha: float = 0.5
               ) -> None:

        assert scrolls <= self.__len__()
        for idx in range(scrolls):
            transformed = self.__getitem__(idx)
            self.__imshow(transformed, alpha)
