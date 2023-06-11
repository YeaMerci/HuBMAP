import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


class Submission:
    def __init__(self, dirpath: str, model):
        self.__model = model
        self.__encoder = EncodeBinaryMask()
        self.__dirpath = dirpath
        self.__filenames = os.listdir(dirpath)

    def __len__(self):
        return len(self.__filenames)

    def __iter__(self):
        for filename in self.__filenames:
            path = self.__get_image_path(filename)
            image = self.__get_image(path)
            mask = self.__forward(image)
            base64_str = self.__encoder(mask)
            yield base64_str

    def __get_image_path(self, filename: str):
        return os.path.join(
            self.__dirpath, filename
        )

    @staticmethod
    def __get_image(path: str):
        image = Image.open(image_path)
        image = np.asarray(image)
        image = torch.tensor(
            image, dtype=torch.float32).permute(2, 0, 1)
        return image

    def __forward(self, image: torch.tensor):
        output = self.model(image)
        return output


class EncodeBinaryMask:
    def __init__(self, dirpath: str = None):
        self.__dirpath = dirpath

    @staticmethod
    def __checking_mask(mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.bool:
            raise ValueError(
                "encode_binary_mask expects a binary mask, received dtype == %s" %
                mask.dtype
            )

        mask = np.squeeze(mask)
        if len(mask.shape) != 2:
            raise ValueError(
                "encode_binary_mask expects a 2d mask, received shape == %s" %
                mask.shape
            )
        return mask

    @staticmethod
    def __convert_mask(mask: np.ndarray):
        mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask_to_encode = mask_to_encode.astype(np.uint8)
        mask_to_encode = np.asfortranarray(mask_to_encode)
        return mask_to_encode

    @staticmethod
    def __compress_encode(encoded_mask):
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str

    def __encode_binary_mask(self, mask: np.ndarray) -> t.Text:
        mask = self.__checking_mask(mask)
        mask_to_encode = self.__convert_mask(mask)
        encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
        base64_str = self.__compress_encode(encoded_mask)
        return base64_str

    def __call__(self, mask: np.ndarray) -> t.Text:
        base64_str = self.__encode_binary_mask(mask)
        return base64_str
