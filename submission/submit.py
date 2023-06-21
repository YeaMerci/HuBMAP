import base64
import numpy as np
import torch
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import pandas as pd
import torchvision.transforms as T


class EncodeBinaryMask:
    @staticmethod
    def __checking_mask(mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.bool:
            raise ValueError(
                "expects a binary mask, received dtype == %s" %
                mask.dtype
            )
        return mask

    @staticmethod
    def __convert_mask(mask: np.ndarray):
        mask_to_encode = mask.astype(np.uint8)
        mask_to_encode = np.asfortranarray(mask_to_encode)
        return mask_to_encode

    @staticmethod
    def __compress_encode(encoded_mask) -> t.Text:
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str

    def __call__(self, mask: np.ndarray) -> t.Text:
        mask = self.__checking_mask(mask)
        mask_to_encode = self.__convert_mask(mask)
        encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
        base64_str = self.__compress_encode(encoded_mask)
        return base64_str


class Submission:
    def __init__(self, dirpath: str, model: torch.nn.Module):
        self.__eval_transforms = self.get_transforms()
        self.__model = model
        self.__encoder = EncodeBinaryMask()
        self.__dirpath = dirpath
        self.__filenames = os.listdir(dirpath)

        self.__submission_dict = {
            "id": [],
            "height": [],
            "width": [],
            "prediction_string": []
        }

        self.submission = None

    @staticmethod
    def get_transforms():
        return T.Compose([
            T.ToTensor(),
            T.Resize(size=(512, 512)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.__filenames)

    def __get_columns(self) -> None:
        for filename in self.__filenames:
            path = self.__get_image_path(filename)
            image = self.__get_image(path)
            masks = self.__forward(image)
            identifier, height, width, prediction_string = self.__get_cells(filename, masks)
            self.__update_columns(identifier, height, width, prediction_string)

    def __update_columns(self, identifier: str, height: int, width: int, prediction_string: str) -> None:
        self.__submission_dict["id"].append(identifier)
        self.__submission_dict["height"].append(height)
        self.__submission_dict["width"].append(width)
        self.__submission_dict["prediction_string"].append(prediction_string)

    def __get_cells(self, filename: str, masks: list):
        prediction_string = ""
        prediction_string = self.__get_prediction_string(masks, prediction_string)
        identifier = filename.split(".")[0]
        height, width = mask.shape
        return identifier, height, width, prediction_string

    def __get_prediction_string(self, masks: list, prediction_string: str) -> str:
        for outputs in masks:
            mask = outputs["mask"].detach().permute(1, 2, 0).cpu().numpy()
            mask = np.where(mask > 0.5, 1, 0).astype(np.bool)
            base64_str = self.__encoder(mask)
            confidence = outputs["confidence"]
            prediction_string += f"0 {confidence} {base64_str.decode('utf-8')} "
        return prediction_string

    def __get_image_path(self, filename: str) -> str:
        return os.path.join(
            self.__dirpath, filename
        )

    def __get_image(self, path: str) -> torch.Tensor:
        image = Image.open(path)
        image = np.asarray(image)
        image = self.__eval_transforms(image)
        return image

    # You must implement this function to work with your custom neural network!
    def __forward(self, image: torch.tensor) -> list:
        """
        This fuction should return list with masks & confidence.
        And each mask we shoulde encode (see EncodeBinaryMask & sample_submission file)

        outputs <- model(image)
        outputs -> [
                    {"mask": mask1, "confidence": confidence1},
                    {"mask": mask2, "confidence": confidence2},
                    ...,
                    {"mask": maskN, "confidence": confidenceN}
                   ]

        Example:
        """
        masks = self.__model(image)  # Mask must have shape 1x512x512
        return masks

    def submit(self) -> None:
        if not self.submission:
            self.__get_columns()
            self.submission = pd.DataFrame(self.__submission_dict)
            self.submission = self.submission.set_index('id')
            self.submission.to_csv("submission.csv")
