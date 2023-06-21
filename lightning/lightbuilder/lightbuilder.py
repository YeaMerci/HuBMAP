import os
import yaml
import wandb
from dataclasses import dataclass


@dataclass()
class Configs:
    datamodule: dict
    lightmodule: dict
    augmodule: dict
    trainer: dict


class LightBuilder:
    def __init__(self,
                 datamodule_config_path: str,
                 lightmodule_config_path: str,
                 augmentations_config_path: str,
                 trainer_config_path: str
                 ):
        self.__datamodule_config = self.__get_config(datamodule_config_path)
        self.__lightmodule_config = self.__get_config(lightmodule_config_path)
        self.__augmentations_config = self.__get_config(augmentations_config_path)
        self.__trainer_config = self.__get_config(trainer_config_path)
        self.configs = self.__get_configs()

    def __get_configs(self):
        return Configs(
            datamodule=self.__datamodule_config,
            lightmodule=self.__lightmodule_config,
            augmodule=self.__augmentations_config,
            trainer=self.__trainer_config
        )

    @staticmethod
    def load_config(template_path) -> dict:
        with open(template_path, mode="r") as f:
            data = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return data

    @staticmethod
    def __check_path(template_path: str) -> None:
        if not os.path.isfile(template_path):
            raise ValueError(
                f"Filename was expected to be a path,"
                f" but obtained: {template_path}"
            )

        if "config" in template_path.split("/"):
            return None

        elif "config" not in template_path.split("/"):
            print("Warning! The configuration file "
                  "is not located in the root directory "
                  "and may not be secure.")

    def __get_config(self,
                     template_path: str,
                     valid_path: bool = True
                     ) -> dict:
        if valid_path:
            self.__check_path(template_path)
        return self.load_config(template_path)
