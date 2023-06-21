import os
import yaml
import wandb
from dataclasses import dataclass


@dataclass()
class Configs:
    datamodule: dict
    augmodule: dict
    trainer: dict


class LightBuilder:
    def __init__(self,
                 datamodule_config_path: str,
                 augmentations_config_path: str,
                 trainer_config_path: str,
                 job_type: str,
                 project: str,
                 tags: list
                 ):

        self.__datamodule_config = self.__get_datamodule_config(datamodule_config_path)
        self.__augmentations_config = self.__get_config(augmentations_config_path)
        self.__trainer_config = self.__get_config(trainer_config_path)

        self.configs = self.__get_configs()
        self.__init_wadnb(job_type, project, tags)

    def __mix_configs(self) -> dict:
        return {
            **self.configs.datamodule,
            **self.configs.trainer,
            **self.configs.augmodule
        }

    def __init_wadnb(self,
                     job_type: str,
                     project: str,
                     tags: list
                     ) -> None:

        config = self.__mix_configs()

        wandb.init(
            job_type=job_type,
            dir="",
            config=config,
            project=project,
            tags=tags
        )

    @staticmethod
    def __get_datamodule_config(datamodule_config_path: str) -> dict:
        datamodule_config = self.__get_config(datamodule_config_path)
        dataset_config = self.__get_config(datamodule_config["config_path"])
        datamodule_config.pop("config_path")
        datamodule_config["config"] = dataset_config
        return datamodule_config

    def __get_configs(self):
        return Configs(
            datamodule=self.__datamodule_config,
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
