import os
import yaml


class LightBuilder:
    def __init__(self, template_path: str):
        self.__template_path = template_path
        self.__check_path()
        self._config = self.load_config()

    def load_config(self) -> dict:
        with open(self.__template_path, mode="r") as f:
            data = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return data

    def __check_path(self) -> None:
        if not os.path.isfile(self.__template_path):
            raise ValueError(
                f"Filename was expected to be a path,"
                f" but obtained: {self.__template_path}"
            )

        if "config" in self.__template_path.split("/"):
            return None

        elif "config" not in self.__template_path.split("/"):
            print("Warning! The configuration file "
                  "is not located in the root directory "
                  "and may not be secure.")
