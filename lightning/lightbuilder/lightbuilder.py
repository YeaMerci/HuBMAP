import os
import yaml


class LightBuilder:
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

    def __call__(self,
                 template_path: str,
                 valid_path: bool = True
                 ) -> dict:
        if valid_path:
            self.__check_path(template_path)
        return self.load_config(template_path)
