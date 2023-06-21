import os
import yaml


class InitialProject:
    def __init__(self,
                 images_dirpath: str,
                 annotations_path: str,
                 root_dirpath: str,
                 config_dirname: str = "config",
                 models_dirname: str = "models",
                 data_dirname: str = "data",
                 ):
        self.__config_struct = ("debug", "runs", "sample")
        self.root_dirpath = root_dirpath
        self.images_dirpath = images_dirpath
        self.annotations_path = annotations_path

        self.config_dirpath = None
        self.models_dirpath = None
        self.data_dirpath = None

        self.__define_struct(
            config_dirname,
            models_dirname,
            data_dirname
        )
        self.__build_struct()
        self.__save_config()

    def __define_struct(self,
                        config_dirname: str,
                        models_dirname: str,
                        data_dirname: str
                        ) -> None:

        self.config_dirpath = self.get_path(config_dirname)
        self.models_dirpath = self.get_path(models_dirname)
        self.data_dirpath = self.get_path(data_dirname)

    def get_path(self, dirname: str) -> str:
        return os.path.join(self.root_dirpath, dirname)

    def __build_struct(self) -> None:
        os.mkdir(self.data_dirpath)
        os.mkdir(self.models_dirpath)
        os.mkdir(self.config_dirpath, e)

        for dirname in self.__config_struct:
            dirpath = os.path.join(self.config_dirpath, dirname)
            os.mkdir(dirpath)

    def __write_config(self,
                       data: dict[dict, ...],
                       filename: str = "project_initial.yaml"
                       ) -> None:

        path = self.get_path(filename)
        with open(path, mode="w") as f:
            yaml.safe_dump(stream=f, data=data)

    def get_config(self) -> dict:
        return {
            "ROOT_DIRPATH": self.root_dirpath,
            "IMAGES_DIRPATH": self.images_dirpath,
            "ANNOTATIONS_PATH": self.annotations_path,
            "CONFIG_DIRPATH": self.config_dirpath,
            "DATA_DIRPATH": self.data_dirpath,
            "MODELS_DIRPATH": self.models_dirpath
        }

    def __save_config(self) -> None:
        data = self.get_config()
        self.__write_config(data)
