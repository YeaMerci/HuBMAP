import os
import yaml


class ImageWriter:
    def generate_pattern(self) -> dict:
        raise NotImplementedError(
            "Child classes must override the base class method generate_pattern()"
        )

    def __get_path(self):
        return os.path.join(os.getcwd(), self._filename)

    def write_config(self, image: dict[dict, ...]) -> None:
        path = self.__get_path()
        with open(path, mode="w") as f:
            yaml.safe_dump(stream=f, data=image)

    def generate_sample(self):
        sample_image = self.generate_pattern()
        self.write_config(sample_image)
