from imagen import (
    DatasetImage,
    DataModuleImage,
    TrainerImage,
    AugModuleImage
)

from argparse import ArgumentParser
import os


def main(module: str, template: dict):
    writer = template[module]
    writer.generate_sample()


if __name__ == '__main__':
    template = {
        "datamodule": DataModuleImage(),
        "dataset": DatasetImage(),
        "trainer": TrainerImage(),
        "augmodule": AugModuleImage()
    }

    parser = ArgumentParser(
        description="Specify one argument:"
                    "module - name module which config image will be used: dataset/datamodule"
    )

    parser.add_argument(
        "module", type=str,
        help="module - name module which config image will be used"
    )

    args = parser.parse_args()
    module = args.module
    main(module, template)
