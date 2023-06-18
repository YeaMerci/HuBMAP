import os
from imagen import DatasetImage, DataModuleImage
from argparse import ArgumentParser


def main(module: str, template: dict):
    writer = template[module]
    writer.generate_sample()


if __name__ == '__main__':
    template = {
        "DataModuleImage": DataModuleImage(),
        "DatasetImage": DatasetImage()
    }

    parser = ArgumentParser(
        description="Specify one argument:"
                    "module - name module which config image will be used"
    )

    parser.add_argument(
        "module", type=str,
        help="module - name module which config image will be used"
    )

    args = parser.parse_args()
    module = args.module
    main(module, template)
