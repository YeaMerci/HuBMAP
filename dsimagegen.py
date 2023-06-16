from dsimage import ImageWriter
from argparse import ArgumentParser


def main(dirpath: str, filename: str):
    writer = ImageWriter(dirpath, filename)
    writer.generate_sample()


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Specify two arguments: "
                    "1) dirpath - path to the directory in which the image file will be saved "
                    "2) filename - image file name. Default value is sample_image.yaml"
    )

    parser.add_argument(
        "dirpath", type=str,
        help="dirpath - path to the directory in which the image file will be saved"
    )

    parser.add_argument(
        "filename", type=str,
        help="filename - image file name. Default value is sample_image.yaml"
    )

    args = parser.parse_args()
    main(args.dirpath, args.filename)

