from lightning import DisplayAugment, AugmentPipeline
from argparse import ArgumentParser
import albumentations as A
from setup import SetupPipline
import os
import yaml


def load_config(path: str) -> dict:
    with open(path, mode="r") as f:
        data = yaml.load(stream=f, Loader=yaml.SafeLoader)
    return data


def main(scrolls: int = 5, alpha: float = 0.6):
    transforms = AugmentPipeline(
        **load_config(os.environ["DEBUG_AUGMODULE_CONFIG"])
    )

    display = DisplayAugment(
        transforms=transforms,
        config=load_config(os.environ["DEBUG_DATASET_CONFIG"]),
        stage="train",
        shuffle=False,
        random_state=42,
        train_size=0.80,
        annotation_path=os.environ["ANNOT_PATH"],
        image_path=os.environ["DATA_PATH"]
    )

    display.scroll(scrolls=scrolls, alpha=alpha)


if __name__ == '__main__':
    setuper = SetupPipline()
    setuper.set_variable()

    parser = ArgumentParser(
        description="Specify two arguments: "
                    "1) --scrolls - the number of images that will be displayed "
                    "2) --alpha - the transparency value. Range from 0 to 1"
    )

    parser.add_argument(
        "--scrolls", type=str,
        help="scrolls - the number of images that will be displayed"
    )

    parser.add_argument(
        "--alpha", type=str,
        help="alpha - the transparency value. Range from 0 to 1"
    )

    args = parser.parse_args()
    if args.scrolls and args.alpha:
        scrolls, alpha = int(args.scrolls), float(args.alpha)
        main(scrolls, alpha)
    else:
        main()
