from lightning import DisplayAugment, AugmentPipeline
from argparse import ArgumentParser
import albumentations as A
from setup import SetupPipline
import os


def main(scrolls: int = 5, alpha: float = 0.6):
    transforms = AugmentPipeline(
        spatial=True,
        debug_mode=True,
    )

    display = DisplayAugment(
        stage="train",
        annotation_path=os.environ["ANNOT_PATH"],
        image_path=os.environ["DATA_PATH"],
        template_path=os.environ["AUGRUN_IMAGE_PATH"],
        transforms=transforms,
        train_size=0.85,
        shuffle=True,
        random_state=42,
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
