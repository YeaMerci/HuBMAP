from initproject import InitialProject
from argparse import ArgumentParser


def main(annotations_path: str,
         images_dirpath: str,
         root_dirpath: str
         ):

    InitialProject(
        annotations_path=annotations_path,
        images_dirpath=images_dirpath,
        root_dirpath=root_dirpath
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Specify two arguments: "
                    "1) annotations_path - the path to the image annotation file "
                    "2) images_dirpath - the path to the directory with the images"
                    "3) root_dirpath - the root directory of the project"
    )

    parser.add_argument(
        "annotations_path", type=str,
        help="annotations_path - the path to the image annotation file "
    )

    parser.add_argument(
        "images_dirpath", type=str,
        help="images_dirpath - the path to the directory with the images"
    )

    parser.add_argument(
        "root_dirpath", type=str,
        help="root_dirpath - the root directory of the project"
    )

    args = parser.parse_args()
    main(args.annotations_path, args.images_dirpath, args.root_dirpath)
