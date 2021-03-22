import os
import shutil
import argparse


def correct_path(value):
    try:
        path = os.path.normpath(value)
        return path
    except TypeError:
        raise argparse.ArgumentTypeError(f"{value} is not a correct path")


def existing_path(value):
    path = correct_path(value)
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{value} path does not exists")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Copy all the base images of a validation dataset from the raw dataset to "
                                     "destination folder.")
    parser.add_argument("val_dataset_path", help="Path to the val dataset directory", type=existing_path)
    parser.add_argument("raw_dataset_path", help="Path to the raw dataset directory", type=existing_path)
    parser.add_argument("--dst", "-d", dest="dst", help="Path to the output directory", type=correct_path)
    parser.add_argument("--extension", '-e', dest="ext", help="Format of the images", default='jpg',
                        choices=['jpg', 'png', 'jp2'])
    parser.add_argument("--annotation_format", '-a', dest="annotation_format", help="Format of the annotation",
                        default='xml', choices=['xml', 'json'])
    args = parser.parse_args()

    datasetPath = args.val_dataset_path
    rawDatasetPath = args.raw_dataset_path
    outputPath = (datasetPath + "_inf_tool") if args.dst is None else args.dst

    images = [imageDir.split('_')[0] for imageDir in os.listdir(datasetPath)
              if os.path.isdir(os.path.join(datasetPath, imageDir))]
    uniqueImages = []
    for image_ in images:
        if image_ not in uniqueImages:
            uniqueImages.append(image_)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath, exist_ok=True)

    for img in uniqueImages:
        for file in [f'{img}.{args.ext}', f'{img}.{args.annotation_format}']:
            shutil.copy2(os.path.join(rawDatasetPath, file), os.path.join(outputPath, file))
    print("Done !")
    exit(0)
