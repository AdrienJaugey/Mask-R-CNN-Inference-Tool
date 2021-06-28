"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import os
import argparse
from datetime import datetime
import cv2


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
    parser = argparse.ArgumentParser("Computes cortices areas for all cleaned images having their cortex mask of a "
                                     "folder and saving them into a csv file.")
    parser.add_argument("src", help="path to the directory containing cleaned images and cortices masks files",
                        type=existing_path)
    parser.add_argument("dst", help="path to the output csv file", nargs='?', type=correct_path)
    args = parser.parse_args()

    sourceDirPath = args.src
    if args.dst is None:
        outputPath = "cortices_areas.csv" if sourceDirPath == '.' \
            else f"cortices_areas_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    else:
        outputPath = args.dst

    files = os.listdir(sourceDirPath)
    with open(outputPath, 'w') as outputFile:
        outputFile.write('image; cortex_area;\n')
        for fileName in files:
            filePath = os.path.join(sourceDirPath, fileName)
            if os.path.isfile(filePath) and "_cortex.jpg" in fileName:
                imageName = fileName.replace('_cortex.jpg', '')
                if f"{imageName}.jpg" in files or f"{imageName}_cleaned.jpg" in files:
                    if f"{imageName}_cleaned.jpg" in files:
                        imagePath = os.path.join(sourceDirPath, f"{imageName}_cleaned.jpg")
                    else:
                        imagePath = os.path.join(sourceDirPath, f"{imageName}.jpg")
                    assert os.path.exists(filePath)
                    assert os.path.exists(imagePath)
                    image = cv2.imread(imagePath)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    height, width = image.shape
                    del image
                    cortex = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
                    cortex = cv2.resize(cortex, (width, height))
                    cortex_area = cv2.countNonZero(cortex)
                    outputFile.write(f"{imageName}; {cortex_area};\n")
    print(f'Cortices areas stored in {outputPath} file!\n')
    exit(0)
