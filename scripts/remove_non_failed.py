"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import os
import shutil
import argparse
import json


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
    parser = argparse.ArgumentParser(description="Remove non-failed images from the input folder of the inference tool")
    parser.add_argument("failed_list", help="path to the json file containing path of images that have failed",
                        type=existing_path)
    parser.add_argument("src", help="path to the directory containing original images and annotations",
                        type=existing_path, nargs='?')
    parser.add_argument("dst", help="path to the directory in which moving the images and annotations",
                        type=correct_path, nargs='?')
    args = parser.parse_args()

    failedListPath = args.failed_list
    sourceDirPath = args.src if args.src is not None else os.curdir
    destinationDirPath = args.dst if args.dst is not None else "successful"

    with open(failedListPath, 'r') as failedListFile:
        failedList = json.load(failedListFile)

    failedList = [os.path.splitext(os.path.basename(p))[0] for p in failedList]

    if not os.path.exists(destinationDirPath):
        os.makedirs(destinationDirPath, exist_ok=True)

    imageList = []
    fileFormat = ['jpg', 'png', 'jp2', 'xml', 'json']
    for aFile in os.listdir(sourceDirPath):
        if os.path.isfile(os.path.join(sourceDirPath, aFile)):
            fileName, extension = os.path.splitext(aFile)
            extension = extension.replace('.', '')
            if extension in fileFormat and fileName not in failedList:
                filePath = os.path.join(sourceDirPath, aFile)
                dstPath = os.path.join(destinationDirPath, aFile)
                shutil.move(filePath, dstPath)
    print("Done !")
    exit(0)
