import os
import shutil
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("failed_list", help="path to the json containing path of images that have failed", type=str)
    parser.add_argument("src", help="path to the directory containing original images and annotations", type=str, nargs='?')
    parser.add_argument("dst", help="path to the directory in which moving the images and annotations", type=str, nargs='?')
    args = parser.parse_args()

    sourceDirPath = os.path.normpath(args.src) if args.src is not None else "."
    destinationDirPath = os.path.normpath(args.dst) if args.dst is not None else "successful"

    if not os.path.exists(args.failed_list):
        print("File containing list of failed images not found, please provide correct path")
        exit(-1)
    else:
        failedListPath = os.path.normpath(args.failed_list)

    with open(failedListPath, 'r') as failedListFile:
        failedList = json.load(failedListFile)

    failedList = ["".join(os.path.basename(p).split('.')[:-1]) for p in failedList]

    if not os.path.exists(destinationDirPath):
        os.makedirs(destinationDirPath, exist_ok=True)

    imageList = []
    fileFormat = ['jpg', 'png', 'jp2', 'xml', 'json']
    for aFile in os.listdir(sourceDirPath):
        if os.path.isfile(os.path.join(sourceDirPath, aFile)):
            part = aFile.split('.')
            if part[-1] in fileFormat:
                fileName = "".join(part[:-1])
                if fileName not in failedList:
                    filePath = os.path.join(sourceDirPath, aFile)
                    dstPath = os.path.join(destinationDirPath, aFile)
                    shutil.move(filePath, dstPath)
    exit(0)
