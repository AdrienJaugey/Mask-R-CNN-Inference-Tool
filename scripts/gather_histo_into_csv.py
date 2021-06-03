import os
import argparse
from datetime import datetime
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
    parser = argparse.ArgumentParser("Gather statistics from all histograms files of a source folder "
                                     "into a csv file.")
    parser.add_argument("src", help="path to the directory containing histograms files", type=existing_path)
    parser.add_argument("dst", help="path to the output csv file", nargs='?', type=correct_path)
    args = parser.parse_args()

    sourceDirPath = args.src
    if args.dst is None:
        outputPath = "histograms.csv" if sourceDirPath == '.' \
            else f"histograms_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    else:
        outputPath = args.dst

    # Getting all the classes that are present
    classes = {}
    for fileName in os.listdir(sourceDirPath):
        filePath = os.path.join(sourceDirPath, fileName)
        if os.path.isfile(filePath) and "_histo.json" in fileName:
            imageName = fileName.replace('_histo.json', '')
            with open(filePath, 'r') as statsFile:
                histo = json.load(statsFile)
            for class_ in histo:
                if class_ == "_comment":
                    continue
                if class_ not in classes:
                    classes.update({class_: []})
                for amount in histo[class_]:
                    amount_int = int(amount)
                    if amount_int not in classes[class_]:
                        classes[class_].append(amount_int)
    for class_ in classes:
        classes[class_].sort()
    if len(classes) == 0:
        print("No histogram found")
        exit(-1)
    with open(outputPath, 'w') as outputFile:
        line = "image; "
        sortedClasses = list(classes.keys())
        sortedClasses.sort()
        for class_ in sortedClasses:
            for amount_ in classes[class_]:
                line += f"{class_} (x{amount_}); "
        outputFile.write(line + "\n")
        for fileName in os.listdir(sourceDirPath):
            filePath = os.path.join(sourceDirPath, fileName)
            if os.path.isfile(filePath) and "_histo.json" in fileName:
                imageName = fileName.replace('_histo.json', '')
                with open(filePath, 'r') as statsFile:
                    histo = json.load(statsFile)
                line = imageName + "; "
                for class_ in sortedClasses:
                    hasClass = class_ in histo
                    for amount_ in classes[class_]:
                        line += ("0" if not hasClass else str(histo[class_].get(str(amount_), 0))) + "; "
                outputFile.write(line + "\n")
    print(f'Histograms gathered in {outputPath} file!\n')
    exit(0)
