import os
import shutil
import argparse
from datetime import datetime
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="path to the directory containing statistics files", type=str)
    parser.add_argument("dst", help="path to the output csv file", nargs='?', type=str)
    args = parser.parse_args()

    sourceDirPath = os.path.normpath(args.src)
    if not os.path.exists(sourceDirPath):
        print("Source folder not found, please provide correct path")
        exit(-1)
    if args.dst is None:
        if sourceDirPath in ['', '.', './', '.\\']:
            outputPath = "statistics.csv"
        else:
            outputPath = f"statistics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    else:
        outputPath = os.path.normpath(args.dst)

    # Getting all the classes that are present
    classes = []
    for fileName in os.listdir(sourceDirPath):
        filePath = os.path.join(sourceDirPath, fileName)
        if os.path.isfile(filePath) and "_stats.json" in fileName:
            imageName = fileName.replace('_stats.json', '')
            with open(filePath, 'r') as statsFile:
                stats = json.load(statsFile)
            for class_ in stats:
                if class_ not in classes:
                    classes.append(class_)
    classes.sort()
    if len(classes) == 0:
        print("No statistics found")
        exit(-1)
    with open(outputPath, 'w') as outputFile:
        line = "image; "
        for class_ in classes:
            line += f"{class_} count; {class_} area; "
        outputFile.write(line + "\n")
        for fileName in os.listdir(sourceDirPath):
            filePath = os.path.join(sourceDirPath, fileName)
            if os.path.isfile(filePath) and "_stats.json" in fileName:
                imageName = fileName.replace('_stats.json', '')
                with open(filePath, 'r') as statsFile:
                    stats = json.load(statsFile)
                line = imageName + "; "
                for class_ in classes:
                    count = 0
                    area = 0
                    if class_ in stats:
                        count = stats[class_]['count']
                        area = stats[class_]['area']
                    line += f"{count}; {area}; "
                outputFile.write(line + "\n")
    print(f'Statistics gathered in {outputPath} file!\n')
    exit(0)
