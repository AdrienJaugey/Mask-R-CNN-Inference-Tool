import os
import argparse
from datetime import datetime
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="path to the directory containing cleaned images and cortices masks files", type=str)
    parser.add_argument("dst", help="path to the output csv file", nargs='?', type=str)
    args = parser.parse_args()

    sourceDirPath = os.path.normpath(args.src)
    if not os.path.exists(sourceDirPath):
        print("Source folder not found, please provide correct path")
        exit(-1)
    if args.dst is None:
        if sourceDirPath in ['', '.', './', '.\\']:
            outputPath = "cortices_areas.csv"
        else:
            outputPath = f"cortices_areas_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    else:
        outputPath = os.path.normpath(args.dst)

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
