import os
import shutil
import argparse

pre_built_path = {
    'cleaned_cortex': os.path.join("{imageDir}", "{imageDir}_fusion", "{imageDir}_cleaned.jpg"),
    'cortex_mask': os.path.join("{imageDir}", "{imageDir}_fusion", "{imageDir}_cortex.jpg"),
    'expected': os.path.join("{imageDir}", "{imageDir}_Expected.{imgFormat}"),
    'predicted': os.path.join("{imageDir}", "{imageDir}_Predicted.{imgFormat}"),
    'predicted_clean': os.path.join("{imageDir}", "{imageDir}_Predicted_clean.{imgFormat}"),
    'stats': os.path.join("{imageDir}", "{imageDir}_stats.json"),
    'annotation': os.path.join("{imageDir}", "{imageDir}.{annotationFormat}")
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="path to the directory containing results of the inferences", type=str)
    parser.add_argument("image", help="Which type of file", choices=list(pre_built_path.keys()))
    parser.add_argument("--dst", "-d", dest="dst", help="path to the directory containing results of the inferences",
                        type=str)
    parser.add_argument("--extension", '-e', dest="ext", help="Format of the images", default='jpg',
                        choices=['jpg', 'png', 'jp2'])
    parser.add_argument("--annotation_format", '-a', dest="annotation_format", help="Format of the annotation",
                        default='xml', choices=['xml', 'json'])
    parser.add_argument("--mode", '-m', dest="mode", help="Whether the images should be copied or moved",
                        default='copy', choices=['copy', 'move'])
    args = parser.parse_args()

    sourceDirPath = os.path.normpath(args.src)
    if not os.path.exists(sourceDirPath):
        print("Source folder not found, please provide correct path")
        exit(-1)
    if args.dst is None:
        if sourceDirPath in ['', '.', './', '.\\']:
            destinationDirPath = "extracted"
        else:
            destinationDirPath = os.path.join(os.path.dirname(sourceDirPath),
                                              os.path.basename(sourceDirPath) + "_extraced")
    else:
        destinationDirPath = os.path.normpath(args.dst)

    if not os.path.exists(destinationDirPath):
        os.makedirs(destinationDirPath, exist_ok=True)

    imageFormat = args.ext
    mode = args.mode
    image_type = args.image
    path = pre_built_path[image_type]
    fileList = []
    for imageDir in os.listdir(sourceDirPath):
        if os.path.isdir(os.path.join(sourceDirPath, imageDir)):
            filePath = os.path.join(sourceDirPath, path.format(imageDir=imageDir, imgFormat=imageFormat,
                                                               annotationFormat=args.annotation_format))
            if os.path.exists(filePath):
                fileList.append(filePath)

    if len(fileList) == 0:
        print("No file found, be sure that the given path is correct\n")
    else:
        method = shutil.copy2 if mode == "copy" else shutil.move
        print(
            f"{'Copy' if args.mode == 'copy' else 'Mov'}ing {len(fileList)} file{'s' if len(fileList) > 0 else ''}...",
            end='', flush=True)
        for filePath in fileList:
            dstPath = os.path.join(destinationDirPath, os.path.basename(filePath).replace('_cleaned.', '.'))
            method(filePath, dstPath)
        print(' Done !\n')
    exit(0)
