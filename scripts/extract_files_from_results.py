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


pre_built_path = {
    'cleaned_cortex': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_fusion", "{imageDir}_cleaned.jpg"),
    'cortex_mask': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_fusion", "{imageDir}_cortex.jpg"),
    'expected': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_Expected.{imgFormat}"),
    'predicted': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_Predicted.{imgFormat}"),
    'predicted_clean': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_Predicted_clean.{imgFormat}"),
    'stats': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_stats.json"),
    'histo': os.path.join("{imageDir}", "{chainDir}", "{imageDir}_histo.json"),
    'annotation': os.path.join("{imageDir}", "{chainDir}", "{imageDir}.{annotationFormat}")
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Copy or move all files of a given type from results folder to destination "
                                     "folder.")
    parser.add_argument("src", help="path to the directory containing results of the inferences", type=existing_path)
    parser.add_argument("file_type", help="Which type of file", choices=list(pre_built_path.keys()))
    parser.add_argument("--dst", "-d", dest="dst", help="path to the directory containing results of the inferences",
                        type=correct_path)
    parser.add_argument("--extension", '-e', dest="ext", help="Format of the images", default='jpg',
                        choices=['jpg', 'png', 'jp2'])
    parser.add_argument("--annotation_format", '-a', dest="annotation_format", help="Format of the annotation",
                        default='xml', choices=['xml', 'json'])
    parser.add_argument("--mode", '-m', dest="mode", help="Whether the images should be copied or moved",
                        default='copy', choices=['copy', 'move'])
    parser.add_argument("--chain", '-c', dest="chain", help="Whether the results are from chain mode or not",
                        default=False, action='store_true')
    parser.add_argument("--inference_mode", '-i', dest="inference_mode", type=str, default=None,
                        help="Specific inference mode of chain results. Disable --chain argument")
    args = parser.parse_args()

    sourceDirPath = args.src
    if args.dst is None:
        destinationDirPath = "extracted" if sourceDirPath == '.' else (sourceDirPath + "_extracted")
    else:
        destinationDirPath = args.dst

    if not os.path.exists(destinationDirPath):
        os.makedirs(destinationDirPath, exist_ok=True)

    imageFormat = args.ext
    mode = args.mode
    specificInfMode = args.inference_mode
    if specificInfMode is not None and specificInfMode.replace(" ", "") == "":
        specificInfMode = None
    chainMode = args.chain if specificInfMode is None else False
    preBuildPath = pre_built_path[args.file_type]
    fileList = []
    for imageDir in os.listdir(sourceDirPath):
        if os.path.isdir(os.path.join(sourceDirPath, imageDir)):
            if chainMode:
                for inferenceMode in os.listdir(os.path.join(sourceDirPath, imageDir)):
                    infModeDirPath = os.path.join(sourceDirPath, imageDir, inferenceMode)
                    if os.path.isdir(infModeDirPath):
                        filePath = os.path.join(
                            sourceDirPath, preBuildPath.format(
                                imageDir=imageDir, imgFormat=imageFormat,
                                annotationFormat=args.annotation_format, chainDir=inferenceMode
                            )
                        )
                        if os.path.exists(filePath):
                            fileList.append(filePath)
            else:
                filePath = os.path.join(sourceDirPath, preBuildPath.format(
                    imageDir=imageDir, imgFormat=imageFormat, annotationFormat=args.annotation_format,
                    chainDir="." if specificInfMode is None else specificInfMode
                ))
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
            if chainMode or specificInfMode is not None:
                infMode = os.path.split(os.path.dirname(filePath))[1]
                dstPath = os.path.join(destinationDirPath, infMode)
                os.makedirs(dstPath, exist_ok=True)
                dstPath = os.path.join(dstPath, os.path.basename(filePath).replace('_cleaned.', '.'))
            else:
                dstPath = os.path.join(destinationDirPath, os.path.basename(filePath).replace('_cleaned.', '.'))
            method(filePath, dstPath)
        print(' Done !\n')
    exit(0)
