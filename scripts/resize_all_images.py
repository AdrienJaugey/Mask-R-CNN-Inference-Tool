import os
import argparse
import cv2

IMAGE_FORMAT = ['jpg', 'png']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resize and/or convert all the images of a folder to given size or " +
                                                 "using given ratio. Can replace images")
    parser.add_argument("src", help="Path to the directory containing the input images.", type=str)
    parser.add_argument("--input_format", '-i', dest="inputFormat", help="Format of the images.", default='jpg',
                        choices=IMAGE_FORMAT)
    parser.add_argument("--dst", "-d", dest="dst", help="Path to the directory containing the output images.",
                        type=str)
    parser.add_argument("output_size", type=float, nargs=2,
                        help="Size of the final image. If you want to pass a ratio instead, add also the --ratio flag.")
    parser.add_argument("--ratio", "-r", dest="ratio", help="If given, output_size will be seen as a ratio.",
                        action="store_true")
    parser.add_argument("--output_format", '-o', dest="outputFormat", help="Format of the output images.",
                        choices=IMAGE_FORMAT)
    parser.add_argument("--jpeg_quality", '-j', dest="jpeg_quality", type=int, default=100,
                        help="Quality of the output JPEG between in [0, 100]. Default is 100.")
    parser.add_argument("--png_compression", '-p', dest="png_compression", type=int, default=9,
                        help="Level of compression of the output PNG image between in [0, 9]. Default is 9.")
    parser.add_argument("--overwrite", '-w', dest="replace", action="store_true",
                        help="If given and using same src/dst directory and format, images will be replaced.")
    args = parser.parse_args()

    # Checking if source directory exists else exiting the program
    sourceDirPath = os.path.normpath(args.src)
    if not os.path.exists(sourceDirPath):
        print("Source folder not found, please provide correct path")
        exit(-1)

    # Choosing the destination directory based on flags
    replace = args.replace
    destinationDirPath = args.dst
    if destinationDirPath is None:
        if replace:
            destinationDirPath = sourceDirPath
        else:
            if sourceDirPath in ['', '.', './', '.\\']:
                destinationDirPath = "resized"
            else:
                destinationDirPath = os.path.join(os.path.dirname(sourceDirPath),
                                                  os.path.basename(sourceDirPath) + "_resized")
    else:
        destinationDirPath = os.path.normpath(destinationDirPath)

    # Creating destination directory if not existing
    if not os.path.exists(destinationDirPath):
        os.makedirs(destinationDirPath, exist_ok=True)

    # Defining resize parameters
    ratio = args.ratio
    size = fx = fy = None
    if ratio:
        fx, fy = args.output_size[:2]
    else:
        size = tuple([round(x) for x in args.output_size])
    resize = not ratio or fx != 1. or fy != 1.

    # Defining behaviour
    inputFormat = args.inputFormat
    outputFormat = inputFormat if args.outputFormat is None else args.outputFormat

    rename = False
    if sourceDirPath == destinationDirPath and inputFormat == outputFormat:
        rename = not replace

    jpeg_quality = max(0, min(args.jpeg_quality, 100))
    png_compression = max(0, min(args.png_compression, 9))
    cv2_write_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality, cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    for item in os.listdir(sourceDirPath):
        name, ext = os.path.splitext(item)
        ext = ext.replace('.', '')
        if ext.lower() == inputFormat:
            imageInputPath = os.path.join(sourceDirPath, item)
            imageOutputPath = os.path.join(destinationDirPath, f"{name}{'_resized' if rename else ''}.{outputFormat}")
            image = cv2.imread(imageInputPath, cv2.IMREAD_UNCHANGED)
            if resize:
                image = cv2.resize(image, size, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(imageOutputPath, image, cv2_write_params)
    print("Done !\n")
