import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def createMask(imgName: str, imgShape, idMask: int, ptsMask, datasetName: str = 'dataset_train',
               maskClass: str = 'masks'):
    """
    Create the mask image based on its polygon points
    :param imgName: name w/o extension of the base image
    :param imgShape: shape of the image
    :param idMask: the ID of the mask, a number not already used for that image
    :param ptsMask: array of [x, y] coordinates which are all the polygon points representing the mask
    :param datasetName: name of the output dataset
    :param maskClass: name of the associated class of the current mask
    :return: None
    """
    # Formatting the suffix for the image representing the mask
    maskName = str('0' if idMask < 100 else '') + str('0' if idMask < 10 else '') + str(idMask)
    # Defining path where the result image will be stored and creating dir if not exists
    output_directory = datasetName + '/' + imgName + '/' + maskClass + '/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # https://www.programcreek.com/python/example/89415/cv2.fillPoly
    # Formatting coordinates matrix to get int
    ptsMask = np.double(ptsMask)
    ptsMask = np.matrix.round(ptsMask)
    ptsMask = np.int32(ptsMask)

    # Creating black matrix with same size than original image and then drawing the mask
    mask = np.uint8(np.zeros((imgShape[0], imgShape[1])))
    cv2.fillPoly(mask, [ptsMask], 255)

    # Saving result image
    output_name = imgName + maskName + '.png'
    cv2.imwrite(output_directory + output_name, mask)


def createMasksOfImage(rawDatasetPath: str, imgName: str, datasetName: str = 'dataset_train'):
    """
    Create all the masks of a given image by parsing xml annotations file
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param imgName: name w/o extension of an image
    :param datasetName: name of the output dataset
    :return: None
    """
    # Getting shape of original image (same for all this masks)
    img = None
    img = cv2.imread(os.path.join(rawDatasetPath, imgName + '.png'))
    if img is None:
        print('Problem with {} image'.format(imgName))
        return
    shape = img.shape

    # Copying the original image in the dataset
    targetDirectoryPath = datasetName + '/' + imgName + '/images/'
    if not os.path.exists(targetDirectoryPath):
        os.makedirs(targetDirectoryPath)
        cv2.imwrite(targetDirectoryPath + imgName + '.png', img)

    # https://www.datacamp.com/community/tutorials/python-xml-elementtree
    tree = ET.parse(os.path.join(rawDatasetPath, imgName + '.xml'))
    root = tree.getroot()
    # Going through the XML tree and getting all Annotation nodes
    for annotation in root.findall('./Annotations/Annotation'):
        maskClass = annotation.attrib.get('PartOfGroup')
        noMask = annotation.attrib.get('Name').replace('Annotation ', '')
        ptsMask = []
        # Going through the Annotation node and getting all Coordinate nodes
        for points in annotation.find('Coordinates'):
            xCoordinate = points.attrib.get('X')
            yCoordinate = points.attrib.get('Y')
            ptsMask.append([xCoordinate, yCoordinate])
        # print('Mask ' + noMask + ': NbPts = ' + str(len(ptsMask)) + '\tclass = ' + maskClass)
        createMask(imgName, shape, int(noMask), ptsMask, datasetName, maskClass)


def fuseCortices(datasetPath: str, imageName: str, deleteBaseMasks=False):
    """
    Fuse each cortex masks into one
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :param deleteBaseMasks: delete the base masks images after fusion
    :return: None
    """
    # Getting the image directory path
    imageDir = os.path.join(datasetPath, imageName)
    cortexDir = os.path.join(imageDir, 'cortex')
    if os.path.exists(cortexDir):
        listCortexImages = os.listdir(cortexDir)
        if len(listCortexImages) > 1:  # Fusing only if strictly more than one cortex mask image present
            print("Fusing {} cortices masks".format(imageName))
            fusion = cv2.imread(os.path.join(cortexDir, listCortexImages[0]), cv2.IMREAD_UNCHANGED)
            for maskName in os.listdir(cortexDir):  # Adding each mask to the same image
                maskPath = os.path.join(cortexDir, maskName)
                mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
                fusion = cv2.add(fusion, mask)
            # temp = cv2.resize(fusion, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_CUBIC)
            # cv2.imshow("fusion_{}".format(imageName), temp)
            # Saving the fused mask image
            cv2.imwrite(os.path.join(cortexDir, imageName + "_cortex.png"), fusion)
            if deleteBaseMasks:
                for maskName in os.listdir(cortexDir):  # Deleting each cortex mask except the fused one
                    if '_cortex.png' not in maskName:
                        maskPath = os.path.join(cortexDir, maskName)
                        os.remove(maskPath)


def cleanCortexDir(datasetPath: str):
    """
    Cleaning all cortex directories in the dataset, keeping only unique file or fused ones
    :param datasetPath: the dataset that have been wrapped
    :return: None
    """
    for imageDir in os.listdir(datasetPath):
        imageDirPath = os.path.join(datasetPath, imageDir)
        cortexDirPath = os.path.join(imageDirPath, 'cortex')
        if os.path.exists(cortexDirPath):
            listCortexImages = os.listdir(cortexDirPath)
            if len(listCortexImages) > 1:  # Deleting only if strictly more than one cortex mask image present
                fusedCortexPresent = False
                for cortexImage in listCortexImages:
                    fusedCortexPresent = fusedCortexPresent or ('_cortex' in cortexImage)
                if fusedCortexPresent:
                    for maskName in os.listdir(cortexDirPath):  # Deleting each cortex mask except the fused one
                        if '_cortex' not in maskName:
                            maskPath = os.path.join(cortexDirPath, maskName)
                            os.remove(maskPath)


def cleanImage(datasetPath: str, imageName: str):
    """
    Creating the full_images directory and cleaning the base image by removing non-cortex areas
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :return: None
    """
    # Defining all the useful paths
    currentImageDirPath = os.path.join(datasetPath, imageName)
    imagesDirPath = os.path.join(currentImageDirPath, 'images')
    imageFileName = os.listdir(imagesDirPath)[0]
    imagePath = os.path.join(imagesDirPath, imageFileName)
    image = cv2.imread(imagePath)

    # Copying the full image into the correct directory
    fullImageDirPath = os.path.join(currentImageDirPath, 'full_images')
    os.makedirs(fullImageDirPath, exist_ok=True)
    cv2.imwrite(os.path.join(fullImageDirPath, imageFileName), image)

    # Getting the cortex image
    cortexDirPath = os.path.join(currentImageDirPath, 'cortex')
    if os.path.exists(cortexDirPath):
        cortexFileName = os.listdir(cortexDirPath)[0]
        cortexFilePath = os.path.join(cortexDirPath, cortexFileName)
        cortex = cv2.imread(cortexFilePath)

        # Cleaning the image
        image = cv2.bitwise_and(image, cortex)
        # resized = cv2.resize(image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("cropped", resized)
        # cv2.waitKey(0)
        cv2.imwrite(imagePath, image)


def convertImage(inputImagePath: str, outputImagePath: str):
    """
    Convert an image from a format to another one
    :param inputImagePath: path to the initial image
    :param outputImagePath: path to the output image
    :return: None
    """
    image = cv2.imread(inputImagePath)
    cv2.imwrite(outputImagePath, image)


def getInfoRawDataset(rawDatasetPath: str, verbose=False):
    """
    Listing all available images, those with missing information
    :param verbose: whether or not print should be executed
    :param rawDatasetPath: path to the raw dataset folder
    :return: list of unique files names, list of available images names, list of missing images names,
    list of missing annotations names
    """
    names = []
    images = []  # list of image that can be used to compute masks
    missingImages = []  # list of missing images
    missingAnnotations = []  # list of missing annotations
    if verbose:
        print("Listing files and creating png if not present")
    for file in os.listdir(rawDatasetPath):
        name = file.split('.')[0]
        if name not in names:  # We want to do this only once per unique file name (without extension)
            names.append(name)

            # Testing if there is an png image with that name
            pngPath = os.path.join(rawDatasetPath, name + '.png')
            pngExists = os.path.exists(pngPath)

            # Same thing with jp2 format
            jp2Path = os.path.join(rawDatasetPath, name + '.jp2')
            jp2Exists = os.path.exists(jp2Path)

            # Testing if annotation file exists for that name
            annotationsExist = os.path.exists(os.path.join(rawDatasetPath, name + '.xml'))
            if pngExists or jp2Exists:  # At least one image exists
                if not annotationsExist:  # Annotations are missing
                    missingAnnotations.append(name)
                else:
                    if verbose:
                        print("Adding {} image".format(name))
                    if not pngExists:  # Only jp2 exists
                        if verbose:
                            print("\tCreating png version")
                        convertImage(jp2Path, pngPath)
                    images.append(name)  # Adding this image to the list
            elif annotationsExist:  # There is no image file but xml found
                missingImages.append(name)
    return names, images, missingImages, missingAnnotations


def startWrapper(rawDatasetPath: str, datasetName: str = 'dataset_train', deleteBaseCortexMasks=False):
    """
    Start wrapping the raw dataset into the wanted format
    :param deleteBaseCortexMasks: delete the base masks images after fusion
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param datasetName: name of the output dataset
    :return: None
    """
    names, images, missingImages, missingAnnotations = getInfoRawDataset(rawDatasetPath, verbose=True)
    for file in os.listdir(rawDatasetPath):
        name = file.split('.')[0]
        if name not in names:  # We want to do this only once per unique file name (without extension)
            names.append(name)

            # Testing if there is an png image with that name
            pngPath = os.path.join(rawDatasetPath, name + '.png')
            pngExists = os.path.exists(pngPath)

            # Same thing with jp2 format
            jp2Path = os.path.join(rawDatasetPath, name + '.jp2')
            jp2Exists = os.path.exists(jp2Path)

            # Testing if annotation file exists for that name
            annotationsExist = os.path.exists(os.path.join(rawDatasetPath, name + '.xml'))
            if pngExists or jp2Exists:  # At least one image exists
                if not annotationsExist:  # Annotations are missing
                    missingAnnotations.append(name)
                else:
                    print("Adding {} image".format(name))
                    if not pngExists:  # Only jp2 exists
                        print("\tCreating png version")
                        image = cv2.imread(jp2Path)  # Opening the jp2 image
                        cv2.imwrite(pngPath, image)  # Saving the jp2 image to png format
                    images.append(name)  # Adding this image to the list
            elif annotationsExist:  # There is no image file but xml found
                missingImages.append(name)

    print()
    # Displaying missing image files
    nbMissingImg = len(missingImages)
    if nbMissingImg > 0:
        print('Missing {} image{} : {}'.format(nbMissingImg, 's' if nbMissingImg > 1 else '', missingImages))

    # Displaying missing annotations files
    nbMissingAnnotations = len(missingAnnotations)
    if nbMissingAnnotations > 0:
        print('Missing {} annotation{} : {}'.format(nbMissingAnnotations, 's' if nbMissingAnnotations > 1 else '',
                                                    missingAnnotations))

    # Checking if there is file that is not image nor annotation
    nbImages = len(images)
    if len(names) - nbMissingImg - nbMissingAnnotations - nbImages != 0:
        print('Be careful, there are not only required dataset files in this folder')

    # Creating masks for any image which has all required files and displaying progress
    for index in range(nbImages):
        file = images[index]
        print('Creating masks for {} image {}/{} ({:.2f}%)'.format(file, index + 1, nbImages,
                                                                   (index + 1) / nbImages * 100))
        createMasksOfImage(rawDatasetPath, file, datasetName)
        fuseCortices(datasetName, file, deleteBaseMasks=deleteBaseCortexMasks)
        cleanImage(datasetName, file)

