import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from datasetTools.datasetDivider import getBWCount
from datasetTools import AnnotationAdapter as adapt
from datasetTools.AnnotationAdapter import AnnotationAdapter

classesInfo = [
    {"id": 0, "name": "Background", "color": "", "ignore": True},
    {"id": 1, "name": "tubule_sain", "color": "#ff007f", "ignore": False},
    {"id": 2, "name": "tubule_atrophique", "color": "#55557f", "ignore": False},
    {"id": 3, "name": "nsg_complet", "color": "#ff557f", "ignore": False},
    {"id": 4, "name": "nsg_partiel", "color": "#55aa7f", "ignore": False},
    {"id": 5, "name": "pac", "color": "#ffaa7f", "ignore": False},
    {"id": 6, "name": "vaisseau", "color": "#55ff7f", "ignore": False},
    {"id": 7, "name": "artefact", "color": "#000000", "ignore": False},
    {"id": 8, "name": "veine", "color": "#0000ff", "ignore": False},
    {"id": 9, "name": "nsg", "color": "#55007f", "ignore": True},
    {"id": 10, "name": "intima", "color": "#aa0000", "ignore": True},
    {"id": 11, "name": "media", "color": "#aa5500", "ignore": True}
]


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
    maskClass = maskClass.replace(" ", "_")
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


def createMasksOfImage(rawDatasetPath: str, imgName: str, datasetName: str = 'dataset_train',
                       adapter: AnnotationAdapter = None):
    """
    Create all the masks of a given image by parsing xml annotations file
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param imgName: name w/o extension of an image
    :param datasetName: name of the output dataset
    :param adapter: the annotation adapter to use to create masks, if None looking for an adapter that can read the file
    :return: None
    """
    # Getting shape of original image (same for all this masks)
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

    # Finding annotation files
    formats = adapt.ANNOTATION_FORMAT
    fileList = os.listdir(rawDatasetPath)
    imageFiles = []
    for file in fileList:
        if imgName in file:
            if file.split('.')[-1] in formats:
                imageFiles.append(file)

    # Choosing the adapter to use (parameters to force it ?)
    file = None
    assert len(imageFiles) > 0
    if adapter is None:
        # No adapter given, we are looking for the adapter with highest priority level that can read an/the annotation
        # file
        adapters = adapt.ANNOTATION_ADAPTERS
        adapterPriority = -1
        for f in imageFiles:
            for a in adapters:
                if a.canRead(os.path.join(rawDatasetPath, f)):
                    if a.getPriorityLevel() > adapterPriority:
                        adapterPriority = a.getPriorityLevel()
                        adapter = a
                        file = f
    else:
        # Using given adapter, we are looking for a file that can be read
        file = None
        for f in imageFiles:
            if adapter.canRead(os.path.join(rawDatasetPath, f)) and file is None:
                file = f

    # Getting the masks data
    masks = adapter.readFile(os.path.join(rawDatasetPath, file))

    # Creating masks
    for noMask, (datasetClass, maskPoints) in enumerate(masks):
        # Converting class id to class name if needed
        if type(datasetClass) is int:
            if classesInfo[datasetClass]["id"] == datasetClass:
                maskClass = classesInfo[datasetClass]["name"]
            else:
                for classInfo in classesInfo:
                    if classInfo["id"] == datasetClass:
                        maskClass = classInfo["name"]
                        break
        else:
            maskClass = datasetClass
        createMask(imgName, shape, noMask, maskPoints, datasetName, maskClass)


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


def cleanImage(datasetPath: str, imageName: str, medullaMinPart=0.05, onlyMasks=False):
    """
    Creating the full_images directory and cleaning the base image by removing non-cortex areas, creating medulla mask
    and cleaning background mask
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :param medullaMinPart: least part of image to represent medulla for it to be kept
    :param onlyMasks: if true, will not create full_images directory and clean image
    :return: None
    """
    # Defining all the useful paths
    currentImageDirPath = os.path.join(datasetPath, imageName)
    imagesDirPath = os.path.join(currentImageDirPath, 'images')
    imageFileName = os.listdir(imagesDirPath)[0]
    imagePath = os.path.join(imagesDirPath, imageFileName)
    image = cv2.imread(imagePath)

    # Getting the cortex image
    medulla = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cortexDirPath = os.path.join(currentImageDirPath, 'cortex')
    cortexPresent = os.path.exists(cortexDirPath)
    if cortexPresent:
        cortexFileName = os.listdir(cortexDirPath)[0]
        cortexFilePath = os.path.join(cortexDirPath, cortexFileName)
        if not onlyMasks:
            # Copying the full image into the correct directory
            fullImageDirPath = os.path.join(currentImageDirPath, 'full_images')
            os.makedirs(fullImageDirPath, exist_ok=True)
            cv2.imwrite(os.path.join(fullImageDirPath, imageFileName), image)

            # Cleaning the image
            cortex = cv2.imread(cortexFilePath)
            image = cv2.bitwise_and(image, cortex)
            cv2.imwrite(imagePath, image)

        # Starting medulla mask creation
        cortex = cv2.imread(cortexFilePath, cv2.IMREAD_UNCHANGED)
        invertedCortex = np.bitwise_not(cortex)
        medulla = cortex

    backgroundDirPath = os.path.join(currentImageDirPath, 'fond')
    if os.path.exists(backgroundDirPath):
        for backgroundFile in os.listdir(backgroundDirPath):
            # Second Medulla mask creation step
            backgroundFilePath = os.path.join(backgroundDirPath, backgroundFile)
            backgroundImage = cv2.imread(backgroundFilePath, cv2.IMREAD_UNCHANGED)
            medulla = np.bitwise_or(medulla, backgroundImage)

            # Cleaning background image if cortex mask is present
            if cortexPresent:
                # background = background && not(cortex)
                backgroundImage = np.bitwise_and(backgroundImage, invertedCortex)
                cv2.imwrite(backgroundFilePath, backgroundImage)

    # Last Medulla mask creation step if medulla > 1% of image
    black, white = getBWCount(medulla)
    bwRatio = black / (black + white)
    if bwRatio >= medullaMinPart:
        # medulla = !(background || cortex)
        medulla = np.bitwise_not(medulla)
        medullaDirPath = os.path.join(currentImageDirPath, 'medullaire')
        os.makedirs(medullaDirPath, exist_ok=True)
        medullaFilePath = os.path.join(medullaDirPath, imageName + '_medulla.png')
        cv2.imwrite(medullaFilePath, medulla)



def convertImage(inputImagePath: str, outputImagePath: str):
    """
    Convert an image from a format to another one
    :param inputImagePath: path to the initial image
    :param outputImagePath: path to the output image
    :return: None
    """
    image = cv2.imread(inputImagePath)
    cv2.imwrite(outputImagePath, image)


def getInfoRawDataset(rawDatasetPath: str, verbose=False, adapter: AnnotationAdapter = None):
    """
    Listing all available images, those with missing information
    :param verbose: whether or not print should be executed
    :param rawDatasetPath: path to the raw dataset folder
    :param adapter:
    :return: list of unique files names, list of available images names, list of missing images names,
    list of missing annotations names
    """
    names = []
    images = []  # list of image that can be used to compute masks
    missingImages = []  # list of missing images
    missingAnnotations = []  # list of missing annotations
    if adapter is None:
        formats = adapt.ANNOTATION_FORMAT
    else:
        formats = [adapter.getAnnotationFormat()]
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
            annotationsExist = False
            for ext in formats:
                annotationsExist = annotationsExist or os.path.exists(os.path.join(rawDatasetPath, name + '.' + ext))
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
    if verbose:
        # Displaying missing image files
        problem = False
        nbMissingImg = len(missingImages)
        if nbMissingImg > 0:
            problem = True
            print('Missing {} image{} : {}'.format(nbMissingImg, 's' if nbMissingImg > 1 else '', missingImages))

        # Displaying missing annotations files
        nbMissingAnnotations = len(missingAnnotations)
        if nbMissingAnnotations > 0:
            problem = True
            print('Missing {} annotation{} : {}'.format(nbMissingAnnotations, 's' if nbMissingAnnotations > 1 else '',
                                                        missingAnnotations))

        # Checking if there is file that is not image nor annotation
        nbImages = len(images)
        if len(names) - nbMissingImg - nbMissingAnnotations - nbImages != 0:
            problem = True
            print('Be careful, there are not only required dataset files in this folder')

        if not problem:
            print("Raw Dataset has no problem. Number of Images : {}".format(nbImages))
    return names, images, missingImages, missingAnnotations


def startWrapper(rawDatasetPath: str, datasetName: str = 'dataset_train', deleteBaseCortexMasks=False,
                 adapter: AnnotationAdapter = None):
    """
    Start wrapping the raw dataset into the wanted format
    :param deleteBaseCortexMasks: delete the base masks images after fusion
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param datasetName: name of the output dataset
    :param adapter: Adapter to use to read annotations, if None compatible adapter will be searched
    :return: None
    """
    names, images, missingImages, missingAnnotations = getInfoRawDataset(rawDatasetPath, verbose=True, adapter=adapter)

    nbImages = len(images)
    # Creating masks for any image which has all required files and displaying progress
    for index in range(nbImages):
        file = images[index]
        print('Creating masks for {} image {}/{} ({:.2f}%)'.format(file, index + 1, nbImages,
                                                                   (index + 1) / nbImages * 100))
        createMasksOfImage(rawDatasetPath, file, datasetName, adapter)
        fuseCortices(datasetName, file, deleteBaseMasks=deleteBaseCortexMasks)
        cleanImage(datasetName, file)
