import os
import cv2
import numpy as np
from datasetTools import AnnotationAdapter as adapt
from datasetTools.AnnotationAdapter import AnnotationAdapter
from datasetTools.datasetDivider import getBWCount, CV2_IMWRITE_PARAM
from mrcnn import utils

NEPHRO_CLASSES = [
    {"id": 0, "name": "Background", "color": "", "ignore": True},
    {"id": 1, "name": "tubule_sain", "color": "#ff007f", "ignore": False},
    {"id": 2, "name": "tubule_atrophique", "color": "#55557f", "ignore": False},
    {"id": 3, "name": "nsg_complet", "color": "#ff557f", "ignore": False},
    {"id": 4, "name": "nsg_partiel", "color": "#55aa7f", "ignore": False},
    {"id": 5, "name": "pac", "color": "#ffaa7f", "ignore": False},
    {"id": 6, "name": "vaisseau", "color": "#55ff7f", "ignore": False},
    {"id": 7, "name": "artefact", "color": "#000000", "ignore": False},
    {"id": 8, "name": "veine", "color": "#0000ff", "ignore": False},
    {"id": 9, "name": "nsg", "color": "#55007f", "ignore": False},
    {"id": 10, "name": "intima", "color": "#aa0000", "ignore": False},
    {"id": 11, "name": "media", "color": "#aa5500", "ignore": False}
]

CORTICES_CLASSES = [
    {"id": 0, "name": "Background", "color": "", "ignore": True},
    {"id": 1, "name": "cortex", "color": "#ffaa00", "ignore": False},
    {"id": 2, "name": "medullaire", "color": "#ff0000", "ignore": False},
    {"id": 3, "name": "capsule", "color": "#ff00ff", "ignore": False}
]


def createMask(imgName: str, imgShape, idMask: int, ptsMask, datasetName: str = 'dataset_train',
               maskClass: str = 'masks', imageFormat="jpg", config=None):
    """
    Create the mask image based on its polygon points
    :param imgName: name w/o extension of the base image
    :param imgShape: shape of the image
    :param idMask: the ID of the mask, a number not already used for that image
    :param ptsMask: array of [x, y] coordinates which are all the polygon points representing the mask
    :param datasetName: name of the output dataset
    :param maskClass: name of the associated class of the current mask
    :param imageFormat: output format of the masks' images
    :return: None
    """
    # Defining path where the result image will be stored and creating dir if not exists
    maskClass = maskClass.replace(" ", "_")
    output_directory = os.path.join(datasetName, imgName, maskClass)
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

    bbox_coordinates = ""
    if config is not None and config.USE_MINI_MASK:
        bbox = utils.extract_bboxes(mask)
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        mask = mask.astype(np.uint8) * 255
        y1, x1, y2, x2 = bbox
        bbox_coordinates = f"_{y1}_{x1}_{y2}_{x2}"
    # Saving result image
    output_name = f"{imgName}_{idMask:03d}{bbox_coordinates}.{imageFormat}"
    cv2.imwrite(os.path.join(output_directory, output_name), mask, CV2_IMWRITE_PARAM)


def getBboxFromName(imageName):
    """
    Return the bbox coordinates stored in the image name
    :param imageName: the image name from which you want the bbox
    :return: the bbox as [y1, x1, y2, x2]
    """
    res = os.path.basename(imageName)
    res = "".join(res.split('.')[:-1])
    res = res.split("_")[2:]
    return np.array([int(x) for x in res])


def resizeMasks(baseMasks, xRatio: float, yRatio: float):
    """
    Resize mask's base points to fit the targeted size
    :param baseMasks array of [x, y] coordinates which are all the polygon points representing the mask
    :param xRatio width ratio that will be applied to coordinates
    :param yRatio height ratio that will be applied to coordinates
    """
    res = []
    for pt in baseMasks:
        xTemp = float(pt[0])
        yTemp = float(pt[1])
        res.append([xTemp * xRatio, yTemp * yRatio])
    return res


def createMasksOfImage(rawDatasetPath: str, imgName: str, datasetName: str = 'dataset_train',
                       adapter: AnnotationAdapter = None, classesInfo: dict = None, imageFormat="jpg", resize=None,
                       config=None):
    """
    Create all the masks of a given image by parsing xml annotations file
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param imgName: name w/o extension of an image
    :param datasetName: name of the output dataset
    :param adapter: the annotation adapter to use to create masks, if None looking for an adapter that can read the file
    :param classesInfo: Information about all classes that are used, by default will be nephrology classes Info
    :param imageFormat: output format of the image and masks
    :param resize: if the image and masks have to be resized
    :param config: config class of Mask R-CNN to use mini masks if enabled
    :return: None
    """
    # Getting shape of original image (same for all this masks)
    if classesInfo is None:
        classesInfo = NEPHRO_CLASSES

    img = cv2.imread(os.path.join(rawDatasetPath, f"{imgName}.{imageFormat}"))
    if img is None:
        print(f'Problem with {imgName} image')
        return
    shape = img.shape
    if resize is not None:
        yRatio = resize[0] / shape[0]
        xRatio = resize[1] / shape[1]
        assert yRatio > 0 and xRatio > 0, f"Error resize ratio not correct ({yRatio:3.2f}, {xRatio:3.2f})"
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
        shape = img.shape

    # Copying the original image in the dataset
    targetDirectoryPath = os.path.join(datasetName, imgName, 'images')
    if not os.path.exists(targetDirectoryPath):
        os.makedirs(targetDirectoryPath)
        cv2.imwrite(os.path.join(targetDirectoryPath, f"{imgName}.{imageFormat}"), img, CV2_IMWRITE_PARAM)

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
            if datasetClass < len(classesInfo) and classesInfo[datasetClass]["id"] == datasetClass:
                maskClass = classesInfo[datasetClass]["name"]
            else:
                for classInfo in classesInfo:
                    if classInfo["id"] == datasetClass:
                        maskClass = classInfo["name"]
                        break
        else:
            maskClass = datasetClass
            if resize is not None:
                resizedMasks = resizeMasks(maskPoints, xRatio, yRatio)
        createMask(imgName, shape, noMask, maskPoints if resize is None else resizedMasks, datasetName, maskClass,
                   imageFormat, config=config)


def fuseCortices(datasetPath: str, imageName: str, imageFormat="jpg", deleteBaseMasks=False):
    """
    Fuse each cortex masks into one
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :param imageFormat: format to use to save the final cortex masks
    :param deleteBaseMasks: delete the base masks images after fusion
    :return: None
    """
    # Getting the image directory path
    imageDir = os.path.join(datasetPath, imageName)
    cortexDir = os.path.join(imageDir, 'cortex')
    imagePath = os.path.join(datasetPath, imageName, "images")
    imagePath = os.path.join(imagePath, os.listdir(imagePath)[0])
    image = cv2.imread(imagePath)
    if os.path.exists(cortexDir):
        listCortexImages = os.listdir(cortexDir)
        print("Fusing {} cortices masks".format(imageName))
        fusion = loadSameResImage(os.path.join(cortexDir, listCortexImages[0]), imageShape=image.shape)
        listCortexImages.remove(listCortexImages[0])
        for maskName in listCortexImages:  # Adding each mask to the same image
            maskPath = os.path.join(cortexDir, maskName)
            mask = loadSameResImage(maskPath, imageShape=image.shape)
            fusion = cv2.add(fusion, mask)
        # Saving the fused mask image
        cv2.imwrite(os.path.join(cortexDir, f"{imageName}_cortex.{imageFormat}"), fusion, CV2_IMWRITE_PARAM)
        if deleteBaseMasks:
            for maskName in os.listdir(cortexDir):  # Deleting each cortex mask except the fused one
                if f'_cortex.{imageFormat}' not in maskName:
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


def cleanImage(datasetPath: str, imageName: str, medullaMinPart=0.05, onlyMasks=False, imageFormat="jpg"):
    """
    Creating the full_images directory and cleaning the base image by removing non-cortex areas, creating medulla mask
    and cleaning background mask
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :param medullaMinPart: least part of image to represent medulla for it to be kept
    :param onlyMasks: if true, will not create full_images directory and clean image
    :param imageFormat: the image format to use to save the image
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
            cv2.imwrite(os.path.join(fullImageDirPath, imageFileName), image, CV2_IMWRITE_PARAM)

            # Cleaning the image and saving it
            cortex = loadSameResImage(cortexFilePath, image.shape)
            cortex = np.repeat(cortex[:, :, np.newaxis], 3, axis=2)
            image = cv2.bitwise_and(image, cortex)
            cv2.imwrite(imagePath, image, CV2_IMWRITE_PARAM)

        # Starting medulla mask creation
        cortex = loadSameResImage(cortexFilePath, image.shape)
        invertedCortex = np.bitwise_not(cortex)
        medulla = cortex

    backgroundDirPath = os.path.join(currentImageDirPath, 'fond')
    if os.path.exists(backgroundDirPath):
        for backgroundFile in os.listdir(backgroundDirPath):
            # Second Medulla mask creation step
            backgroundFilePath = os.path.join(backgroundDirPath, backgroundFile)
            backgroundImage = loadSameResImage(backgroundFilePath, image.shape)
            medulla = np.bitwise_or(medulla, backgroundImage)

            # Cleaning background image if cortex mask is present
            if cortexPresent:
                # background = background && not(cortex)
                backgroundImage = np.bitwise_and(backgroundImage, invertedCortex)
                cv2.imwrite(backgroundFilePath, backgroundImage, CV2_IMWRITE_PARAM)

    # Last Medulla mask creation step if medulla > 1% of image
    black, white = getBWCount(medulla)
    bwRatio = black / (black + white)
    if bwRatio >= medullaMinPart:
        # medulla = !(background || cortex)
        medulla = np.bitwise_not(medulla)
        medullaDirPath = os.path.join(currentImageDirPath, 'medullaire')
        os.makedirs(medullaDirPath, exist_ok=True)
        medullaFilePath = os.path.join(medullaDirPath, f"{imageName}_medulla.{imageFormat}")
        cv2.imwrite(medullaFilePath, medulla, CV2_IMWRITE_PARAM)


def loadSameResImage(imagePath, imageShape):
    mask = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if mask.shape[0] != imageShape[0] or mask.shape[1] != imageShape[1]:
        bbox = getBboxFromName(imagePath)
        mask = utils.expand_mask(bbox, mask, image_shape=imageShape)
        mask = mask.astype(np.uint8) * 255
    return mask


def convertImage(inputImagePath: str, outputImagePath: str):
    """
    Convert an image from a format to another one
    :param inputImagePath: path to the initial image
    :param outputImagePath: path to the output image
    :return: None
    """
    image = cv2.imread(inputImagePath)
    cv2.imwrite(outputImagePath, image, CV2_IMWRITE_PARAM)


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
    usedFormat = None
    if adapter is None:
        formats = adapt.ANNOTATION_FORMAT
    else:
        formats = [adapter.getAnnotationFormat()]
    if verbose:
        print("Listing files")
    for file in os.listdir(rawDatasetPath):
        name = file.split('.')[0]
        if name not in names:  # We want to do this only once per unique file name (without extension)
            names.append(name)

            # Testing if there is an jpg image with that name
            jpgPath = os.path.join(rawDatasetPath, name + '.jpg')
            jpgExists = os.path.exists(jpgPath)
            if usedFormat is None and jpgExists:
                usedFormat = "jpg"

            # Same thing with png format
            pngPath = os.path.join(rawDatasetPath, name + '.png')
            pngExists = os.path.exists(pngPath)
            if usedFormat is None and pngExists:
                usedFormat = "png"

            # Same thing with jp2 format
            jp2Path = os.path.join(rawDatasetPath, name + '.jp2')
            jp2Exists = os.path.exists(jp2Path)

            # Testing if annotation file exists for that name
            annotationsExist = False
            for ext in formats:
                annotationsExist = annotationsExist or os.path.exists(os.path.join(rawDatasetPath, name + '.' + ext))
            if pngExists or jpgExists or jp2Exists:  # At least one image exists
                if not annotationsExist:  # Annotations are missing
                    missingAnnotations.append(name)
                else:
                    if verbose:
                        print("Adding {} image".format(name))
                    outPath = pngPath if usedFormat == "png" else jpgPath
                    outExists = pngExists if usedFormat == "png" else jpgExists
                    if not outExists:  # Only jp2 exists
                        if verbose:
                            print(f"\tConverting to {usedFormat}")
                        convertImage(jpgPath if jpgExists else jp2Path if jp2Exists else pngPath, outPath)
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
    return names, images, missingImages, missingAnnotations, usedFormat


def startWrapper(rawDatasetPath: str, datasetName: str = 'dataset_train', deleteBaseCortexMasks=False,
                 adapter: AnnotationAdapter = None, resize=None, cortexMode=False, classesInfo=None):
    """
    Start wrapping the raw dataset into the wanted format
    :param deleteBaseCortexMasks: delete the base masks images after fusion
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param datasetName: name of the output dataset
    :param adapter: Adapter to use to read annotations, if None compatible adapter will be searched
    :param resize: If tuple given, the images and their masks will be resized to the tuple value
    :param cortexMode: If in cortex mode, will not clean the image nor fuse cortex masks
    :param classesInfo: Information about the classes that will be used
    :return: None
    """
    names, images, missingImages, missingAnnotations, usedFormat = getInfoRawDataset(rawDatasetPath, verbose=True,
                                                                                     adapter=adapter)
    if classesInfo is None:
        if cortexMode:
            classesInfo = CORTICES_CLASSES
        else:
            classesInfo = NEPHRO_CLASSES
    nbImages = len(images)
    # Creating masks for any image which has all required files and displaying progress
    for index in range(nbImages):
        file = images[index]
        print('Creating masks for {} image {}/{} ({:.2f}%)'.format(file, index + 1, nbImages,
                                                                   (index + 1) / nbImages * 100))
        createMasksOfImage(rawDatasetPath, file, datasetName, adapter, classesInfo=classesInfo, resize=resize,
                           imageFormat=usedFormat)
        if not cortexMode:
            fuseCortices(datasetName, file, deleteBaseMasks=deleteBaseCortexMasks)
            cleanImage(datasetName, file)
