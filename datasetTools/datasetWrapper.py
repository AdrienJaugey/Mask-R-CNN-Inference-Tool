import os
import shutil
from time import time

import cv2
import numpy as np
from skimage.io import imread

from common_utils import progressBar, formatTime
from datasetTools import AnnotationAdapter as adapt
from datasetTools.AnnotationAdapter import AnnotationAdapter
from datasetTools.datasetDivider import getBWCount, CV2_IMWRITE_PARAM
from mrcnn.Config import Config
from mrcnn.utils import extract_bboxes, expand_mask, minimize_mask

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

MESTC_GLOM_CLASSES = [
    {"id": 0, "name": "Background", "color": "", "ignore": True},
    {"id": 1, "name": "hile", "color": "#64FE2E", "ignore": False},
    {"id": 2, "name": "M", "color": "#55007f", "ignore": False},
    {"id": 3, "name": "E", "color": "#ff007f", "ignore": False},
    {"id": 4, "name": "S", "color": "#55557f", "ignore": False},
    {"id": 5, "name": "C", "color": "#ff557f", "ignore": False},
    {"id": 6, "name": "necrose_fib", "color": "#55aa7f", "ignore": False},
    {"id": 7, "name": "artefact", "color": "#ffaa7f", "ignore": False}
]


def createMask(imgName: str, imgShape, idMask: int, ptsMask, datasetName: str = 'dataset_train',
               maskClass: str = 'masks', imageFormat="jpg", config: Config = None):
    """
    Create the mask image based on its polygon points
    :param imgName: name w/o extension of the base image
    :param imgShape: shape of the image
    :param idMask: the ID of the mask, a number not already used for that image
    :param ptsMask: array of [x, y] coordinates which are all the polygon points representing the mask
    :param datasetName: name of the output dataset
    :param maskClass: name of the associated class of the current mask
    :param imageFormat: output format of the masks' images
    :param config: config object
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
    if config is not None and config.is_using_mini_mask():
        bbox = extract_bboxes(mask)
        mask = minimize_mask(bbox, mask, config.get_mini_mask_shape())
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
    lastParts = os.path.splitext(os.path.basename(imageName))[0].split('_')[-4:]
    return np.array([int(x) for x in lastParts])


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
                       config: Config = None):
    """
    Create all the masks of a given image by parsing xml annotations file
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param imgName: name w/o extension of an image
    :param datasetName: name of the output dataset
    :param adapter: the annotation adapter to use to create masks, if None looking for an adapter that can read the file
    :param classesInfo: Information about all classes that are used, by default will be nephrology classes Info
    :param imageFormat: output format of the image and masks
    :param resize: if the image and masks have to be resized
    :param config: config object
    :return: None
    """
    # Getting shape of original image (same for all this masks)
    if classesInfo is None:
        classesInfo = NEPHRO_CLASSES if config is None else config.get_classes_info()

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
        # TODO use file copy if unchanged else cv2
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
        adapters = list(adapt.ANNOTATION_ADAPTERS.values())
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
            if maskClass == "None":
                print(f" /!\\ {imgName} : None class present /!\\ ")
            if resize is not None:
                resizedMasks = resizeMasks(maskPoints, xRatio, yRatio)
        createMask(imgName, shape, noMask, maskPoints if resize is None else resizedMasks, datasetName, maskClass,
                   imageFormat, config=config)


def fuseClassMasks(datasetPath: str, imageName: str, targetedClass, imageFormat="jpg", deleteBaseMasks=False,
                   silent=False):
    """
    Fuse each cortex masks into one
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :param targetedClass: the class of the masks that have to be fused
    :param imageFormat: format to use to save the final cortex masks
    :param deleteBaseMasks: delete the base masks images after fusion
    :param silent: if True will not print
    :return: None
    """
    # Getting the image directory path
    imageDir = os.path.join(datasetPath, imageName)
    classDir = os.path.join(imageDir, targetedClass)
    imagePath = os.path.join(datasetPath, imageName, "images")
    imagePath = os.path.join(imagePath, os.listdir(imagePath)[0])
    image = cv2.imread(imagePath)
    if os.path.exists(classDir):
        listCortexImages = os.listdir(classDir)
        if not silent:
            print(f"Fusing {imageName} {targetedClass} class masks")
        fusion = loadSameResImage(os.path.join(classDir, listCortexImages[0]), imageShape=image.shape)
        listCortexImages.remove(listCortexImages[0])
        for maskName in listCortexImages:  # Adding each mask to the same image
            maskPath = os.path.join(classDir, maskName)
            mask = loadSameResImage(maskPath, imageShape=image.shape)
            fusion = cv2.add(fusion, mask)
        # Saving the fused mask image
        cv2.imwrite(os.path.join(classDir, f"{imageName}_{targetedClass}.{imageFormat}"), fusion, CV2_IMWRITE_PARAM)
        if deleteBaseMasks:
            for maskName in os.listdir(classDir):  # Deleting each cortex mask except the fused one
                if f'_{targetedClass}.{imageFormat}' not in maskName:
                    maskPath = os.path.join(classDir, maskName)
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


def cleanImage(datasetPath: str, imageName: str, cleaningClass: str, imageFormat="jpg", cleanMasks=False,
               minAreaThreshold=300, config: Config = None):
    """
    Creating the full_images directory and cleaning the base image by removing non-cleaning-class areas
    :param datasetPath: the dataset that have been wrapped
    :param imageName: the image you want its cortex to be fused
    :param cleaningClass: the class to use to clean the image
    :param cleanMasks: if true, will clean masks based on the cleaning-class-mask
    :param imageFormat: the image format to use to save the image
    :param minAreaThreshold: remove mask if its area is smaller than this threshold
    :param config: config object
    :return: None
    """
    assert cleaningClass is not None and cleaningClass != "", "Cleaning class is required."

    # Getting the base image
    path = os.path.join(datasetPath, imageName, '{folder}', f"{imageName}.{imageFormat}")
    imagePath = path.format(folder='images')
    fullImagePath = path.format(folder='full_images')
    image = cv2.imread(imagePath)

    # Fusing all the cleaning-class masks and then cleaning the image and if needed the masks
    cleaningClassDirPath = os.path.join(datasetPath, imageName, cleaningClass)
    cleaningClassExists = os.path.exists(cleaningClassDirPath)
    if cleaningClassExists and len(os.listdir(cleaningClassDirPath)) > 0:
        first = True
        for cleaningClassMaskName in os.listdir(cleaningClassDirPath):
            cleaningClassMaskPath = os.path.join(cleaningClassDirPath, cleaningClassMaskName)
            if first:
                cleaningClassMasks = loadSameResImage(cleaningClassMaskPath, image.shape)
                first = False
            else:  # Adding additional masks
                temp = loadSameResImage(cleaningClassMaskPath, image.shape)
                cleaningClassMasks = cv2.bitwise_or(cleaningClassMasks, temp)
                del temp

        # Copying the full image into the correct directory
        os.makedirs(os.path.dirname(fullImagePath), exist_ok=True)
        shutil.copy2(imagePath, fullImagePath)

        # Cleaning the image and saving it
        image = cv2.bitwise_and(image, np.repeat(cleaningClassMasks[:, :, np.newaxis], 3, axis=2))
        cv2.imwrite(imagePath, image, CV2_IMWRITE_PARAM)

        # Cleaning masks so that they cannot exist elsewhere
        if cleanMasks:
            folderToRemove = []
            for folder in os.listdir(os.path.join(datasetPath, imageName)):
                folderPath = os.path.join(datasetPath, imageName, folder)
                # Checking only for the other classes folder
                if os.path.isdir(folderPath) and folder not in [cleaningClass, "images", "full_images"]:
                    # For each mask of the folder
                    for maskImageFileName in os.listdir(folderPath):
                        maskImagePath = os.path.join(folderPath, maskImageFileName)
                        mask = loadSameResImage(maskImagePath, image.shape)
                        areaBefore = getBWCount(mask)[1]

                        # If mask is not empty
                        if areaBefore > 0:
                            # Cleaning it with the cleaning-class masks
                            mask = cv2.bitwise_and(mask, cleaningClassMasks)
                            areaAfter = getBWCount(mask)[1]
                        else:
                            areaAfter = areaBefore

                        # If mask was empty or too small after cleaning, we remove it
                        if areaBefore == 0 or areaAfter < minAreaThreshold:
                            os.remove(maskImagePath)
                        elif areaBefore != areaAfter:
                            # If mask has is different after cleaning, we replace the original one
                            try:
                                try:
                                    idMask = int(maskImageFileName.split('.')[0].split('_')[1])
                                except ValueError:
                                    # If we could not retrieve the original mask ID, give it a unique one
                                    idMask = int(time())

                                # If mini-mask are enabled, we minimize it before saving it
                                bbox_coordinates = ""
                                if config is not None and config.is_using_mini_mask():
                                    bbox = extract_bboxes(mask)
                                    mask = minimize_mask(bbox, mask, config.get_mini_mask_shape())
                                    mask = mask.astype(np.uint8) * 255
                                    y1, x1, y2, x2 = bbox
                                    bbox_coordinates = f"_{y1}_{x1}_{y2}_{x2}"

                                # Saving cleaned mask
                                outputName = f"{imageName}_{idMask:03d}{bbox_coordinates}.{imageFormat}"
                                cv2.imwrite(os.path.join(folderPath, outputName), mask, CV2_IMWRITE_PARAM)
                                if outputName != maskImageFileName:  # Remove former mask if not the same name
                                    os.remove(maskImagePath)
                            except Exception:
                                print(f"Error on {maskImagePath} update")

                    if len(os.listdir(folderPath)) == 0:
                        folderToRemove.append(folderPath)
            for folderPath in folderToRemove:
                shutil.rmtree(folderPath, ignore_errors=True)
            pass


def loadSameResImage(imagePath, imageShape):
    mask = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if mask.shape[0] != imageShape[0] or mask.shape[1] != imageShape[1]:
        bbox = getBboxFromName(imagePath)
        mask = expand_mask(bbox, mask, image_shape=imageShape)
        mask = mask.astype(np.uint8) * 255
    return mask


def convertImage(inputImagePath: str, outputImagePath: str):
    """
    Convert an image from a format to another one
    :param inputImagePath: path to the initial image
    :param outputImagePath: path to the output image
    :return: None
    """
    image = imread(inputImagePath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outputImagePath, image, CV2_IMWRITE_PARAM)


def getInfoRawDataset(rawDatasetPath: str, verbose=False, adapter: AnnotationAdapter = None, mainFormat="jpg"):
    """
    Listing all available images, those with missing information
    :param verbose: whether or not print should be executed
    :param rawDatasetPath: path to the raw dataset folder
    :param adapter:
    :param mainFormat: the format to use for the dataset
    :return: list of unique files names, list of available images names, list of missing images names,
    list of missing annotations names
    """
    names = []
    images = []  # list of image that can be used to compute masks
    missingImages = []  # list of missing images
    missingAnnotations = []  # list of missing annotations
    inputFormats = ["jpg", "jp2", "png"]
    if adapter is None:
        annotationFormats = adapt.ANNOTATION_FORMAT
    else:
        annotationFormats = [adapter.getAnnotationFormat()]
    fileList = os.listdir(rawDatasetPath)
    if verbose:
        progressBar(0, len(fileList), "Listing files")
    for idx, file in enumerate(fileList):
        if verbose and (idx % 10 == 0 or idx + 1 == len(fileList)):
            lastIdx = idx
            progressBar(idx + 1, len(fileList), "Listing files")
        name = file.split('.')[0]
        if name not in names:  # We want to do this only once per unique file name (without extension)
            names.append(name)

            availableFormat = []
            for format in inputFormats:
                imgPath = os.path.join(rawDatasetPath, f"{name}.{format}")
                if os.path.exists(imgPath):
                    availableFormat.append(format)

            # Testing if annotation file exists for that name
            annotationsExist = False
            for ext in annotationFormats:
                annotationsExist = annotationsExist or os.path.exists(os.path.join(rawDatasetPath, f"{name}.{ext}"))
            if len(availableFormat) > 0:  # At least one image exists
                if not annotationsExist:  # Annotations are missing
                    missingAnnotations.append(name)
                else:
                    if mainFormat not in availableFormat:
                        for format in inputFormats:
                            if format in availableFormat:
                                sourcePath = os.path.join(rawDatasetPath, f"{name}.{format}")
                                outputPath = os.path.join(rawDatasetPath, f"{name}.{mainFormat}")
                                convertImage(sourcePath, outputPath)
                                break
                    images.append(name)  # Adding this image to the list
            elif annotationsExist:  # There is no image file but xml found
                missingImages.append(name)
    if verbose:
        if lastIdx + 1 != len(fileList):
            progressBar(1, 1, "Listing files")
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
                 adapter: AnnotationAdapter = None, resize=None, mode="main", classesInfo=None, imageFormat="jpg"):
    """
    Start wrapping the raw dataset into the wanted format
    :param rawDatasetPath: path to the folder containing images and associated annotations
    :param datasetName: name of the output dataset
    :param deleteBaseCortexMasks: delete the base masks images after fusion
    :param adapter: Adapter to use to read annotations, if None compatible adapter will be searched
    :param resize: If tuple given, the images and their masks will be resized to the tuple value
    :param mode: Mode to use, will not clean the image nor fuse cortex masks
    :param classesInfo: Information about the classes that will be used
    :param imageFormat: the image format to use in the dataset
    :return: None
    """
    names, images, missingImages, missingAnnotations = getInfoRawDataset(rawDatasetPath, verbose=True,
                                                                         adapter=adapter, mainFormat=imageFormat)
    if classesInfo is None:
        if mode == 'main':
            classesInfo = NEPHRO_CLASSES
        elif mode == "cortex":
            classesInfo = CORTICES_CLASSES
        elif mode == "mestc_glom":
            classesInfo = MESTC_GLOM_CLASSES
    # Creating masks for any image which has all required files and displaying progress
    start_time = time()
    for index, file in enumerate(images):
        progressBar(index, len(images), prefix='Creating masks',
                    suffix=f" {formatTime(round(time() - start_time))} Current : {file}")
        createMasksOfImage(rawDatasetPath, file, datasetName, adapter, classesInfo=classesInfo,
                           resize=resize, imageFormat=imageFormat)
        if mode == "main":
            fuseClassMasks(datasetName, file, cortex, deleteBaseMasks=deleteBaseCortexMasks, silent=True)
            cleanImage(datasetName, file, cleaningClass='cortex')
        elif mode == "mest_glom":
            cleanImage(datasetName, file, cleaningClass='nsg', cleanMasks=True)
    progressBar(1, 1, prefix='Creating masks', suffix=f"{formatTime(round(time() - start_time))}" + " " * 25)
