import os
from shutil import move, copy, copytree
import numpy as np

from datasetTools import datasetWrapper as dW, AnnotationAdapter
from datasetTools import datasetDivider as dD
from datasetTools.ASAPAdapter import ASAPAdapter
from datasetTools.LabelMeAdapter import LabelMeAdapter


def infoNephrologyDataset(datasetPath: str, silent=False):
    """
    Print information about a dataset
    :param datasetPath: path to the dataset
    :param silent: if true nothing will be printed
    :return: nbImg, histogram, cortexMissing, multiCortices, maxClasses, maxClassesNoCortex
    """
    print()
    histogram = {}
    maxNbClasses = 0
    maxClasses = []
    maxNbClassesNoCortex = 0
    maxClassesNoCortex = []
    cortexMissing = []
    printedCortexMissing = []
    multiCortices = []
    missingDataImages = []
    nbImg = 0
    for imageDir in os.listdir(datasetPath):
        nbImg += 1
        imagePath = os.path.join(datasetPath, imageDir)
        cortex = False
        cortexDivided = False
        localHisto = {}
        missingData = True
        for maskDir in os.listdir(imagePath):
            if maskDir == 'cortex':
                cortex = True
                missingData = False
                cortexDivided = len(os.listdir(os.path.join(os.path.join(datasetPath, imageDir), maskDir))) > 1
            if maskDir not in ["images", "full_images"]:
                # Removing spaces, this should not happen actually but it did
                if maskDir in ["medullaire", "fond"]:
                    missingData = False
                if " " in maskDir:
                    newMaskDir = maskDir.replace(" ", "_")
                    maskDirPath = os.path.join(imagePath, maskDir)
                    newMaskDirPath = os.path.join(imagePath, newMaskDir)
                    move(maskDirPath, newMaskDirPath)
                    maskDir = newMaskDir
                if maskDir not in histogram.keys():
                    histogram[maskDir] = 0
                histogram[maskDir] += 1
                if maskDir not in localHisto.keys():
                    localHisto[maskDir] = 0
                localHisto[maskDir] += 1
        if not cortex:
            if "_" in imageDir:
                name = imageDir.split('_')[0] + "_*"
            else:
                name = imageDir
            if name not in printedCortexMissing:
                printedCortexMissing.append(name)
            cortexMissing.append(imageDir)
        if missingData:
            missingDataImages.append(name)
        if cortexDivided:
            multiCortices.append(imageDir)

        nbClasses = 0
        nbClassesNoCortex = 0
        for objectClass in localHisto:
            if localHisto[objectClass] > 0:
                if objectClass != 'cortex':
                    nbClassesNoCortex += 1
                nbClasses += 1
        if nbClasses >= maxNbClasses:
            if nbClasses > maxNbClasses:
                maxNbClasses = nbClasses
                maxClasses = []
            maxClasses.append(imageDir)

        if nbClassesNoCortex >= maxNbClassesNoCortex:
            if nbClassesNoCortex > maxNbClassesNoCortex:
                maxNbClassesNoCortex = nbClassesNoCortex
                maxClassesNoCortex = []
            maxClassesNoCortex.append(imageDir)

    if not silent:
        print("{} dataset Informations :".format(datasetPath))
        print("\tNb Images : {}".format(nbImg))
        print("\tHistogram : {}".format(histogram))
        print("\tMissing cortices ({}) : {}".format(len(cortexMissing), printedCortexMissing))
        print("\tMissing data ({}) : {}".format(len(missingDataImages), missingDataImages))
        print("\tMulti cortices ({}) : {}".format(len(multiCortices), multiCortices))
        print("\tMax Classes w/ cortex  ({}) :\t{}".format(maxNbClasses, maxClasses))
        print("\tMax Classes w/o cortex ({}) :\t{}".format(maxNbClassesNoCortex, maxClassesNoCortex))
    return nbImg, histogram, cortexMissing, multiCortices, maxClasses, maxClassesNoCortex, missingDataImages


def sortImages(datasetPath: str, createCortexDataset=False, cortexDatasetPath: str = None, unusedDirPath: str = None):
    """
    Move images that cannot be used in main training/inference, can also create cortex dataset
    :param datasetPath: path to the base dataset
    :param createCortexDataset: whether the cortex dataset should be created or not
    :param cortexDatasetPath: path to the cortex dataset
    :param unusedDirPath: path to the directory where unused files will be moved
    :return: None
    """
    # Setting paths if not given
    if unusedDirPath is None:
        unusedDirPath = datasetPath + '_unused'

    TO_UNCOUNT = ['images', 'full_images', 'cortex', 'medullaire', 'background']
    if createCortexDataset and cortexDatasetPath is None:
        cortexDatasetPath = datasetPath + '_cortex'

    # Getting list of images directories without data
    info = infoNephrologyDataset(datasetPath, silent=True)
    noCortex = info[2]
    noData = info[6]
    toBeMoved = []
    toBeMoved.extend(noData)
    toBeMoved.extend(noCortex)

    # For each image directory that is not already in toBeMoved list
    for imageDir in os.listdir(datasetPath):
        # If image has no cortex, medulla nor background, skip it
        if imageDir in noData:
            continue

        masksDirList = os.listdir(os.path.join(datasetPath, imageDir))
        if imageDir not in noCortex:
            for uncount in TO_UNCOUNT:
                if uncount in masksDirList:
                    masksDirList.remove(uncount)
            if len(masksDirList) == 0:
                toBeMoved.append(imageDir)

        if createCortexDataset:
            # Init : getting paths & making dest dir
            imageDirPath = os.path.join(datasetPath, imageDir)
            dstDirPath = os.path.join(cortexDatasetPath, imageDir)
            destImagesDirPath = os.path.join(dstDirPath, 'images')
            os.makedirs(destImagesDirPath, exist_ok=True)

            # Copy image from full_images (images if not present) dir to dest
            imageFileName = imageDir + '.png'
            if os.path.exists(os.path.join(imageDirPath, 'full_images')):
                srcImageFilePath = os.path.join(imageDirPath, 'full_images', imageFileName)
            else:
                srcImageFilePath = os.path.join(imageDirPath, 'images', imageFileName)
            fullImageDstPath = os.path.join(destImagesDirPath, imageFileName)
            copy(srcImageFilePath, fullImageDstPath)

            # Copy cortex directory to dest if exists
            cortexDirPath = os.path.join(imageDirPath, 'cortex')
            if os.path.exists(cortexDirPath):
                copytree(cortexDirPath, os.path.join(dstDirPath, 'cortex'))

            # Move background & medulla directories to dest if present
            medullaDirPath = os.path.join(imageDirPath, 'medullaire')
            if os.path.exists(medullaDirPath):
                move(medullaDirPath, dstDirPath)
            backgroundDirPath = os.path.join(imageDirPath, 'fond')
            if os.path.exists(backgroundDirPath):
                move(backgroundDirPath, dstDirPath)

    # Moving directories that will not be used in main dataset
    if len(toBeMoved) > 0:
        os.makedirs(unusedDirPath, exist_ok=True)
        print("Moving {} non-usable images directories into correct folder".format(len(toBeMoved)))
        for imageWithoutCortexDir in toBeMoved:
            srcPath = os.path.join(datasetPath, imageWithoutCortexDir)
            dstPath = os.path.join(unusedDirPath, imageWithoutCortexDir)
            move(srcPath, dstPath)


def createValDataset(datasetPath: str, valDatasetPath: str = None, valDatasetSizePart=0.1, valDatasetMinSize=30,
                     rename=False, customRename: str = None):
    """
    Create the validation dataset by moving a random set of base dataset's images
    :param datasetPath: path to the base dataset
    :param valDatasetPath: path to the val dataset
    :param valDatasetSizePart: the part of the base dataset to be moved to the val one
    :param valDatasetMinSize: the minimum size of the validation dataset
    :param rename: whether you want to rename the base dataset after creation of the val one
    :param customRename: the new name of the training dataset
    :return: None
    """
    assert 0 < valDatasetSizePart < 1
    if valDatasetPath is None:
        valDatasetPath = datasetPath + '_val'

    fullList = os.listdir(datasetPath)
    valDatasetSize = round(len(fullList) * valDatasetSizePart)
    if valDatasetSize < valDatasetMinSize:
        valDatasetSize = valDatasetMinSize
    assert len(fullList) > valDatasetSize
    toBeMoved = np.random.choice(fullList, valDatasetSize, replace=False)

    os.makedirs(valDatasetPath, exist_ok=True)
    if len(toBeMoved) > 0:
        print("Moving {} images directories into val dataset".format(len(toBeMoved)))
        for dirName in toBeMoved:
            move(os.path.join(datasetPath, dirName), os.path.join(valDatasetPath, dirName))
    if rename:
        newName = (datasetPath + '_train') if customRename is None else customRename
        move(datasetPath, newName)


def checkNSG(datasetPath: str):
    totalDiff = 0
    for imageDir in os.listdir(datasetPath):
        dirPath = os.path.join(datasetPath, imageDir)
        nsgDirectoriesPath = [os.path.join(dirPath, 'nsg'), os.path.join(dirPath, 'nsg_partiel'),
                              os.path.join(dirPath, 'nsg_complet')]
        count = [0, 0, 0]
        for index, dir in enumerate(nsgDirectoriesPath):
            if os.path.exists(dir):
                count[index] = len(os.listdir(dir))
        nsg = count[0]
        completAndPartiel = count[1] + count[2]
        if nsg != completAndPartiel:
            diff = abs(count[0] - count[1] - count[2])
            print(
                "{} : {} {} manquant{}".format(imageDir, diff, 'nsg' if nsg < completAndPartiel else 'complet/partiel',
                                               's' if diff > 1 else ''))
            totalDiff += diff
    print("Total : {}".format(totalDiff))


def createDataset(rawDataset='raw_dataset', tempDataset='temp_dataset',
                  cortexDatasetPath='nephrology_cortex_dataset', unusedDirPath='nephrology_dataset_unused',
                  mainDataset='main_dataset', mainDatasetUnusedDirPath='main_dataset_unused',
                  deleteBaseCortexMasks=True, createCortexDataset=False, adapter: AnnotationAdapter = None,
                  separateDivInsteadOfImage=False, divisionSize=1024, minDivisionOverlapping=0.33,
                  cleanBeforeStart=False):
    """
    Generates datasets folder from a base directory, all paths are customizable, and it can also remove previous
    directories
    :param rawDataset: path to the base directory
    :param tempDataset: path to a temporary directory
    :param cortexDatasetPath: path to the cortex directory, used to also define cortex training and validation directories
    :param unusedDirPath: path to the unused files' directory
    :param mainDataset: path to the main dataset directory, used to also define main training and validation directories
    :param mainDatasetUnusedDirPath: path to unused files' directory of main dataset
    :param deleteBaseCortexMasks: whether to delete base cortex masks or not
    :parma createCortexDataset: whether to create cortex dataset or not, default is False
    :param adapter: the adapter used to read annotations files, if None, will detect automatically which one to use
    :param separateDivInsteadOfImage: if True, divisions of same image can be separated into training and val directories
    :param divisionSize: the size of a division, default is 1024
    :param minDivisionOverlapping: the min overlapping between two divisions, default is 33%
    :param cleanBeforeStart: if True, will delete previous directories that could still exist
    :return:
    """
    if cleanBeforeStart:
        # Removing temp directories
        import shutil
        dirToDel = [tempDataset, unusedDirPath,
                    cortexDatasetPath, cortexDatasetPath + '_train', cortexDatasetPath + '_val',
                    'temp_' + mainDataset + '_val', mainDataset + '_val', mainDataset + '_train']
        for directory in dirToDel:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)

    dW.getInfoRawDataset(rawDataset, verbose=True, adapter=adapter)
    # Creating masks and making per image directories
    dW.startWrapper(rawDataset, tempDataset, deleteBaseCortexMasks=deleteBaseCortexMasks, adapter=adapter)
    infoNephrologyDataset(tempDataset)
    checkNSG(tempDataset)

    # Sorting images to keep those that can be used to train cortex
    sortImages(datasetPath=tempDataset,
               createCortexDataset=createCortexDataset, cortexDatasetPath=cortexDatasetPath,
               unusedDirPath=unusedDirPath)
    if createCortexDataset:
        infoNephrologyDataset(cortexDatasetPath)

        # Taking some images from the cortex dataset to make the validation cortex dataset
        createValDataset(cortexDatasetPath, rename=True)
        infoNephrologyDataset(cortexDatasetPath + '_train')
        infoNephrologyDataset(cortexDatasetPath + '_val')

    if separateDivInsteadOfImage:
        # Dividing main dataset in 1024*1024 divisions
        dD.divideDataset(tempDataset, mainDataset,
                         squareSideLength=divisionSize, min_overlap_part=minDivisionOverlapping)
        infoNephrologyDataset(mainDataset)

        # # If you want to keep all cortex files comment dW.cleanCortexDir() lines
        # # If you want to check them and then delete them, comment these lines too and after checking use them
        # # dW.cleanCortexDir(tempDataset)
        # # dW.cleanCortexDir(mainDataset)

        # Removing unusable images by moving them into a specific directory
        sortImages(mainDataset, unusedDirPath=mainDatasetUnusedDirPath)
        # Taking some images from the main dataset to make the validation dataset
        createValDataset(mainDataset, rename=True)
    else:  # To avoid having divisions of same image to be dispatched in main and validation dataset
        # Removing unusable images by moving them into a specific directory
        sortImages(tempDataset, unusedDirPath=mainDatasetUnusedDirPath)
        # Taking some images from the main dataset to make the validation dataset
        createValDataset(tempDataset, valDatasetPath='temp_' + mainDataset + '_val', rename=False)

        # Dividing the main dataset after having separated images for the validation dataset
        # then removing unusable divisions
        dD.divideDataset(tempDataset, mainDataset + '_train',
                         squareSideLength=divisionSize, min_overlap_part=minDivisionOverlapping)
        sortImages(mainDataset + '_train', unusedDirPath=mainDatasetUnusedDirPath)

        # Same thing with the validation dataset directly
        dD.divideDataset('temp_' + mainDataset + '_val', mainDataset + '_val',
                         squareSideLength=divisionSize, min_overlap_part=minDivisionOverlapping)
        sortImages(mainDataset + '_val', unusedDirPath=mainDatasetUnusedDirPath)

    infoNephrologyDataset(mainDataset + '_train')
    infoNephrologyDataset(mainDataset + '_val')
    print("\nDataset made, nothing left to do")


if __name__ == "__main__":
    createDataset(mainDataset='nephrology_dataset', mainDatasetUnusedDirPath='nephrology_dataset_unused',
                  createCortexDataset=False, cleanBeforeStart=True)
