import os
from shutil import move
import numpy as np

import datasetWrapper as dW
import datasetDivider as dD


def infoNephrologyDataset(datasetPath: str, silent=False):
    """
    Print information about a dataset
    :param datasetPath: path to the dataset
    :param silent: if true nothing will be printed
    :return: nbImg, histogram, cortexMissing, multiCortices, maxClasses, maxClassesNoCortex
    """
    print()
    histogram = {'tubule_atrophique': 0, 'vaisseau': 0, 'pac': 0, 'nsg_complet': 0,
                 'nsg_partiel': 0, 'tubule_sain': 0, 'cortex': 0}
    maxNbClasses = 0
    maxClasses = []
    maxNbClassesNoCortex = 0
    maxClassesNoCortex = []
    cortexMissing = []
    printedCortexMissing = []
    multiCortices = []
    nbImg = 0
    for imageDir in os.listdir(datasetPath):
        nbImg += 1
        imagePath = os.path.join(datasetPath, imageDir)
        cortex = False
        cortexDivided = False
        localHisto = {'tubule_atrophique': 0, 'vaisseau': 0, 'pac': 0, 'nsg_complet': 0,
                      'nsg_partiel': 0, 'tubule_sain': 0, 'cortex': 0}
        for maskDir in os.listdir(imagePath):
            if maskDir == 'cortex':
                cortex = True
                cortexDivided = len(os.listdir(os.path.join(os.path.join(datasetPath, imageDir), maskDir))) > 1
            if maskDir != "images" and maskDir != "full_images":
                histogram[maskDir] += 1
                localHisto[maskDir] += 1
        if not cortex:
            name = imageDir.split('_')[0] + "_*"
            if name not in printedCortexMissing:
                printedCortexMissing.append(name)
            cortexMissing.append(imageDir)

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
        print("\tMulti cortices ({}) : {}".format(len(multiCortices), multiCortices))
        print("\tMax Classes w/ cortex  ({}) :\t{}".format(maxNbClasses, maxClasses))
        print("\tMax Classes w/o cortex ({}) :\t{}".format(maxNbClassesNoCortex, maxClassesNoCortex))
    return nbImg, histogram, cortexMissing, multiCortices, maxClasses, maxClassesNoCortex


def separateCortexDataset(datasetPath: str, cortexDatasetPath: str = None):
    """
    Move each image folder with no cortex present to the cortex dataset folder
    :param datasetPath: path to the base dataset
    :param cortexDatasetPath: path to the cortex dataset
    :return: None
    """
    if cortexDatasetPath is None:
        cortexDatasetPath = datasetPath + '_cortex'

    _, _, toBeMoved, _, _, _ = infoNephrologyDataset(datasetPath, silent=True)
    os.makedirs(cortexDatasetPath, exist_ok=True)
    if len(toBeMoved) > 0:
        print("Moving {} non-cortex images directories into correct dataset".format(len(toBeMoved)))
        for imageWithoutCortexDir in toBeMoved:
            move(os.path.join(datasetPath, imageWithoutCortexDir),
                 os.path.join(cortexDatasetPath, imageWithoutCortexDir))


def createEvalDataset(datasetPath: str, evalDatasetPath: str = None, evalDatasetSizePart=0.1, evalDatasetMinSize=30, rename=False, customRename: str = None):
    """
    Create the evaluation dataset by moving a random set of base dataset's images
    :param datasetPath: path to the base dataset
    :param evalDatasetPath: path to the eval dataset
    :param evalDatasetSizePart: the part of the base dataset to be moved to the eval one
    :param evalDatasetMinSize: the minimum size of the evaluation dataset
    :param rename: whether or not you want to rename the base dataset after creation of the eval one
    :param customRename: new name of the training dataset
    :return: None
    """
    assert 0 < evalDatasetSizePart < 1
    if evalDatasetPath is None:
        evalDatasetPath = datasetPath + '_eval'

    fullList = os.listdir(datasetPath)
    evalDatasetSize = round(len(fullList) * evalDatasetSizePart)
    if evalDatasetSize < evalDatasetMinSize:
        evalDatasetSize = evalDatasetMinSize
    assert len(fullList) > evalDatasetSize
    toBeMoved = np.random.choice(fullList, evalDatasetSize, replace=False)

    os.makedirs(evalDatasetPath, exist_ok=True)
    if len(toBeMoved) > 0:
        print("Moving {} images directories into eval dataset".format(len(toBeMoved)))
        for dirName in toBeMoved:
            move(os.path.join(datasetPath, dirName), os.path.join(evalDatasetPath, dirName))
    if rename:
        newName = (datasetPath + '_train') if customRename is None else customRename
        move(datasetPath, newName)


dW.startWrapper('raw_dataset', 'temp_nephrology_dataset', deleteBaseCortexMasks=False)
infoNephrologyDataset('temp_nephrology_dataset')
dD.divideDataset('temp_nephrology_dataset', 'nephrology_dataset', squareSideLength=1024)
infoNephrologyDataset('nephrology_dataset')

# If you want to keep all cortex files comment dW.cleanCortexDir() lines
# If you want to check them and then delete them, comment these lines too and after checking use them
# dW.cleanCortexDir('temp_nephrology_dataset')
# dW.cleanCortexDir('nephrology_dataset')


separateCortexDataset('nephrology_dataset', 'nephrology_cortex_dataset')
createEvalDataset('nephrology_dataset', rename=True)
infoNephrologyDataset('nephrology_dataset_train')
infoNephrologyDataset('nephrology_cortex_dataset')
infoNephrologyDataset('nephrology_dataset_eval')
