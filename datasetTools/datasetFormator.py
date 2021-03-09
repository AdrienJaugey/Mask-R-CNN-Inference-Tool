import json
import os
from shutil import move, copy, copytree
import numpy as np

from common_utils import formatDate
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
                cortexDivided = len(os.listdir(os.path.join(datasetPath, imageDir, maskDir))) > 1
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
                histogram[maskDir] += len(os.listdir(os.path.join(datasetPath, imageDir, maskDir)))
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


def infoPatients(rawDataset, mode: str = "main"):
    names = dW.getInfoRawDataset(rawDatasetPath=rawDataset)[0]
    patients = []
    patients_biopsy = []
    patients_nephrectomy = []
    for name in names:
        patient = name[2:6]
        if patient not in patients:
            patients.append(patient)
        if mode == "main":
            biopsie = name[6] == "B"
            if biopsie:
                if patient not in patients_biopsy:
                    patients_biopsy.append(patient)
            elif patient not in patients_nephrectomy:
                patients_nephrectomy.append(patient)
    patients.sort()
    patients_biopsy.sort()
    patients_nephrectomy.sort()
    return patients, patients_biopsy, patients_nephrectomy


def selectPatients(patientsBiopsie, patientsNephrectomie, nbPatientBiopsie=5, nbPatientNephrectomie=4):
    communs = []
    for pb in patientsBiopsie:
        for pn in patientsNephrectomie:
            if pb == pn and pb not in communs:
                communs.append(pb)
    for patient in communs:
        patientsBiopsie.remove(patient)
        patientsNephrectomie.remove(patient)

    patientsBiopsie = np.array(patientsBiopsie)
    patientsNephrectomie = np.array(patientsNephrectomie)
    if max(len(communs), nbPatientBiopsie, nbPatientNephrectomie) == len(communs):
        communs = np.array(communs)
        selected = np.random.choice(communs, size=max(nbPatientBiopsie, nbPatientNephrectomie), replace=False)
    else:
        selected = communs.copy()
        nbPatientBiopsie -= len(communs)
        if nbPatientBiopsie > 0:
            temp = np.random.choice(patientsBiopsie, size=nbPatientBiopsie, replace=False)
            for patient in temp:
                selected.append(patient)
        nbPatientNephrectomie -= len(communs)
        if nbPatientNephrectomie > 0:
            temp = np.random.choice(patientsNephrectomie, size=nbPatientNephrectomie, replace=False)
            for patient in temp:
                selected.append(patient)
    return selected


def sortImages(datasetPath: str, unusedDirPath: str = None, mode: str = "main"):
    """
    Move images that cannot be used in main training/inference, can also create cortex dataset
    :param datasetPath: path to the base dataset
    :param unusedDirPath: path to the directory where unused files will be moved
    :param mode: Choosing between sort for main segmentation or sort for cortices segmentation
    :return: None
    """
    # Setting paths if not given
    if unusedDirPath is None:
        unusedDirPath = datasetPath + '_unused'

    NOT_TO_COUNT = ['images', 'full_images']
    if mode == "main":
        NOT_TO_COUNT.extend(['cortex', 'medullaire', 'capsule'])
    else:
        NOT_TO_COUNT.extend(["nsg", "nsg_complet", "nsg_partiel", "tubule_sain", "tubule_atrophique", "vaisseau",
                             "intima", "media", "pac", "artefact", "veine"])

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
        if imageDir in toBeMoved:
            continue

        masksDirList = os.listdir(os.path.join(datasetPath, imageDir))
        for uncount in NOT_TO_COUNT:
            if uncount in masksDirList:
                masksDirList.remove(uncount)
        if len(masksDirList) == 0:
            toBeMoved.append(imageDir)

    # Moving directories that will not be used in main dataset
    if len(toBeMoved) > 0:
        os.makedirs(unusedDirPath, exist_ok=True)
        print("Moving {} non-usable images directories into correct folder".format(len(toBeMoved)))
        for imageWithoutCortexDir in toBeMoved:
            srcPath = os.path.join(datasetPath, imageWithoutCortexDir)
            dstPath = os.path.join(unusedDirPath, imageWithoutCortexDir)
            move(srcPath, dstPath)


def createValDataset(datasetPath: str, valDatasetPath: str = None, valDatasetSizePart=0.1, valDatasetMinSize=30,
                     rename=False, customRename: str = None, recreateInfo=None):
    """
    Create the validation dataset by moving a random set of base dataset's images
    :param datasetPath: path to the base dataset
    :param valDatasetPath: path to the val dataset
    :param valDatasetSizePart: the part of the base dataset to be moved to the val one
    :param valDatasetMinSize: the minimum size of the validation dataset
    :param rename: whether you want to rename the base dataset after creation of the val one
    :param customRename: the new name of the training dataset
    :param recreateInfo: list of images to use to create val dataset
    :return: None
    """
    assert 0 < valDatasetSizePart < 1
    if valDatasetPath is None:
        valDatasetPath = datasetPath + '_val'

    if recreateInfo is None or len(recreateInfo) == 0:
        fullList = os.listdir(datasetPath)
        valDatasetSize = max(valDatasetMinSize, round(len(fullList) * valDatasetSizePart))
        assert len(fullList) > valDatasetSize
        toBeMoved = np.random.choice(fullList, valDatasetSize, replace=False)
    else:
        toBeMoved = recreateInfo

    os.makedirs(valDatasetPath, exist_ok=True)
    if len(toBeMoved) > 0:
        print("Moving {} images directories into val dataset".format(len(toBeMoved)))
        for dirName in toBeMoved:
            if os.path.exists(os.path.join(datasetPath, dirName)):
                move(os.path.join(datasetPath, dirName), os.path.join(valDatasetPath, dirName))
    if rename:
        newName = (datasetPath + '_train') if customRename is None else customRename
        move(datasetPath, newName)
    return toBeMoved


def createValDatasetByPeople(rawDataset, datasetPath: str, valDatasetPath: str = None, nbPatientBiopsie=5,
                             nbPatientNephrectomy=4, recreateInfo=None):
    """
    Create the validation dataset by moving a set of base dataset's images from a few patients
    If a patient has biopsy and nephrectomy images, it will count in both categories
    :param rawDataset: path to the raw dataset
    :param datasetPath: path to the dataset where to take images'directories
    :param valDatasetPath: path to the val dataset, if None, datasetPath + _val is used
    :param nbPatientBiopsie: number of patient with a biopsy to use as validation data
    :param nbPatientNephrectomy: number of patient with a nephrectomy to use as validation data
    :param recreateInfo: list of images to use to create val dataset
    :return: None
    """
    if valDatasetPath is None:
        valDatasetPath = datasetPath + "_val"
    os.makedirs(valDatasetPath, exist_ok=True)

    if recreateInfo is not None and len(recreateInfo) > 0:
        for dirName in os.listdir(datasetPath):
            patientDir = [p in dirName for p in recreateInfo]
            if any(patientDir):
                move(os.path.join(datasetPath, dirName), os.path.join(valDatasetPath, dirName))
        return recreateInfo
    else:
        _, pB, pN = infoPatients(rawDataset)
        selected = selectPatients(pB, pN, nbPatientBiopsie, nbPatientNephrectomy)
        toMove = []
        imagesFolder = os.listdir(datasetPath)
        for folder in imagesFolder:
            for patient in selected:
                if patient in folder:
                    toMove.append(folder)

        if len(toMove) > 0:
            print("Moving {} images directories into val dataset".format(len(toMove)))
            for dirName in toMove:
                # print(os.path.join(datasetPath, dirName), " vers ", os.path.join(valDatasetPath, dirName))
                move(os.path.join(datasetPath, dirName), os.path.join(valDatasetPath, dirName))
        return selected


def checkNSG(datasetPath: str):
    totalDiff = 0
    print()
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
            print(f"{imageDir} : {diff} {'nsg' if nsg < completAndPartiel else 'complet/partiel'} manquant{'s' if diff > 1 else ''}")
            totalDiff += diff
    print(f"Total : {totalDiff}")


def generateDataset(rawDataset='raw_dataset', tempDataset='temp_dataset', unusedDirPath='nephrology_dataset_unused',
                    mainDataset='main_dataset', mainDatasetUnusedDirPath='main_dataset_unused',
                    deleteBaseCortexMasks=True, adapter: AnnotationAdapter = None, imageFormat="jpg",
                    recreateValList=None, separateDivInsteadOfImage=False, separateByPatient=True,
                    divisionSize=1024, minDivisionOverlapping=0.33, cleanBeforeStart=False):
    """
    Generates datasets folder from a base directory, all paths are customizable, and it can also remove previous
    directories
    :param rawDataset: path to the base directory
    :param tempDataset: path to a temporary directory
    :param unusedDirPath: path to the unused files' directory
    :param mainDataset: path to the main dataset directory, used to also define main training and validation directories
    :param mainDatasetUnusedDirPath: path to unused files' directory of main dataset
    :param deleteBaseCortexMasks: whether to delete base cortex masks or not
    :param adapter: the adapter used to read annotations files, if None, will detect automatically which one to use
    :param imageFormat: the image format to use for the datasets
    :param recreateValList: list of images to use to recreate val dataset
    :param separateDivInsteadOfImage: if True, divisions of same image can be separated into training and val directories
    :param separateByPatient: if True and not separateDivInsteadOfImage, will create validation directory based on patient
    :param divisionSize: the size of a division, default is 1024
    :param minDivisionOverlapping: the min overlapping between two divisions, default is 33%
    :param cleanBeforeStart: if True, will delete previous directories that could still exist
    :return:
    """
    if cleanBeforeStart:
        # Removing temp directories
        import shutil
        dirToDel = [tempDataset, unusedDirPath,
                    'temp_' + mainDataset + '_val', mainDataset + '_val', mainDataset + '_train']
        for directory in dirToDel:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)

    # Creating masks and making per image directories
    dW.startWrapper(rawDataset, tempDataset, deleteBaseCortexMasks=deleteBaseCortexMasks,
                    adapter=adapter, imageFormat=imageFormat)
    infoNephrologyDataset(tempDataset)
    checkNSG(tempDataset)

    # Sorting images to keep those that can be used to train cortex
    sortImages(datasetPath=tempDataset, unusedDirPath=unusedDirPath)

    recreateInfo = {"mode": "main", "temp_dataset": tempDataset, "unused_dir_path": unusedDirPath,
                    "main_dataset": mainDataset, "main_dataset_unused_path": mainDatasetUnusedDirPath,
                    "delete_base_cortex_masks": deleteBaseCortexMasks, "image_format": imageFormat,
                    "separate": "div" if separateDivInsteadOfImage else ("patient" if separateByPatient else "images"),
                    "division_size": divisionSize, "min_overlap_part": minDivisionOverlapping,
                    "clean_before_start": cleanBeforeStart, "val_dataset": []}
    if separateDivInsteadOfImage:
        # Dividing main dataset in 1024*1024 divisions
        dD.divideDataset(tempDataset, mainDataset, squareSideLength=divisionSize,
                         min_overlap_part=minDivisionOverlapping, verbose=1)
        infoNephrologyDataset(mainDataset)

        # # If you want to keep all cortex files comment dW.cleanCortexDir() lines
        # # If you want to check them and then delete them, comment these lines too and after checking use them
        # # dW.cleanCortexDir(tempDataset)
        # # dW.cleanCortexDir(mainDataset)

        # Removing unusable images by moving them into a specific directory
        sortImages(mainDataset, unusedDirPath=mainDatasetUnusedDirPath)
        # Taking some images from the main dataset to make the validation dataset
        recreateInfo["val_dataset"] = createValDataset(mainDataset, rename=True, recreateInfo=recreateValList)
    else:  # To avoid having divisions of same image to be dispatched in main and validation dataset
        # Removing unusable images by moving them into a specific directory
        if separateByPatient:
            recreateInfo["val_dataset"] = createValDatasetByPeople(rawDataset=rawDataset, datasetPath=tempDataset,
                                                                   valDatasetPath='temp_' + mainDataset + '_val',
                                                                   nbPatientBiopsie=7, nbPatientNephrectomy=3,
                                                                   recreateInfo=recreateValList)
        else:
            # Taking some images from the main dataset to make the validation dataset
            recreateInfo["val_dataset"] = createValDataset(tempDataset, valDatasetPath='temp_' + mainDataset + '_val',
                                                           rename=False, recreateInfo=recreateValList)

        # Dividing the main dataset after having separated images for the validation dataset
        # then removing unusable divisions
        dD.divideDataset(tempDataset, mainDataset + '_train', squareSideLength=divisionSize,
                         min_overlap_part=minDivisionOverlapping, verbose=1)
        sortImages(mainDataset + '_train', unusedDirPath=mainDatasetUnusedDirPath)

        # Same thing with the validation dataset directly
        dD.divideDataset('temp_' + mainDataset + '_val', mainDataset + '_val', squareSideLength=divisionSize,
                         min_overlap_part=minDivisionOverlapping, verbose=1)
        sortImages(mainDataset + '_val', unusedDirPath=mainDatasetUnusedDirPath)

    infoNephrologyDataset(mainDataset + '_train')
    infoNephrologyDataset(mainDataset + '_val')
    if recreateValList is None or len(recreateValList) == 0:
        with open(f"dataset_{formatDate()}.json", 'w') as recreateFile:
            json.dump(recreateValList, recreateFile, indent="\t")
    print("\nDataset made, nothing left to do")


def generateCortexDataset(rawDataset: str, outputDataset="nephrology_cortex_dataset", cleanBeforeStart=True,
                          resize=(2048, 2048), overlap=0., recreateValList=None, adapter: AnnotationAdapter = None):
    """
    Generates datasets folder from a base directory, all paths are customizable, and it can also remove previous
    directories
    :param rawDataset: path to the base directory
    :param outputDataset: path to the output cortex dataset
    :param cleanBeforeStart: if True, will delete previous directories that could still exist
    :param resize:
    :param overlap:
    :param recreateValList:
    :param adapter:
    :return:
    """
    recreateInfo = {"mode": "cortex", "output_dataset": outputDataset,
                    "clean_before_start": cleanBeforeStart, "cortex_resize": list(resize),
                    "min_overlap_part": overlap, "val_dataset": []}
    # Removing former dataset directories
    if cleanBeforeStart:
        import shutil
        dirToDel = ["temp_" + outputDataset, outputDataset, outputDataset + '_train', outputDataset + '_val']
        for directory in dirToDel:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
    # Creating masks for cortices images
    dW.startWrapper(rawDataset, "temp_" + outputDataset, resize=resize, cortexMode=True, adapter=adapter)
    # If size is greater than 1024x1024, dataset must be divided
    if resize is not None and not resize[0] == resize[1] == 1024:
        dD.divideDataset("temp_" + outputDataset, outputDataset, squareSideLength=1024, min_overlap_part=overlap,
                         mode="cortex")
    # Creating val dataset by
    recreateInfo["val_dataset"] = createValDataset(outputDataset, valDatasetPath=outputDataset + '_val',
                                                   rename=True, valDatasetSizePart=0.05, valDatasetMinSize=10,
                                                   recreateInfo=recreateValList)
    infoNephrologyDataset(outputDataset + '_train')
    infoNephrologyDataset(outputDataset + '_val')
    if recreateValList is None or len(recreateValList) == 0:
        with open(f"dataset_cortex_{formatDate()}.json", 'w') as recreateFile:
            json.dump(recreateValList, recreateFile, indent="\t")
    print("\nDataset made, nothing left to do")


def regenerateDataset(rawDataset, recreateFilePath, adapter: AnnotationAdapter = None):
    with open(recreateFilePath, 'r') as recreateFile:
        recreateInfo = json.load(recreateFile)
    if recreateInfo["mode"] == "main":
        generateDataset(rawDataset=rawDataset, tempDataset=recreateInfo["temp_dataset"],
                        unusedDirPath=recreateInfo["unused_dir_path"], mainDataset=recreateInfo["main_dataset"],
                        mainDatasetUnusedDirPath=recreateInfo["main_dataset_unused_path"],
                        deleteBaseCortexMasks=recreateInfo["delete_base_cortex_masks"], adapter=adapter,
                        imageFormat=recreateInfo["image_format"], recreateValList=recreateInfo["val_dataset"],
                        separateDivInsteadOfImage=recreateInfo["separate"] == "div",
                        separateByPatient=recreateInfo["separate"] == "patient",
                        divisionSize=recreateInfo["division_size"],
                        minDivisionOverlapping=recreateInfo["min_overlap_part"],
                        cleanBeforeStart=recreateInfo["clean_before_start"])
    elif recreateInfo["mode"] == "cortex":
        generateCortexDataset(rawDataset=rawDataset, outputDataset=recreateInfo["output_dataset"],
                              cleanBeforeStart=recreateInfo["clean_before_start"],
                              resize=tuple(recreateInfo["cortex_resize"]), overlap=recreateInfo["min_overlap_part"],
                              recreateValList=recreateInfo["val_dataset"], adapter=adapter)
    else:
        raise NotImplementedError(f"{recreateInfo['mode']} dataset mode not available")


if __name__ == "__main__":
    generateDataset(mainDataset='nephrology_dataset', mainDatasetUnusedDirPath='nephrology_dataset_unused',
                    cleanBeforeStart=True)
