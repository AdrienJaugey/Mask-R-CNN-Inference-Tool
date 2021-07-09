"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project
Dataset tools

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import json
import os
from shutil import move
import numpy as np

from common_utils import formatDate
from datasetTools import datasetWrapper as dW, AnnotationAdapter
from datasetTools import datasetDivider as dD
from datasetTools import datasetIsolator as dI


def infoNephrologyDataset(datasetPath: str, baseClass=None, silent=False):
    """
    Print information about a dataset
    :param datasetPath: path to the dataset
    :param baseClass: the class representing the area where to look for other objects
    :param silent: if true nothing will be printed
    :return: nbImg, histogram, baseClassMissing, multiBaseClassMasks, maxClasses, maxClassesNoBaseMask
    """
    print()
    histogram = {}
    maxNbClasses = 0
    maxClasses = []
    maxNbClassesNoBase = 0
    maxClassesNoBaseMask = []
    missingBaseClass = []
    printedBaseClassMissing = []
    multiBaseMasks = []
    missingDataImages = []
    nbImg = 0
    for imageDir in os.listdir(datasetPath):
        nbImg += 1
        imagePath = os.path.join(datasetPath, imageDir)
        baseClassPresent = True if baseClass is None else False
        multipleBaseClassMasks = False
        localHisto = {}
        missingData = False if baseClass is None else True
        for maskDir in os.listdir(imagePath):
            if maskDir == baseClass:
                baseClassPresent = True
                missingData = False
                multipleBaseClassMasks = len(os.listdir(os.path.join(datasetPath, imageDir, maskDir))) > 1
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
        if not baseClassPresent:
            if imageDir not in printedBaseClassMissing:
                printedBaseClassMissing.append(imageDir)
            missingBaseClass.append(imageDir)
        if missingData:
            missingDataImages.append(imageDir)
        if multipleBaseClassMasks:
            multiBaseMasks.append(imageDir)

        nbClasses = 0
        nbClassesNoBaseClass = 0
        for objectClass in localHisto:
            if localHisto[objectClass] > 0:
                if objectClass != baseClass:
                    nbClassesNoBaseClass += 1
                nbClasses += 1
        if nbClasses >= maxNbClasses:
            if nbClasses > maxNbClasses:
                maxNbClasses = nbClasses
                maxClasses = []
            maxClasses.append(imageDir)

        if nbClassesNoBaseClass >= maxNbClassesNoBase:
            if nbClassesNoBaseClass > maxNbClassesNoBase:
                maxNbClassesNoBase = nbClassesNoBaseClass
                maxClassesNoBaseMask = []
            maxClassesNoBaseMask.append(imageDir)

    if not silent:
        print(f"{datasetPath} dataset Informations :")
        print(f"\tNb Images : {nbImg}")
        print(f"\tHistogram : {histogram}")
        print(f"\tMissing {baseClass} ({len(missingBaseClass)}) : {printedBaseClassMissing}")
        print(f"\tMissing data ({len(missingDataImages)}) : {missingDataImages}")
        print(f"\tMulti {baseClass} masks ({len(multiBaseMasks)}) : {multiBaseMasks}")
        if baseClass is not None:
            print(f"\tMax Classes w/ {baseClass}  ({maxNbClasses}) :\t{maxClasses}")
            print(f"\tMax Classes w/o {baseClass} ({maxNbClassesNoBase}) :\t{maxClassesNoBaseMask}")
        else:
            print(f"\tMax Classes ({maxNbClasses}) :\t{maxClasses}")
    return {"image_count": nbImg, "classes_histogram": histogram, "missing_base_class": missingBaseClass,
            "multi_base_class_masks": multiBaseMasks, "max_classes_images": maxClasses,
            "max_classes_images_no_base_class": maxClassesNoBaseMask, "missing_data_images": missingDataImages}


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


def selectPatients(patientsBiopsie, patientsNephrectomie, nbPatientBiopsie=8, nbPatientNephrectomie=2):
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
            selected.extend(np.random.choice(patientsBiopsie, size=nbPatientBiopsie, replace=False))

        nbPatientNephrectomie -= len(communs)
        if nbPatientNephrectomie > 0:
            selected.extend(np.random.choice(patientsNephrectomie, size=nbPatientNephrectomie, replace=False))
    return selected


def sortImages(datasetPath: str, unusedDirPath: str = None, mode: str = "main"):
    """
    Move images that cannot be used in main training/inference
    :param datasetPath: path to the base dataset
    :param unusedDirPath: path to the directory where unused files will be moved
    :param mode: the inference mode
    :return: None
    """
    # Setting paths if not given
    if unusedDirPath is None:
        unusedDirPath = datasetPath + '_unused'

    NOT_TO_COUNT = ['images', 'full_images']
    baseClass = None
    if mode == "cortex":
        NOT_TO_COUNT.extend(["nsg", "nsg_complet", "nsg_partiel", "tubule_sain", "tubule_atrophique", "vaisseau",
                             "intima", "media", "pac", "artefact", "veine"])
    elif mode == "main":
        NOT_TO_COUNT.extend(['cortex', 'medullaire', 'capsule'])
        baseClass = "cortex"
    elif mode == "mest_main":
        NOT_TO_COUNT.extend(["cortex", "medullaire", "capsule", "nsg_complet",
                             "nsg_partiel", "intima", "media", "artefact"])
        baseClass = "cortex"
    elif mode == "mest_glom":
        NOT_TO_COUNT.extend(["nsg_complet", "nsg_partiel", "tubule_sain", "tubule_atrophique", "vaisseau",
                             "intima", "media", "pac", "artefact", "veine", "medullaire", "capsule"])
        baseClass = "nsg"
    elif mode == "inflammation":
        NOT_TO_COUNT.extend(["cortex", "tubule_sain", "tubule_atrophique", "pac", "vaisseau",
                             "artefact", "veine", "nsg", "intima", "media"])
        baseClass = "cortex"
    else:
        print(f"Mode {mode} is not yet supported")
        return

    # Getting list of images directories without data
    info = infoNephrologyDataset(datasetPath, baseClass=baseClass, silent=True)
    toBeMoved = []
    toBeMoved.extend(info["missing_data_images"])
    toBeMoved.extend(info["missing_base_class"])

    # For each image directory that is not already in toBeMoved list
    for imageDir in os.listdir(datasetPath):
        # If image has no base mask or no other classes masks : moving it
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
        for imageWithoutBaseDir in toBeMoved:
            srcPath = os.path.join(datasetPath, imageWithoutBaseDir)
            dstPath = os.path.join(unusedDirPath, imageWithoutBaseDir)
            try:
                move(srcPath, dstPath)
            except FileNotFoundError:
                pass


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
        toBeMoved = list(np.random.choice(fullList, valDatasetSize, replace=False))
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
    toBeMoved.sort()
    return toBeMoved


def createValDatasetByPeople(rawDataset, datasetPath: str, valDatasetPath: str = None, nbPatientBiopsie: int = 5,
                             nbPatientNephrectomy: int = 4, recreateInfo: list = None):
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
        recreateInfo.sort()
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
                # print(os.path.join(datasetPath, dirName), " to ", os.path.join(valDatasetPath, dirName))
                move(os.path.join(datasetPath, dirName), os.path.join(valDatasetPath, dirName))
        selected.sort()
        return selected


def checkNSG(datasetPath: str):
    totalDiff = 0
    print()
    for imageDir in os.listdir(datasetPath):
        dirPath = os.path.join(datasetPath, imageDir)
        nsgDirectoriesPath = [os.path.join(dirPath, 'nsg'), os.path.join(dirPath, 'nsg_partiel'),
                              os.path.join(dirPath, 'nsg_complet')]
        count = [0, 0, 0]
        for index, folder in enumerate(nsgDirectoriesPath):
            if os.path.exists(folder):
                count[index] = len(os.listdir(folder))
        nsg = count[0]
        completAndPartiel = count[1] + count[2]
        if nsg != completAndPartiel:
            diff = abs(count[0] - count[1] - count[2])
            print(f"{imageDir} : {diff} {'nsg' if nsg < completAndPartiel else 'complet/partiel'} "
                  f"manquant{'s' if diff > 1 else ''}")
            totalDiff += diff
    print(f"Total : {totalDiff}")


def generateDataset(rawDataset='raw_dataset', tempDataset='temp_dataset', unusedDirPath='nephrology_dataset_unused',
                    mainDataset='main_dataset', mainDatasetUnusedDirPath='main_dataset_unused',
                    deleteBaseMasks=True, adapter: AnnotationAdapter = None, imageFormat="jpg",
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
    :param deleteBaseMasks: whether to delete base masks or not
    :param adapter: the adapter used to read annotations files, if None, will detect automatically which one to use
    :param imageFormat: the image format to use for the datasets
    :param recreateValList: list of images to use to recreate val dataset
    :param separateDivInsteadOfImage: if True, divisions of same image can be separated into training and val
                                      directories
    :param separateByPatient: if True and not separateDivInsteadOfImage, will create validation directory based on
                              patient
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
    dW.startWrapper(rawDataset, tempDataset, deleteBaseMasks=deleteBaseMasks,
                    adapter=adapter, imageFormat=imageFormat, mode="main")
    infoNephrologyDataset(tempDataset, baseClass='cortex')
    checkNSG(tempDataset)

    # Sorting images to keep those that can be used
    sortImages(datasetPath=tempDataset, unusedDirPath=unusedDirPath, mode="main")

    recreateInfo = {"mode": "main", "temp_dataset": tempDataset, "unused_dir_path": unusedDirPath,
                    "main_dataset": mainDataset, "main_dataset_unused_path": mainDatasetUnusedDirPath,
                    "delete_base_masks": deleteBaseMasks, "image_format": imageFormat,
                    "separate": "div" if separateDivInsteadOfImage else ("patient" if separateByPatient else "images"),
                    "division_size": divisionSize, "min_overlap_part": minDivisionOverlapping,
                    "clean_before_start": cleanBeforeStart, "val_dataset": []}
    if separateDivInsteadOfImage:
        # Dividing main dataset in 1024*1024 divisions
        dD.divideDataset(tempDataset, mainDataset, squareSideLength=divisionSize,
                         min_overlap_part=minDivisionOverlapping, verbose=1)
        infoNephrologyDataset(mainDataset, baseClass='cortex')

        # If you want to keep all cortex files comment dW.cleanCortexDir() lines
        # If you want to check them and then delete them, comment these lines too and after checking use them
        # dW.cleanFusedClassDir(tempDataset, 'cortex')
        # dW.cleanFusedClassDir(mainDataset, 'cortex')

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

    infoNephrologyDataset(mainDataset + '_train', baseClass='cortex')
    infoNephrologyDataset(mainDataset + '_val', baseClass='cortex')
    if recreateValList is None or len(recreateValList) == 0:
        with open(f"dataset_{formatDate()}.json", 'w') as recreateFile:
            json.dump(recreateInfo, recreateFile, indent="\t")
    print("\nDataset made, nothing left to do")


def generateCortexDataset(rawDataset: str, outputDataset="nephrology_cortex_dataset", cleanBeforeStart=True,
                          resize=(2048, 2048), overlap=0., separateDivInsteadOfImage=False, recreateValList=None,
                          adapter: AnnotationAdapter = None):
    """
    Generates datasets folder from a base directory, all paths are customizable, and it can also remove previous
    directories
    :param rawDataset: path to the base directory
    :param outputDataset: path to the output cortex dataset
    :param cleanBeforeStart: if True, will delete previous directories that could still exist
    :param resize: the size of the output images and masks before dividing
    :param overlap: the least overlap between two divisions
    :param separateDivInsteadOfImage: if True, divisions of same image can be separated into training and val
                                      directories
    :param recreateValList: list of the images to use to recreate cortex dataset
    :param adapter: the adapter to use if given, else it will be chosen depending on the annotations found
    :return: None
    """
    recreateInfo = {"mode": "cortex", "output_dataset": outputDataset,
                    "clean_before_start": cleanBeforeStart, "resize": list(resize),
                    "separate": "div" if separateDivInsteadOfImage else "images",
                    "min_overlap_part": overlap, "val_dataset": []}
    # Removing former dataset directories
    if cleanBeforeStart:
        import shutil
        dirToDel = ["temp_" + outputDataset, outputDataset, outputDataset + '_train', outputDataset + '_val']
        for directory in dirToDel:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
    # Creating masks for cortices images
    dW.startWrapper(rawDataset, "temp_" + outputDataset, resize=resize, mode="cortex", adapter=adapter)
    if not separateDivInsteadOfImage:
        recreateInfo["val_dataset"] = createValDataset("temp_" + outputDataset,
                                                       valDatasetPath="temp_" + outputDataset + '_val',
                                                       rename=True, valDatasetSizePart=0.05, valDatasetMinSize=10,
                                                       recreateInfo=recreateValList)
    # If size is greater than 1024x1024, dataset must be divided
    if resize is not None and not resize[0] == resize[1] == 1024:
        if separateDivInsteadOfImage:
            divide = {"temp_" + outputDataset: outputDataset}
        else:
            divide = {"temp_" + outputDataset + '_val': outputDataset + '_val',
                      "temp_" + outputDataset + '_train': outputDataset + '_train'}
        for inputPath, outputPath in divide.items():
            dD.divideDataset(inputPath, outputPath, squareSideLength=1024, min_overlap_part=overlap,
                             mode="cortex", verbose=1)
    if separateDivInsteadOfImage:
        # Creating val dataset by
        recreateInfo["val_dataset"] = createValDataset(outputDataset, valDatasetPath=outputDataset + '_val',
                                                       rename=True, valDatasetSizePart=0.05, valDatasetMinSize=10,
                                                       recreateInfo=recreateValList)
    for datasetPath in [outputDataset + '_train', outputDataset + '_val']:
        sortImages(datasetPath, outputDataset + '_unused', mode="cortex")
    infoNephrologyDataset(outputDataset + '_train')
    infoNephrologyDataset(outputDataset + '_val')
    if recreateValList is None or len(recreateValList) == 0:
        with open(f"dataset_cortex_{formatDate()}.json", 'w') as recreateFile:
            json.dump(recreateInfo, recreateFile, indent="\t")
    print("\nDataset made, nothing left to do")


def generateMESTCDataset(rawDataset: str, outputDataset="nephrology_mest_{mode}_dataset", cleanBeforeStart=True,
                         mode="glom", imageFormat='jpg', divisionSize=1024, overlap=0.33, separate="images",
                         recreateValList=None, adapter: AnnotationAdapter = None):
    """
    Generates datasets folder from a base directory, all paths are customizable, and it can also remove previous
    directories
    :param rawDataset: path to the base directory
    :param outputDataset: path to the output dataset
    :param cleanBeforeStart: if True, will delete previous directories that could still exist
    :param mode: the mode to use : glom or main
    :param imageFormat: the image format to look for and to use
    :param divisionSize: size of the output images
    :param overlap: the least overlap between two divisions
    :param separate: if True, divisions of same image can be separated into training and val directories
    :param recreateValList: list of the images to use to recreate val dataset
    :param adapter: the adapter to use if given, else it will be chosen depending on the annotations found
    :return: None
    """
    outputDataset = outputDataset.format(mode=mode)
    recreateInfo = {"mode": "mest", "submode": mode, "output_dataset": outputDataset,
                    "clean_before_start": cleanBeforeStart, "image_format": imageFormat,
                    "division_size": divisionSize, "min_overlap_part": separate, "val_dataset": []}
    # Removing former dataset directories
    if cleanBeforeStart:
        import shutil
        dirToDel = ["temp_" + outputDataset, "temp_" + outputDataset + '_train', "temp_" + outputDataset + '_val',
                    outputDataset, outputDataset + '_train', outputDataset + '_val', outputDataset + '_unused']
        for directory in dirToDel:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
    # Creating masks for cortices images
    dW.startWrapper(rawDataset, "temp_" + outputDataset, mode=f"mest_{mode}", adapter=adapter)
    if mode == "glom":
        if not separate == "div":
            recreateInfo["val_dataset"] = createValDataset("temp_" + outputDataset,
                                                           valDatasetPath="temp_" + outputDataset + '_val',
                                                           rename=True, valDatasetSizePart=0.05, valDatasetMinSize=10,
                                                           recreateInfo=recreateValList)
            for datasetPart in ["train", "val"]:
                dI.isolateClass(f"temp_{outputDataset}_{datasetPart}", f"{outputDataset}_{datasetPart}", 'nsg',
                                image_size=divisionSize, imageFormat=imageFormat, verbose=3, silent=False)
                sortImages(f"{outputDataset}_{datasetPart}", f"{outputDataset}_unused", mode="mest_glom")
        else:
            dI.isolateClass("temp_" + outputDataset, outputDataset, 'nsg', image_size=divisionSize,
                            imageFormat=imageFormat, verbose=3, silent=False)
            sortImages(outputDataset, f"{outputDataset}_unused", mode="mest_glom")
            recreateInfo["val_dataset"] = createValDataset(outputDataset,
                                                           valDatasetPath=outputDataset + '_val',
                                                           rename=True, valDatasetSizePart=0.05, valDatasetMinSize=10,
                                                           recreateInfo=recreateValList)
    elif mode == "main":
        if separate in ["images", "patient"]:
            if separate == "images":
                recreateInfo["val_dataset"] = createValDataset("temp_" + outputDataset,
                                                               valDatasetPath="temp_" + outputDataset + '_val',
                                                               rename=False, valDatasetSizePart=0.05,
                                                               valDatasetMinSize=10, recreateInfo=recreateValList)
            else:
                recreateInfo["val_dataset"] = createValDatasetByPeople(rawDataset=rawDataset,
                                                                       datasetPath="temp_" + outputDataset,
                                                                       valDatasetPath='temp_' + outputDataset + '_val',
                                                                       nbPatientBiopsie=8, nbPatientNephrectomy=2,
                                                                       recreateInfo=recreateValList)
        # Dividing the dataset
        if separate == "div":
            divide = {"temp_" + outputDataset: outputDataset}
        else:
            divide = {"temp_" + outputDataset + '_val': outputDataset + '_val',
                      "temp_" + outputDataset: outputDataset + '_train'}
        for inputPath, outputPath in divide.items():
            dD.divideDataset(inputPath, outputPath, squareSideLength=divisionSize, min_overlap_part=overlap,
                             mode=f"mest_{mode}", verbose=1)

        if separate == "div":
            # Creating val dataset by
            recreateInfo["val_dataset"] = createValDataset(outputDataset, valDatasetPath=outputDataset + '_val',
                                                           rename=True, valDatasetSizePart=0.05, valDatasetMinSize=10,
                                                           recreateInfo=recreateValList)
        for datasetPath in [outputDataset + '_train', outputDataset + '_val']:
            sortImages(datasetPath, outputDataset + '_unused', mode="mest_main")

    infoNephrologyDataset(outputDataset + '_train', baseClass='nsg' if mode == "glom" else "cortex")
    infoNephrologyDataset(outputDataset + '_val', baseClass="glom" if mode == "glom" else "cortex")
    if recreateValList is None or len(recreateValList) == 0:
        with open(f"dataset_mest_{mode}_{formatDate()}.json", 'w') as recreateFile:
            json.dump(recreateInfo, recreateFile, indent="\t")
    print("\nDataset made, nothing left to do")


def generateInflammationDataset(rawDataset: str, outputDataset="nephrology_inflammation_dataset", cleanBeforeStart=True,
                                imageFormat='jpg', divisionSize=1024, overlap=0.33, separate="images",
                                recreateValList=None, adapter: AnnotationAdapter = None):
    """
    Generates datasets folder from a base directory, all paths are customizable, and it can also remove previous
    directories
    :param rawDataset: path to the base directory
    :param outputDataset: path to the output dataset
    :param cleanBeforeStart: if True, will delete previous directories that could still exist
    :param imageFormat: the image format to look for and to use
    :param divisionSize: size of the output images
    :param overlap: the least overlap between two divisions
    :param separate: if True, divisions of same image can be separated into training and val directories
    :param recreateValList: list of the images to use to recreate val dataset
    :param adapter: the adapter to use if given, else it will be chosen depending on the annotations found
    :return: None
    """
    base_class = "cortex"
    if cleanBeforeStart:
        # Removing temp directories
        import shutil
        dirToDel = ["temp_" + outputDataset, "temp_" + outputDataset + '_val', outputDataset,
                    outputDataset + '_train', outputDataset + '_val', outputDataset + '_unused']
        for directory in dirToDel:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)

    # Creating masks and making per image directories
    dW.startWrapper(rawDataset, "temp_" + outputDataset, deleteBaseMasks=True,
                    adapter=adapter, imageFormat=imageFormat, mode="inflammation")
    infoNephrologyDataset("temp_" + outputDataset, baseClass=base_class)

    # Sorting images to keep those that can be used
    sortImages(datasetPath="temp_" + outputDataset, unusedDirPath=outputDataset + '_unused', mode="main")

    recreateInfo = {"mode": "inflammation", "output_dataset": outputDataset, "clean_before_start": cleanBeforeStart,
                    "image_format": imageFormat, "division_size": divisionSize, "min_overlap_part": overlap,
                    "separate": separate, "val_dataset": []}
    if separate == "div":
        # Dividing dataset
        dD.divideDataset("temp_" + outputDataset, outputDataset, squareSideLength=divisionSize,
                         min_overlap_part=overlap, verbose=1)
        infoNephrologyDataset(outputDataset, baseClass=base_class)

        # Removing unusable images by moving them into a specific directory
        sortImages(outputDataset, unusedDirPath=outputDataset + '_unused')
        # Taking some images from the main dataset to make the validation dataset
        recreateInfo["val_dataset"] = createValDataset(outputDataset, rename=True, recreateInfo=recreateValList)
    else:  # To avoid having divisions of same image to be dispatched in main and validation dataset
        # Removing unusable images by moving them into a specific directory
        if separate == "patient":
            recreateInfo["val_dataset"] = createValDatasetByPeople(rawDataset=rawDataset,
                                                                   datasetPath="temp_" + outputDataset,
                                                                   valDatasetPath='temp_' + outputDataset + '_val',
                                                                   nbPatientBiopsie=7, nbPatientNephrectomy=3,
                                                                   recreateInfo=recreateValList)
        else:
            # Taking some images from the main dataset to make the validation dataset
            recreateInfo["val_dataset"] = createValDataset("temp_" + outputDataset,
                                                           valDatasetPath='temp_' + outputDataset + '_val',
                                                           rename=False, recreateInfo=recreateValList)

        # Dividing the main dataset after having separated images for the validation dataset
        # then removing unusable divisions
        dD.divideDataset("temp_" + outputDataset, outputDataset + '_train', squareSideLength=divisionSize,
                         min_overlap_part=overlap, verbose=1)
        sortImages(outputDataset + '_train', unusedDirPath=outputDataset + '_unused')

        # Same thing with the validation dataset directly
        dD.divideDataset('temp_' + outputDataset + '_val', outputDataset + '_val', squareSideLength=divisionSize,
                         min_overlap_part=overlap, verbose=1)
        sortImages(outputDataset + '_val', unusedDirPath=outputDataset + '_unused')

    infoNephrologyDataset(outputDataset + '_train', baseClass=base_class)
    infoNephrologyDataset(outputDataset + '_val', baseClass=base_class)
    if recreateValList is None or len(recreateValList) == 0:
        with open(f"dataset_{formatDate()}.json", 'w') as recreateFile:
            json.dump(recreateInfo, recreateFile, indent="\t")
    print("\nDataset made, nothing left to do")


def regenerateDataset(rawDataset, recreateFilePath, adapter: AnnotationAdapter = None):
    with open(recreateFilePath, 'r') as recreateFile:
        recreateInfo = json.load(recreateFile)
    if recreateInfo["mode"] == "main":
        generateDataset(rawDataset=rawDataset, tempDataset=recreateInfo["temp_dataset"],
                        unusedDirPath=recreateInfo["unused_dir_path"], mainDataset=recreateInfo["main_dataset"],
                        mainDatasetUnusedDirPath=recreateInfo["main_dataset_unused_path"],
                        deleteBaseMasks=recreateInfo["delete_base_masks"],
                        adapter=adapter, imageFormat=recreateInfo["image_format"],
                        recreateValList=recreateInfo["val_dataset"],
                        separateDivInsteadOfImage=recreateInfo["separate"] == "div",
                        separateByPatient=recreateInfo["separate"] == "patient",
                        divisionSize=recreateInfo["division_size"],
                        minDivisionOverlapping=recreateInfo["min_overlap_part"],
                        cleanBeforeStart=recreateInfo["clean_before_start"])
    elif recreateInfo["mode"] == "cortex":
        generateCortexDataset(rawDataset=rawDataset, outputDataset=recreateInfo["output_dataset"],
                              cleanBeforeStart=recreateInfo["clean_before_start"],
                              resize=tuple(recreateInfo["resize"]), overlap=recreateInfo["min_overlap_part"],
                              separateDivInsteadOfImage=recreateInfo["separate"] == "div",
                              recreateValList=recreateInfo["val_dataset"], adapter=adapter)
    elif recreateInfo["mode"] == "mest":
        generateMESTCDataset(rawDataset=rawDataset, outputDataset=recreateInfo["output_dataset"],
                             cleanBeforeStart=recreateInfo["clean_before_start"], mode=recreateInfo["submode"],
                             imageFormat=recreateInfo["image_format"], divisionSize=recreateInfo["division_size"],
                             overlap=recreateInfo["min_overlap_part"], separate=recreateInfo["separate"],
                             recreateValList=recreateInfo["val_dataset"], adapter=adapter)
    elif recreateInfo["mode"] == "inflammation":
        generateInflammationDataset(rawDataset=rawDataset, outputDataset=recreateInfo["output_dataset"],
                                    cleanBeforeStart=recreateInfo["clean_before_start"],
                                    imageFormat=recreateInfo["image_format"],
                                    divisionSize=recreateInfo["division_size"],
                                    overlap=recreateInfo["min_overlap_part"], separate=recreateInfo["separate"],
                                    recreateValList=recreateInfo["val_dataset"], adapter=adapter)
    else:
        raise NotImplementedError(f"{recreateInfo['mode']} dataset mode not available")
