"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import sys
import os
import json
import re
import traceback
import shutil
import warnings
import gc

from datasetTools.AnnotationAdapter import export_annotations
from datasetTools.CustomDataset import CustomDataset
from mrcnn.Config import Config, DynamicMethod

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from common_utils import progressBar, formatTime, formatDate, progressText
    from datasetTools.datasetDivider import CV2_IMWRITE_PARAM
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.io import imsave
    from datasetTools import datasetDivider as dD, datasetWrapper as dW, AnnotationAdapter, datasetIsolator as dI

    from mrcnn import utils
    from mrcnn.TensorflowDetector import TensorflowDetector
    from mrcnn import visualize
    from mrcnn import post_processing as pp
    from mrcnn import statistics as stats


def get_ax(rows=1, cols=1, size=8):
    return plt.subplots(rows, cols, figsize=(size * cols, size * rows), frameon=False)


def find_latest_weight(weight_path):
    """
    Return the weight file path with the highest id
    :param weight_path: weight path with %LAST% as the id part of the path
    :return: the weight path if found else None
    """
    if "%LAST%" not in weight_path:
        return weight_path
    folder = os.path.dirname(os.path.abspath(weight_path))
    folder = '.' if folder == "" else folder
    name = os.path.basename(weight_path)
    name_part1, name_part2 = name.split("%LAST%")
    regex = f"^{name_part1}([0-9]+){name_part2}$"
    maxID = -1
    maxTxtID = ""
    for weight_file in os.listdir(folder):
        regex_res = re.search(regex, weight_file)
        if regex_res:
            nbTxt = regex_res.group(1)
            nb = int(nbTxt)
            if nb > maxID:
                maxID = nb
                maxTxtID = nbTxt
    return None if maxID == -1 else weight_path.replace("%LAST%", maxTxtID)


def listAvailableImage(dirPath: str):
    files = os.listdir(dirPath)
    image = []
    for file in files:
        if file in image:
            continue
        name = file.split('.')[0]
        extension = file.split('.')[-1]
        if extension == 'jp2':
            if (name + '.png') not in image and (name + '.jpg') not in image:
                if os.path.exists(os.path.join(dirPath, name + '.jpg')):
                    image.append(name + '.jpg')
                elif os.path.exists(os.path.join(dirPath, name + '.png')):
                    image.append(name + '.png')
                else:
                    image.append(file)
        elif extension in ['png', 'jpg']:
            image.append(file)
        elif extension == 'skinet':
            with open(os.path.join(dirPath, file), 'r') as skinetFile:
                fusionInfo = json.load(skinetFile)
                fusionDir = fusionInfo['image'] + "_fusion"
                if fusionDir in files:
                    divExists = False
                    divPath = os.path.join(fusionDir, fusionInfo['image'] + '_{}.jpg')
                    for divID in range(len(fusionInfo["divisions"])):
                        if fusionInfo["divisions"][str(divID)]["used"] and os.path.exists(
                                os.path.join(dirPath, divPath.format(divID))):
                            image.append(divPath.format(divID))
                            divExists = True
                        elif fusionInfo["divisions"][str(divID)]["used"]:
                            print(f"Div n°{divID} of {fusionInfo['image']} missing")
                    if divExists:
                        image.append(file)

    for i in range(len(image)):
        image[i] = os.path.join(dirPath, image[i])
    return image


class InferenceTool:

    def __init__(self, configPath: str, low_memory=False):
        self.__STEP = "init"
        self.__CONFIG = Config(configPath)
        self.__CONFIG_PATH = configPath
        self.__LOW_MEMORY = low_memory
        self.__READY = False
        self.__MODE = None
        self.__MODEL_PATH = None
        self.__MODEL = None

        self.__CLASS_2_ID = None
        self.__ID_2_CLASS = None
        self.__CLASSES_INFO = None
        self.__NB_CLASS = 0
        self.__CUSTOM_CLASS_NAMES = None
        self.__VISUALIZE_NAMES = None
        self.__COLORS = None

        self.__DIVISION_SIZE = None
        self.__MIN_OVERLAP_PART = None
        self.__RESIZE = None

        self.__CONFUSION_MATRIX = None

        self.__BASE = None
        self.__PREVIOUS_RES = None

    def load(self, mode: str, forceFullSizeMasks=False, forceModelPath=None):
        # If mode is already loaded, nothing to do
        if self.__MODE == mode:
            print(f"{mode} mode is already loaded.\n")
            return

        self.__MODEL_PATH = find_latest_weight(self.__CONFIG.get_param(mode)['weight_file']
                                               if forceModelPath is None else forceModelPath)

        # Testing only for one of the format, as Error would have already been raised if modelPath was not correct
        isExportedModelDir = os.path.exists(os.path.join(self.__MODEL_PATH, 'saved_model'))
        if isExportedModelDir:
            self.__MODEL_PATH = os.path.join(self.__MODEL_PATH, 'saved_model')

        self.__CLASSES_INFO = self.__CONFIG.get_classes_info(mode)
        self.__CLASS_2_ID = self.__CONFIG.get_mode_config(mode)['class_to_id']

        label_map = {c["id"]: c for c in self.__CONFIG.get_classes_info(mode)}
        if self.__MODEL is None:
            self.__MODEL = TensorflowDetector(self.__MODEL_PATH, label_map)
        else:
            self.__MODEL.load(self.__MODEL_PATH, label_map)
        self.__READY = self.__MODEL.isLoaded()
        if not self.__MODEL.isLoaded():
            raise ValueError("Please provide correct path to model.")

        self.__CONFIG.set_current_mode(mode, forceFullSizeMasks)
        self.__MODE = mode

        self.__DIVISION_SIZE = self.__CONFIG.get_param()['roi_size']
        self.__MIN_OVERLAP_PART = self.__CONFIG.get_param()['min_overlap_part']
        if self.__CONFIG.get_param().get('resize', None) is None:
            self.__RESIZE = None
        else:
            self.__RESIZE = tuple(self.__CONFIG.get_param()['resize'])

        self.__NB_CLASS = len(self.__CLASSES_INFO)
        self.__CUSTOM_CLASS_NAMES = [classInfo["name"] for classInfo in self.__CLASSES_INFO]
        self.__VISUALIZE_NAMES = ['Background']
        self.__VISUALIZE_NAMES.extend([classInfo.get('display_name', classInfo['name'])
                                       for classInfo in self.__CLASSES_INFO])

        self.__COLORS = [classInfo["color"] for classInfo in self.__CLASSES_INFO]

        # Configurations
        self.__CONFUSION_MATRIX = {'pixel': np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64),
                                   'mask': np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64),
                                   'count': 0}
        print()

    def __prepare_image__(self, imagePath, results_path, chainMode=False, silent=False):
        """
        Creating png version if not existing, dataset masks if annotation found and get some information
        :param imagePath: path to the image to use
        :param results_path: path to the results dir to create the image folder and paste it in
        :param silent: No display
        :return: image, imageInfo = {"PATH": str, "DIR_PATH": str, "FILE_NAME": str, "NAME": str, "HEIGHT": int,
        "WIDTH": int, "NB_DIV": int, "X_STARTS": list, "Y_STARTS": list, "HAS_ANNOTATION": bool}
        """
        if not silent:
            print(" - Preparing image before inference")
        image = None
        fullImage = None
        imageInfo = None
        image_results_path = None
        roi_mode = self.__CONFIG.get_param()["roi_mode"]
        suffix = ""
        if chainMode and self.__CONFIG.get_previous_mode() is not None:
            image_name, extension = os.path.splitext(os.path.basename(imagePath))
            extension = extension.replace('.', '')
            if extension not in ['png', 'jpg']:
                extension = 'jpg'
            if self.__CONFIG.get_param('previous').get('resize', None) is not None:
                suffix = "_base"
            imagePath = os.path.join(results_path, image_name, self.__CONFIG.get_previous_mode(),
                                     f"{image_name}{suffix}.{extension}")
        if os.path.exists(imagePath):
            imageInfo = {
                'PATH': imagePath,
                'DIR_PATH': os.path.dirname(imagePath),
                'FILE_NAME': os.path.basename(imagePath),
                'HAS_ANNOTATION': False
            }
            imageInfo['NAME'] = imageInfo['FILE_NAME'].split('.')[0]
            if suffix != "":
                # Replacing only the last occurence https://stackoverflow.com/a/2556252/9962046
                imageInfo['NAME'] = ''.join(imageInfo['NAME'].rsplit(suffix, maxsplit=1))
            imageInfo['IMAGE_FORMAT'] = imageInfo['FILE_NAME'].split('.')[-1]

            # Reading input image in RGB color order
            imageChanged = False
            if self.__RESIZE is not None:  # If in mode, resize image to lower resolution
                imageInfo['ORIGINAL_IMAGE'] = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = imageInfo['ORIGINAL_IMAGE'].shape
                fullImage = cv2.resize(imageInfo['ORIGINAL_IMAGE'], self.__RESIZE)
                imageChanged = True
            else:
                fullImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = fullImage.shape
            imageInfo['HEIGHT'] = int(height)
            imageInfo['WIDTH'] = int(width)

            if self.__CONFIG.get_param().get('base_class', None) is not None \
                    and self.__CONFIG.get_previous_mode() is not None:
                imageInfo['BASE_CLASS'] = self.__CONFIG.get_param()['base_class']

            if 'BASE_CLASS' in imageInfo:
                imageInfo['BASE_AREA'] = height * width
                imageInfo['BASE_COUNT'] = 1

            # Conversion of the image if format is not png or jpg
            if imageInfo['IMAGE_FORMAT'] not in ['png', 'jpg']:
                imageInfo['IMAGE_FORMAT'] = 'jpg'
                imageChanged = True
                tempPath = os.path.join(imageInfo['PATH'], f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                imsave(tempPath, fullImage)
                imageInfo['PATH'] = tempPath

            # Creating the result dir if given and copying the base image in it
            if results_path is not None:
                image_results_path = os.path.join(os.path.normpath(results_path), imageInfo['NAME'])
                if chainMode:  # Saving into the specific inference mode
                    image_results_path = os.path.join(image_results_path, self.__MODE)
                if not os.path.exists(image_results_path):
                    os.makedirs(image_results_path, exist_ok=True)
                imageInfo['PATH'] = os.path.join(image_results_path, f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                if not imageChanged:
                    shutil.copy2(imagePath, imageInfo['PATH'])
                else:
                    imsave(imageInfo['PATH'], fullImage)
                if self.__RESIZE is not None:
                    originalImagePath = os.path.join(image_results_path,
                                                     f"{imageInfo['NAME']}_base.{imageInfo['IMAGE_FORMAT']}")
                    if imageInfo['IMAGE_FORMAT'] in imagePath:
                        shutil.copy2(imagePath, originalImagePath)
                    else:
                        imsave(originalImagePath, imageInfo['ORIGINAL_IMAGE'])
            else:
                image_results_path = None

            # If annotations found, create masks and clean image if possible
            annotationExists = False
            if not chainMode:
                for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                    annotationExists = annotationExists or os.path.exists(os.path.join(imageInfo['DIR_PATH'],
                                                                                       imageInfo['NAME'] + '.' + ext))
            if self.__CONFIG.get_param().get('allow_empty_annotations', False):
                imageInfo['HAS_ANNOTATION'] = annotationExists

            hasToExclude = self.__CONFIG.get_param().get('exclude_class', None) is not None
            hasBase = hasExcluded = usingPreviousBase = False
            fusedMask = fusedExcludedMask = None
            if chainMode and ('BASE_CLASS' in imageInfo or hasToExclude):
                if 'BASE_CLASS' in imageInfo:
                    if self.__CONFIG.get_param().get('base_class', None) == \
                            self.__CONFIG.get_param("previous").get('base_class', None):
                        # If base class is the same as previous mode (no test on base_class is None, made previously)
                        # TODO support previous BASE masks with centered mode (i.e. storing each individual masks)
                        usingPreviousBase = True
                        fusedMask = self.__BASE['masks']
                        hasBase = True
                    else:
                        baseClassId = self.__CONFIG.get_class_id(imageInfo['BASE_CLASS'], 'previous')
                        if self.__PREVIOUS_RES is not None and baseClassId in self.__PREVIOUS_RES['class_ids']:
                            fusedMask, hasBase = self.__gather_masks__(self.__PREVIOUS_RES, baseClassId, "previous",
                                                                       height, width)

                if hasToExclude:
                    classesToExclude = self.__CONFIG.get_param()['exclude_class']
                    if classesToExclude == "all":
                        classesToExclude = [c["id"] for c in self.__CONFIG.get_classes_info("previous")]
                    else:
                        if type(classesToExclude) in [str, int]:
                            classesToExclude = [classesToExclude]
                        classesToExclude = [self.__CONFIG.get_class_id('previous', c) for c in classesToExclude
                                            if self.__CONFIG.get_class_id('previous', c) != -1]
                    fusedExcludedMask, hasExcluded = self.__gather_masks__(self.__PREVIOUS_RES, classesToExclude,
                                                                           "previous", height, width)
                    if hasExcluded:
                        fusedExcludedMask = cv2.bitwise_not(fusedExcludedMask)

                if hasBase or hasExcluded:
                    if hasBase and hasExcluded:
                        fusedMask = cv2.bitwise_and(fusedMask, fusedExcludedMask)
                    elif hasExcluded:
                        fusedMask = fusedExcludedMask
                    image = cv2.bitwise_and(fullImage, np.repeat(fusedMask[..., np.newaxis], 3, axis=2))
                else:
                    if results_path is not None:
                        shutil.rmtree(image_results_path, ignore_errors=True)
                    return (None,) * 4

                if hasBase:
                    crop_to_remaining = self.__CONFIG.get_param().get('crop_to_remaining', False)
                    if crop_to_remaining:
                        fusedBbox = utils.extract_bboxes(fusedMask)
                        image = image[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3], :]
                        fullImage = fullImage[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3], :]
                        imsave(imageInfo['PATH'], fullImage)
                        height, width = image.shape[:2]
                        offset = np.array([fusedBbox[0], fusedBbox[1]] * 2)
                        if self.__CONFIG.has_to_return() and "base" in self.__CONFIG.get_return_param():
                            # Save base mask for next mode if needed
                            self.__BASE = {
                                'rois': np.array([0, 0, height, width], dtype=np.int32),
                                'masks': fusedMask[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3]]
                            }
                    elif self.__CONFIG.has_to_return() and "base" in self.__CONFIG.get_return_param():
                        self.__BASE = {
                            'rois': utils.extract_bboxes(fusedMask),
                            'masks': fusedMask
                        }

                    # If RoI mode is 'centered', inference will be done on base-class masks
                    if roi_mode == 'centered':
                        if usingPreviousBase:
                            raise NotImplementedError('Centered mode and same base as previous '
                                                      'mode is not currently supported')
                        if self.__CONFIG.get_param().get('fuse_base_class', False):
                            if crop_to_remaining:
                                imageInfo['ROI_COORDINATES'] = fusedBbox - offset
                            else:
                                imageInfo['ROI_COORDINATES'] = utils.extract_bboxes(fusedMask)
                        else:
                            indices = np.arange(len(self.__PREVIOUS_RES['class_ids']))
                            indices = indices[np.isin(self.__PREVIOUS_RES['class_ids'], [baseClassId])]
                            imageInfo['ROI_COORDINATES'] = self.__PREVIOUS_RES['rois'][indices]
                            if crop_to_remaining:
                                imageInfo['ROI_COORDINATES'] -= offset
                        for idx, bbox in enumerate(imageInfo['ROI_COORDINATES']):
                            imageInfo['ROI_COORDINATES'][idx] = dI.center_mask(bbox, (height, width),
                                                                               self.__DIVISION_SIZE, verbose=0)
                        imageInfo['NB_DIV'] = len(imageInfo['ROI_COORDINATES'])

                    # Getting count and area of base-class masks
                    if self.__CONFIG.get_param().get('fuse_base_class', False) or usingPreviousBase:
                        imageInfo.update({'BASE_AREA': dD.getBWCount(fusedMask)[1], 'BASE_COUNT': 1})
                    else:
                        indices = np.arange(len(self.__PREVIOUS_RES['class_ids']))
                        indices = indices[np.isin(self.__PREVIOUS_RES['class_ids'], [baseClassId])]
                        for idx in indices:
                            mask = self.__PREVIOUS_RES['masks'][..., idx]
                            if self.__CONFIG.get_mini_mask_shape('previous') is not None and \
                                    self.__PREVIOUS_RES['masks'].shape[:2] == \
                                    self.__CONFIG.get_mini_mask_shape('previous'):
                                bbox = self.__PREVIOUS_RES['rois'][idx]
                                if self.__CONFIG.get_param('previous').get('resize', None) is not None:
                                    mask = utils.expand_mask(
                                        bbox, mask, tuple(self.__CONFIG.get_param('previous')['resize'])
                                    )
                                    mask = cv2.resize(mask, (imageInfo['HEIGHT'], imageInfo['WIDTH']))
                                else:
                                    mask = utils.expand_mask(bbox, mask, (imageInfo['HEIGHT'], imageInfo['WIDTH']))
                            imageInfo['BASE_AREA'] += dD.getBWCount(mask)[1]
                            imageInfo['BASE_COUNT'] += 1
                            del mask
                    imageInfo['HEIGHT'] = int(height)
                    imageInfo['WIDTH'] = int(width)
            elif annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible",
                          flush=True, end='')
                    start_wrapping = time()
                dW.createMasksOfImage(imageInfo['DIR_PATH'], imageInfo['NAME'], 'data',
                                      classesInfo=self.__CLASSES_INFO, imageFormat=imageInfo['IMAGE_FORMAT'],
                                      resize=self.__RESIZE, config=self.__CONFIG)
                maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                if 'BASE_CLASS' in imageInfo and imageInfo['BASE_CLASS'] in maskDirs:
                    # Fusing masks of base class if needed, then cleaning image using it/them
                    if self.__CONFIG.get_param().get('fuse_base_class', False):
                        dW.fuseClassMasks('data', imageInfo['NAME'], imageInfo['BASE_CLASS'],
                                          imageFormat=imageInfo['IMAGE_FORMAT'], deleteBaseMasks=True, silent=True)
                    if hasToExclude:
                        classesToExclude = self.__CONFIG.get_param()['exclude_class']
                        if classesToExclude == "all":
                            classesToExclude = [c["name"] for c in self.__CONFIG.get_classes_info("previous")]
                        elif type(classesToExclude) is str:
                            classesToExclude = [classesToExclude]
                    else:
                        classesToExclude = None
                    dW.cleanImage('data', imageInfo['NAME'], cleaningClasses=imageInfo['BASE_CLASS'],
                                  excludeClasses=classesToExclude, imageFormat=imageInfo['IMAGE_FORMAT'],
                                  cleanMasks=False)
                    maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                    # If RoI mode is 'centered', inference will be done on base-class masks
                    if roi_mode == 'centered':
                        imageInfo['ROI_COORDINATES'] = dI.getCenteredClassBboxes(
                            datasetPath='data', imageName=imageInfo['NAME'], classToCenter=imageInfo['BASE_CLASS'],
                            image_size=self.__DIVISION_SIZE, imageFormat=imageInfo['IMAGE_FORMAT'],
                            allow_oversized=True, config=self.__CONFIG
                        )
                        imageInfo['NB_DIV'] = len(imageInfo['ROI_COORDINATES'])

                    # Getting count and area of base-class masks
                    if imageInfo['BASE_CLASS'] not in self.__CUSTOM_CLASS_NAMES:
                        maskDirs.remove(imageInfo['BASE_CLASS'])
                    imageInfo.update({'BASE_AREA': 0, 'BASE_COUNT': 0})
                    baseClassDirPath = os.path.join('data', imageInfo['NAME'], imageInfo['BASE_CLASS'])
                    for baseClassMask in os.listdir(baseClassDirPath):
                        baseMask = dW.loadSameResImage(os.path.join(baseClassDirPath, baseClassMask),
                                                       fullImage.shape)
                        imageInfo['BASE_AREA'] += dD.getBWCount(baseMask)[1]
                        imageInfo['BASE_COUNT'] += 1
                        del baseMask

                    # If full_images directory exists: image has been cleaned so we have to get it another time
                    if 'full_images' in maskDirs:
                        imagesDirPath = os.path.join('data', imageInfo['NAME'], 'images')
                        imageFilePath = os.listdir(imagesDirPath)[0]
                        imageInfo['cleaned_image_path'] = os.path.join(imagesDirPath, imageFilePath)
                        image = cv2.cvtColor(cv2.imread(imageInfo['cleaned_image_path']), cv2.COLOR_BGR2RGB)

                # TODO TIME Wrapper/Clean
                if not silent:
                    end_wrapping = time()
                    print(f' Duration = {formatTime(end_wrapping - start_wrapping)}')
                for notAClass in ['images', 'full_images']:
                    if notAClass in maskDirs:
                        maskDirs.remove(notAClass)

                # Image has annotation if at least a class that we want to predict has a mask (i.e. is present)
                if not imageInfo['HAS_ANNOTATION']:
                    imageInfo['HAS_ANNOTATION'] = any([d in self.__CUSTOM_CLASS_NAMES for d in maskDirs])

                if imageInfo['HAS_ANNOTATION'] and not silent:
                    print("    - Confusion matrix will be computed")

            if roi_mode == "divided" or (roi_mode == 'centered' and 'ROI_COORDINATES' not in imageInfo):
                imageInfo['X_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=width if self.__RESIZE is None else self.__RESIZE[0],
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=0.33 if self.__MIN_OVERLAP_PART is None else self.__MIN_OVERLAP_PART
                )
                imageInfo['Y_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=height if self.__RESIZE is None else self.__RESIZE[0],
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=0.33 if self.__MIN_OVERLAP_PART is None else self.__MIN_OVERLAP_PART
                )
                imageInfo['NB_DIV'] = dD.getDivisionsCount(imageInfo['X_STARTS'], imageInfo['Y_STARTS'])
            elif roi_mode is None:
                imageInfo['X_STARTS'] = imageInfo['Y_STARTS'] = [0]
                imageInfo['NB_DIV'] = 1
            elif roi_mode != "centered":
                raise NotImplementedError(f'\'{roi_mode}\' RoI mode is not implemented.')

        return image, fullImage, imageInfo, image_results_path

    def __gather_masks__(self, results, classId, mode, height, width):
        if type(classId) is list:
            classId_ = classId
        else:
            classId_ = [classId]
        indices = np.arange(len(results['class_ids']))
        indices = indices[np.isin(results['class_ids'], classId_)]
        fusedMask = None
        if len(indices) > 0:
            temp = self.__CONFIG.get_mini_mask_shape(mode)
            modeUsingMiniMask = temp is not None and results['masks'].shape[:2] == temp
            hasResize = self.__CONFIG.get_param(mode).get('resize', None)
            fusedMask = np.zeros((height, width), dtype=np.uint8)
            for idx in indices:
                mask = results['masks'][:, :, idx].astype(np.uint8) * 255
                my1, mx1, my2, mx2 = y1, x1, y2, x2 = bbox = results['rois'][idx]
                if modeUsingMiniMask:
                    if hasResize is not None:
                        mask = utils.expand_mask(bbox, mask, tuple(hasResize)).astype(np.uint8) * 255
                        mask = cv2.resize(mask, (width, height))
                    else:
                        shifted = utils.shift_bbox(bbox)
                        my1, mx1, my2, mx2 = shifted
                        mask = utils.expand_mask(shifted, mask, shifted[2:]).astype(np.uint8) * 255
                elif hasResize:
                    yRatio = height / hasResize[0]
                    xRatio = width / hasResize[1]
                    bbox = np.around(bbox * [yRatio, xRatio, yRatio, xRatio]).astype(np.int)
                    my1, mx1, my2, mx2 = y1, x1, y2, x2 = bbox
                    mask = cv2.resize(mask, (width, height))
                # TODO : Optimize fusion by working only on mask's bbox
                fusedMask[y1:y2, x1:x2] = np.bitwise_or(fusedMask[y1:y2, x1:x2], mask[my1:my2, mx1:mx2])
        return fusedMask, len(indices) > 0

    @staticmethod
    def __init_results_dir__(results_path):
        if results_path is None or results_path in ['', '.', './', "/"]:
            lastDir = "results"
            remainingPath = ""
        else:
            results_path = os.path.normpath(results_path)
            lastDir = os.path.basename(results_path)
            remainingPath = os.path.dirname(results_path)
        results_path = os.path.normpath(os.path.join(remainingPath, f"{lastDir}_{formatDate()}"))
        os.makedirs(results_path)
        print(f"Results will be saved to {results_path}")
        logsPath = os.path.join(results_path, 'inference_data.csv')
        with open(logsPath, 'w') as results_log:
            results_log.write(f"Image; Duration (s); Inference Mode\n")
        return results_path, logsPath

    def inference(self, images: list, results_path=None, chainMode=False, save_results=True, displayOnlyStats=False,
                  saveDebugImages=False, chainModeForceFullSize=False, verbose=0, debugMode=False):

        if len(images) == 0:
            print("Images list is empty, no inference to perform.")
            return

        # If results have to be saved, setting the results path and creating directory
        if save_results:
            results_path, logsPath = self.__init_results_dir__(results_path)
        else:
            print("No result will be saved")
            results_path = None

        if not chainMode:
            self.__CONFUSION_MATRIX['mask'].fill(0)
            self.__CONFUSION_MATRIX['pixel'].fill(0)
            self.__CONFUSION_MATRIX['count'] = 0
        total_start_time = time()
        failedImages = []
        for img_idx, IMAGE_PATH in enumerate(images):
            try:
                print(f"Using {IMAGE_PATH} image file {progressText(img_idx + 1, len(images))}")
                image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
                if chainMode:
                    nextMode = self.__CONFIG.get_first_mode()
                    stay = True
                    while nextMode is not None and stay:
                        print(f"[{nextMode} mode]")
                        self.load(nextMode, forceFullSizeMasks=chainModeForceFullSize)
                        nextMode = self.__CONFIG.get_next_mode()
                        stay = self.__process_image__(IMAGE_PATH, results_path, chainMode, saveDebugImages,
                                                      save_results, displayOnlyStats, logsPath, verbose=verbose)
                        if not stay:
                            print(f"{image_name} does not have required data (ex : needed class from previous mode is "
                                  f"missing) to continue current and/or next inference modes processing.\n")
                else:
                    self.__process_image__(IMAGE_PATH, results_path, chainMode, saveDebugImages, save_results,
                                           displayOnlyStats, logsPath, verbose=verbose)
            except Exception as e:
                failedImages.append(os.path.basename(IMAGE_PATH))
                try:
                    with open(os.path.join(results_path, "failed.json"), 'w') as failedJsonFile:
                        json.dump(failedImages, failedJsonFile, indent="\t")
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                print(f"\n/!\\ Failed {IMAGE_PATH} at \"{self.__STEP}\"\n", flush=True)
                if debugMode:
                    raise e
                else:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    sys.stderr.flush()
                if save_results and self.__STEP not in ["image preparation", "finalizing"]:
                    with open(logsPath, 'a') as results_log:
                        results_log.write(f"{image_name}; -1;FAILED ({self.__STEP});\n")

        if not chainMode and self.__CONFUSION_MATRIX['count'] > 1:
            cmap = plt.cm.get_cmap('hot')
            for mat_type in ['mask', 'pixel']:
                for normalized in [False, True]:
                    name = f"Final Confusion Matrix ({mat_type.capitalize()}){' (Normalized)' if normalized else ''}"
                    confusionMatrixFileName = os.path.join(results_path, name.replace('(', '')
                                                           .replace(')', '')
                                                           .replace(' ', '_'))
                    visualize.display_confusion_matrix(self.__CONFUSION_MATRIX[mat_type],
                                                       self.__VISUALIZE_NAMES.copy(), title=name, cmap=cmap,
                                                       show=False, normalize=normalized,
                                                       fileName=confusionMatrixFileName)
            plt.close('all')
        total_time = round(time() - total_start_time)
        print(f"All inferences done in {formatTime(total_time)}")
        if save_results:
            with open(logsPath, 'a') as results_log:
                results_log.write(f"GLOBAL; {total_time};\n")

        # Displaying failed images list if not empty
        if len(failedImages) > 0:
            print(f"\n{len(failedImages)} image{'s' if len(failedImages) > 1 else ''} failed :")
            print("\n".join(failedImages))

    def __process_image__(self, image_path, results_path, chainMode=False, saveDebugImages=False, save_results=True,
                          displayOnlyStats=False, logsPath=None, verbose=0):
        allowSparse = self.__CONFIG.get_param().get('allow_sparse', True)
        start_time = time()
        self.__STEP = "image preparation"
        image, fullImage, imageInfo, image_results_path = self.__prepare_image__(image_path, results_path, chainMode,
                                                                                 silent=displayOnlyStats)
        if fullImage is None and imageInfo is None:
            return False

        if imageInfo["HAS_ANNOTATION"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ground_truth = self.__get_ground_truth__(image_results_path, fullImage,
                                                         imageInfo, displayOnlyStats, save_results)
            if save_results:
                if self.__LOW_MEMORY:
                    del fullImage
                    gc.collect()
                    fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

        debugIterator = -1
        res = self.__inference__(image, fullImage, imageInfo, allowSparse, displayOnlyStats)

        """####################
        ### Post-Processing ###
        ####################"""

        def gather_dynamic_args(method: DynamicMethod):
            dynamic_args = {}
            for dynamic_arg in method.dynargs():
                if dynamic_arg == 'image':
                    dynamic_args[dynamic_arg] = image if image is not None else fullImage
                elif dynamic_arg == 'image_info':
                    dynamic_args[dynamic_arg] = imageInfo
                elif dynamic_arg == 'save':
                    dynamic_args[dynamic_arg] = image_results_path
                elif dynamic_arg == 'base_res' and chainMode:
                    dynamic_args[dynamic_arg] = self.__PREVIOUS_RES
                elif dynamic_arg == 'base_res':
                    dynamic_args[dynamic_arg] = self.__get_ground_truth__(image_results_path, fullImage, imageInfo,
                                                                          displayOnlyStats, False, base_gt=True)
                else:
                    raise NotImplementedError(f'Dynamic argument \'{dynamic_arg}\' is not implemented.')
            return dynamic_args

        if len(res['class_ids']) > 0:
            for methodInfo in self.__CONFIG.get_post_processing_method():
                self.__STEP = f"post-processing ({methodInfo['method']})"
                # if saveDebugImages and idx != 0:
                if saveDebugImages:
                    debugIterator += 1
                    self.__save_debug_image__(f"pre_{methodInfo['method']}", debugIterator, fullImage,
                                              imageInfo, res, image_results_path, silent=displayOnlyStats)
                    if self.__LOW_MEMORY:
                        del fullImage
                        gc.collect()
                        fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

                ppMethod = pp.PostProcessingMethod(methodInfo['method'])
                dynargs = gather_dynamic_args(ppMethod)

                res = ppMethod.method(results=res, config=self.__CONFIG, args=methodInfo, dynargs=dynargs,
                                      display=not displayOnlyStats, verbose=verbose)

        """###############
        ### Evaluation ###
        ###############"""
        if imageInfo["HAS_ANNOTATION"]:
            self.__compute_confusion_matrices__(image_results_path, imageInfo, res, ground_truth,
                                                save_results, displayOnlyStats)
            del ground_truth

        """###############
        ### Statistics ###
        ###############"""
        res_stats = {}
        if len(res['class_ids']) > 0:
            for methodInfo in self.__CONFIG.get_statistics_method():
                self.__STEP = f"statistics ({methodInfo['method']})"
                ppMethod = stats.StatisticsMethod(methodInfo['method'])
                dynargs = gather_dynamic_args(ppMethod)

                res_stats[methodInfo['method']] = ppMethod.method(
                    results=res, config=self.__CONFIG, args=methodInfo, dynargs=dynargs,
                    display=not displayOnlyStats, verbose=verbose
                )

        """#################
        ### Finalization ###
        #################"""
        if save_results:
            # Cleaned image export
            if self.__CONFIG.has_to_export_cleaned_img():
                self.__clean_image__(fullImage, image, imageInfo, res, image_results_path, displayOnlyStats)

            # Annotations Extraction
            if self.__CONFIG.has_to_export():
                self.__export_annotations__(image_results_path, imageInfo, res, displayOnlyStats)

            if len(res['class_ids']) > 0:
                if not displayOnlyStats:
                    print(" - Applying masks on image")
                self.__STEP = "saving predicted image"
                self.__draw_masks__(image_results_path, fullImage, imageInfo, res,
                                    title=f"{imageInfo['NAME']} Predicted")
            elif not displayOnlyStats:
                print(" - No mask to apply on image")

        final_time = round(time() - start_time)
        print(f" Done in {formatTime(final_time)}\n")
        plt.clf()
        plt.close('all')

        """########################################
        ### Extraction of needed classes' masks ###
        ########################################"""
        continueChain = True
        if chainMode:
            self.__PREVIOUS_RES = {}
            if self.__CONFIG.has_to_return():
                return_param = self.__CONFIG.get_return_param().copy()
                if type(return_param) is str:
                    return_param = [return_param]
                if "base" in return_param:
                    return_param.remove("base")
                if len(return_param) > 0:
                    if "all" in return_param:
                        classIdsToGet = [c['id'] for c in self.__CLASSES_INFO]
                    else:
                        classIdsToGet = [self.__CLASS_2_ID[c] for c in return_param]
                    indices = np.arange(len(res['class_ids']))
                    indices = indices[np.isin(res['class_ids'], classIdsToGet)]
                    if len(indices) == 0:
                        continueChain = False
                    else:
                        for key in res:
                            if key == 'masks':
                                self.__PREVIOUS_RES[key] = res[key][..., indices]
                            else:
                                self.__PREVIOUS_RES[key] = res[key][indices, ...]

        self.__STEP = "finalizing"
        if save_results:
            with open(logsPath, 'a') as results_log:
                results_log.write(f"{imageInfo['NAME']}; {final_time}; {self.__MODEL_PATH};\n")
        del res, imageInfo, fullImage
        gc.collect()
        return continueChain

    def __inference__(self, image, fullImage, imageInfo, allowSparse, displayOnlyStats):
        res = []
        total_px = self.__CONFIG.get_param()['roi_size'] ** 2
        skipped = 0
        skippedText = ""
        inference_start_time = time()
        if not displayOnlyStats:
            progressBar(0, imageInfo["NB_DIV"], prefix=' - Inference')
        for divId in range(imageInfo["NB_DIV"]):
            self.__STEP = f"{divId} div processing"
            forceInference = False
            if 'X_STARTS' in imageInfo and 'Y_STARTS' in imageInfo:
                division = dD.getImageDivision(fullImage if image is None else image, imageInfo["X_STARTS"],
                                               imageInfo["Y_STARTS"], divId, self.__DIVISION_SIZE)
            elif 'ROI_COORDINATES' in imageInfo:
                currentRoI = imageInfo['ROI_COORDINATES'][divId]
                division = (fullImage if image is None else image)[
                           currentRoI[0]:currentRoI[2], currentRoI[1]:currentRoI[3], :
                           ]
                forceInference = True
            else:
                raise ValueError('Cannot find image areas to use')

            if not forceInference:
                grayDivision = cv2.cvtColor(division, cv2.COLOR_RGB2GRAY)
                colorPx = cv2.countNonZero(grayDivision)
                del grayDivision
            if forceInference or colorPx / total_px > 0.1:
                self.__STEP = f"{divId} div inference"
                results = self.__MODEL.process(division, normalizedCoordinates=False,
                                               score_threshold=self.__CONFIG.get_param()['min_confidence'])
                results["div_id"] = divId
                if self.__CONFIG.is_using_mini_mask():
                    res.append(utils.reduce_memory(results.copy(), config=self.__CONFIG, allow_sparse=allowSparse))
                else:
                    res.append(results.copy())
                del results
            elif not displayOnlyStats:
                skipped += 1
                skippedText = f"({skipped} empty division{'s' if skipped > 1 else ''} skipped) "
            del division
            gc.collect()

            if not displayOnlyStats:
                if divId + 1 == imageInfo["NB_DIV"]:
                    inference_duration = round(time() - inference_start_time)
                    skippedText += f"Duration = {formatTime(inference_duration)}"
                progressBar(divId + 1, imageInfo["NB_DIV"], prefix=' - Inference', suffix=skippedText)
        if not displayOnlyStats:
            print(" - Fusing results of all divisions")
        self.__STEP = "fusing results"
        res = pp.fuse_results(res, imageInfo, division_size=self.__DIVISION_SIZE, resize=self.__RESIZE,
                              config=self.__CONFIG)
        return res

    def __draw_masks__(self, image_results_path, img, image_info, masks_data, title=None, cleaned_image=True):
        if title is None:
            title = f"{image_info['NAME']} Masked"
        fileName = os.path.join(image_results_path, title.replace(' ', '_').replace('(', '').replace(')', ''))

        # Computing figure size
        figsize = self.__CONFIG.get_param().get('resize', None)
        if figsize is None:
            figsize = (image_info['WIDTH'] / 100, image_info['HEIGHT'] / 100)
        else:
            figsize = (figsize[0] / 100, figsize[1] / 100)

        # No need of reloading or passing copy of image as it is the final drawing
        visualize.display_instances(
            img, masks_data['rois'], masks_data['masks'], masks_data['class_ids'],
            self.__VISUALIZE_NAMES, masks_data['scores'] if 'scores' in masks_data else None, colors=self.__COLORS,
            colorPerClass=True, fileName=fileName, save_cleaned_img=cleaned_image, silent=True, title=title,
            figsize=figsize, image_format=image_info['IMAGE_FORMAT'], config=self.__CONFIG
        )

    def __get_ground_truth__(self, image_results_path, fullImage, image_info,
                             displayOnlyStats, save_results, base_gt=False):
        """
        Get ground-truth annotations and applies masks on image if enabled
        :param image_results_path: the current image output folder
        :param fullImage: the original image
        :param image_info: info about the current image
        :param displayOnlyStats: if True, will not print anything
        :param save_results: if True, will apply masks
        :return: gt_bbox, gt_class_id, gt_mask
        """
        datasetType = "base" if base_gt else "current"
        self.__STEP = f"{datasetType} dataset creation"
        dataset_val = CustomDataset("InferenceTool", image_info, self.__CONFIG,
                                    previous_mode=base_gt, enable_occlusion=False)
        dataset_val.load_images()
        dataset_val.prepare()
        self.__STEP = f"loading {datasetType} annotated masks"
        image_id = dataset_val.image_ids[0]
        gt_mask, gt_class_id, gt_bbox = dataset_val.load_mask(image_id)
        gt = {'rois': gt_bbox, 'class_ids': gt_class_id, 'masks': gt_mask}
        if save_results:
            if len(gt_class_id) > 0:
                if not displayOnlyStats:
                    print(" - Applying annotations on file to get expected results")
                self.__STEP = "applying expected masks"
                self.__draw_masks__(image_results_path, fullImage if self.__LOW_MEMORY else fullImage.copy(),
                                    image_info, gt, title=f"{image_info['NAME']}_Expected", cleaned_image=False)
            elif not displayOnlyStats:
                print(" - No mask expected, not saving expected image")
        return gt

    def __compute_statistics__(self, image_results_path, image_info, predicted, save_results=True):
        """
        Computes area and counts masks of each class
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param save_results: Whether to save statistics in a file or not
        :return: None
        """
        self.__STEP = "computing statistics"
        _ = stats.get_count_and_area(predicted, image_info=image_info, selected_classes="all",
                                     save=image_results_path if save_results else None, display=True,
                                     config=self.__CONFIG)

    def __compute_confusion_matrices__(self, image_results_path, image_info, predicted, ground_truth,
                                       save_results=True, displayOnlyStats=False):
        """
        Computes confusion matrix
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param ground_truth: the ground truth results dictionary
        :param save_results: Whether to save files or not
        :param displayOnlyStats: Whether to display only stats or also current steps
        :return:
        """
        if not displayOnlyStats:
            print(" - Computing Confusion Matrices")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__STEP = "computing mask confusion matrix"
            conf_matrix = {
                'mask': utils.compute_matches(
                    gt_boxes=ground_truth['rois'], gt_class_ids=ground_truth['class_ids'],
                    gt_masks=ground_truth['masks'], pred_boxes=predicted["rois"],
                    pred_class_ids=predicted["class_ids"], pred_masks=predicted['masks'],
                    pred_scores=predicted["scores"], nb_class=self.__NB_CLASS,
                    min_iou_to_count=0.3, ap_iou_threshold=0.5, confusion_iou_threshold=0.5,
                    classes_hierarchy=self.__CONFIG.get_classes_hierarchy(),
                    confusion_background_class=True, confusion_only_best_match=False)[-1]
            }
            self.__CONFUSION_MATRIX['mask'] = np.add(self.__CONFUSION_MATRIX['mask'], conf_matrix['mask'])

            self.__STEP = "computing pixel confusion matrix"
            conf_matrix['pixel'] = utils.compute_confusion_matrix(
                image_shape=self.__RESIZE if self.__RESIZE is not None else (image_info['HEIGHT'], image_info['WIDTH']),
                expectedResults=ground_truth, predictedResults=predicted,
                num_classes=self.__NB_CLASS, config=self.__CONFIG
            )
            self.__CONFUSION_MATRIX['pixel'] = np.add(self.__CONFUSION_MATRIX['pixel'], conf_matrix['pixel'])

            self.__CONFUSION_MATRIX['count'] += 1
            cmap = plt.cm.get_cmap('hot')
            if save_results:
                self.__STEP = "saving confusion matrices"
                for mat_type in ['mask', 'pixel']:
                    for normalized in [False, True]:
                        name = (f"{image_info['NAME']} Confusion Matrix ({mat_type.capitalize()})"
                                f"{' (Normalized)' if normalized else ''}")
                        confusionMatrixFileName = os.path.join(
                            image_results_path, name.replace('(', '').replace(')', '').replace(' ', '_')
                        )
                        visualize.display_confusion_matrix(
                            conf_matrix[mat_type], self.__VISUALIZE_NAMES, title=name, cmap=cmap,
                            show=False, normalize=normalized, fileName=confusionMatrixFileName
                        )

    def __save_debug_image__(self, step, debugIterator, fullImage, image_info, res, image_results_path, silent=True):
        if not silent:
            print(f" - Saving {step} image")
        title = f"{image_info['NAME']} Inference debug {debugIterator:02d} {step}"
        self.__draw_masks__(image_results_path, fullImage if self.__LOW_MEMORY else fullImage.copy(),
                            image_info, res, title, cleaned_image=False)

    def __export_annotations__(self, image_results_path, image_info, predicted, silent, verbose=0):
        """
        Exports predicted results as annotations files
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param silent: Whether to print text or not
        :return:
        """
        self.__STEP = "saving annotations"
        if not silent:
            print(" - Saving predicted annotations files")
        adapters = []
        export_param = self.__CONFIG.get_export_param()
        if type(export_param) is str:
            if export_param.lower() == "all":
                adapters = list(AnnotationAdapter.ANNOTATION_ADAPTERS.values())
            else:
                export_param = [export_param]
        if type(export_param) is list:
            for adapterName in export_param:
                adapter = AnnotationAdapter.getAdapterFromName(adapterName)
                if adapter is not None:
                    adapters.append(adapter)
        for adapter in adapters:
            try:
                if not silent:
                    print(' - ', end='')
                export_annotations(image_info, predicted, adapter, save_path=image_results_path,
                                   config=self.__CONFIG, verbose=verbose if silent else 1)
            except Exception:
                print(f"Failed to export using {adapter.getName()} adapter")

    def __clean_image__(self, fullImage, image, imageInfo, res, image_results_path, displayOnlyStats):
        """ Cleans the image following the config parameter """
        self.__STEP = "exporting cleaned image"
        if not displayOnlyStats:
            print(' - Exporting cleaned image')
        cleaning_params = self.__CONFIG.get_export_param_cleaned_img()
        if type(cleaning_params) is not list:
            cleaning_params = [cleaning_params]

        for idx, cleaning_param in enumerate(cleaning_params):
            name = cleaning_param.get('name', None)
            if name is None:
                name = f"{idx:02d}"
            # Gathering base masks if needed
            base_param = cleaning_param.get('base_class', None)
            if type(base_param) is not list:
                base_param = [base_param]
            baseMask, reuseImage = (None, False)
            if 'base' in base_param or ('BASE_CLASS' in imageInfo and imageInfo['BASE_CLASS'] in base_param):
                # TODO Add support for resized image
                reuseImage = hasBase = True
            else:
                if "all" in base_param:
                    classIDs = [c['id'] for c in self.__CONFIG.get_classes_info()]
                else:
                    classIDs = [self.__CONFIG.get_class_id(c) for c in base_param]
                baseMask, hasBase = self.__gather_masks__(
                    results=res, classId=classIDs,
                    mode="current", height=imageInfo['HEIGHT'], width=imageInfo['WIDTH']
                )
            if not hasBase:
                baseMask = np.ones((imageInfo['HEIGHT'], imageInfo['WIDTH']), dtype=np.uint8)

            # Gathering excluded masks if needed
            exclude_param = cleaning_param.get('exclude_class', None)
            excludedMask, hasExcluded = (None, False)
            if exclude_param is not None:
                if type(exclude_param) is not list:
                    exclude_param = [exclude_param]
                if "all" in exclude_param:
                    classIDs = [c['id'] for c in self.__CONFIG.get_classes_info()]
                else:
                    classIDs = [self.__CONFIG.get_class_id(c) for c in exclude_param]
                excludedMask, hasExcluded = self.__gather_masks__(
                    results=res, classId=classIDs,
                    mode="current", height=imageInfo['HEIGHT'], width=imageInfo['WIDTH']
                )
                if hasExcluded:
                    excludedMask = cv2.bitwise_not(excludedMask)

            # Cleaning the image and resizing if needed
            resImage = None
            cropToRemaining = cleaning_param.get('crop_to_remaining', False)
            if reuseImage:
                if hasExcluded:
                    y1, x1, y2, x2 = (0, 0, imageInfo['HEIGHT'], imageInfo['WIDTH'])
                    if cropToRemaining:
                        y1, x1, y2, x2 = utils.extract_bboxes(excludedMask)
                    resImage = cv2.bitwise_and(
                        image[y1:y2, x1:x2, :],
                        np.repeat(excludedMask[y1:y2, x1:x2, np.newaxis], 3, axis=2)
                    )
                else:
                    resImage = image
            elif hasBase or hasExcluded:
                if hasExcluded:
                    baseMask = cv2.bitwise_and(baseMask, excludedMask)
                y1, x1, y2, x2 = (0, 0, imageInfo['HEIGHT'], imageInfo['WIDTH'])
                if cropToRemaining:
                    y1, x1, y2, x2 = utils.extract_bboxes(baseMask)
                resImage = cv2.bitwise_and(
                    (imageInfo['ORIGINAL_IMAGE'] if 'ORIGINAL_IMAGE' in imageInfo else fullImage)[y1:y2, x1:x2, :],
                    np.repeat(baseMask[y1:y2, x1:x2, np.newaxis], 3, axis=2)
                )

            # Saving the image
            if resImage is not None:
                cv2.imwrite(
                    os.path.join(image_results_path, f"{imageInfo['NAME']}_cleaned_{name}.jpg"),
                    cv2.cvtColor(resImage, cv2.COLOR_RGB2BGR), CV2_IMWRITE_PARAM
                )
