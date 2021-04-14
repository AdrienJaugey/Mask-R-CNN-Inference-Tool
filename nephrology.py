import json
import os
import re
import traceback
import shutil
import sys
import warnings
import gc

from datasetTools.CustomDataset import SkinetCustomDataset

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from common_utils import progressBar, formatTime, formatDate, progressText
    from datasetTools.datasetDivider import CV2_IMWRITE_PARAM
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.io import imread, imsave
    from datasetTools import datasetDivider as dD, datasetWrapper as dW, AnnotationAdapter, datasetIsolator as dI

    from mrcnn import utils
    from mrcnn.TensorflowDetector import TensorflowDetector
    from mrcnn import visualize
    from mrcnn import post_processing as pp


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


class NephrologyInferenceModel:

    def __init__(self, classesInfo, modelPath, min_confidence=0.5, divisionSize=1024, min_overlap_part_main=0.33,
                 min_overlap_part_cortex=0.5, cortex_size=None, mini_mask_size=96, forceFullSizeMasks=False,
                 low_memory=False):
        print("Initialisation")
        self.__STEP = "init"

        # to capture mode + version : ^skinet_([a-zA-Z_]+)_v([0-9]+|%LAST%).*$
        self.__MODEL_PATH = find_latest_weight(modelPath)
        reg = re.compile(r'^skinet_([a-zA-Z_]+)_v[0-9]+.*$')
        # Testing only for one of the format, as Error would have already been raised if modelPath was not correct
        isExportedModelDir = os.path.exists(os.path.join(modelPath, 'saved_model'))
        if isExportedModelDir:
            self.__MODEL_PATH = os.path.join(self.__MODEL_PATH, 'saved_model')
        self.__MODE = reg.search(os.path.basename(self.__MODEL_PATH)).group(1)
        cortex_mode = self.__MODE == "cortex"

        self.__MODEL = TensorflowDetector(self.__MODEL_PATH)
        if not self.__MODEL.isLoaded():
            raise ValueError("Please provide correct path to model.")
        # TODO Build classesInfo from model's config ?
        self.__CLASSES_INFO = classesInfo

        self.__DIVISION_SIZE = divisionSize
        self.__MIN_OVERLAP_PART_MAIN = min_overlap_part_main
        self.__MIN_OVERLAP_PART_CORTEX = min_overlap_part_cortex
        self.__MIN_OVERLAP_PART = min_overlap_part_cortex if cortex_mode else min_overlap_part_main
        self.__CORTEX_SIZE = None if not cortex_mode else (1024, 1024) if cortex_size is None else cortex_size
        self.__LOW_MEMORY = low_memory
        self.__CUSTOM_CLASS_NAMES = []
        for classInfo in classesInfo:
            if not classInfo["ignore"]:
                self.__CUSTOM_CLASS_NAMES.append(classInfo["name"])
        self.__NB_CLASS = len(self.__CUSTOM_CLASS_NAMES)
        self.__VISUALIZE_NAMES = self.__CUSTOM_CLASS_NAMES.copy()
        self.__VISUALIZE_NAMES.insert(0, 'background')

        self.__COLORS = [[0, 0, 0]] * self.__NB_CLASS
        model_config = self.__MODEL.getConfig()
        for classId in model_config:
            self.__COLORS[classId - 1] = model_config[classId]["color"]
        # Root directory of the project
        '''self.__ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "logs")'''

        # Configurations
        nbClass = self.__NB_CLASS
        divSize = 1024 if self.__DIVISION_SIZE == "noDiv" else self.__DIVISION_SIZE
        min_overlap_part = self.__MIN_OVERLAP_PART
        self.__CONFUSION_MATRIX = np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64)
        self.__APs = []

        class SkinetConfig:
            NAME = "skinet"
            NUM_CLASSES = 1 + nbClass
            IMAGE_SIDE = divSize
            MIN_OVERLAP = min_overlap_part
            DETECTION_MIN_CONFIDENCE = min_confidence
            USE_MINI_MASK = not cortex_mode and not forceFullSizeMasks
            MINI_MASK_SHAPE = (mini_mask_size, mini_mask_size)  # (height, width) of the mini-mask

        self.__CONFIG = SkinetConfig()

        '''# Recreate the model in inference mode
        self.__MODEL = modellib.MaskRCNN(mode="inference", config=self.__CONFIG, model_dir=self.__MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        assert self.__MODEL_PATH is not None and self.__MODEL_PATH != "", "Provide path to trained weights"
        print("Loading weights from", self.__MODEL_PATH)
        self.__MODEL.load_weights(self.__MODEL_PATH, by_name=True)'''
        print()

    def prepare_image(self, imagePath, results_path, silent=False):
        """
        Creating png version if not existing, dataset masks if annotation found and get some information
        :param imagePath: path to the image to use
        :param results_path: path to the results dir to create the image folder and paste it in
        :param silent: No display
        :return: image, imageInfo = {"PATH": str, "DIR_PATH": str, "FILE_NAME": str, "NAME": str, "HEIGHT": int,
        "WIDTH": int, "NB_DIV": int, "X_STARTS": v, "Y_STARTS": list, "HAS_ANNOTATION": bool}
        """
        image = None
        fullImage = None
        imageInfo = None
        image_results_path = None
        cortex_mode = self.__MODE == "cortex"
        if os.path.exists(imagePath):
            imageInfo = {
                'PATH': imagePath,
                'DIR_PATH': os.path.dirname(imagePath),
                'FILE_NAME': os.path.basename(imagePath)
            }
            imageInfo['NAME'] = imageInfo['FILE_NAME'].split('.')[0]
            imageInfo['IMAGE_FORMAT'] = imageInfo['FILE_NAME'].split('.')[-1]

            # Reading input image in RGB color order
            imageChanged = False
            if cortex_mode:  # If in cortex mode, resize image to lower resolution
                imageInfo['FULL_RES_IMAGE'] = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = imageInfo['FULL_RES_IMAGE'].shape
                fullImage = cv2.resize(imageInfo['FULL_RES_IMAGE'], self.__CORTEX_SIZE)
                imageChanged = True
            else:
                fullImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = fullImage.shape
            imageInfo['HEIGHT'] = int(height)
            imageInfo['WIDTH'] = int(width)

            if "main" in self.__MODE:
                imageInfo['BASE_CLASS'] = "cortex"
            elif self.__MODE == "mest_glom":
                imageInfo['BASE_CLASS'] = "nsg"
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
                os.makedirs(image_results_path, exist_ok=True)
                imageInfo['PATH'] = os.path.join(image_results_path, f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                if not imageChanged:
                    shutil.copy2(imagePath, imageInfo['PATH'])
                else:
                    imsave(imageInfo['PATH'], fullImage)
            else:
                image_results_path = None

            # Computing divisions coordinates if needed and total number of div
            if self.__DIVISION_SIZE == "noDiv":
                imageInfo['X_STARTS'] = imageInfo['Y_STARTS'] = [0]
            else:
                imageInfo['X_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=self.__CORTEX_SIZE[0] if cortex_mode else width,
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=self.__MIN_OVERLAP_PART
                )
                imageInfo['Y_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=self.__CORTEX_SIZE[1] if cortex_mode else height,
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=self.__MIN_OVERLAP_PART
                )
            imageInfo['NB_DIV'] = dD.getDivisionsCount(imageInfo['X_STARTS'], imageInfo['Y_STARTS'])

            # If annotations found, create masks and clean image if possible
            annotationExists = False

            for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                annotationExists = annotationExists or os.path.exists(os.path.join(imageInfo['DIR_PATH'],
                                                                                   imageInfo['NAME'] + '.' + ext))
            imageInfo['HAS_ANNOTATION'] = annotationExists if self.__MODE == "mest_glom" else False
            if annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible", flush=True)
                dW.createMasksOfImage(imageInfo['DIR_PATH'], imageInfo['NAME'], 'data', classesInfo=self.__CLASSES_INFO,
                                      imageFormat=imageInfo['IMAGE_FORMAT'], resize=self.__CORTEX_SIZE,
                                      config=self.__CONFIG)
                if self.__MODE in ['main', 'mest_main']:
                    dW.fuseCortices('data', imageInfo['NAME'], imageFormat=imageInfo['IMAGE_FORMAT'],
                                    deleteBaseMasks=True, silent=True)
                    dW.cleanImage('data', imageInfo['NAME'], cleaningClass="cortex", cleanMasks=False,
                                  imageFormat=imageInfo['IMAGE_FORMAT'])
                elif self.__MODE == "mest_glom":
                    dW.cleanImage('data', imageInfo['NAME'], cleaningClass='nsg', cleanMasks=False,
                                  imageFormat=imageInfo['IMAGE_FORMAT'])
                    imageInfo.pop('X_STARTS')
                    imageInfo.pop('Y_STARTS')
                    imageInfo['ROI_COORDINATES'] = dI.getCenteredClassBboxes(
                        datasetPath='data', imageName=imageInfo['NAME'], classToCenter='nsg',
                        image_size=self.__DIVISION_SIZE, imageFormat=imageInfo['IMAGE_FORMAT'],
                        allow_oversized=True, config=self.__CONFIG
                    )
                    imageInfo['NB_DIV'] = len(imageInfo['ROI_COORDINATES'])
                maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                if 'BASE_CLASS' in imageInfo:
                    if imageInfo['BASE_CLASS'] in maskDirs:
                        imageInfo.update({'BASE_AREA': 0, 'BASE_COUNT': 0})
                        baseClassDirPath = os.path.join('data', imageInfo['NAME'], imageInfo['BASE_CLASS'])
                        for baseClassMask in os.listdir(baseClassDirPath):
                            baseMask = dW.loadSameResImage(os.path.join(baseClassDirPath, baseClassMask),
                                                           fullImage.shape)
                            imageInfo['BASE_AREA'] += dD.getBWCount(baseMask)[1]
                            imageInfo['BASE_COUNT'] += 1
                            del baseMask
                    # If full_images directory exists it means than image has been cleaned so we have to get it another
                    # time
                    if 'full_images' in maskDirs:
                        imagesDirPath = os.path.join('data', imageInfo['NAME'], 'images')
                        imageFilePath = os.listdir(imagesDirPath)[0]
                        imageInfo['cleaned_image_path'] = os.path.join(imagesDirPath, imageFilePath)
                        image = cv2.cvtColor(cv2.imread(imageInfo['cleaned_image_path']), cv2.COLOR_BGR2RGB)

                # We want to know if image has annotation, if we don't want to detect cortex and this mask exist
                # as we need it to clean the image, we remove it from the mask list before checking if a class
                # we want to predict has an annotated mask
                if self.__MODE in ['main', 'mest_main', 'mest_glom'] and not imageInfo['HAS_ANNOTATION']:
                    if imageInfo['BASE_CLASS'] not in self.__CUSTOM_CLASS_NAMES and imageInfo['BASE_CLASS'] in maskDirs:
                        maskDirs.remove(imageInfo['BASE_CLASS'])
                        imageInfo['HAS_ANNOTATION'] = any([d in self.__CUSTOM_CLASS_NAMES for d in maskDirs])
                if imageInfo['HAS_ANNOTATION'] and not silent:
                    print("    - AP and confusion matrix will be computed")

        return image, fullImage, imageInfo, image_results_path

    def init_results_dir(self, results_path):
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
            results_log.write(f"Image; Duration (s); Precision; {os.path.basename(self.__MODEL_PATH)}\n")
        return results_path, logsPath

    def inference(self, images: list, results_path=None, save_results=True,
                  fusion_bb_threshold=0., fusion_mask_threshold=0.1,
                  filter_bb_threshold=0.5, filter_mask_threshold=0.9,
                  priority_table=None, nbMaxDivPerAxis=3, fusionDivThreshold=0.1,
                  displayOnlyAP=False, savePreFusionImage=False, savePreFilterImage=False,
                  allowSparse=True, minMaskArea=300, on_border_threshold=0.25, perPixelConfMatrix=True,
                  enableCortexFusionDiv=True):

        if len(images) == 0:
            print("Images list is empty, no inference to perform.")
            return

        # If results have to be saved, setting the results path and creating directory
        if save_results:
            results_path, logsPath = self.init_results_dir(results_path)
        else:
            print("No result will be saved")
            results_path = None

        self.__CONFUSION_MATRIX.fill(0)
        self.__APs.clear()
        total_start_time = time()
        failedImages = []
        cortex_mode = self.__MODE == "cortex"
        for img_idx, IMAGE_PATH in enumerate(images):
            try:
                # Last step of full image inference
                if '_fusion_info.skinet' in IMAGE_PATH:
                    if cortex_mode:
                        continue
                    image_name = os.path.basename(IMAGE_PATH).replace('_fusion_info.skinet', '')
                    print(f"Finalising {image_name} image {progressText(img_idx + 1, len(images))}")
                    self.merge_fusion_div(results_path, IMAGE_PATH)
                else:
                    print(f"Using {IMAGE_PATH} image file {progressText(img_idx + 1, len(images))}")
                    image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
                    self.process_image(IMAGE_PATH, results_path, allowSparse, savePreFusionImage, fusion_bb_threshold,
                                       fusion_mask_threshold, savePreFilterImage, filter_bb_threshold,
                                       filter_mask_threshold, priority_table, on_border_threshold, minMaskArea,
                                       enableCortexFusionDiv, fusionDivThreshold, nbMaxDivPerAxis, perPixelConfMatrix,
                                       save_results, displayOnlyAP, logsPath)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                failedImages.append(os.path.basename(IMAGE_PATH))
                print(f"/!\\ Failed {IMAGE_PATH} at \"{self.__STEP}\"\n")
                if save_results and self.__STEP not in ["image preparation", "finalizing"]:
                    with open(logsPath, 'a') as results_log:
                        results_log.write(f"{image_name}; -1; -1;FAILED ({self.__STEP});\n")
        # Saving failed images list if not empty
        if len(failedImages) > 0:
            try:
                with open(os.path.join(results_path, "failed.json"), 'w') as failedJsonFile:
                    json.dump(failedImages, failedJsonFile, indent="\t")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print("Failed to save failed image(s) list. Following is the list itself :")
                print(failedImages)

        if len(self.__APs) > 1:
            mAP = np.mean(self.__APs)
            print(f"Mean Average Precision is about {mAP:06.2%}")
            name = "Final Confusion Matrix"
            name2 = "Final Confusion Matrix (Normalized)"
            cmap = plt.cm.get_cmap('hot')
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, self.__CUSTOM_CLASS_NAMES.copy(), title=name,
                                               cmap=cmap, show=False,
                                               fileName=(os.path.join(results_path, name.replace(' ', '_'))
                                                         if save_results else None))
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, self.__CUSTOM_CLASS_NAMES.copy(), title=name2,
                                               cmap=cmap, show=False, normalize=True,
                                               fileName=(os.path.join(results_path, name2.replace('(', '')
                                                                      .replace(')', '')
                                                                      .replace(' ', '_'))
                                                         if save_results else None))
            plt.close('all')
        else:
            mAP = -1
        total_time = round(time() - total_start_time)
        print(f"All inferences done in {formatTime(total_time)}")
        if save_results:
            with open(logsPath, 'a') as results_log:
                mapText = f"{mAP:4.3f}".replace(".", ",")
                results_log.write(f"GLOBAL; {total_time}; {mapText}%;\n")

    def process_image(self, image_path, results_path, allowSparse, savePreFusionImage, fusion_bb_threshold,
                      fusion_mask_threshold, savePreFilterImage, filter_bb_threshold, filter_mask_threshold,
                      priority_table, on_border_threshold, minMaskArea, enableCortexFusionDiv, fusionDivThreshold,
                      nbMaxDivPerAxis, perPixelConfMatrix, save_results, displayOnlyAP, logsPath):
        cortex_mode = self.__MODE == "cortex"
        start_time = time()
        self.__STEP = "image preparation"
        image, fullImage, imageInfo, image_results_path = self.prepare_image(image_path, results_path,
                                                                             silent=displayOnlyAP)
        if imageInfo["HAS_ANNOTATION"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ground_truth = self.get_ground_truth(image_results_path, fullImage,
                                                     imageInfo, displayOnlyAP, save_results)
            if save_results:
                if self.__LOW_MEMORY:
                    del fullImage
                    gc.collect()
                    fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

        # Getting predictions for each division
        res = []
        total_px = self.__CONFIG.IMAGE_SIDE ** 2
        skipped = 0
        debugIterator = -1
        skippedText = ""
        inference_start_time = time()
        if not displayOnlyAP:
            progressBar(0, imageInfo["NB_DIV"], prefix=' - Inference')
        for divId in range(imageInfo["NB_DIV"]):
            self.__STEP = f"{divId} div processing"
            if 'X_STARTS' in imageInfo and 'Y_STARTS' in imageInfo:
                division = dD.getImageDivision(fullImage if image is None else image, imageInfo["X_STARTS"],
                                               imageInfo["Y_STARTS"], divId, self.__DIVISION_SIZE)
            elif 'ROI_COORDINATES' in imageInfo:
                currentRoI = imageInfo['ROI_COORDINATES'][divId]
                division = (fullImage if image is None else image)[currentRoI[0]:currentRoI[2],
                           currentRoI[1]:currentRoI[3], :]
            else:
                raise ValueError('Cannot find image areas to use')

            grayDivision = cv2.cvtColor(division, cv2.COLOR_RGB2GRAY)
            colorPx = cv2.countNonZero(grayDivision)
            del grayDivision
            if colorPx / total_px > 0.1:
                self.__STEP = f"{divId} div inference"
                results = self.__MODEL.process(division, normalizedCoordinates=False,
                                               score_threshold=self.__CONFIG.DETECTION_MIN_CONFIDENCE)
                results["div_id"] = divId
                if self.__CONFIG.USE_MINI_MASK:
                    res.append(utils.reduce_memory(results.copy(), config=self.__CONFIG, allow_sparse=allowSparse))
                else:
                    res.append(results.copy())
                del results
            elif not displayOnlyAP:
                skipped += 1
                skippedText = f"({skipped} empty division{'s' if skipped > 1 else ''} skipped) "
            del division
            gc.collect()

            if not displayOnlyAP:
                if divId + 1 == imageInfo["NB_DIV"]:
                    inference_duration = round(time() - inference_start_time)
                    skippedText += f"Duration = {formatTime(inference_duration)}"
                progressBar(divId + 1, imageInfo["NB_DIV"], prefix=' - Inference', suffix=skippedText)

        # Post-processing of the predictions
        if not displayOnlyAP:
            print(" - Fusing results of all divisions")
        # TODO Test fuse_results
        self.__STEP = "fusing results"
        # res = pp.fuse_results(res, fullImage.shape, division_size=self.__DIVISION_SIZE,
        #                       min_overlap_part=self.__MIN_OVERLAP_PART)
        res = pp.fuse_results(res, imageInfo, division_size=self.__DIVISION_SIZE, config=self.__CONFIG)

        if len(res['class_ids']) > 0:
            # TODO dyn call and args as dict for post-processing automation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.__STEP = "fusing masks"
                progressBarPrefix = " - Fusing overlapping masks" if not displayOnlyAP else None
                res = pp.fuse_masks(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                    config=self.__CONFIG, displayProgress=progressBarPrefix, verbose=0)

                if "main" in self.__MODE:
                    if savePreFilterImage:
                        debugIterator += 1
                        self.save_debug_image("pre border filter", debugIterator, fullImage, imageInfo, res,
                                              image_results_path, silent=displayOnlyAP)
                        if self.__LOW_MEMORY:
                            del fullImage
                            gc.collect()
                            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)
                    self.__STEP = "removing border masks"
                    progressBarPrefix = " - Removing border masks" if not displayOnlyAP else None
                    if self.__MODE == "main":
                        classes_to_check = [7, 8, 9, 10]
                    else:
                        classes_to_check = [5, 6]
                    res = pp.filter_on_border_masks(res, fullImage if image is None else image,
                                                    onBorderThreshold=on_border_threshold,
                                                    classes=classes_to_check, config=self.__CONFIG,
                                                    displayProgress=progressBarPrefix, verbose=0)

                if self.__MODE == "main":
                    if savePreFilterImage:
                        debugIterator += 1
                        self.save_debug_image("pre orphan filter", debugIterator, fullImage, imageInfo, res,
                                              image_results_path, silent=displayOnlyAP)
                        if self.__LOW_MEMORY:
                            del fullImage
                            gc.collect()
                            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)
                        # TODO : Build automatically per_mask_conf_hierarchy

                    per_mask_conf_hierarchy = {
                        3: {"contains": [4, 5], "keep_if_no_child": False},
                        8: {"contains": [9, 10], "keep_if_no_child": True}
                    }
                    self.__STEP = "filtering orphan masks (pass 1)"
                    progressBarPrefix = " - Removing orphan masks" if not displayOnlyAP else None
                    res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                 mask_threshold=filter_mask_threshold,
                                                 classes_hierarchy=per_mask_conf_hierarchy,
                                                 displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                 verbose=0)
                del image

                if type(priority_table[0][0]) is not bool or (type(priority_table[0][0]) is bool
                                                              and any([any(row) for row in priority_table])):
                    if savePreFilterImage:
                        debugIterator += 1
                        self.save_debug_image("pre filter", debugIterator, fullImage, imageInfo, res,
                                              image_results_path, silent=displayOnlyAP)
                        if self.__LOW_MEMORY:
                            del fullImage
                            gc.collect()
                            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

                    self.__STEP = "filtering masks"
                    progressBarPrefix = " - Removing non-sense masks" if not displayOnlyAP else None
                    res = pp.filter_masks(res, bb_threshold=filter_bb_threshold, priority_table=priority_table,
                                          mask_threshold=filter_mask_threshold, included_threshold=0.9,
                                          including_threshold=0.6, verbose=0,
                                          displayProgress=progressBarPrefix, config=self.__CONFIG)

                if self.__MODE == "main":
                    if savePreFilterImage:
                        debugIterator += 1
                        self.save_debug_image("pre orphan filter (Pass 2)", debugIterator, fullImage, imageInfo,
                                              res, image_results_path, silent=displayOnlyAP)
                        if self.__LOW_MEMORY:
                            del fullImage
                            gc.collect()
                            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

                    self.__STEP = "filtering orphan masks (pass 2)"
                    progressBarPrefix = " - Removing orphan masks" if not displayOnlyAP else None
                    res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                 mask_threshold=filter_mask_threshold,
                                                 classes_hierarchy=per_mask_conf_hierarchy,
                                                 displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                 verbose=0)

                    if savePreFusionImage:
                        debugIterator += 1
                        self.save_debug_image("pre class fusion", debugIterator, fullImage, imageInfo, res,
                                              image_results_path, silent=displayOnlyAP)
                        if self.__LOW_MEMORY:
                            del fullImage
                            gc.collect()
                            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)
                    self.__STEP = "fusing classes"
                    progressBarPrefix = " - Fusing overlapping equivalent masks" if not displayOnlyAP else None
                    classes_compatibility = [[4, 5]]  # Nsg partiel + nsg complet
                    res = pp.fuse_class(res, bb_threshold=fusion_bb_threshold,
                                        mask_threshold=fusion_mask_threshold,
                                        classes_compatibility=classes_compatibility, config=self.__CONFIG,
                                        displayProgress=progressBarPrefix, verbose=0)

                if savePreFilterImage:
                    debugIterator += 1
                    self.save_debug_image("pre small masks removal", debugIterator, fullImage, imageInfo,
                                          res, image_results_path, silent=displayOnlyAP)
                    if self.__LOW_MEMORY:
                        del fullImage
                        gc.collect()
                        fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)
                self.__STEP = "removing small masks"
                progressBarPrefix = " - Removing small masks" if not displayOnlyAP else None
                res = pp.filter_small_masks(res, min_size=minMaskArea, config=self.__CONFIG,
                                            displayProgress=progressBarPrefix, verbose=0)
                if savePreFusionImage:
                    debugIterator += 1
                    self.save_debug_image("pre fusion", debugIterator, fullImage, imageInfo, res,
                                          image_results_path, silent=displayOnlyAP)
                    if self.__LOW_MEMORY:
                        del fullImage
                        gc.collect()
                        fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if imageInfo["HAS_ANNOTATION"]:
                # TODO : Build automatically per_mask_conf_hierarchy
                per_mask_conf_hierarchy = None
                per_pixel_conf_hierarchy = None
                if self.__MODE == 'cortex':
                    per_pixel_conf_hierarchy = [2, {3: 1}]  # Ensures that cortex overrides other classes
                elif self.__MODE == 'main':
                    if len(self.__CUSTOM_CLASS_NAMES) == 10:
                        per_mask_conf_hierarchy = {3: [4, 5], 8: [9, 10], 10: [9]}
                    per_pixel_conf_hierarchy = [1, 2, {3: [4, 5]}, 6, 7, {8: [{10: 9}, 9]}]

                AP = self.compute_ap_and_conf_matrix(image_results_path, imageInfo, res, ground_truth,
                                                     per_mask_conf_hierarchy, per_pixel_conf_hierarchy,
                                                     perPixelConfMatrix, save_results, displayOnlyAP)
                del ground_truth
            if not cortex_mode:
                self.compute_statistics(image_results_path, imageInfo, res, save_results)
        if save_results:
            if cortex_mode:
                self.finalize_cortex_mode(image_results_path, imageInfo, res, enableCortexFusionDiv, fusionDivThreshold,
                                          nbMaxDivPerAxis, displayOnlyAP)
            if len(res['class_ids']) > 0:
                if not displayOnlyAP:
                    print(" - Applying masks on image")
                self.__STEP = "saving predicted image"
                self.draw_masks(image_results_path, fullImage, imageInfo, res, title=f"{imageInfo['NAME']} Predicted")
            elif not displayOnlyAP:
                print(" - No mask to apply on image")

            # Annotations Extraction
            self.export_annotations(image_results_path, imageInfo, res, displayOnlyAP)
        final_time = round(time() - start_time)
        print(f" Done in {formatTime(final_time)}\n")
        if not imageInfo['HAS_ANNOTATION']:
            AP = -1
        self.__STEP = "finalizing"
        if save_results:
            with open(logsPath, 'a') as results_log:
                apText = f"{AP:4.3f}".replace(".", ",")
                results_log.write(f"{imageInfo['NAME']}; {final_time}; {apText}%;\n")
        del res, imageInfo, fullImage
        plt.clf()
        plt.close('all')
        gc.collect()
        return AP

    def draw_masks(self, image_results_path, img, image_info, masks_data, title=None, cleaned_image=True):
        if title is None:
            title = f"{image_info['NAME']} Masked"
        fileName = os.path.join(image_results_path, title.replace(' ', '_').replace('(', '').replace(')', ''))
        # No need of reloading or passing copy of image as it is the final drawing
        visualize.display_instances(
            img, masks_data['rois'], masks_data['masks'], masks_data['class_ids'],
            self.__VISUALIZE_NAMES, masks_data['scores'] if 'scores' in masks_data else None, colors=self.__COLORS,
            colorPerClass=True, fileName=fileName, save_cleaned_img=cleaned_image, silent=True, title=title, figsize=(
                (1024 if self.__MODE == "cortex" else image_info["WIDTH"]) / 100,
                (1024 if self.__MODE == "cortex" else image_info["HEIGHT"]) / 100
            ), image_format=image_info['IMAGE_FORMAT'], config=self.__CONFIG
        )

    def merge_fusion_div(self, results_path, fusion_info_path, image_format='jpg'):
        """
        Last process of the full image prediction using fusion divisions
        :param results_path: path to the results folder
        :param fusion_info_path: path to the fusion info file
        :param image_format: the image format to use
        :return:
        """
        # Loading fusion info that are saved as a JSON formatted file
        self.__STEP = "loading fusion info"
        with open(fusion_info_path, 'r') as skinetFile:
            fusionInfo = json.load(skinetFile)

        # Loading the cleaned version of the original image
        self.__STEP = "loading cleaned image"
        cleaned_image_path = os.path.join(os.path.dirname(fusion_info_path), f"{fusionInfo['image']}_fusion",
                                          f"{fusionInfo['image']}_cleaned.jpg")
        cleanedImage = cv2.imread(cleaned_image_path)

        # Generating output folder
        image_results_path = os.path.join(results_path, fusionInfo['image'])
        os.makedirs(image_results_path, exist_ok=True)

        # Pasting each divisions and merging all the stats
        globalStats = None
        for fusionDivFolder in os.listdir(results_path):  # Going through each image dir in the results folder
            self.__STEP = f"{fusionDivFolder} div fusion"
            fusion_div_folder_path = os.path.join(results_path, fusionDivFolder)

            # If the current one is a fusion division of the same image
            if f"{fusionInfo['image']}_" in fusionDivFolder:
                divID = fusionDivFolder.split('_')[-1]  # Getting the fusion division ID

                # Checking that the division had to be inferred
                if fusionInfo["divisions"][divID]["used"]:
                    # Loading statistics of the fusion division to merge them with statistics of other divisions
                    statsPath = os.path.join(fusion_div_folder_path, f"{fusionDivFolder}_stats.json")
                    with open(statsPath, 'r') as tempStatsFile:
                        tempStats = json.load(tempStatsFile)

                    # Overwriting cortex area with correct value from fusion info file
                    if "cortex" in tempStats:
                        tempStats["cortex"]["area"] = fusionInfo["divisions"][divID]["cortex_area"]
                        with open(statsPath, 'w') as tempStatsFile:
                            json.dump(tempStats, tempStatsFile, indent="\t")

                    # Merging statistics of all divisions
                    if globalStats is None:
                        globalStats = tempStats
                    else:
                        for className in tempStats:
                            if className == "cortex":
                                continue
                            globalStats[className]["count"] += tempStats[className]["count"]
                            globalStats[className]["area"] += tempStats[className]["area"]

                    # Copy-Pasting the fusion div onto the cleaned original image
                    imagePath = os.path.join(fusion_div_folder_path,
                                             f"{fusionDivFolder}_predicted_clean.{image_format}")
                    divImage = cv2.imread(imagePath)
                    coo = fusionInfo["divisions"][divID]["coordinates"]
                    cleanedImage[coo["y1"]:coo["y2"], coo["x1"]:coo["x2"], :] = divImage

                    # Moving the fusion div folder to "divisions" sub folder of the output folder
                    shutil.move(fusion_div_folder_path, os.path.join(image_results_path, "divisions", fusionDivFolder))

        # Saving final image
        self.__STEP = "saving full predicted image"
        cv2.imwrite(os.path.join(image_results_path, f"{fusionInfo['image']}_full_prediction.jpg"),
                    cleanedImage, CV2_IMWRITE_PARAM)

        # Adding cortex area from fusion file to stats and saving them
        self.__STEP = "saving full stats file"
        with open(os.path.join(image_results_path, f"{fusionInfo['image']}_full_prediction_stats.json"),
                  'w') as globalStatsFile:
            globalStats["cortex"] = {"count": 1, "area": fusionInfo["cortex_area"]}
            json.dump(globalStats, globalStatsFile, indent="\t")

        del cleanedImage
        print("Done\n")

    def get_ground_truth(self, image_results_path, fullImage, image_info, displayOnlyAP, save_results):
        """
        Get ground-truth annotations and applies masks on image if enabled
        :param image_results_path: the current image output folder
        :param fullImage: the original image
        :param image_info: info about the current image
        :param displayOnlyAP: if True, will not print anything
        :param save_results: if True, will apply masks
        :return: gt_bbox, gt_class_id, gt_mask
        """
        self.__STEP = "dataset creation"
        dataset_val = SkinetCustomDataset(self.__CUSTOM_CLASS_NAMES, self.__MODE, self.__CORTEX_SIZE,
                                          self.__CONFIG, image_info, enable_occlusion=False)
        dataset_val.load_images()
        dataset_val.prepare()
        self.__STEP = "loading annotated masks"
        image_id = dataset_val.image_ids[0]
        gt_mask, gt_class_id, gt_bbox = dataset_val.load_mask(image_id)
        gt = {'rois': gt_bbox, 'class_ids': gt_class_id, 'masks': gt_mask}
        if save_results:
            if len(gt_class_id) > 0:
                if not displayOnlyAP:
                    print(" - Applying annotations on file to get expected results")
                self.__STEP = "applying expected masks"
                self.draw_masks(image_results_path, fullImage if self.__LOW_MEMORY else fullImage.copy(),
                                image_info, gt, title=f"{image_info['NAME']}_Expected", cleaned_image=False)
            elif not displayOnlyAP:
                print(" - No mask expected, not saving expected image")
        return gt

    def compute_statistics(self, image_results_path, image_info, predicted, save_results=True):
        """
        Computes area and counts masks of each class
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param save_results: Whether to save statistics in a file or not
        :return: None
        """
        self.__STEP = "computing statistics"
        print(" - Computing statistics on predictions")
        stats = pp.getCountAndArea(predicted, classesInfo=self.__CLASSES_INFO,
                                   selectedClasses=self.__CUSTOM_CLASS_NAMES, config=self.__CONFIG)
        if 'BASE_CLASS' in image_info:
            if 'BASE_CLASS' in image_info:
                stats[image_info['BASE_CLASS']] = {"count": image_info['BASE_COUNT'],
                                                   "area": image_info["BASE_AREA"]}
        for className in stats:
            stat = stats[className]
            print(f"    - {className} : count = {stat['count']}, area = {stat['area']} px")
        if save_results:
            with open(os.path.join(image_results_path, f"{image_info['NAME']}_stats.json"),
                      "w") as saveFile:
                try:
                    json.dump(stats, saveFile, indent='\t')
                except TypeError:
                    print("    Failed to save statistics", flush=True)

    def compute_ap_and_conf_matrix(self, image_results_path, image_info, predicted, ground_truth,
                                   per_mask_conf_hierarchy, per_pixel_conf_hierarchy=None, perPixelConfMatrix=True,
                                   save_results=True, displayOnlyAP=False):
        """
        Computes AP and confusion matrix
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param ground_truth: the ground truth results dictionary
        :param per_mask_conf_hierarchy: classes hierarchy for per mask confusion matrix
        :param per_pixel_conf_hierarchy: classes hierarchy for per pixel confusion matrix
        :param perPixelConfMatrix: If True, confusion matrix will represents pixel instead of masks
        :param save_results: Whether to save files or not
        :param displayOnlyAP: Whether to display only AP or also current steps
        :return:
        """
        if not displayOnlyAP:
            print(" - Computing Average Precision and Confusion Matrix")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__STEP = "computing confusion matrix"
            AP, _, _, _, confusion_matrix = utils.compute_ap(gt_boxes=ground_truth['rois'],
                                                             gt_class_ids=ground_truth['class_ids'],
                                                             gt_masks=ground_truth['masks'],
                                                             pred_boxes=predicted["rois"],
                                                             pred_class_ids=predicted["class_ids"],
                                                             pred_masks=predicted['masks'],
                                                             pred_scores=predicted["scores"],
                                                             nb_class=-1 if perPixelConfMatrix else self.__NB_CLASS,
                                                             score_threshold=0.3,
                                                             iou_threshold=0.5,
                                                             confusion_iou_threshold=0.5,
                                                             classes_hierarchy=per_mask_conf_hierarchy,
                                                             confusion_background_class=True,
                                                             confusion_only_best_match=False)
            if perPixelConfMatrix:
                if per_pixel_conf_hierarchy is None:
                    per_pixel_conf_hierarchy = [i + 1 for i in range(self.__NB_CLASS)]
                if self.__CORTEX_SIZE is not None:
                    image_shape = self.__CORTEX_SIZE
                else:
                    image_shape = (image_info['HEIGHT'], image_info['WIDTH'])
                confusion_matrix = utils.compute_confusion_matrix(
                    image_shape=image_shape, config=self.__CONFIG,
                    expectedResults=ground_truth, predictedResults=predicted,
                    classes_hierarchy=per_pixel_conf_hierarchy, num_classes=self.__NB_CLASS
                )
            print(f"{'' if displayOnlyAP else '   '} - Average Precision is about {AP:06.2%}")
            self.__CONFUSION_MATRIX = np.add(self.__CONFUSION_MATRIX, confusion_matrix)
            self.__APs.append(AP)
            cmap = plt.cm.get_cmap('hot')
            if save_results:
                self.__STEP = "saving confusion matrix"
                name = f"{image_info['NAME']} Confusion Matrix"
                confusionMatrixFileName = os.path.join(image_results_path, name.replace(' ', '_'))
                name2 = f"{image_info['NAME']} Confusion Matrix (Normalized)"
                confusionMatrixFileName2 = os.path.join(image_results_path, name2.replace('(', '')
                                                        .replace(')', '')
                                                        .replace(' ', '_'))
                visualize.display_confusion_matrix(confusion_matrix, self.__CUSTOM_CLASS_NAMES.copy(),
                                                   title=name,
                                                   cmap=cmap, show=False,
                                                   fileName=confusionMatrixFileName)
                visualize.display_confusion_matrix(confusion_matrix, self.__CUSTOM_CLASS_NAMES.copy(),
                                                   title=name2,
                                                   cmap=cmap, show=False, normalize=True,
                                                   fileName=confusionMatrixFileName2)
        return AP

    def finalize_cortex_mode(self, image_results_path, image_info, predicted, enableCortexFusionDiv, fusionDivThreshold,
                             nbMaxDivPerAxis, displayOnlyAP):
        self.__STEP = "cleaning full resolution image"
        if not displayOnlyAP:
            print(" - Cleaning full resolution image and saving statistics")
        allCortices = None
        # Gathering every cortex masks into one
        for idxMask, classMask in enumerate(predicted['class_ids']):
            if classMask == 1:
                if allCortices is None:  # First mask found
                    allCortices = predicted['masks'][:, :, idxMask].copy() * 255
                else:  # Additional masks found
                    allCortices = cv2.bitwise_or(allCortices, predicted['masks'][:, :, idxMask] * 255)
        # To avoid cleaning an image without cortex
        if allCortices is not None:
            # Cleaning original image with cortex mask(s) and saving stats
            allCorticesArea, allCorticesSmall = self.keep_only_cortex(
                image_results_path, image_info, allCortices
            )
            if enableCortexFusionDiv:
                # Preparing to export all "fusion" divisions with stats
                self.generate_fusion_div_ressources(image_results_path, image_info, allCorticesSmall,
                                                    allCorticesArea, fusionDivThreshold,
                                                    nbMaxDivPerAxis)

    def keep_only_cortex(self, image_results_path, image_info, allCortices):
        """
        Cleans the original biopsy/nephrectomy with cortex masks and extracts the cortex area
        :param image_results_path: the current image output folder
        :param image_info: info about the current image
        :param allCortices: fused-cortices mask
        :return: allCorticesArea, allCorticesSmall
        """
        self.__STEP = "init fusion dir"
        fusion_dir = os.path.join(image_results_path, f"{image_info['NAME']}_fusion")
        os.makedirs(fusion_dir, exist_ok=True)

        # Saving useful part of fused-cortices mask
        self.__STEP = "crop & save fused-cortices mask"
        allCorticesROI = utils.extract_bboxes(allCortices)
        allCorticesSmall = allCortices[allCorticesROI[0]:allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3]]
        cv2.imwrite(os.path.join(fusion_dir, f"{image_info['NAME']}_cortex.jpg"), allCorticesSmall, CV2_IMWRITE_PARAM)

        # Computing coordinates at full resolution
        yRatio = image_info['HEIGHT'] / self.__CORTEX_SIZE[0]
        xRatio = image_info['WIDTH'] / self.__CORTEX_SIZE[1]
        allCorticesROI[0] = int(allCorticesROI[0] * yRatio)
        allCorticesROI[1] = int(allCorticesROI[1] * xRatio)
        allCorticesROI[2] = int(allCorticesROI[2] * yRatio)
        allCorticesROI[3] = int(allCorticesROI[3] * xRatio)

        # Resizing fused-cortices mask and computing its area
        allCortices = cv2.resize(np.uint8(allCortices), (image_info['WIDTH'], image_info['HEIGHT']),
                                 interpolation=cv2.INTER_CUBIC)
        allCorticesArea = dD.getBWCount(allCortices)[1]
        with open(os.path.join(image_results_path, f"{image_info['NAME']}_stats.json"), "w") as saveFile:
            stats = {"cortex": {"count": 1, "area": allCorticesArea}}
            try:
                json.dump(stats, saveFile, indent='\t')
            except TypeError:
                print("    Failed to save statistics", flush=True)

        # Masking the image and saving it
        temp = np.repeat(allCortices[:, :, np.newaxis], 3, axis=2)
        image_info['FULL_RES_IMAGE'] = cv2.bitwise_and(
            image_info['FULL_RES_IMAGE'][allCorticesROI[0]: allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3], :],
            temp[allCorticesROI[0]: allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3], :]
        )
        cv2.imwrite(os.path.join(fusion_dir, f"{image_info['NAME']}_cleaned.jpg"),
                    cv2.cvtColor(image_info['FULL_RES_IMAGE'], cv2.COLOR_RGB2BGR),
                    CV2_IMWRITE_PARAM)
        return allCorticesArea, allCorticesSmall

    def generate_fusion_div_ressources(self, image_results_path, image_info, allCorticesSmall,
                                       allCorticesArea, fusionDivThreshold, nbMaxDivPerAxis):
        """
        Generates files to perform full image prediction using fusion divisions
        :param image_results_path: the output folder of the current image
        :param image_info: info about the current image
        :param allCorticesSmall: fusion of all cortices masks
        :param allCorticesArea: total area of the fusion of all cortices (correct value, not reduced)
        :param fusionDivThreshold: Whether the fusion div will be saved or not (least part of cortex to be kept)
        :param nbMaxDivPerAxis: the number of div per axis when dividing the fusion div with main mode parameters
        :return: None
        """
        fusion_dir = os.path.join(image_results_path, f"{image_info['NAME']}_fusion")
        fusionInfo = {"image": image_info["NAME"]}

        # Computing ratio between full resolution image and the low one
        self.__STEP = "computing image to full size image ratio"
        height, width, _ = image_info['FULL_RES_IMAGE'].shape
        smallHeight, smallWidth = allCorticesSmall.shape
        xRatio = width / smallWidth
        yRatio = height / smallHeight

        # Computing divisions coordinates for full and low resolution images
        self.__STEP = "computing fusion div coordinates"
        divisionSize = dD.getMaxSizeForDivAmount(nbMaxDivPerAxis, self.__DIVISION_SIZE, self.__MIN_OVERLAP_PART_MAIN)
        xStarts = dD.computeStartsOfInterval(width, intervalLength=divisionSize, min_overlap_part=0)
        yStarts = dD.computeStartsOfInterval(height, intervalLength=divisionSize, min_overlap_part=0)
        xStartsEquivalent = [round(x / xRatio) for x in xStarts]
        yStartsEquivalent = [round(y / yRatio) for y in yStarts]
        xDivSide = round(divisionSize / xRatio)
        yDivSide = round(divisionSize / yRatio)

        # Preparing informations to write into the .skinet file
        fusionInfo["division_size"] = divisionSize
        fusionInfo["min_overlap_part"] = 0
        fusionInfo["xStarts"] = xStarts
        fusionInfo["yStarts"] = yStarts
        fusionInfo["max_div_per_axis"] = nbMaxDivPerAxis
        fusionInfo["cortex_area"] = allCorticesArea
        fusionInfo["divisions"] = {}

        # Extracting and saving all divisions
        for divID in range(dD.getDivisionsCount(xStarts, yStarts)):
            self.__STEP = f"testing fusion div {divID}"
            # Getting the corresponding part of small cortex mask
            cortexDiv = dD.getImageDivision(allCorticesSmall, xStartsEquivalent, yStartsEquivalent,
                                            divID, divisionSize=(xDivSide, yDivSide))
            black, white = dD.getBWCount(cortexDiv.astype(np.uint8))
            partOfDiv = white / (white + black)
            used = partOfDiv > fusionDivThreshold
            fusionInfo["divisions"][divID] = {"cortex_area": white, "cortex_representative_part": partOfDiv,
                                              "used": used}

            # If there is enough cortex in this division, saving corresponding part of the image as an image to infer
            if used:
                self.__STEP = f"saving fusion div {divID}"
                x, xEnd, y, yEnd = dD.getDivisionByID(xStarts, yStarts, divID, divisionSize)
                fusionInfo["divisions"][divID]["coordinates"] = {"x1": x, "x2": xEnd, "y1": y, "y2": yEnd}
                imageDivision = dD.getImageDivision(image_info['FULL_RES_IMAGE'], xStarts, yStarts, divID, divisionSize)
                cv2.imwrite(os.path.join(fusion_dir, f"{image_info['NAME']}_{divID}.jpg"),
                            cv2.cvtColor(imageDivision, cv2.COLOR_RGB2BGR), CV2_IMWRITE_PARAM)

        # Writing informations into the .skinet file
        self.__STEP = "saving fusion info file"
        fusion_info_file_path = os.path.join(image_results_path, f"{image_info['NAME']}_fusion_info.skinet")
        with open(fusion_info_file_path, 'w') as fusionInfoFile:
            try:
                json.dump(fusionInfo, fusionInfoFile, indent="\t")
            except TypeError:
                print("    Failed to save fusion info file", file=sys.stderr, flush=True)

    def save_debug_image(self, step, debugIterator, fullImage, image_info, res, image_results_path, silent=True):
        if not silent:
            print(f" - Saving {step} image")
        title = f"{image_info['NAME']} Inference debug {debugIterator:02d} {step}"
        self.draw_masks(image_results_path, fullImage if self.__LOW_MEMORY else fullImage.copy(),
                        image_info, res, title, cleaned_image=False)

    def export_annotations(self, image_results_path, image_info, predicted, silent):
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
        for adapter in AnnotationAdapter.ANNOTATION_ADAPTERS:
            try:
                pp.export_annotations(image_info, predicted, self.__CLASSES_INFO,
                                      adapter, save_path=image_results_path,
                                      config=self.__CONFIG, verbose=0 if silent else 1)
            except Exception:
                print(f"Failed to export using {adapter.__qualname__} adapter")
