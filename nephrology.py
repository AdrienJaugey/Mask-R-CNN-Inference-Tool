import json
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import sys
import warnings

from common_utils import progressBar, formatTime, formatDate, progressText
from datasetTools.datasetDivider import CV2_IMWRITE_PARAM

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.io import imread, imsave
    from datasetTools import datasetDivider as dD, datasetWrapper as dW, AnnotationAdapter
    from datasetTools.ASAPAdapter import ASAPAdapter
    from datasetTools.LabelMeAdapter import LabelMeAdapter

    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
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
    folder = os.path.dirname(weight_path)
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
                 min_overlap_part_cortex=0.5, cortex_size=None, mini_mask_size=96, forceFullSizeMasks=False):
        print("Initialisation")
        self.__CLASSES_INFO = classesInfo
        self.__CORTEX_MODE = not classesInfo[0]["ignore"]
        cortex_mode = self.__CORTEX_MODE
        self.__MODEL_PATH = find_latest_weight(modelPath)
        self.__DIVISION_SIZE = divisionSize
        self.__MIN_OVERLAP_PART_MAIN = min_overlap_part_main
        self.__MIN_OVERLAP_PART_CORTEX = min_overlap_part_cortex
        self.__MIN_OVERLAP_PART = min_overlap_part_cortex if self.__CORTEX_MODE else min_overlap_part_main
        self.__CORTEX_SIZE = None if not self.__CORTEX_MODE else (1024, 1024) if cortex_size is None else cortex_size
        self.__CUSTOM_CLASS_NAMES = []
        for classInfo in classesInfo:
            if not classInfo["ignore"]:
                self.__CUSTOM_CLASS_NAMES.append(classInfo["name"])
        self.__NB_CLASS = len(self.__CUSTOM_CLASS_NAMES)
        # Root directory of the project
        self.__ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "logs")

        # Configurations
        nbClass = self.__NB_CLASS
        divSize = 1024 if self.__DIVISION_SIZE == "noDiv" else self.__DIVISION_SIZE
        self.__CONFUSION_MATRIX = np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int32)
        self.__APs = []

        class SkinetConfig(Config):
            NAME = "skinet"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + nbClass
            IMAGE_MIN_DIM = divSize
            IMAGE_MAX_DIM = divSize
            RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)
            TRAIN_ROIS_PER_IMAGE = 800
            DETECTION_MIN_CONFIDENCE = min_confidence
            STEPS_PER_EPOCH = 400
            VALIDATION_STEPS = 50
            USE_MINI_MASK = not cortex_mode and not forceFullSizeMasks
            MINI_MASK_SHAPE = (mini_mask_size, mini_mask_size)  # (height, width) of the mini-mask

        self.__CONFIG = SkinetConfig()

        # Recreate the model in inference mode
        self.__MODEL = modellib.MaskRCNN(mode="inference", config=self.__CONFIG, model_dir=self.__MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        assert self.__MODEL_PATH is not None and self.__MODEL_PATH != "", "Provide path to trained weights"
        print("Loading weights from", self.__MODEL_PATH)
        self.__MODEL.load_weights(self.__MODEL_PATH, by_name=True)
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
        if os.path.exists(imagePath):
            image_file_name = os.path.basename(imagePath)
            dirPath = os.path.dirname(imagePath)
            image_name = image_file_name.split('.')[0]
            image_extension = image_file_name.split('.')[-1]

            # Reading input image in RGB color order
            fullImage = cv2.imread(imagePath)
            fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2RGB)
            height, width, _ = fullImage.shape
            height = int(height)
            width = int(width)
            cortexArea = height * width

            if self.__CORTEX_MODE:  # If in cortex mode, resize image to lower resolution
                fullResImage = fullImage
                fullImage = cv2.resize(fullResImage, self.__CORTEX_SIZE)
            image = fullImage.copy()

            # Conversion of the image if format is not png or jpg
            if image_extension not in ["png", "jpg"]:
                image_extension = "jpg"
                tempPath = os.path.join(dirPath, f"{image_name}.{image_extension}")
                imsave(tempPath, fullImage)
                imagePath = tempPath

            # Creating the result dir if given and copying the base image in it
            if results_path is not None:
                results_path = os.path.normpath(results_path)
                image_results_path = os.path.join(results_path, image_name)
                os.makedirs(image_results_path, exist_ok=True)
                fullImagePath = os.path.join(image_results_path, f"{image_name}.{image_extension}")
                imsave(fullImagePath, fullImage)
            else:
                image_results_path = None

            # Computing divisions coordinates if needed and total number of div
            if self.__DIVISION_SIZE == "noDiv":
                xStarts = yStarts = [0]
            else:
                xStarts = dD.computeStartsOfInterval(maxVal=self.__CORTEX_SIZE[0] if self.__CORTEX_MODE else width,
                                                     intervalLength=self.__DIVISION_SIZE,
                                                     min_overlap_part=self.__MIN_OVERLAP_PART)
                yStarts = dD.computeStartsOfInterval(maxVal=self.__CORTEX_SIZE[1] if self.__CORTEX_MODE else height,
                                                     intervalLength=self.__DIVISION_SIZE,
                                                     min_overlap_part=self.__MIN_OVERLAP_PART)
            nbDiv = dD.getDivisionsCount(xStarts, yStarts)

            # If annotations found, create masks and clean image if possible
            annotationExists = False
            has_annotation = False
            for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                annotationExists = annotationExists or os.path.exists(os.path.join(dirPath, image_name + '.' + ext))
            if annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible")
                dW.createMasksOfImage(dirPath, image_name, 'data', classesInfo=self.__CLASSES_INFO,
                                      imageFormat=image_extension, resize=self.__CORTEX_SIZE, config=self.__CONFIG)
                if not self.__CORTEX_MODE:
                    dW.fuseCortices('data', image_name, imageFormat=image_extension, deleteBaseMasks=True, silent=True)
                    dW.cleanImage('data', image_name, onlyMasks=False)
                maskDirs = os.listdir(os.path.join('data', image_name))
                if not self.__CORTEX_MODE:
                    if "cortex" in maskDirs:
                        cortexDirPath = os.path.join('data', image_name, 'cortex')
                        cortexImageFilePath = os.listdir(cortexDirPath)[0]
                        cortex = dW.loadSameResImage(os.path.join(cortexDirPath, cortexImageFilePath), fullImage.shape)
                        cortexArea = dD.getBWCount(cortex)[1]
                    # If full_images directory exists it means than image has been cleaned so we have to get it another
                    # time
                    if 'full_images' in maskDirs:
                        imagesDirPath = os.path.join('data', image_name, 'images')
                        imageFilePath = os.listdir(imagesDirPath)[0]
                        image = cv2.imread(os.path.join(imagesDirPath, imageFilePath))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # We want to know if image has annotation, if we don't want to detect cortex and this mask exist
                # as we need it to clean the image, we remove it from the mask list before checking if a class
                # we want to predict has an annotated mask
                if "cortex" not in self.__CUSTOM_CLASS_NAMES and "cortex" in maskDirs:
                    maskDirs.remove("cortex")
                for classToPredict in self.__CUSTOM_CLASS_NAMES:
                    if classToPredict in maskDirs:
                        has_annotation = True
                        break
                if has_annotation and not silent:
                    print("    - AP and confusion matrix will be computed")

            imageInfo = {
                "PATH": imagePath,
                "DIR_PATH": dirPath,
                "FILE_NAME": image_file_name,
                "IMAGE_FORMAT": image_extension,
                "NAME": image_name,
                "HEIGHT": height,
                "WIDTH": width,
                "NB_DIV": nbDiv,
                "X_STARTS": xStarts,
                "Y_STARTS": yStarts,
                "HAS_ANNOTATION": has_annotation,
                "CORTEX_AREA": cortexArea
            }
            if self.__CORTEX_MODE:
                imageInfo['FULL_RES_IMAGE'] = fullResImage
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
                  allowSparse=True, minMaskArea=300):

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
        cmap = plt.cm.get_cmap('hot')
        total_start_time = time()
        for i, IMAGE_PATH in enumerate(images):
            # Last step of full image inference
            if '_fusion_info.skinet' in IMAGE_PATH:
                if self.__CORTEX_MODE:
                    continue
                with open(IMAGE_PATH, 'r') as skinetFile:
                    fusionInfo = json.load(skinetFile)
                print(f"Finalising {fusionInfo['image']} image {progressText(i + 1, len(images))}")
                fusionDir = fusionInfo["image"] + '_fusion'
                fusionDirPath = os.path.join(os.path.dirname(IMAGE_PATH), fusionDir)
                # TODO Dynamic image extension !!!
                cleaned_image_path = os.path.join(fusionDirPath, fusionInfo["image"] + "_cleaned.jpg")
                cleanedImage = cv2.imread(cleaned_image_path)
                image_results_path = os.path.join(results_path, fusionInfo['image'])
                os.makedirs(image_results_path, exist_ok=True)

                # Pasting each divisions and merging all the stats
                globalStats = None
                for imageFolder in os.listdir(results_path):
                    if (fusionInfo['image'] + '_') in imageFolder:
                        divID = imageFolder.split('_')[-1]
                        if fusionInfo["divisions"][divID]["used"]:
                            imagePath = os.path.join(results_path, imageFolder,
                                                     imageFolder + "_predicted_clean.png")
                            statsPath = os.path.join(results_path, imageFolder, imageFolder + "_stats.json")
                            with open(statsPath, 'r') as tempStatsFile:
                                tempStats = json.load(tempStatsFile)
                            if "cortex" in tempStats:
                                tempStats["cortex"]["area"] = fusionInfo["divisions"][divID]["cortex_area"]
                                with open(statsPath, 'w') as tempStatsFile:
                                    json.dump(tempStats, tempStatsFile, indent="\t")
                            if globalStats is None:
                                globalStats = tempStats
                            else:
                                for className in tempStats:
                                    if className == "cortex":
                                        continue
                                    globalStats[className]["count"] += tempStats[className]["count"]
                                    globalStats[className]["area"] += tempStats[className]["area"]
                            image = cv2.imread(imagePath)
                            coo = fusionInfo["divisions"][divID]["coordinates"]
                            cleanedImage[coo["y1"]:coo["y2"], coo["x1"]:coo["x2"], :] = image
                            shutil.move(os.path.join(results_path, imageFolder),
                                        os.path.join(image_results_path, "divisions", imageFolder))
                cv2.imwrite(os.path.join(image_results_path, fusionInfo['image'] + "_full_prediction.jpg"),
                            cleanedImage, CV2_IMWRITE_PARAM)
                with open(os.path.join(image_results_path, fusionInfo['image'] + "_full_prediction_stats.json"),
                          'w') as globalStatsFile:
                    globalStats["cortex"] = {"count": 1, "area": fusionInfo["cortex_area"]}
                    json.dump(globalStats, globalStatsFile, indent="\t")
                print("Done\n")
            else:
                start_time = time()
                print(f"Using {IMAGE_PATH} image file {progressText(i + 1, len(images))}")
                image, fullImage, imageInfo, image_results_path = self.prepare_image(IMAGE_PATH, results_path,
                                                                                     silent=displayOnlyAP)
                if imageInfo["HAS_ANNOTATION"]:
                    class CustomDataset(utils.Dataset):

                        def __init__(self, custom_class_names, cortex_mode, cortex_size, config, image_format):
                            super().__init__()
                            self.__CUSTOM_CLASS_NAMES = custom_class_names.copy()
                            self.__CORTEX_MODE = cortex_mode
                            self.__CORTEX_SIZE = cortex_size
                            self.__CONFIG = config
                            self.__IMAGE_FORMAT = image_format

                        def get_class_names(self):
                            return self.__CUSTOM_CLASS_NAMES.copy()

                        def load_images(self):
                            # Add classes
                            for class_id, class_name in enumerate(self.__CUSTOM_CLASS_NAMES):
                                self.add_class("skinet", class_id + 1, class_name)
                            image_name = imageInfo["NAME"]
                            img_path = os.path.join('data', image_name, "images", f"{image_name}.{self.__IMAGE_FORMAT}")
                            self.add_image("skinet", image_id=imageInfo["NAME"], path=img_path)

                        def image_reference(self, image_id):
                            """Return the skinet data of the image."""
                            info = self.image_info[image_id]
                            if info["source"] == "skinet":
                                return info["skinet"]
                            else:
                                super(self.__class__).image_reference(self, image_id)

                        def load_mask(self, image_id):
                            """Generate instance masks for cells of the given image ID.
                            """
                            info = self.image_info[image_id]
                            info = info.get("id")

                            path = os.path.join('data', info)

                            # Counting masks for current image
                            number_of_masks = 0
                            for masks_dir in os.listdir(path):
                                # For each directory excepting /images
                                if masks_dir not in self.__CUSTOM_CLASS_NAMES:
                                    continue
                                temp_DIR = os.path.join(path, masks_dir)
                                # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
                                number_of_masks += len([name_ for name_ in os.listdir(temp_DIR)
                                                        if os.path.isfile(os.path.join(temp_DIR, name_))])
                            if self.__CORTEX_MODE:
                                masks_shape = (self.__CORTEX_SIZE[0], self.__CORTEX_SIZE[1], number_of_masks)
                            elif self.__CONFIG.USE_MINI_MASK:
                                masks_shape = (self.__CONFIG.MINI_MASK_SHAPE[0], self.__CONFIG.MINI_MASK_SHAPE[1],
                                               number_of_masks)
                            else:
                                masks_shape = (imageInfo["HEIGHT"], imageInfo["WIDTH"], number_of_masks)
                            masks = np.zeros(masks_shape, dtype=np.uint8)
                            bboxes = np.zeros((number_of_masks, 4), dtype=np.int32)
                            iterator = 0
                            class_ids = np.zeros((number_of_masks,), dtype=int)
                            for masks_dir in os.listdir(path):
                                if masks_dir not in self.__CUSTOM_CLASS_NAMES:
                                    continue
                                temp_class_id = self.__CUSTOM_CLASS_NAMES.index(masks_dir) + 1
                                masks_dir_path = os.path.join(path, masks_dir)
                                for mask_file in os.listdir(masks_dir_path):
                                    mask = imread(os.path.join(masks_dir_path, mask_file))
                                    masks[:, :, iterator] = mask
                                    if self.__CONFIG.USE_MINI_MASK:
                                        bboxes[iterator] = dW.getBboxFromName(mask_file)
                                    else:
                                        bboxes[iterator] = utils.extract_bboxes(mask)
                                    class_ids[iterator] = temp_class_id
                                    iterator += 1
                            # Handle occlusions /!\ In our case there is no possible occlusion (part of object that is
                            # hidden), all objects are complete (some are parts of other)
                            # occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
                            # for i in range(number_of_masks - 2, -1, -1):
                            #     masks[:, :, i] = masks[:, :, i] * occlusion
                            #     occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
                            return masks, class_ids.astype(np.int32), bboxes

                    dataset_val = CustomDataset(self.__CUSTOM_CLASS_NAMES, self.__CORTEX_MODE,
                                                self.__CORTEX_SIZE, self.__CONFIG, imageInfo["IMAGE_FORMAT"])
                    dataset_val.load_images()
                    dataset_val.prepare()

                    image_id = dataset_val.image_ids[0]
                    gt_mask, gt_class_id, gt_bbox = dataset_val.load_mask(image_id)
                    if save_results:
                        if not displayOnlyAP:
                            print(" - Applying annotations on file to get expected results")
                        fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Expected")
                        visualize.display_instances(fullImage, gt_bbox, gt_mask, gt_class_id, dataset_val.class_names,
                                                    colorPerClass=True,
                                                    figsize=((1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                                             (1024 if self.__CORTEX_MODE else imageInfo[
                                                                 "HEIGHT"]) / 100),
                                                    image_format=imageInfo['IMAGE_FORMAT'],
                                                    title=f"{imageInfo['NAME']} Expected",
                                                    fileName=fileName, silent=True, config=self.__CONFIG)

                # Getting predictions for each division

                res = []
                total_px = self.__CONFIG.IMAGE_MAX_DIM * self.__CONFIG.IMAGE_MIN_DIM
                skipped = 0
                debugIterator = -1
                skippedText = ""
                inference_start_time = time()
                if not displayOnlyAP:
                    progressBar(0, imageInfo["NB_DIV"], prefix=' - Inference ')
                for divId in range(imageInfo["NB_DIV"]):
                    division = dD.getImageDivision(image, imageInfo["X_STARTS"], imageInfo["Y_STARTS"], divId)
                    grayDivision = cv2.cvtColor(division, cv2.COLOR_RGB2GRAY)
                    colorPx = cv2.countNonZero(grayDivision)
                    if colorPx / total_px > 0.1:
                        results = self.__MODEL.detect([division])
                        results = results[0]
                        results["div_id"] = divId
                        if self.__CONFIG.USE_MINI_MASK:
                            res.append(utils.reduce_memory(results, config=self.__CONFIG, allow_sparse=allowSparse))
                        else:
                            res.append(results)
                    elif not displayOnlyAP:
                        skipped += 1
                        skippedText = f" ({skipped} empty division{'s' if skipped > 1 else ''} skipped)"
                    if not displayOnlyAP:
                        if divId + 1 == imageInfo["NB_DIV"]:
                            inference_duration = round(time() - inference_start_time)
                            skippedText += f" Duration = {formatTime(inference_duration)}"
                        progressBar(divId + 1, imageInfo["NB_DIV"], prefix=' - Inference ', suffix=skippedText)
                if len(res) > 0:
                    del results
                # Post-processing of the predictions
                if not displayOnlyAP:
                    print(" - Fusing results of all divisions")

                res = pp.fuse_results(res, image.shape, division_size=self.__DIVISION_SIZE,
                                      min_overlap_part=self.__MIN_OVERLAP_PART)

                if savePreFusionImage:
                    step = "pre fusion"
                    debugIterator += 1
                    self.save_debug_image(step, debugIterator, fullImage, imageInfo, res, image_results_path,
                                          silent=displayOnlyAP)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    progressBarPrefix = " - Fusing overlapping masks " if not displayOnlyAP else None
                    res = pp.fuse_masks(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                        config=self.__CONFIG, displayProgress=progressBarPrefix, verbose=0)

                    if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                        if savePreFilterImage:
                            step = "pre orphan filter"
                            debugIterator += 1
                            self.save_debug_image(step, debugIterator, fullImage, imageInfo, res, image_results_path,
                                                  silent=displayOnlyAP)
                        # TODO : Build automatically classes_hierarchy
                        classes_hierarchy = {
                            3: {"contains": [4, 5], "keep_if_no_child": False},
                            8: {"contains": [9, 10], "keep_if_no_child": True}
                        }
                        progressBarPrefix = " - Removing orphan masks " if not displayOnlyAP else None
                        res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                     mask_threshold=filter_mask_threshold,
                                                     classes_hierarchy=classes_hierarchy,
                                                     displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                     verbose=0)
                    if savePreFilterImage:
                        step = "pre filter"
                        debugIterator += 1
                        self.save_debug_image(step, debugIterator, fullImage, imageInfo, res, image_results_path,
                                              silent=displayOnlyAP)

                    progressBarPrefix = " - Removing non-sense masks " if not displayOnlyAP else None
                    res = pp.filter_masks(res, bb_threshold=filter_bb_threshold, priority_table=priority_table,
                                          mask_threshold=filter_mask_threshold, included_threshold=0.9,
                                          including_threshold=0.6, verbose=0,
                                          displayProgress=progressBarPrefix, config=self.__CONFIG)

                    if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                        if savePreFilterImage:
                            step = "pre orphan filter (Pass 2)"
                            debugIterator += 1
                            self.save_debug_image(step, debugIterator, fullImage, imageInfo, res, image_results_path,
                                                  silent=displayOnlyAP)
                        # TODO : Build automatically classes_hierarchy
                        classes_hierarchy = {
                            3: {"contains": [4, 5], "keep_if_no_child": False},
                            8: {"contains": [9, 10], "keep_if_no_child": True}
                        }
                        progressBarPrefix = " - Removing orphan masks " if not displayOnlyAP else None
                        res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                     mask_threshold=filter_mask_threshold,
                                                     classes_hierarchy=classes_hierarchy,
                                                     displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                     verbose=0)

                    if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                        if savePreFusionImage:
                            step = "pre class fusion"
                            debugIterator += 1
                            self.save_debug_image(step, debugIterator, fullImage, imageInfo, res, image_results_path,
                                                  silent=displayOnlyAP)
                        progressBarPrefix = " - Fusing overlapping equivalent masks " if not displayOnlyAP else None
                        classes_compatibility = [[4, 5]]  # Nsg partiel + nsg complet
                        res = pp.fuse_class(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                            classes_compatibility=classes_compatibility, config=self.__CONFIG,
                                            displayProgress=progressBarPrefix, verbose=0)
                        if savePreFilterImage:
                            step = "pre small masks removal"
                            debugIterator += 1
                            self.save_debug_image(step, debugIterator, fullImage, imageInfo, res, image_results_path,
                                                  silent=displayOnlyAP)
                        progressBarPrefix = " - Removing small masks " if not displayOnlyAP else None
                        res = pp.filter_small_masks(res, min_size=minMaskArea, config=self.__CONFIG,
                                                    displayProgress=progressBarPrefix, verbose=0)

                if imageInfo["HAS_ANNOTATION"]:
                    if not displayOnlyAP:
                        print(" - Computing Average Precision and Confusion Matrix")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # TODO : Build automatically classes_hierarchy
                        classes_hierarchy = None
                        if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                            classes_hierarchy = {3: [4, 5], 8: [9, 10], 9: [10]}
                        AP, _, _, _, confusion_matrix = utils.compute_ap(gt_boxes=gt_bbox, gt_class_ids=gt_class_id,
                                                                         gt_masks=gt_mask, pred_boxes=res["rois"],
                                                                         pred_class_ids=res["class_ids"],
                                                                         pred_scores=res["scores"],
                                                                         pred_masks=res['masks'],
                                                                         nb_class=self.__NB_CLASS, score_threshold=0.3,
                                                                         iou_threshold=0.5, confusion_iou_threshold=0.3,
                                                                         classes_hierarchy=classes_hierarchy,
                                                                         confusion_background_class=True,
                                                                         confusion_only_best_match=False)

                        print(f"{'' if displayOnlyAP else '   '} - Average Precision is about {AP:06.2%}")
                        self.__CONFUSION_MATRIX = np.add(self.__CONFUSION_MATRIX, confusion_matrix)
                        self.__APs.append(AP)
                        if save_results:
                            name = f"{imageInfo['NAME']} Confusion Matrix"
                            confusionMatrixFileName = os.path.join(image_results_path, name.replace(' ', '_'))
                            name2 = f"{imageInfo['NAME']} Confusion Matrix (Normalized)"
                            confusionMatrixFileName2 = os.path.join(image_results_path, name2.replace('(', '')
                                                                    .replace(')', '')
                                                                    .replace(' ', '_'))
                            visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                               title=name,
                                                               cmap=cmap, show=False,
                                                               fileName=confusionMatrixFileName)
                            visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                               title=name2,
                                                               cmap=cmap, show=False, normalize=True,
                                                               fileName=confusionMatrixFileName2)

                if not self.__CORTEX_MODE:
                    print(" - Computing statistics on predictions")
                    stats = pp.getCountAndArea(res, classesInfo=self.__CLASSES_INFO,
                                               selectedClasses=self.__CUSTOM_CLASS_NAMES, config=self.__CONFIG)
                    print(f"    - cortex : area = {imageInfo['CORTEX_AREA']}")
                    for className in self.__CUSTOM_CLASS_NAMES:
                        stat = stats[className]
                        print(f"    - {className} : count = {stat['count']}, area = {stat['area']} px")
                    if save_results:
                        with open(os.path.join(image_results_path, f"{imageInfo['NAME']}_stats.json"), "w") as saveFile:
                            stats["cortex"] = {"count": 1, "area": imageInfo["CORTEX_AREA"]}
                            try:
                                json.dump(stats, saveFile, indent='\t')
                            except TypeError:
                                print("    Failed to save statistics", flush=True)

                if not displayOnlyAP:
                    print(" - Applying masks on image")

                if save_results:
                    if self.__CORTEX_MODE:
                        if not displayOnlyAP:
                            print(" - Cleaning full resolution image")
                        allCortices = None
                        allMedulla = None
                        # Gathering every cortex masks into one
                        for idxMask, classMask in enumerate(res['class_ids']):
                            if classMask == 1:
                                if allCortices is None:  # First mask found
                                    allCortices = res['masks'][:, :, idxMask].copy() * 255
                                else:  # Additional masks found
                                    allCortices = cv2.bitwise_or(allCortices, res['masks'][:, :, idxMask] * 255)
                            elif classMask == 2:
                                if allMedulla is None:
                                    allMedulla = res['masks'][:, :, idxMask].copy() * 255
                                else:
                                    allMedulla = cv2.bitwise_or(allMedulla, res['masks'][:, :, idxMask] * 255)

                        # To avoid cleaning an image without cortex
                        if allCortices is not None:
                            # If Medulla mask found, we clean the cortex with those
                            if allMedulla is not None:
                                allMedulla = cv2.bitwise_not(allMedulla.astype(np.uint8))
                                allCortices = cv2.bitwise_and(allCortices.astype(np.uint8), allMedulla)

                            # Extracting the new Bbox
                            allCorticesROI = utils.extract_bboxes(allCortices)

                            fusion_dir = os.path.join(image_results_path, f"{imageInfo['NAME']}_fusion")
                            os.makedirs(fusion_dir, exist_ok=True)
                            allCorticesSmall = allCortices[allCorticesROI[0]:allCorticesROI[2],
                                               allCorticesROI[1]:allCorticesROI[3]]
                            cv2.imwrite(os.path.join(fusion_dir, f"{imageInfo['NAME']}_cortex.jpg"),
                                        allCorticesSmall, CV2_IMWRITE_PARAM)
                            # Computing coordinates at full resolution
                            yRatio = imageInfo['HEIGHT'] / self.__CORTEX_SIZE[0]
                            xRatio = imageInfo['WIDTH'] / self.__CORTEX_SIZE[1]
                            allCorticesROI[0] = int(allCorticesROI[0] * yRatio)
                            allCorticesROI[1] = int(allCorticesROI[1] * xRatio)
                            allCorticesROI[2] = int(allCorticesROI[2] * yRatio)
                            allCorticesROI[3] = int(allCorticesROI[3] * xRatio)

                            # Resizing and adding the 2 missing channels of the cortices mask
                            allCortices = cv2.resize(np.uint8(allCortices), (imageInfo['WIDTH'], imageInfo['HEIGHT']),
                                                     interpolation=cv2.INTER_CUBIC)
                            temp = np.repeat(allCortices[:, :, np.newaxis], 3, axis=2)

                            # Masking the image and saving it
                            imageInfo['FULL_RES_IMAGE'] = cv2.bitwise_and(
                                imageInfo['FULL_RES_IMAGE'][allCorticesROI[0]: allCorticesROI[2],
                                allCorticesROI[1]:allCorticesROI[3], :],
                                temp[allCorticesROI[0]: allCorticesROI[2],
                                allCorticesROI[1]:allCorticesROI[3], :])
                            cv2.imwrite(os.path.join(fusion_dir, f"{imageInfo['NAME']}_cleaned.jpg"),
                                        cv2.cvtColor(imageInfo['FULL_RES_IMAGE'], cv2.COLOR_RGB2BGR, CV2_IMWRITE_PARAM))

                            #########################################################
                            # Preparing to export all "fusion" divisions with stats #
                            #########################################################
                            fusion_info_file_path = os.path.join(image_results_path,
                                                                 f"{imageInfo['NAME']}_fusion_info.skinet")
                            fusionInfo = {"image": imageInfo["NAME"]}

                            # Computing ratio between full resolution image and the low one
                            height, width, _ = imageInfo['FULL_RES_IMAGE'].shape
                            smallHeight, smallWidth = allCorticesSmall.shape
                            xRatio = width / smallWidth
                            yRatio = height / smallHeight

                            # Computing divisions coordinates for full and low resolution images
                            divisionSize = dD.getMaxSizeForDivAmount(nbMaxDivPerAxis, self.__DIVISION_SIZE,
                                                                     self.__MIN_OVERLAP_PART_MAIN)
                            xStarts = dD.computeStartsOfInterval(width, intervalLength=divisionSize,
                                                                 min_overlap_part=0)
                            yStarts = dD.computeStartsOfInterval(height, intervalLength=divisionSize,
                                                                 min_overlap_part=0)

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
                            fusionInfo["cortex_area"] = dD.getBWCount(allCortices)[1]
                            fusionInfo["divisions"] = {}

                            # Extracting and saving all divisions
                            for divID in range(dD.getDivisionsCount(xStarts, yStarts)):
                                cortexDiv = dD.getImageDivision(allCorticesSmall, xStartsEquivalent, yStartsEquivalent,
                                                                divID, divisionSize=(xDivSide, yDivSide))
                                black, white = dD.getBWCount(cortexDiv.astype(np.uint8))
                                partOfDiv = white / (white + black)
                                fusionInfo["divisions"][divID] = {"cortex_area": white,
                                                                  "cortex_representative_part": partOfDiv,
                                                                  "used": partOfDiv > fusionDivThreshold}
                                if partOfDiv > fusionDivThreshold:
                                    x, xEnd, y, yEnd = dD.getDivisionByID(xStarts, yStarts, divID, divisionSize)
                                    fusionInfo["divisions"][divID]["coordinates"] = {"x1": x, "x2": xEnd, "y1": y,
                                                                                     "y2": yEnd}
                                    imageDivision = dD.getImageDivision(imageInfo['FULL_RES_IMAGE'], xStarts, yStarts,
                                                                        divID, divisionSize)
                                    cv2.imwrite(os.path.join(fusion_dir, f"{imageInfo['NAME']}_{divID}.jpg"),
                                                cv2.cvtColor(imageDivision, cv2.COLOR_RGB2BGR, CV2_IMWRITE_PARAM))

                            # Writing informations into the .skinet file
                            with open(fusion_info_file_path, 'w') as fusionInfoFile:
                                try:
                                    json.dump(fusionInfo, fusionInfoFile, indent="\t")
                                except TypeError:
                                    print("    Failed to save fusion info file", file=sys.stderr, flush=True)

                    fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Predicted")
                    names = self.__CUSTOM_CLASS_NAMES.copy()
                    names.insert(0, 'background')
                    visualize.display_instances(fullImage, res['rois'], res['masks'], res['class_ids'], names,
                                                res['scores'],
                                                colorPerClass=True, fileName=fileName, onlyImage=True, silent=True,
                                                figsize=((1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                                         (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100),
                                                image_format=imageInfo['IMAGE_FORMAT'],
                                                config=self.__CONFIG)

                    # Annotations Extraction
                    if not displayOnlyAP:
                        print(" - Saving predicted annotations files")
                    pp.export_annotations(imageInfo, res, self.__CLASSES_INFO,
                                          ASAPAdapter, save_path=image_results_path,
                                          config=self.__CONFIG, verbose=0 if displayOnlyAP else 1)
                    pp.export_annotations(imageInfo, res, self.__CLASSES_INFO,
                                          LabelMeAdapter, save_path=image_results_path,
                                          config=self.__CONFIG, verbose=0 if displayOnlyAP else 1)
                final_time = round(time() - start_time)
                print(f" Done in {formatTime(final_time)}\n")
                if not imageInfo['HAS_ANNOTATION']:
                    AP = -1
                if save_results:
                    with open(logsPath, 'a') as results_log:
                        apText = f"{AP:4.3f}".replace(".", ",")
                        results_log.write(f"{imageInfo['NAME']}; {final_time}; {apText}%;\n")
                plt.close('all')

        if len(self.__APs) > 1:
            mAP = np.mean(self.__APs)
            print(f"Mean Average Precision is about {mAP:06.2%}")
            name = "Final Confusion Matrix"
            name2 = "Final Confusion Matrix (Normalized)"
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name,
                                               cmap=cmap, show=False,
                                               fileName=(os.path.join(results_path, name.replace(' ', '_'))
                                                         if save_results else None))
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name2,
                                               cmap=cmap, show=False, normalize=True,
                                               fileName=(os.path.join(results_path, name2.replace('(', '')
                                                                      .replace(')', '')
                                                                      .replace(' ', '_'))
                                                         if save_results else None))
        else:
            mAP = -1
        total_time = round(time() - total_start_time)
        print(f"All inferences done in {formatTime(total_time)}")
        if save_results:
            with open(logsPath, 'a') as results_log:
                mapText = f"{mAP:4.3f}".replace(".", ",")
                results_log.write(f"GLOBAL; {total_time}; {mapText}%;\n")

    def save_debug_image(self, step, debugIterator, fullImage, imageInfo, res, image_results_path, silent=True):
        if not silent:
            print(f" - Saving {step} image")
        step = step.replace(' ', '_').replace('(', '').replace(')', '')
        names = self.__CUSTOM_CLASS_NAMES.copy()
        names.insert(0, 'background')
        fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Inference_debug_{debugIterator:02d}_{step}")
        visualize.display_instances(fullImage, res['rois'], res['masks'], res['class_ids'], names,
                                    res['scores'],
                                    colorPerClass=True, fileName=fileName, onlyImage=False, silent=True,
                                    figsize=((1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                             (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100),
                                    image_format=imageInfo['IMAGE_FORMAT'],
                                    config=self.__CONFIG)
