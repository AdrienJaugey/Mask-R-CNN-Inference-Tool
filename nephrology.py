import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sys
    import random
    import math
    import re
    import time
    import numpy as np
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    import json
    from time import time, ctime
    from skimage.io import imread, imsave, imshow, imread_collection, concatenate_images
    from skimage.transform import resize
    from datasetTools import datasetDivider as div, AnnotationAdapter
    from datasetTools import datasetWrapper as wr
    from datasetTools.ASAPAdapter import ASAPAdapter
    from datasetTools.LabelMeAdapter import LabelMeAdapter
    from datetime import datetime
    from mrcnn import config
    from mrcnn import utils
    from mrcnn import model
    from mrcnn import visualize

    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
    from mrcnn import visualize
    from mrcnn.model import log


def get_ax(rows=1, cols=1, size=8):
    return plt.subplots(rows, cols, figsize=(size * cols, size * rows), frameon=False)


def listAvailableImage(dirPath: str):
    files = os.listdir(dirPath)
    image = []
    for file in files:
        if file in image:
            continue
        name = file.split('.')[0]
        extension = file.split('.')[-1]
        if extension == 'jp2':
            if (name + '.png') not in image:
                if os.path.exists(os.path.join(dirPath, name + '.png')):
                    image.append(name + '.png')
                else:
                    image.append(file)
        elif extension == 'png':
            image.append(file)
    for i in range(len(image)):
        image[i] = os.path.join(dirPath, image[i])
    return image


class NephrologyInferenceModel:

    def __init__(self, classesInfo, modelPath, divisionSize=1024):
        print("Initialisation")
        self.__CLASSES_INFO = classesInfo
        self.__MODEL_PATH = modelPath
        self.__DIVISION_SIZE = divisionSize
        self.__CELLS_CLASS_NAMES = []
        for classInfo in classesInfo:
            if not classInfo["ignore"]:
                self.__CELLS_CLASS_NAMES.append(classInfo["name"])
        self.__NB_CLASS = len(self.__CELLS_CLASS_NAMES)
        # Root directory of the project
        self.__ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "logs")

        # Configurations
        nbClass = self.__NB_CLASS
        divSize = 1024 if self.__DIVISION_SIZE == "noDiv" else self.__DIVISION_SIZE
        self.__CONFUSION_MATRIX = np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int32)
        self.__APs = []

        class CellsConfig(Config):
            NAME = "cells"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + nbClass
            IMAGE_MIN_DIM = divSize
            IMAGE_MAX_DIM = divSize
            RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)
            TRAIN_ROIS_PER_IMAGE = 800
            STEPS_PER_EPOCH = 400
            VALIDATION_STEPS = 50

        config = CellsConfig()

        class InferenceConfig(CellsConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        self.__MODEL = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=self.__MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        assert self.__MODEL_PATH != "", "Provide path to trained weights"
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
        fullImage = None
        imageInfo = None
        image_results_path = None
        if os.path.exists(imagePath):
            image_file_name = imagePath.split('/')[-1]
            dirPath = imagePath.replace(image_file_name, '')
            image_name = image_file_name.split('.')[0]
            image_extension = image_file_name.split('.')[-1]

            fullImage = imread(imagePath)
            if image_extension != "png":
                # print('Converting to png')
                tempPath = dirPath + image_name + '.png'
                imsave(tempPath, fullImage)
                imagePath = tempPath

            # Creating the result dir if given and copying the base image in it
            if results_path is not None:
                image_results_path = results_path + image_name + "/"
                os.makedirs(image_results_path, exist_ok=True)
                imsave(os.path.join(image_results_path, image_name + ".png"), fullImage)
            else:
                image_results_path = None

            fullImage = cv2.imread(imagePath)
            fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2RGB)
            image = fullImage.copy()
            height, width, _ = fullImage.shape
            cortexArea = height * width
            xStarts = [0] if self.__DIVISION_SIZE == "noDiv" else div.computeStartsOfInterval(
                maxVal=width, intervalLength=self.__DIVISION_SIZE
            )
            yStarts = [0] if self.__DIVISION_SIZE == "noDiv" else div.computeStartsOfInterval(
                maxVal=height, intervalLength=self.__DIVISION_SIZE
            )

            nbDiv = div.getDivisionsCount(xStarts, yStarts)

            annotationExists = False
            has_annotation = False
            for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                annotationExists = annotationExists or os.path.exists(dirPath + image_name + '.' + ext)
            if annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible")
                wr.createMasksOfImage(dirPath, image_name, 'data', classesInfo=self.__CLASSES_INFO)
                wr.fuseCortices('data', image_name, deleteBaseMasks=True)
                wr.cleanImage('data', image_name, onlyMasks=False)
                maskDirs = os.listdir(os.path.join('data', image_name))
                if "cortex" in maskDirs:
                    cortexDirPath = os.path.join('data', image_name, 'cortex')
                    cortexImageFilePath = os.listdir(cortexDirPath)[0]
                    cortex = cv2.imread(os.path.join(cortexDirPath, cortexImageFilePath), cv2.IMREAD_UNCHANGED)
                    cortexArea = div.getBWCount(cortex)[1]
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
                if "cortex" not in self.__CELLS_CLASS_NAMES and "cortex" in maskDirs:
                    maskDirs.remove("cortex")
                for classToPredict in self.__CELLS_CLASS_NAMES:
                    if classToPredict in maskDirs:
                        has_annotation = True
                        break
                if has_annotation and not silent:
                    print("    - AP and confusion matrix will be computed")

            imageInfo = {
                "PATH": imagePath,
                "DIR_PATH": dirPath,
                "FILE_NAME": image_file_name,
                "NAME": image_name,
                "HEIGHT": height,
                "WIDTH": width,
                "NB_DIV": nbDiv,
                "X_STARTS": xStarts,
                "Y_STARTS": yStarts,
                "HAS_ANNOTATION": has_annotation,
                "CORTEX_AREA": cortexArea
            }
        return image, fullImage, imageInfo, image_results_path

    def inference(self, images: list, results_path=None, save_results=True,
                  fusion_bb_threshold=0., fusion_mask_threshold=0.1,
                  filter_bb_threshold=0.5, filter_mask_threshold=0.9,
                  priority_table=None, displayOnlyAP=False):

        assert len(images) > 0, "images list cannot be empty"

        # If results have to be saved, setting the results path and creating directory
        if save_results:
            if results_path is None or results_path in ['.', './', "/"]:
                lastDir = "results"
                remainingPath = ""
            else:
                if results_path[-1] == '/':
                    index = -2
                else:
                    index = -1
                lastDir = results_path.split('/')[index].replace('/', '')
                remainingPath = results_path[0:results_path.index(lastDir)]
            results_path = remainingPath + "{}_{}/".format(lastDir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(results_path)
            print("Results will be saved to {}".format(results_path))
            logsPath = os.path.join(results_path, 'inference_data.csv')
            with open(logsPath, 'w') as results_log:
                results_log.write("Image; Duration (s); Precision;\n")
        else:
            print("No result will be saved")
            results_path = None

        self.__CONFUSION_MATRIX.fill(0)
        self.__APs.clear()
        cmap = plt.cm.get_cmap('hot')
        total_start_time = time()
        for i, IMAGE_PATH in enumerate(images):
            start_time = time()
            print("Using {} image file ({}/{} | {:4.2f}%)".format(IMAGE_PATH, i + 1, len(images),
                                                                  (i + 1) / len(images) * 100))
            image, fullImage, imageInfo, image_results_path = self.prepare_image(IMAGE_PATH, results_path,
                                                                                 silent=displayOnlyAP)
            if imageInfo["HAS_ANNOTATION"]:
                class LightCellsDataset(utils.Dataset):

                    def __init__(self, cells_class_names):
                        super().__init__()
                        self.__CELLS_CLASS_NAMES = cells_class_names.copy()

                    def get_class_names(self):
                        return self.__CELLS_CLASS_NAMES.copy()

                    def load_cells(self):
                        # Add classes
                        for class_id, class_name in enumerate(self.__CELLS_CLASS_NAMES):
                            self.add_class("cells", class_id + 1, class_name)

                        img_path = 'data/' + imageInfo["NAME"] + '/images/'
                        self.add_image("cells", image_id=imageInfo["NAME"], path=img_path)

                    def load_image(self, image_id):

                        info = self.image_info[image_id]
                        info = info.get("id")

                        img = imread('data/' + info + '/images/' + info + '.png')[:, :, :3]

                        return img

                    def image_reference(self, image_id):
                        """Return the cells data of the image."""
                        info = self.image_info[image_id]
                        if info["source"] == "cells":
                            return info["cells"]
                        else:
                            super(self.__class__).image_reference(self, image_id)

                    def load_mask(self, image_id):
                        """Generate instance masks for cells of the given image ID.
                        """
                        info = self.image_info[image_id]
                        info = info.get("id")

                        path = 'data/' + info

                        # Counting masks for current image
                        number_of_masks = 0
                        for masks_dir in os.listdir(path):
                            # For each directory excepting /images
                            if masks_dir not in self.__CELLS_CLASS_NAMES:
                                continue
                            temp_DIR = path + '/' + masks_dir
                            # Adding length of directory https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
                            number_of_masks += len(
                                [name for name in os.listdir(temp_DIR) if
                                 os.path.isfile(os.path.join(temp_DIR, name))])

                        mask = np.zeros([imageInfo["HEIGHT"], imageInfo["WIDTH"], number_of_masks], dtype=np.uint8)
                        iterator = 0
                        class_ids = np.zeros((number_of_masks,), dtype=int)
                        for masks_dir in os.listdir(path):
                            if masks_dir not in self.__CELLS_CLASS_NAMES:
                                continue
                            temp_class_id = self.__CELLS_CLASS_NAMES.index(masks_dir) + 1
                            for mask_file in next(os.walk(path + '/' + masks_dir + '/'))[2]:
                                mask_ = imread(path + '/' + masks_dir + '/' + mask_file)
                                mask[:, :, iterator] = mask_
                                class_ids[iterator] = temp_class_id
                                iterator += 1
                        # Handle occlusions
                        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
                        for i in range(number_of_masks - 2, -1, -1):
                            mask[:, :, i] = mask[:, :, i] * occlusion
                            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
                        return mask, class_ids.astype(np.int32)

                dataset_val = LightCellsDataset(self.__CELLS_CLASS_NAMES)
                dataset_val.load_cells()
                dataset_val.prepare()

                image_id = dataset_val.image_ids[0]
                gt_mask, gt_class_id = dataset_val.load_mask(image_id)
                gt_bbox = utils.extract_bboxes(gt_mask)
                if save_results:
                    if not displayOnlyAP:
                        print(" - Applying annotations on file to get expected results")
                    fileName = image_results_path + "{} Expected".format(imageInfo["NAME"])
                    visualize.display_instances(fullImage, gt_bbox, gt_mask, gt_class_id, dataset_val.class_names,
                                                colorPerClass=True, figsize=(imageInfo["WIDTH"] / 100,
                                                                             imageInfo["HEIGHT"] / 100),
                                                title="{} Expected".format(imageInfo["NAME"]),
                                                fileName=fileName, silent=True)

            # Getting predictions for each division

            res = []
            if not displayOnlyAP:
                print(" - Starting Inference")
            for divId in range(imageInfo["NB_DIV"]):
                division = div.getImageDivision(image, imageInfo["X_STARTS"], imageInfo["Y_STARTS"], divId)
                if not displayOnlyAP:
                    print('    - Inference {}/{}'.format(divId + 1, imageInfo["NB_DIV"]))
                results = self.__MODEL.detect([division])
                res.append(results[0])

            # Post-processing of the predictions

            if not displayOnlyAP:
                print(" - Fusing results of all divisions")
            res = utils.fuse_results(res, image, division_size=self.__DIVISION_SIZE)

            if not displayOnlyAP:
                print(" - Fusing overlapping masks")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = utils.fuse_masks(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                       verbose=1)

                if not displayOnlyAP:
                    print(" - Removing non-sense masks")
                res = utils.filter_fused_masks(res,
                                               bb_threshold=filter_bb_threshold,
                                               mask_threshold=filter_mask_threshold,
                                               priority_table=priority_table)

            if imageInfo["HAS_ANNOTATION"]:
                if not displayOnlyAP:
                    print(" - Computing Average Precision and Confusion Matrix")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    AP, _, _, _, confusion_matrix = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                                     res["rois"], res["class_ids"],
                                                                     res["scores"], res['masks'],
                                                                     nb_class=self.__NB_CLASS,
                                                                     confusion_iou_threshold=0.1)

                    print(
                        "{} - Average Precision is about {:5.2f}%".format("" if displayOnlyAP else "   ", AP * 100))
                    self.__CONFUSION_MATRIX = np.add(self.__CONFUSION_MATRIX, confusion_matrix)
                    self.__APs.append(AP)
                    if save_results:
                        name = "{} Confusion Matrix".format(imageInfo["NAME"])
                        name2 = "{} Normalized Confusion Matrix".format(imageInfo["NAME"])
                        visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                           title=name,
                                                           cmap=cmap, show=False,
                                                           fileName=image_results_path + name)
                        visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                           title=name2,
                                                           cmap=cmap, show=False, normalize=True,
                                                           fileName=image_results_path + name2)

            print(" - Computing statistics on predictions")
            stats = utils.getCountAndArea(res, classesInfo=self.__CLASSES_INFO,
                                          selectedClasses=self.__CELLS_CLASS_NAMES)
            print("    - cortex : area = {}".format(imageInfo["CORTEX_AREA"]))
            for classStats in stats:
                print("    - {} : count = {}, area = {} px".format(classStats["name"], classStats["count"],
                                                                   classStats["area"]))

            if not displayOnlyAP:
                print(" - Applying masks on image")

            if save_results:
                fileName = image_results_path + "{} Predicted".format(imageInfo["NAME"])
                names = self.__CELLS_CLASS_NAMES.copy()
                names.insert(0, 'background')
                visualize.display_instances(fullImage, res['rois'], res['masks'], res['class_ids'], names,
                                            res['scores'],
                                            colorPerClass=True, fileName=fileName, onlyImage=True, silent=True,
                                            figsize=(imageInfo["WIDTH"] / 100, imageInfo["HEIGHT"] / 100))

                # Annotations Extraction
                if not displayOnlyAP:
                    print(" - Saving predicted annotations files")
                utils.export_annotations(imageInfo["NAME"], res, self.__CLASSES_INFO,
                                         ASAPAdapter,
                                         save_path=image_results_path, verbose=0 if displayOnlyAP else 1)
                utils.export_annotations(imageInfo["NAME"], res, self.__CLASSES_INFO,
                                         LabelMeAdapter,
                                         save_path=image_results_path, verbose=0 if displayOnlyAP else 1)
            final_time = round(time() - start_time)
            m = final_time // 60
            s = final_time % 60
            print(" Done in {:02d}m {:02d}s\n".format(m, s))
            if not imageInfo['HAS_ANNOTATION']:
                AP = -1
            if save_results:
                with open(logsPath, 'a') as results_log:
                    apText = "{:4.3f}".format(AP * 100).replace(".", ",")
                    results_log.write("{}; {}; {}%;\n".format(imageInfo["NAME"], final_time, apText))
            plt.close('all')

        if len(self.__APs) > 1:
            mAP = np.mean(self.__APs) * 100
            print("Mean Average Precision is about {:5.2f}%".format(mAP))
            name = "Final Confusion Matrix"
            name2 = "Final Normalized Confusion Matrix"
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name,
                                               cmap=cmap, show=False,
                                               fileName=(results_path + name) if save_results else None)
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name2,
                                               cmap=cmap, show=False, normalize=True,
                                               fileName=(results_path + name2) if save_results else None)
        else:
            mAP = -1
        total_time = round(time() - total_start_time)
        h = total_time // 3600
        m = (total_time % 3600) // 60
        s = final_time % 60
        print("All inferences done in {:02d}h {:02d}m {:02d}s".format(h, m, s))
        if save_results:
            with open(logsPath, 'a') as results_log:
                mapText = "{:4.3f}".format(mAP).replace(".", ",")
                results_log.write("GLOBAL; {}; {}%;\n".format(total_time, mapText))
