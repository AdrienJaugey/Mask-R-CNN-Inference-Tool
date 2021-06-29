"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project
Dataset tools

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import os
from abc import ABC, abstractmethod
import xml.etree.ElementTree as et
import json
import numpy as np
import cv2

from mrcnn.Config import Config
from mrcnn.utils import shift_bbox, expand_mask


class AnnotationAdapter(ABC):
    # TODO: Saving annotations to adapter specific folder to avoid overwriting some annotations if using same file name

    def __init__(self, imageInfo: dict, verbose=0):
        """
        Init Annotation exporter
        :param imageInfo: {"name": Image Name, "height": image height, "width": image width}
        """
        self.imageInfo = imageInfo
        self.verbose = verbose

    @staticmethod
    def getName():
        """
        Returns the name of the Annotation format or Software that reads this format
        :return: name of the annotation format
        """
        return "Base"

    @abstractmethod
    def addAnnotation(self, classInfo: {}, points):
        """
        Adding an annotation to the Annotations file
        :param classInfo: {"name" : Class name, "id" : Class ID }
        :param points: 2D array of polygon points representing the annotated area : [[x, y]]
        :return: None
        Not casting data to default type could cause crashes while printing or saving to file
        """
        pass

    @abstractmethod
    def addAnnotationClass(self, classInfo: {}):
        """
        Adding the description of a prediction class to the Annotations file
        :param classInfo: {"name" : Class name, "id" : Class ID }
        :return: None
        Not casting data to default type could cause crashes while printing or saving to file
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def getAnnotationFormat():
        return None

    @staticmethod
    def getPriorityLevel():
        return -1

    @abstractmethod
    def getSaveFileName(self, fileName):
        pass

    def saveToFile(self, savePath, fileName):
        """
        Saving the current Annotations to a file
        :param savePath: path to the save directory
        :param fileName: name of the save file
        :return: None
        """
        filePath = os.path.join(savePath, self.getSaveFileName(fileName))
        with open(filePath, 'w') as file:
            file.write(str(self))
            if self.verbose > 0:
                print("Annotations saved to {}".format(filePath))

    @staticmethod
    def canRead(filePath):
        """
        Test if class is able to read an annotation file format
        :param filePath: file path to the annotation file
        :return: True if the class is able to read it, else False
        """
        return False

    @staticmethod
    def readFile(filePath):
        """
        Read an annotation file and extract masks information
        :param filePath: file path to the annotation file
        :return: [ ( maskClass, [[x, y]] ) ]
        """
        return None

    @staticmethod
    def offsetAnnotations(filePath, xOffset=0, yOffset=0, outputFilePath=None):
        """
        Offsets annotations of a file
        :param filePath: file path to the annotation file
        :param xOffset: the x-axis offset to apply to annotations
        :param yOffset: the y-axis offset to apply to annotations
        :param outputFilePath: path to the output file, if None, will modify the base file
        :return: None
        """
        return None


class XMLAdapter(AnnotationAdapter, ABC):

    def __init__(self, imageInfo: dict, rootName, verbose=0):
        super().__init__(imageInfo, verbose=verbose)
        self.root = et.Element(rootName)
        self.root.tail = "\n"
        self.root.text = "\n\t"

    @staticmethod
    def getName():
        return "XML"

    def addToRoot(self, node):
        """
        Adding the given node to the XML root element
        :param node: the node to add
        :return: None
        """
        self.root.append(node)

    @staticmethod
    def getAnnotationFormat():
        return "xml"

    def getSaveFileName(self, fileName):
        return fileName + '.xml'

    def __str__(self):
        if 'xml_declaration' in et.tostring.__code__.co_varnames:
            return et.tostring(self.root, encoding='unicode', method='xml', xml_declaration=True)
        else:
            return et.tostring(self.root, encoding='unicode', method='xml')

    @staticmethod
    def canRead(filePath):
        return 'xml' in os.path.split(filePath)[1]


class JSONAdapter(AnnotationAdapter, ABC):

    def __init__(self, imageInfo: dict, verbose=0):
        super().__init__(imageInfo, verbose)
        self.data = {}

    @staticmethod
    def getName():
        return "JSON"

    @staticmethod
    def getAnnotationFormat():
        return "json"

    def getSaveFileName(self, fileName):
        return fileName + '.json'

    def __str__(self):
        return json.dumps(self.data, indent='\t')

    @staticmethod
    def canRead(filePath):
        return 'json' in os.path.split(filePath)[1]


from datasetTools.ASAPAdapter import ASAPAdapter
from datasetTools.LabelMeAdapter import LabelMeAdapter

ANNOTATION_ADAPTERS = [ASAPAdapter, LabelMeAdapter]
ANNOTATION_ADAPTERS = {c.getName().lower(): c for c in ANNOTATION_ADAPTERS}
ANNOTATION_FORMAT = []
for adapter in ANNOTATION_ADAPTERS.values():
    annotation_format = adapter.getAnnotationFormat()
    if annotation_format not in ANNOTATION_FORMAT:
        ANNOTATION_FORMAT.append(annotation_format)


def getAdapterFromName(name: str):
    return ANNOTATION_ADAPTERS.get(name.lower(), None)


def getPoints(mask, xOffset=0, yOffset=0, epsilon=1, show=False, waitSeconds=10, info=False):
    """
    Return a list of points describing the given mask as a polygon
    :param mask: the mask you want the points
    :param xOffset: if using a RoI the x-axis offset used
    :param yOffset: if using a RoI the y-axis offset used
    :param epsilon: epsilon parameter of cv2.approxPolyDP() method
    :param show: whether you want or not to display the approximated mask so you can see it
    :param waitSeconds: time in seconds to wait before closing automatically the displayed masks, or press ESC to close
    :param info: whether you want to display some information (mask size, number of predicted points, number of
    approximated points...) or not
    :return: 2D-array of points coordinates : [[x, y]]
    """
    pts = None
    contours, _ = cv2.findContours(mask, method=cv2.RETR_TREE, mode=cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # https://stackoverflow.com/questions/41879315/opencv-visualize-polygonal-curves-extracted-with-cv2-approxpolydp
        # Finding biggest area
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        res = cv2.approxPolyDP(cnt, epsilon, True)
        pts = []
        for point in res:
            # Casting coordinates to int, not doing this makes crash json dump
            pts.append([int(point[0][0] + xOffset), int(point[0][1] + yOffset)])

        if info:
            maskHeight, maskWidth = mask.shape
            nbPtPred = contours[0].shape[0]
            nbPtApprox = len(pts)
            print("Mask size : {}x{}".format(maskWidth, maskHeight))
            print("Nb points prediction : {}".format(nbPtPred))
            print("Nb points approx : {}".format(nbPtApprox))
            print("Compression rate : {:5.2f}%".format(nbPtPred / nbPtApprox * 100))
            temp = np.array(pts)
            xMin = np.amin(temp[:, 0])
            xMax = np.amax(temp[:, 0])
            yMin = np.amin(temp[:, 1])
            yMax = np.amax(temp[:, 1])
            print("{} <= X <= {}".format(xMin, xMax))
            print("{} <= Y <= {}".format(yMin, yMax))
            print()

        if show:
            img = np.zeros(mask.shape, np.int8)
            img = cv2.drawContours(img, [res], -1, 255, 2)
            cv2.imshow('before {}'.format(img.shape), mask * 255)
            cv2.imshow("approxPoly", img * 255)
            cv2.waitKey(max(waitSeconds, 1) * 1000)

    return pts


def export_annotations(image_info: dict, results: dict, adapterClass: AnnotationAdapter.__class__,
                       save_path="predicted", config: Config = None, verbose=0):
    """
    Exports predicted results to an XML annotation file using given XMLExporter
    :param image_info: Dict with at least {"NAME": str, "HEIGHT": int, "WIDTH": int} about the inferred image
    :param results: inference results of the image
    :param adapterClass: class inheriting XMLExporter
    :param save_path: path to the dir you want to save the annotation file
    :param config: the config to get mini_mask informations
    :param verbose: verbose level of the method (0 = nothing, 1 = information)
    :return: None
    """
    if config is None:
        print("Cannot export annotations as config is not given.")
        return

    rois = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']
    height = masks.shape[0]
    width = masks.shape[1]
    adapter_instance = adapterClass({"name": image_info['NAME'], "height": image_info['HEIGHT'],
                                     'width': image_info['WIDTH'], 'format': image_info['IMAGE_FORMAT']},
                                    verbose=verbose)
    if verbose > 0:
        print(f"Exporting to {adapter_instance.getName()} annotation file format.")
    # For each prediction
    for i in range(masks.shape[2]):
        if config is not None and config.is_using_mini_mask():
            shifted_roi = shift_bbox(rois[i])
            shifted_roi += [5, 5, 5, 5]
            image_size = shifted_roi[2:] + [5, 5]
            mask = expand_mask(shifted_roi, masks[:, :, i], image_size)
            yStart, xStart = rois[i][:2] - [5, 5]
        else:
            # Getting the RoI coordinates and the corresponding area
            # y1, x1, y2, x2
            yStart, xStart, yEnd, xEnd = rois[i]
            yStart = max(yStart - 10, 0)
            xStart = max(xStart - 10, 0)
            yEnd = min(yEnd + 10, height)
            xEnd = min(xEnd + 10, width)
            mask = masks[yStart:yEnd, xStart:xEnd, i]

        # Getting list of points coordinates and adding the prediction to XML
        points = getPoints(np.uint8(mask), xOffset=xStart, yOffset=yStart, show=False, waitSeconds=0, info=False)
        if points is None:
            continue
        adapter_instance.addAnnotation(config.get_classes_info()[class_ids[i] - 1], points)

    for classInfo in config.get_classes_info():
        adapter_instance.addAnnotationClass(classInfo)

    os.makedirs(save_path, exist_ok=True)
    adapter_instance.saveToFile(save_path, image_info['NAME'])
