import os
from abc import ABC, abstractmethod
import xml.etree.ElementTree as et
import json


class AnnotationAdapter(ABC):
    # TODO: Saving annotations to adapter specific folder to avoid overwriting some annotations if using same file name

    def __init__(self, imageInfo: dict, verbose=0):
        """
        Init Annotation exporter
        :param imageInfo: {"name": Image Name, "height": image height, "width": image width}
        """
        self.imageInfo = imageInfo
        self.verbose = verbose

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
        return et.tostring(self.root, encoding='unicode', method='xml', xml_declaration=True)

    @staticmethod
    def canRead(filePath):
        return 'xml' in os.path.split(filePath)[1]


class JSONAdapter(AnnotationAdapter, ABC):

    def __init__(self, imageInfo: dict, verbose=0):
        super().__init__(imageInfo, verbose)
        self.data = {}

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
ANNOTATION_FORMAT = []
for adapter in ANNOTATION_ADAPTERS:
    format = adapter.getAnnotationFormat()
    if format not in ANNOTATION_FORMAT:
        ANNOTATION_FORMAT.append(format)
