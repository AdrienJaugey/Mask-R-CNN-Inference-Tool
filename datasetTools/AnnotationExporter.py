import os
from abc import ABC, abstractmethod
import xml.etree.ElementTree as et
import json


class AnnotationExporter(ABC):

    def __init__(self, imageInfo: dict):
        """
        Init Annotation exporter
        :param imageInfo: {"name": Image Name, "height": image height, "width": image width}
        """
        self.imageInfo = imageInfo

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
            print("Annotations saved to {}".format(filePath))


class XMLExporter(AnnotationExporter, ABC):

    def __init__(self, imageInfo: dict, rootName):
        super().__init__(imageInfo)
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

    def getSaveFileName(self, fileName):
        return fileName + '.xml'

    def __str__(self):
        return "<?xml version=\"1.0\"?>\n" + et.tostring(self.root, encoding='unicode', method='xml')


class JSONExporter(AnnotationExporter, ABC):

    def __init__(self, imageInfo: dict):
        super().__init__(imageInfo)
        self.data = {}

    def getSaveFileName(self, fileName):
        return fileName + '.json'

    def __str__(self):
        return json.dumps(self.data, indent='\t')
