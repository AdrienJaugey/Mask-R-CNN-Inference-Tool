from abc import ABC, abstractmethod
import xml.etree.ElementTree as et


class XMLExporter(ABC):

    def __init__(self, rootName):
        super().__init__()
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

    @abstractmethod
    def addAnnotation(self, className, points):
        """
        Adding an annotation to the XML
        :param className: name of the annotation class
        :param points: 2D array of polygon points representing the annotated area : [[x, y]]
        :return: None
        """
        pass

    @abstractmethod
    def addAnnotationClass(self, className):
        """
        Adding the description of a prediction class to the XML
        :param className: name of the class
        :return: None
        """
        pass

    def __str__(self):
        return et.tostring(self.root, encoding='unicode', method='xml')

    def saveToFile(self, filePath):
        """
        Saving the current XML to a file
        :param filePath: the path to the saved file
        :return: None
        """
        with open(filePath, 'w') as file:
            file.write("<?xml version=\"1.0\"?>\n" + str(self))
            print("XML saved to {}".format(filePath))