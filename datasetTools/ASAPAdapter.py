"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project
Dataset tools

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import shutil
import xml.etree.ElementTree as et

from common_utils import rgb_to_hex
from datasetTools.AnnotationAdapter import XMLAdapter


class ASAPAdapter(XMLAdapter):
    """
    Exports predictions to ASAP annotation format
    """
    '''
        <?xml version="1.0"?>
        <ASAP_Annotations>
            <Annotations>
                <Annotation Name="Annotation {MASK_NUM:d}" Type="Polygon" PartOfGroup="{CLASS_NAME}" Color="#F4FA58">
                    <Coordinates>
                        <Coordinate Order="{PT_NUM:d}" X="{PT_X:d}" Y="{PT_Y:d}" />
                    </Coordinates>
                </Annotation>
            </Annotations>
            <AnnotationGroups>
                <Group Name="{CLASS_NAME}" PartOfGroup="None" Color="{CORRESPONDING_CLASS_COLOR}">
                    <Attributes />
                </Group>
            </AnnotationGroups>
        </ASAP_Annotations>
    '''

    def __init__(self, imageInfo: dict, verbose=0):
        super().__init__(imageInfo, "ASAP_Annotations", verbose=verbose)
        self.annotations = et.Element('Annotations')
        self.annotations.text = "\n\t\t"
        self.annotations.tail = "\n\t"
        self.addToRoot(self.annotations)
        self.groups = et.Element('AnnotationGroups')
        self.groups.text = "\n\t\t"
        self.groups.tail = "\n"
        self.addToRoot(self.groups)
        self.nbAnnotation = 0
        self.nbGroup = 0
        self.classCount = {}

    @staticmethod
    def getName():
        return "ASAP"

    def addAnnotation(self, classInfo: {}, points):
        if classInfo["name"] not in self.classCount:
            self.classCount[classInfo["name"]] = 0

        mask = et.Element('Annotation')
        mask.set('Name', f"{classInfo['name']} {self.classCount[classInfo['name']]} ({self.nbAnnotation})")
        mask.set("Type", "Polygon")
        mask.set("PartOfGroup", classInfo["name"])
        mask.set("Color", "#F4FA58")
        mask.text = "\n\t\t\t"
        mask.tail = "\n\t\t"
        self.annotations.append(mask)

        coordinates = et.Element('Coordinates')
        coordinates.text = "\n\t\t\t\t"
        coordinates.tail = "\n\t\t"
        mask.append(coordinates)

        for i, pt in enumerate(points):
            coordinate = et.Element('Coordinate')
            coordinate.set("Order", str(i))
            coordinate.set("X", str(pt[0]))
            coordinate.set("Y", str(pt[1]))
            coordinate.tail = "\n\t\t\t" + ("\t" if i != len(points) - 1 else "")
            coordinates.append(coordinate)

        self.classCount[classInfo["name"]] += 1
        self.nbAnnotation += 1

    def addAnnotationClass(self, classInfo: {}):
        group = et.Element('Group')
        group.set('Name', classInfo["name"])
        group.set('PartOfGroup', "None")
        group.set("Color", classInfo.get("asap_color", rgb_to_hex(classInfo["color"])))
        group.text = "\n\t\t\t"
        group.tail = "\n\t\t"

        attribute = et.Element('Attributes')
        attribute.tail = "\n\t\t"
        group.append(attribute)

        self.nbGroup += 1
        self.groups.append(group)

    def __str__(self):
        # Fix indentation of Annotations and AnnotationGroups closing tags
        if self.nbAnnotation == 0:
            self.annotations.text = ""
        else:
            self.annotations[-1].tail = "\n\t"
        if self.nbGroup == 0:
            self.groups.text = ""
        else:
            self.groups[-1].tail = "\n\t"
        return super().__str__()

    @staticmethod
    def getPriorityLevel():
        return 10

    @staticmethod
    def canRead(filePath):
        canRead = XMLAdapter.canRead(filePath)
        if canRead:
            tree = et.parse(filePath)
            root = tree.getroot()
            canRead = root.tag == "ASAP_Annotations"
        return canRead

    @staticmethod
    def readFile(filePath):
        canRead = ASAPAdapter.canRead(filePath)
        if not canRead:
            raise TypeError('This file is not an ASAP annotation file')
        tree = et.parse(filePath)
        root = tree.getroot()
        masks = []
        # Going through the XML tree and getting all Annotation nodes
        for annotation in root.findall('./Annotations/Annotation'):
            maskClass = annotation.attrib.get('PartOfGroup')
            ptsMask = []
            # Going through the Annotation node and getting all Coordinate nodes
            for points in annotation.find('Coordinates'):
                xCoordinate = points.attrib.get('X')
                yCoordinate = points.attrib.get('Y')
                ptsMask.append([xCoordinate, yCoordinate])
            masks.append((maskClass, ptsMask))
        return masks

    @staticmethod
    def offsetAnnotations(filePath, xOffset=0, yOffset=0, outputFilePath=None):
        canRead = ASAPAdapter.canRead(filePath)
        if not canRead:
            raise TypeError('This file is not an ASAP annotation file')
        if xOffset == yOffset == 0:
            if outputFilePath is not None and outputFilePath != filePath:
                shutil.copyfile(filePath, outputFilePath)
            else:
                return None
        else:
            tree = et.parse(filePath)
            root = tree.getroot()
            # Going through the XML tree and updating all Annotation nodes
            for annotation in root.findall('./Annotations/Annotation'):
                # Going through the Annotation node and updating all Coordinate nodes
                for points in annotation.find('Coordinates'):
                    xCoordinate = points.attrib.get('X')
                    yCoordinate = points.attrib.get('Y')
                    points.set('X', str(int(xCoordinate) + xOffset))
                    points.set('Y', str(int(yCoordinate) + yOffset))
            if 'xml_declaration' in tree.write.__code__.co_varnames:
                tree.write(filePath if outputFilePath is None else outputFilePath,
                           encoding='unicode', xml_declaration=True)
            else:
                tree.write(filePath if outputFilePath is None else outputFilePath, encoding='unicode')
        return None

    @staticmethod
    def fuseAnnotationsFiles(filePaths: list, savePath: str):
        """
        Fuses multiple
        :param filePaths:
        :param savePath:
        :return:
        """
        canRead = all([ASAPAdapter.canRead(filePath) for filePath in filePaths])
        if not canRead:
            raise TypeError('At least one given file path is not an ASAP annotation file')
        if len(filePaths) < 2:
            raise ValueError('You must provide at least two file paths')
        tree = et.parse(filePaths[0])
        root = tree.getroot()
        groups = root.findall('./AnnotationGroups/Group')
        annotations = root.findall('./Annotations/Annotation')
        classes = [aClass.attrib.get('Name') for aClass in groups]

        # Merging AnnotationsGroups and annotations
        for filePath in filePaths[1:]:
            tempTree = et.parse(filePath)
            tempRoot = tempTree.getroot()
            for aClass in tempRoot.findall('./AnnotationGroups/Group'):
                className = aClass.attrib.get('Name')
                if className not in classes:
                    classes.append(className)
                    groups.append(aClass)

            for anAnnotation in tempRoot.findall('./Annotations/Annotation'):
                annotations.append(anAnnotation)

        newTree = et.ElementTree(element=et.Element("ASAP_Annotations"))
        newRoot = newTree.getroot()
        annotationsNode = et.Element('Annotations')
        annotationsNode.text = "\n\t\t"
        annotationsNode.tail = "\n\t"
        for node in annotations:
            annotationsNode.append(node)
        newRoot.append(annotationsNode)
        groupsNode = et.Element('AnnotationGroups')
        groupsNode.text = "\n\t\t"
        groupsNode.tail = "\n"
        for node in groups:
            groupsNode.append(node)
        newRoot.append(groupsNode)
        if 'xml_declaration' in tree.write.__code__.co_varnames:
            newTree.write(savePath, encoding='unicode', xml_declaration=True)
        else:
            newTree.write(savePath, encoding='unicode')
