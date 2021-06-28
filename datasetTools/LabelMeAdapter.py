"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project
Dataset tools

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import shutil

from datasetTools.AnnotationAdapter import JSONAdapter
import jsonschema as sch
import json

LABELME_SCHEMA = {
    "type": "object",
    "required": [
        "version",
        "imageHeight",
        "imagePath",
        "imageData",
        "imageWidth",
        "shapes",
        "flags"
    ],
    "properties": {
        "version": {"type": "string"},
        "imageHeight": {"type": "integer"},
        "imagePath": {"type": "string"},
        "imageData": {"type": ["null", "string"]},
        "imageWidth": {"type": "integer"},
        "shapes": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "required": [
                            "points",
                            "group_id",
                            "flags",
                            "shape_type",
                            "label"
                        ],
                        "properties": {
                            "points": {
                                "type": "array",
                                "items": {
                                    "anyOf": [
                                        {
                                            "type": "array",
                                            "items": {"anyOf": [{"type": "number"}]}
                                        }
                                    ]
                                }
                            },
                            "group_id": {"type": ["null", "integer"]},
                            "flags": {"type": "object"},
                            "shape_type": {"type": "string"},
                            "label": {"type": "string"}
                        }
                    }
                ]
            }
        },
        "flags": {"type": "object"}
    }
}

LABELME_VERSION = "4.2.10"


class LabelMeAdapter(JSONAdapter):
    """
    Export annotations to LabelMe format
    """
    '''
    {
      "version": "4.2.10",
      "flags": {},
      "shapes": [
        {
          "label": "Annotation {MASK_NUM:d}",
          "points": [
            [
              {PT_X:d},
              {PT_Y:d}
            ]
          ],
          "group_id": {CLASS_ID:d},
          "shape_type": "polygon",
          "flags": {}
        }
      ],
      "imagePath": {IMAGE_PATH:s},
      "imageData": null
      "imageHeight": {IMAGE_HEIGHT:d},
      "imageWidth": {IMAGE_WIDTH:d}
    }
    '''

    def __init__(self, imageInfo: dict, verbose=0):
        super().__init__(imageInfo, verbose=verbose)
        self.data = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": [],
            "imageHeight": self.imageInfo["height"],
            "imageWidth": self.imageInfo["width"],
            "imageData": None,
            "imagePath": f"{self.imageInfo['name']}.{self.imageInfo['format']}"
        }
        self.classCount = {}
        self.nbAnnotation = 0

    @staticmethod
    def getName():
        return "LabelMe"

    def addAnnotation(self, classInfo: {}, points):
        if classInfo["name"] not in self.classCount:
            self.classCount[classInfo["name"]] = 0
        mask = {
            "label": f"{classInfo['name']} {self.classCount[classInfo['name']]} ({self.nbAnnotation})",
            "points": points,
            "group_id": int(classInfo.get('labelme_id', classInfo['id'])),
            "shape_type": "polygon",
            "flags": {}
        }
        self.data["shapes"].append(mask)
        self.classCount[classInfo["name"]] += 1
        self.nbAnnotation += 1

    def addAnnotationClass(self, classInfo: {}):
        pass

    def saveToFile(self, savePath, fileName):
        super().saveToFile(savePath, fileName)

    @staticmethod
    def getPriorityLevel():
        return 9

    @staticmethod
    def canRead(filePath):
        canRead = JSONAdapter.canRead(filePath)
        if canRead:
            with open(filePath, 'r') as file:
                data = json.load(file)
                try:
                    sch.validate(instance=data, schema=LABELME_SCHEMA)
                    canRead = True
                except sch.exceptions.ValidationError as err:
                    canRead = False
        return canRead

    @staticmethod
    def readFile(filePath):
        canRead = LabelMeAdapter.canRead(filePath)
        if not canRead:
            raise TypeError('This file is not a LabelMe annotation file')
        masks = []
        with open(filePath, 'r') as file:
            data = json.load(file)
        if data['version'] != LABELME_VERSION:
            print(f"{filePath} version ({data['version']}) of annotation file is different from the one used "
                  f"to implement LabelMe annotation reader ({LABELME_VERSION}).")
            print("Errors may occur so consider updating LabelMeAdapter::readFile().")
        for i, shape in enumerate(data["shapes"]):
            ptsMask = []
            for coordinates in shape["points"]:
                ptsMask.append([coordinates[0], coordinates[1]])
            masks.append((shape["group_id"], ptsMask))
        return masks

    @staticmethod
    def offsetAnnotations(filePath, xOffset=0, yOffset=0, outputFilePath=None):
        canRead = LabelMeAdapter.canRead(filePath)
        if not canRead:
            raise TypeError('This file is not a LabelMe annotation file')
        if xOffset == yOffset == 0:
            if outputFilePath is not None and outputFilePath != filePath:
                shutil.copyfile(filePath, outputFilePath)
            else:
                return None
        else:
            with open(filePath, 'r') as file:
                data = json.load(file)
            if data['version'] != LABELME_VERSION:
                print(f"{filePath} version ({data['version']}) of annotation file is different from the one used "
                      f"to implement LabelMe annotation reader ({LABELME_VERSION}).")
                print("Errors may occur so consider updating LabelMeAdapter::readFile().")
            for i, shape in enumerate(data["shapes"]):
                for coordinates in shape["points"]:
                    coordinates[0] += xOffset
                    coordinates[1] += yOffset
            with open(filePath if outputFilePath is None else outputFilePath, 'w') as file:
                json.dump(data, file, indent='\t')
        return None
