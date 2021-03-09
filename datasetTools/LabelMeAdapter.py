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
                            "group_id": {"type": "integer"},
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

    def addAnnotation(self, classInfo: {}, points):
        if classInfo["name"] not in self.classCount:
            self.classCount[classInfo["name"]] = 0
        mask = {
            "label": "{} {} ({})".format(classInfo["name"], self.classCount[classInfo["name"]], self.nbAnnotation),
            "points": points,
            "group_id": int(classInfo["id"]),
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
        assert canRead
        masks = []
        with open(filePath, 'r') as file:
            data = json.load(file)
            if data['version'] != LABELME_VERSION:
                print("{} version ({}) of annotation file is different from the".format(filePath, data["version"]),
                      "one used to implement LabelMe annotation reader ({}).".format(LABELME_VERSION))
                print("Errors may occur so consider updating LabelMeAdapter::readFile().")
            for i, shape in enumerate(data["shapes"]):
                ptsMask = []
                for coordinates in shape["points"]:
                    ptsMask.append([coordinates[0], coordinates[1]])
                masks.append((shape["group_id"], ptsMask))
        return masks
