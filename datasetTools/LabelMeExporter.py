from datasetTools.AnnotationExporter import JSONExporter


class LabelMeExporter(JSONExporter):
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
            "version": "4.2.10",
            "flags": {},
            "shapes": [],
            "imageHeight": self.imageInfo["height"],
            "imageWidth": self.imageInfo["width"],
            "imageData": None,
            "imagePath": "{}.png".format(self.imageInfo["name"])
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
