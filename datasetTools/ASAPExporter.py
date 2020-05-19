import xml.etree.ElementTree as et
from datasetTools.AnnotationExporter import XMLExporter


class ASAPExporter(XMLExporter):
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

    def __init__(self, imageInfo: dict):
        super().__init__(imageInfo, "ASAP_Annotations")
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

    def addAnnotation(self, classInfo: {}, points):
        if classInfo["name"] not in self.classCount:
            self.classCount[classInfo["name"]] = 0

        mask = et.Element('Annotation')
        mask.set('Name', "{} {} ({})".format(classInfo["name"], self.classCount[classInfo["name"]], self.nbAnnotation))
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
        group.set("Color", classInfo["color"])
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
