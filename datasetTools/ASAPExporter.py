import xml.etree.ElementTree as et
from datasetTools.XMLExporter import XMLExporter


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

    def __init__(self):
        super().__init__("ASAP_Annotations")
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

    def addAnnotation(self, className, points):
        mask = et.Element('Annotation')
        self.annotations.append(mask)
        mask.set('Name', "Annotation {}".format(self.nbAnnotation))
        self.nbAnnotation += 1
        mask.set("Type", "Polygon")
        mask.set("PartOfGroup", className)
        mask.set("Color", "#F4FA58")
        mask.text = "\n\t\t\t"
        mask.tail = "\n\t\t"
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

    def addAnnotationClass(self, className, color):
        self.nbGroup += 1
        group = et.Element('Group')
        group.set('Name', className)
        group.set('PartOfGroup', "None")
        group.set("Color", color)
        group.text = "\n\t\t\t"
        group.tail = "\n\t\t"
        attribute = et.Element('Attributes')
        attribute.tail = "\n\t\t"
        group.append(attribute)
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
