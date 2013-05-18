"""
Simple parser of xml files provided in PASCAL VOC 2007
http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/

Outputs arff table (with information about images or
image labels) 

Author: Lukasz Kidzinski 2011
"""

from xml.dom import minidom
import os

def parse_images(filename):
    imgid = filename.split(".")[0]

    DOMTree = minidom.parse(filename)
    obj = DOMTree.childNodes[0]

    database = obj.getElementsByTagName("database")[0].childNodes[0].toxml()
    annotation = obj.getElementsByTagName("annotation")[0].childNodes[0].toxml()
    image = obj.getElementsByTagName("image")[0].childNodes[0].toxml()
    imageid = obj.getElementsByTagName("flickrid")[0].childNodes[0].toxml()
    ownername = obj.getElementsByTagName("name")[0].childNodes[0].toxml()
    ownerid = obj.getElementsByTagName("flickrid")[1].childNodes[0].toxml()
    width = obj.getElementsByTagName("width")[0].childNodes[0].toxml()
    height = obj.getElementsByTagName("height")[0].childNodes[0].toxml()
    depth = obj.getElementsByTagName("depth")[0].childNodes[0].toxml()

    print("'%s','%s','%s','%s','%s','%s','%s',%s,%s,%s" % (imgid,database,annotation,image,imageid,ownername,ownerid,width,height,depth))

def parse_labels(filename):
    imgid = filename.split(".")[0]

    DOMTree = minidom.parse(filename)
    nodes = DOMTree.childNodes
    objects = nodes[0].getElementsByTagName("object")
    for obj in objects:
        name = obj.getElementsByTagName("name")[0].childNodes[0].toxml()
        pose = obj.getElementsByTagName("pose")[0].childNodes[0].toxml()
        difficult = obj.getElementsByTagName("difficult")[0].childNodes[0].toxml()
        bndbox = obj.getElementsByTagName("bndbox")[0]

        xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].toxml()
        ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].toxml()
        xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].toxml()
        ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].toxml()

        print("'%s','%s','%s',%s,%s,%s,%s,%s" % (imgid,name,pose,difficult,xmin,ymin,xmax,ymax))

PARSE_LABELS = True

if PARSE_LABELS:
    print("@relation labels\n")
    attr = ["id","name","pose","difficult","xmin","ymin","xmax","ymax"]
    attr_types = ["string","string","string","numeric","numeric","numeric","numeric","numeric"]
else:
    print("@relation images\n")
    attr = ["id","database","annotation","image","imageid","ownername","ownerid","width","height","depth"]
    attr_types = ["string","string","string","string","string","string","string","numeric","numeric","numeric"]

for key in range(0,len(attr)):
    print("@attribute %s %s" % (attr[key], attr_types[key]))
    
print("\n@data")

for filename in os.listdir("."):
    if filename.split(".")[1] == "xml":
        if PARSE_LABELS:
            parse_labels(filename)
        else:
            parse_images(filename)
