"""
Simple parser of xml files provided in PASCAL VOC 2006
http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2006/

With small modifications (concerning image file name
expression) it works also for 2005.

Outputs arff table

Author: Lukasz Kidzinski 2011
"""

import re, os

reg_rect = "\"(?P<name>.*)\" \(.*\) : \((?P<xmin>[0-9]*), (?P<ymin>[0-9]*)\) - \((?P<xmax>[0-9]*), (?P<ymax>[0-9]*)\)"

reg_filename = "(?P<filename>[0-9]*)\.png\""

def parse_labels(filename):
    f = open(filename)
    imgfile = ""

    for line in f.readlines():
        m = re.search(reg_filename, line)
        if m:
            imgfile = m.group('filename')
        m = re.search(reg_rect, line)
        if m:
            name = m.group('name')
            pose = "Undefined"
            difficult = 0
            xmin = m.group('xmin')
            ymin = m.group('ymin')
            xmax = m.group('xmax')
            ymax = m.group('ymax')
            print "'%s','%s','%s',%s,%s,%s,%s,%s" % (imgfile,name,pose,difficult,xmin,ymin,xmax,ymax)

def print_attr_names(parse_labels):
    if parse_labels:
        print "@relation labels\n"
        attr = ["id","name","pose","difficult","xmin","ymin","xmax","ymax"]
        attr_types = ["string","string","string","numeric","numeric","numeric","numeric","numeric"]
    else:
        print "@relation images\n"
        attr = ["id","database","annotation","image","imageid","ownername","ownerid","width","height","depth"]
        attr_types = ["string","string","string","string","string","string","string","numeric","numeric","numeric"]

    for key in range(0,len(attr)):
        print "@attribute %s %s" % (attr[key], attr_types[key])
    print "\n@data"

print_attr_names(1)

for filename in os.listdir("."):
    if filename.split(".")[1] == "txt":
        parse_labels(filename)

