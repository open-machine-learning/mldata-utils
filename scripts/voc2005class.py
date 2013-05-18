"""
Parser for voc 2005 images
reads ImageSet directory and gets classes of images
than writes it to arff file
"""
import re, os

reg_rect = "\"(?P<name>.*)\" \(.*\) : \((?P<xmin>[0-9]*), (?P<ymin>[0-9]*)\) - \((?P<xmax>[0-9]*), (?P<ymax>[0-9]*)\)"

reg_filename = "filename : \"(?P<filename>.*)\""
res = {}
FILES = list(range(0,10000))
T = {}

MAP = {
    'PASposterClutter': 'dummy',
    'PASbicycle': 'bicycle',
    'PASskyRegion': 'dummy',
    'PASbuildingPart': 'dummy',
    'PASbicycleSide': 'bicycle',
    'PAStreePart': 'dummy',
    'PASpersonSitting': 'person',
    'PASstreet': 'dummy',
    'PASmotorbike': 'motorbike',
    'PASwindow': 'dummy',
    'PAScarPart': 'car',
    'PAStreeWhole': 'dummy',
    'PAStrafficlight': 'dummy',
    'PASbuildingWhole': 'dummy',
    'PAStrash': 'dummy',
    'PAStreeRegion': 'dummy',
    'PASstreetlight': 'dummy',
    'PAScarRear': 'car',
    'PASstreetSign': 'dummy',
    'PASposter': 'dummy',
    'PASdoor': 'dummy',
    'PASmotorbikeSide': 'motorbike',
    'PASstopSign': 'dummy',
    'PASbuilding': 'dummy',
    'PAScarSide': 'car',
    'PAScarFrontal': 'car',
    'PAScar': 'car',
    'PAStree': 'dummy',
    'PASpersonStanding': 'dummy',
    'PASpersonWalking': 'dummy',
    'PASperson': 'person',
    'PASbuildingRegion': 'dummy'
}
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
            if imgfile not in res:
                res[imgfile] = {}
#            if not res[cls].has_key(imgfile):
            res[imgfile][MAP[name]] = 1
            T[MAP[name]] = 1

for dirname in os.listdir("."):
    if (not os.path.isdir(dirname)) or (len(dirname) < 3):
        continue
    for filename in os.listdir(dirname):
        if filename.split(".")[1] == "txt":
            parse_labels(dirname + "/" + filename)

print("@relation classification\n")
print("@attribute image string")
CLSS = [key for key in list(T.keys()) if not key == "dummy"]
for key in CLSS:
    print("@attribute %s numeric" % (key))

print("\n@data")

for key in list(res.keys()):
    line = "'" + key + "',"
    for cls in CLSS:
        if cls in res[key]:
            line += res[key][cls].__str__() + ","
        else:
            line += "-1,"
    line = line[:-1]
    print(line)
