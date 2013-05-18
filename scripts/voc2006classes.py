"""
Parser for vox 2006-2007 images
reads ImageSet directory and gets classes of images
than writes it to arff file
"""
import os

SIZE = 5305
maxint = 0
res = {}

for subdir, dirs, files in os.walk('.'):
    for file in files:
	if file[-3:] != "txt":
            continue

        cls,dummy = file.split("_")

        f=open(file, 'r')
        lines=f.readlines()

        if cls not in res:
            res[cls] = list(range(0,SIZE))

        for line in lines:
            id,val = line.strip().replace("  "," ").split(" ")
            res[cls][int(id)] = val
            if maxint < int(id):
                maxint = int(id)

print("@relation classification\n")
print("@attribute image string")
KEYS=list(res.keys())
for key in KEYS:
    print("@attribute %s numeric" % (key))

print("\n@data")

for i in range(1,SIZE):
    line = "'%06d'," % (i,)
    for cls in KEYS:
        line += res[cls][i].__str__() + ","
    line = line[:-1]
    print(line)
