"""
Merger for http://datam.i2r.a-star.edu.sg/datasets/krbd/OvarianCancer/OvarianCancer-NCI-QStar.html
"""

print "@relation 'ovarian'\n"
files = 10

f = range(0,files)
fl = range(0,files)
cur_line = range(0,files)

for i in range(0,files):
    f[i] = open("all%d.arff" % (i))
    fl[i] = f[i].readlines()
    j = 0
    for line in fl[i]:
        j += 1
        if line[:5] == "@attr" and not line[:16] == "@attribute Class":
            print line.strip().replace("@attribute ","@attribute A").replace(".","C")
        if line[:5] == "@data":
            cur_line[i] = j
            break
print "@attribute Class {Cancer,Normal}"
stop = False
for j in range(0,100000):
    newline = ""
    for i in range(0,files):
        lines = fl[i]
        idx = cur_line[i]
        try:
            nl = lines[idx+j]
        except IndexError:
            stop = True
            break
        nl = nl.strip()
        if i < files-1:
            nl = nl.replace(",Normal","")
            nl = nl.replace(",Cancer","")
        newline += nl
        newline += ","

    if stop:
        break
    
    if len(newline) > 200:
        print newline[:-1]
    else:
        print ""

for i in range(0,files):
    f[i].close()

