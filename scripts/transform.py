"""
Module join train and test into one file and creates a split information file.

Used for http://datam.i2r.a-star.edu.sg/datasets/krbd/

Should be run in parent directory containg directories with files:
[NAME]-train.arff
[NAME]-test.arff

Outputs:
[NAME].arff
[NAME]_split.arff
"""

import sys
import os
import getopt
import sys, glob, operator

def merge(filename):
    train_file = filename + "-train.arff"
    test_file = filename + "-test.arff"

    full_file = filename + ".arff"
    split_file = filename + "_split.txt"

    full = open(full_file, 'w')
    train = open(train_file, 'r')
    test = open(test_file, 'r')
    split = open(split_file, 'w')

    w = False
    counter = 0
    line = 1
    while (line):
        line = train.readline()
        if not line:
            break
        if len(line)==0 and w:
            continue
        full.write(line)

        if line[:5] == "@data":
            w = True
            counter = -1
        counter = counter + 1

    old = counter - 1
    split.write("0:%d\n" % (old))

    w = False
    counter = 0

    line = 1
    while (line):
        line = test.readline()
        if not line:
            break
        if len(line)==0 and w:
            continue
        if w == True:
            full.write(line)

        if line[:5] == "@data":
            counter = -1
            w = True
        counter = counter + 1
    
    split.write("%d:%d\n" % (old,old+counter-1))

    full.close()
    train.close()
    test.close()
    split.close()

"""
    skip directories polya and TIS
"""
def process(args):
    dirname = '.'
    dirs = [dirname + "/" + f for f in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, f))]
    for dirname in dirs:
        if dirname == './polya' or dirname == './TIS':
            continue
        arffs = [f for f in os.listdir(dirname) if f[-10:] == "-test.arff"]
        for filename in arffs:
            if filename == 'MLL-test.arff':
                continue
            merge(dirname + "/" + filename[:-10])
	print dirname, " ", arffs    
    return

def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
    # process arguments
    process(args) # process() is defined elsewhere

if __name__ == "__main__":
    main()

