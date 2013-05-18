#!/usr/bin/env python
"""Convert from one supported format to another.

Example usage:
python validate.py mydata.h5
python validate.py otherdata.arff
python validate.py problemdata.h5 output.csv
"""

import sys
import ml2h5.fileformat
import ml2h5.converter
from ml2h5.converter.h5_csv import H5_CSV
from ml2h5.converter.h5_arff import H5_ARFF
from ml2h5.converter.h5_libsvm import H5_LibSVM
from ml2h5.converter.h5_mat import H5_MAT
from ml2h5.converter.h5_octave import H5_OCTAVE
from ml2h5.converter.h5_rdata import H5_RData
from ml2h5.converter.h5_uci import H5_UCI
from ml2h5.converter.basehandler import BaseHandler

converter_factory = {'h5': BaseHandler,
                     'csv': H5_CSV,
                     'arff': H5_ARFF,
                     'libsvm': H5_LibSVM,
                     'Rdata': H5_RData,
                     'data': H5_UCI,
                     'octave': H5_OCTAVE,
                     'matlab': H5_MAT,
                     }

def usage():
    print("""Usage: """ + sys.argv[0] + """ filename [filename_out]""")

def convert(file_in, file_out):
    """Convert mldata data file from file_in to file_out.
    Only format conversion and check that output file can be read.
    """
    print('Converting ' + file_in + ' to ' + file_out)
    format_out = ml2h5.fileformat.get(file_out)
    conv = ml2h5.converter.Converter(file_in, file_out)
    conv.run(verify=True)
    check = converter_factory[format_out](file_out)
    data = check.read()

def validate(filename):
    """Detect file format, then convert"""
    format_in = ml2h5.fileformat.get(filename)
    if format_in == 'h5':
        for format_out in ml2h5.converter.FROM_H5:
            try:
                convert(filename, 'validated.' + format_out)
            except:
                print('Conversion failed')
    elif format_in in ml2h5.converter.TO_H5:
        try:
            convert(filename, 'validated.h5')
        except:
            print('Conversion failed')

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc != 2 and argc != 3:
        usage()
        sys.exit(1)
    if argc == 2:
        validate(sys.argv[1])
    else:
        convert(sys.argv[1], sys.argv[2])
    
