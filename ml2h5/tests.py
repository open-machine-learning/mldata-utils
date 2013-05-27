"""Test module.

Coverts correctness and performance
"""
import random
import os
import unittest
import sys
import getopt
import datetime
import numpy

import ml2h5
from mleval import other
from ml2h5 import converter
from ml2h5 import fileformat
from ml2h5.converter.h5_csv import H5_CSV
from ml2h5.converter.h5_arff import H5_ARFF
from ml2h5.converter.h5_libsvm import H5_LibSVM
from ml2h5.converter.h5_mat import H5_MAT
from ml2h5.converter.h5_octave import H5_OCTAVE
from ml2h5.converter.h5_rdata import H5_RData
from ml2h5.converter.h5_uci import H5_UCI
from ml2h5.converter.basehandler import BaseHandler


__doc__

FIXTURES = {
    'csv': '../fixtures/test.csv',
    'arff': '../fixtures/iris.arff',
    'uci': '../fixtures/iris.data',
    'R': '../fixtures/iris.Rdata',
    'libsvm': '../fixtures/breast-cancer.txt',
    'h5': '../examples/uci-20070111-zoo.h5',
    'octave': '../fixtures/test.octave',
    'mat': '../fixtures/test.mat',

    'big-arff': '../fixtures/breastCancer.arff',
    'big-csv': '../fixtures/breastCancer.csv',
}

RESULTS = {
    'h5': '../fixtures/res.h5',
    'generic': '../fixtures/res.',           
}

class TestConversion(unittest.TestCase):
    fixtures = FIXTURES
    result = RESULTS
    
    def assertCanRead(self, filename, Con):
        conv = Con(filename)
        data = conv.read()

    def setUp(self):
        pass

    # checking fileformat from string is used in Converter
    def test_reading_fileformats(self):
        ff = fileformat.get('non-existsing-file.h5')
        self.assertEqual('h5', ff,
                         'wrong format found')
    
    def test_read_csv(self):
        conv = H5_CSV(self.fixtures['csv'])
        data = conv.read()
        self.assertEqual(data['data']['int3'][3], 4,
                         'wrong first integer')

    def test_read_arff(self):
        conv = H5_ARFF(self.fixtures['arff'])
        data = conv.read()
        self.assertEqual(data['data']['sepallength'][3], 4.6,
                         'wrong first integer')

    def test_read_uci(self):
        conv = H5_UCI(self.fixtures['uci'])
        data = conv.read()
        self.assertEqual(data['data']['sepallength'][3], 4.6,
                         'wrong first integer')

    def test_read_rdata(self):
        pass
        # Rdata read not supported yet

    def test_read_octave(self):
        conv = H5_OCTAVE(self.fixtures['octave'])
        data = conv.read()
        self.assertEqual(data['data']['int2'][0,0], 172076,
                         'wrong first integer')

    def test_read_matlab(self):
        conv = H5_MAT(self.fixtures['mat'])
        data = conv.read()
        self.assertEqual(data['data']['int2'][0,0], 172076,
                         'wrong first integer')

    def test_read_libsvm(self):
        conv = H5_LibSVM(self.fixtures['libsvm'])
        data = conv.read()
        self.assertEqual(data['data']['data'][3][2], 1,
                         'wrong first integer')
        
    # CONVERSIONS IN
    def conversion_in(self, file_in, file_out):
        conv = converter.Converter(file_in, file_out)
        conv.run(verify=True)
        self.assertCanRead(file_out, BaseHandler)
        
    def test_h5in(self):
        self.conversion_in(self.fixtures['csv'],self.result['h5'])
        
    def test_arff2h5(self):
        self.conversion_in(self.fixtures['arff'],self.result['h5'])
        
    @unittest.skipUnless(os.path.exists(FIXTURES['big-arff']),"No big arff in fixtures")
    def test_bigarff2h5(self):
#        self.conversion_in(self.fixtures['big-arff'],self.result['h5'])
        self.conversion_in(self.fixtures['big-csv'],self.result['h5'])
        
    def test_libsvm2h5(self):
        self.conversion_in(self.fixtures['libsvm'],self.result['h5'])
        
    def test_octave2h5(self):
        self.conversion_in(self.fixtures['octave'],self.result['h5'])

    def test_matlab2h5(self):
        self.conversion_in(self.fixtures['mat'],self.result['h5'])
        
    def test_rdata2h5(self):
        # Rdata reading not supported
        pass

    def test_uci2h5(self):
        self.conversion_in(self.fixtures['uci'],self.result['h5'])

    # CONVERSIONS OUT
    def conversion_out(self, type, Conv):
        filename = self.result['generic'] + type
        conv = converter.Converter(self.fixtures['h5'], filename)
        conv.run(verify=True)
        self.assertCanRead(filename, Conv)
        
    def test_h5out(self):
        self.conversion_out("csv",H5_CSV)
        self.conversion_out("arff",H5_ARFF)
        #self.conversion_out("libsvm",H5_LibSVM)
        #self.conversion_out("Rdata",H5_RData)
        #self.conversion_out("data",H5_UCI)
        self.conversion_out("octave",H5_OCTAVE)
        self.conversion_out("mat",H5_MAT)

class PerformanceTests:
    def generate_arff(self, fname, attributes=20000, instances=1):
        """Generates the test arff file of given size
    
        @param fname: filename of new file
        @type fname: string
        @param attributes: num of attributes
        @type attributes: integer
        @param instances: num of instances
        @type instances: integer
        """
        f = open(fname, 'w')
        f.write("@relation 'rel'\n\n")
        for i in range(0,attributes):
            f.write("@attribute a%d numeric\n" % i)

        f.write("\n@data\n")
        for j in range(0,instances):
            for i in range(0,attributes):
                if i > 0:
                    f.write(",")
                if i%100 == 0:
                    f.write("%d" % i)
                else:
                    f.write("%f" % float(i))
            f.write("\n")
        f.close()
        
    def setUp(self):
        """Set up the testing enviroment
        """
        if not os.path.exists(FIXTURES['big-arff']):
            self.generate_arff(FIXTURES['big-arff'])
            
    def __init__(self):
        """Initialize by setting up the testing enviroment
        """
        self.setUp()
        
    def start_test(self, name=""):
        """Name the test and record current time.

        @param name: test name
        @type name: string
        @return: test name display string
        @rtype: string
        """
        self._start = datetime.datetime.now()
        return name
        
    def stop_test(self):
        """Gets the running time since the last start_test

        @return: time delta since start
        @rtype: datatime.delta
        """
        return (datetime.datetime.now() - self._start)

    def test_many_attributes_conversion(self):
        """Measure time of arff -> h5 conversion
        """
        print(self.start_test("Arff conversion"))

        conv = converter.Converter(FIXTURES['big-arff'], RESULTS['h5'])
        conv.run(verify=True)

        print(self.stop_test())
        
    def test_many_attributes_types(self):
        """Measure time of getting the attributes
        """
        print(self.start_test("Many attributes - get types"))

        ml2h5.data.get_attribute_types(RESULTS['h5'])

        print(self.stop_test())
        
    def main(self):
        """Run tests
        TODO: Should be more generic similar to unittest module
        """
        self.test_many_attributes_conversion()
        self.test_many_attributes_types()
        
__usage__ = """Usage:
  python tests.py             - runs correctness tests
  python tests.py performance - runs performance tests
"""
def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error as msg:
        print(msg)
        print("for help use --help")
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print(__usage__)
            sys.exit(0)
    
    if len(sys.argv) <= 1:
        unittest.main()
    elif args[0]=="performance":
        per = PerformanceTests()
        per.main()
    else:
        print(__usage__)

if __name__ == '__main__':
    main()
