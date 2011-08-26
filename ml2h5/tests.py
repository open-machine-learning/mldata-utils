import random
import unittest

from converter.h5_csv import H5_CSV
from converter.h5_arff import H5_ARFF
from converter.h5_libsvm import H5_LibSVM
from converter.h5_mat import H5_MAT
from converter.h5_octave import H5_OCTAVE
from converter.h5_rdata import H5_RData
from converter.h5_uci import H5_UCI

class TestConversion(unittest.TestCase):

    fixtures = {
        'csv': '../fixtures/test.csv',
        'arff': '../fixtures/iris.arff',
        'uci': '../fixtures/iris.data',
        'octave': '../fixtures/iris.data',
        'R': '../fixtures/iris.Rdata',
    }
    

    def setUp(self):
        pass

    def test_csv2h5(self):
        converter = H5_CSV(self.fixtures['csv'])
        data = converter.read()
        self.assertEqual(data['data']['int3'][3], 4,
                         'wrong first integer')

    def test_arff2h5(self):
        converter = H5_ARFF(self.fixtures['arff'])
        data = converter.read()
        self.assertEqual(data['data']['sepallength'][3], 4.6,
                         'wrong first integer')

    def test_uci(self):
        converter = H5_UCI(self.fixtures['uci'])
        data = converter.read()
        self.assertEqual(data['data']['sepallength'][3], 4.6,
                         'wrong first integer')

    def test_rdata(self):
        converter = H5_RData(self.fixtures['R'])
#        data = converter.read()
#        print data
#        self.assertEqual(data['data']['sepallength'][3], 4.6,
#                         'wrong first integer')

if __name__ == '__main__':
    unittest.main()