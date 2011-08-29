import random
import os
import unittest

from converter.h5_csv import H5_CSV
from converter.h5_arff import H5_ARFF
from converter.h5_libsvm import H5_LibSVM
from converter.h5_mat import H5_MAT
from converter.h5_octave import H5_OCTAVE
from converter.h5_rdata import H5_RData
from converter.h5_uci import H5_UCI
from converter.basehandler import BaseHandler

import converter
import fileformat

FIXTURES = {
    'csv': '../fixtures/test.csv',
    'arff': '../fixtures/iris.arff',
    'uci': '../fixtures/iris.data',
    'R': '../fixtures/iris.Rdata',
    'libsvm': '../fixtures/breast-cancer.txt',
    'h5': '../examples/uci-20070111-zoo.h5',
    'octave': '../fixtures/test.octave',
    'mat': '../fixtures/test.mat',
#        'matlab': '../examples/PDXprecip.dat',

    'big-arff': '../fixtures/breastCancer.arff',
}

class TestConversion(unittest.TestCase):
    fixtures = FIXTURES

    result = {
        'h5': '../fixtures/res.h5',
        'generic': '../fixtures/res.',
    }
    
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
        self.conversion_in(self.fixtures['big-arff'],self.result['h5'])
        
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
        

if __name__ == '__main__':
    unittest.main()