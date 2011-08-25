import random
import unittest

from converter.h5_csv import H5_CSV

class TestConversion(unittest.TestCase):

    fixtures = {
        'csv': '../fixtures/test.csv',
    }
    

    def setUp(self):
        pass

    def test_csv2h5(self):
        converter = H5_CSV(self.fixtures['csv'])
        data = converter.read()
        self.assertEqual(data['data']['int3'][3], 4,
                         'wrong first integer')

if __name__ == '__main__':
    unittest.main()