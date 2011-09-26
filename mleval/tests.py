"""Test module.

Coverts correctness and performance
"""
import random
import os
import unittest
import sys
import getopt
import datetime
import ml2h5
from mleval import other
import numpy

class TestsCorrectness(unittest.TestCase):
    def test_vco_eval(self):
        r1 = [1,2,4,3]
        self.assertEqual(other.rectangle_area(r1), 3, 'wrong rectangle area measurement')
        r2 = [2,1,7,4]
        self.assertEqual(other.intersection_area(r1,r2), 2, 'wrong intersect area measurement')

        out = numpy.array([["abc",'dog',10,20,50,50],["bbc",'dog',10,20,50,50],["bbc",'cat',0,0,10,10]])
        lab = numpy.array([["abc",'dog',10,20,50,50],["abd",'cat',10,20,50,50],["bbc",'dog',0,0,10,10],["abc",'cat',10,20,30,50]])
        res = other.voc_detection(out,lab)
        self.assertLess(abs(res-0.2), 0.01, 'wrong intersect area measurement')
        
        out = numpy.array([[ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASpersonStanding', '270', '101', '314', '243'],])
        lab = numpy.array([[ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASbicycle', '106', '157', '191', '297'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASbicycle', '220', '173', '296', '250'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASbicycle', '305', '203', '422', '311'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASperson', '61', '127', '116', '258'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASperson', '104', '113', '196', '262'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASpersonStanding', '191', '94', '246', '252'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASpersonStanding', '270', '101', '314', '243'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASpersonStanding', '310', '91', '359', '270'],
 [ 'VOC2005_2/PNGImages/bicycle/branchburgnewspapertripodcomjan20editionjan20picsbicycle.png',
  'PASpersonStanding', '337', '83', '416', '322'],
 [ 'VOC2005_2/PNGImages/bicycle/homepage2niftycomhosonumabicyclepantanitdfIMG12741.png',
  'PASbicycleSide', '9', '12', '381', '235']])
        res = other.voc_detection(out,lab)
        self.assertLess(abs(res-0.1), 0.01, 'wrong intersect area measurement')

def main():
    unittest.main()

if __name__ == '__main__':
    main()