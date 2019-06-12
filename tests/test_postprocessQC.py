# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import sys
from methQC import postprocessQC as postQC

data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder

class TestPostProcessQC(unittest.TestCase):

    def test_importCoefHannum(self):
        self.assertEqual(postQC._importCoefHannum().shape, (71, 2))
        self.assertEqual(set(postQC._importCoefHannum().columns),
                         set(['Marker', 'Coefficient']))

    def test_DNAmAgeHannumFunction(self):
        dat2 = postQC.DNAmAgeHannumFunction(data)
        self.assertIn([c for c in data.columns if c !=
                       'CGidentifier'], dat2.index.values)
        self.assertEqual(set(['DNAmAgeHannum']), set(dat2.columns.values))
