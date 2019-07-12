# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import sys
#app
from ..methQC import *

data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder

class TestPostProcessQC(unittest.TestCase):

    pass
