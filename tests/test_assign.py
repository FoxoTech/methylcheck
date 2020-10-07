# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import sys
import seaborn as sns
#patching
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch
#app
import methylcheck

@patch("methylcheck.samples.postprocessQC.plt.show")
@patch('methylcheck.samples.assign_groups.get_input', return_value=1)
def test_assign(mock1, mock2):
    # assigns group 1 to everything, just to be sure the function runs
    data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder
    data = data.set_index('CGidentifier')
    #monkeypatch.setattr('builtins.input', lambda _: 1)
    methylcheck.sample_plot(data)
    user_defined_groups = methylcheck.assign(data)
    methylcheck.plot_assigned_groups(data, user_defined_groups)
