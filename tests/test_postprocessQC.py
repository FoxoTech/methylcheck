# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import sys
#app
import methQC

data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder
df = pd.read_pickle('docs/test_betas.pkl')

todo = ['beta_density_plot',
    'beta_mds_plot',
    'cumulative_sum_beta_distribution',
    'detect_array',
    'exclude_probes',
    'exclude_sex_control_probes',
    'list_problem_probes',
    'mean_beta_plot',
    'mean_beta_compare']

#patching
from unittest.mock import patch

class TestPostProcessQC(unittest.TestCase):

    @staticmethod
    def test_df_read_pickle():
        df = pd.read_pickle('../docs/test_betas.pkl')
        if df.shape == (485512, 6):
            raise AssertionError()

    @staticmethod
    @patch("mean_beta_plot.plt.show")
    def test_mean_beta_plot():
        methQC.postprocessQC.mean_beta_plot(self.df, verbose=False, save=False)

    @staticmethod
    @patch("beta_density_plot.plt.show")
    def test_beta_density_plot():
        methQC.postprocessQC.beta_density_plot(self.df, verbose=False, save=False)

    @staticmethod
    @patch("mean_beta_compare.plt.show")
    def test_mean_beta_compare():
        methQC.postprocessQC.mean_beta_compare(self.df, self.df, verbose=False, save=False)

    @staticmethod
    def cumulative_sum_beta_distribution():
        df2 = methQC.postprocessQC.cumulative_sum_beta_distribution(self.df, cutoff=0.7, plot=False, verbose=False, save=False)
