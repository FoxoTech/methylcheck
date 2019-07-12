# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import sys
#patching
from unittest.mock import patch
#app
import methQC

data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder


class TestPostProcessQC(unittest.TestCase):
    df = pd.read_pickle('docs/test_betas.pkl')

    def test_df_read_pickle(self):
        df = pd.read_pickle('docs/test_betas.pkl')
        if df.shape != (485512, 6):
            raise AssertionError()

    @patch("methQC.postprocessQC.plt.show")
    def test_mean_beta_plot(self, mock):
        methQC.postprocessQC.mean_beta_plot(self.df, verbose=False, save=False)

    @patch("methQC.postprocessQC.plt.show")
    def test_beta_density_plot(self, mock):
        methQC.postprocessQC.beta_density_plot(self.df, verbose=False, save=False)

    @patch("methQC.postprocessQC.plt.show")
    def test_mean_beta_compare(self, mock):
        methQC.postprocessQC.mean_beta_compare(self.df, self.df, verbose=False, save=False)

    def test_cumulative_sum_beta_distribution(self):
        df2 = methQC.postprocessQC.cumulative_sum_beta_distribution(self.df, cutoff=0.7, plot=False, verbose=False, save=False)

    def test_beta_mds_plot(self):
        df2 = methQC.postprocessQC.beta_mds_plot(self.df, filter_stdev=2, verbose=False, silent=True, save=False)

    def test_detect_array(self):
        ARRAY = methQC.cli.detect_array(self.df)
        if ARRAY != '450k':
            raise AssertionError()

    def test_list_problem_probes_epic(self):
        probes = methQC.filters.list_problem_probes('EPIC', criteria=None, custom_list=None)
        if len(probes) != 389050:
            raise AssertionError()

    def test_list_problem_probes_450(self):
        probes = methQC.filters.list_problem_probes('450k', criteria=None, custom_list=None)
        if len(probes) != 341057:
            raise AssertionError()

    def test_list_problem_probes_reason(self):
        probes = methQC.filters.list_problem_probes('EPIC', criteria=['BaseColorChange'], custom_list=None)
        if len(probes) != 406:
            raise AssertionError()

    def test_list_problem_probes_pub(self):
        probes = methQC.filters.list_problem_probes('450k', criteria=['Chen2013'], custom_list=None)
        if len(probes) != 265410:
            raise AssertionError()

    def test_exclude_sex_control_probes(self):
        df2 = methQC.filters.exclude_sex_control_probes(self.df, 'EPIC', no_sex=True, no_control=True, verbose=False)
        if len(df2) != 474929:
            raise AssertionError()
