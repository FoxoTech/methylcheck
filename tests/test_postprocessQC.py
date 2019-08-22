# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import sys
#patching
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch
#app
import methylcheck

data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder


class TestPostProcessQC(unittest.TestCase):
    df = pd.read_pickle('docs/test_betas.pkl')

    def test_df_read_pickle(self):
        df = pd.read_pickle('docs/test_betas.pkl')
        if df.shape != (485512, 6):
            raise AssertionError()

    @patch("methylcheck.postprocessQC.plt.show")
    def test_mean_beta_plot(self, mock):
        methylcheck.postprocessQC.mean_beta_plot(self.df, verbose=False, save=False)

    @patch("methylcheck.postprocessQC.plt.show")
    def test_beta_density_plot(self, mock):
        methylcheck.postprocessQC.beta_density_plot(self.df, verbose=False, save=False)

    @patch("methylcheck.postprocessQC.plt.show")
    def test_mean_beta_compare(self, mock):
        methylcheck.postprocessQC.mean_beta_compare(self.df, self.df, verbose=False, save=False)

    def test_cumulative_sum_beta_distribution(self):
        df2 = methylcheck.postprocessQC.cumulative_sum_beta_distribution(self.df, cutoff=0.7, verbose=False, save=False, silent=True)

    def test_beta_mds_plot(self):
        df2 = methylcheck.postprocessQC.beta_mds_plot(self.df, filter_stdev=2, verbose=False, save=False, silent=True)

    def test_detect_array(self):
        ARRAY = methylcheck.cli.detect_array(self.df)
        if ARRAY != '450k':
            raise AssertionError()

    def test_list_problem_probes_epic(self):
        probes = methylcheck.filters.list_problem_probes('EPIC', criteria=None, custom_list=None)
        if len(probes) != 389050:
            raise AssertionError()

    def test_list_problem_probes_450(self):
        probes = methylcheck.filters.list_problem_probes('450k', criteria=None, custom_list=None)
        if len(probes) != 341057:
            raise AssertionError()

    def test_list_problem_probes_reason(self):
        probes = methylcheck.filters.list_problem_probes('EPIC', criteria=['BaseColorChange'], custom_list=None)
        if len(probes) != 406:
            raise AssertionError()

    def test_list_problem_probes_pub(self):
        probes = methylcheck.filters.list_problem_probes('450k', criteria=['Chen2013'], custom_list=None)
        if len(probes) != 265410:
            raise AssertionError()

    def test_exclude_sex_control_probes(self):
        df2 = methylcheck.filters.exclude_sex_control_probes(self.df, 'EPIC', no_sex=True, no_control=True, verbose=False)
        if len(df2) != 474929:
            raise AssertionError()

    def test_cli_all_plots_silent(self):
        testargs = ["__program__", '-d', 'docs/test_betas.pkl', '--exclude_all', '--silent', '--verbose']
        with patch.object(sys, 'argv', testargs):
            results = methylcheck.cli.cli_parser()
