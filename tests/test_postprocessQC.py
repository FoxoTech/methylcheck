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

data = pd.read_csv('tests/test_data.csv') # pytest runs tests as if it is in the package root folder


class TestPostProcessQC(unittest.TestCase):
    df = pd.read_pickle('docs/test_betas.pkl')

    def test_df_read_pickle(self):
        df = pd.read_pickle('docs/test_betas.pkl')
        if df.shape != (485512, 6):
            raise AssertionError()

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_mean_beta_plot(self, mock):
        methylcheck.mean_beta_plot(self.df, verbose=False, save=False)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_mean_beta_plot_transposed(self, mock):
        methylcheck.mean_beta_plot(self.df.transpose(), verbose=False, save=False)

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_mean_beta_compare(self, mock):
        methylcheck.mean_beta_compare(self.df, self.df, verbose=False, save=False)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_mean_beta_compare_transposed_1(self, mock):
        methylcheck.mean_beta_compare(self.df.transpose(), self.df, verbose=False, save=False)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_mean_beta_compare_transposed_2(self, mock):
        methylcheck.mean_beta_compare(self.df, self.df.transpose(), verbose=False, save=False)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_mean_beta_compare_transposed_both(self, mock):
        methylcheck.mean_beta_compare(self.df.transpose(), self.df.transpose(), verbose=False, save=False)

    def test_cumulative_sum_beta_distribution(self):
        df2 = methylcheck.cumulative_sum_beta_distribution(self.df, cutoff=0.7, verbose=False, save=False, silent=True)
    def test_cumulative_sum_beta_distribution_transposed(self):
        df2 = methylcheck.cumulative_sum_beta_distribution(self.df.transpose(), cutoff=0.7, verbose=False, save=False, silent=True)

    def test_beta_mds_plot(self):
        df2 = methylcheck.beta_mds_plot(self.df, filter_stdev=2, verbose=False, save=False, silent=True)
    def test_beta_mds_plot_transposed(self):
        df2 = methylcheck.beta_mds_plot(self.df.transpose(), filter_stdev=2, verbose=False, save=False, silent=True)

    def test_combine_mds(self):
        df2 = methylcheck.combine_mds(self.df, self.df,
            save=False, silent=True, verbose=False)

    def test_drop_nan_probes(self):
        df2 = methylcheck.probes.filters.drop_nan_probes(self.df, silent=True, verbose=False)
    def test_drop_nan_probes_transposed(self):
        df2 = methylcheck.probes.filters.drop_nan_probes(self.df.transpose(), silent=True, verbose=False)

    def test_detect_array(self):
        ARRAY = methylcheck.cli.detect_array(self.df)
        if ARRAY != '450k':
            raise AssertionError()

    def test_exclude_sex_control_probes(self):
        df2 = methylcheck.exclude_sex_control_probes(self.df, 'EPIC', no_sex=True, no_control=True, verbose=False)
        if len(df2) != 474929:
            raise AssertionError()

    def test_cli_all_plots_silent(self):
        testargs = ["__program__", '-d', 'docs/test_betas.pkl', '--exclude_all', '--silent', '--verbose']
        with patch.object(sys, 'argv', testargs):
            results = methylcheck.cli.cli_parser()

    def test_exclude_probes(self):
        probe_list = methylcheck.list_problem_probes('450k', criteria=None, custom_list=None)
        df2 = methylcheck.exclude_probes(self.df, probe_list)
    def test_exclude_probes_transpose(self):
        probe_list = methylcheck.list_problem_probes('450k', criteria=None, custom_list=None)
        df2 = methylcheck.exclude_probes(self.df.transpose(), probe_list)

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_beta_density_plot(self, mock):
        methylcheck.beta_density_plot(self.df, verbose=False, save=False)

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_beta_density_plot_transposed(self, mock):
        methylcheck.beta_density_plot(self.df.transpose(), verbose=False, save=False)

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_beta_density_plot_kwargs(self, mock):
        tips = sns.load_dataset("tips") # ~250 rows, two numeric columns
        methylcheck.beta_density_plot(tips[['total_bill','tip']])
        methylcheck.beta_density_plot(tips[['total_bill','tip']], show_labels=True, highlight_samples='total_bill')
        methylcheck.beta_density_plot(tips[['total_bill','tip']], show_labels=False, highlight_samples='tip')
        methylcheck.beta_density_plot(tips[['total_bill','tip']], silent=False, verbose=True)
        methylcheck.beta_density_plot(tips[['total_bill','tip']], silent=True, verbose=False)
        methylcheck.beta_density_plot(tips[['total_bill','tip']], reduce=1.0)
        methylcheck.beta_density_plot(tips[['total_bill','tip']], reduce=None)
        methylcheck.beta_density_plot(tips[['total_bill','tip']], reduce=0.5)
        methylcheck.beta_density_plot(tips[['total_bill','tip']], plot_title='testing tips low ymax', ymax=0.05)
        fig = methylcheck.beta_density_plot(tips[['total_bill','tip']], return_fig=True, full_range=True, show_labels=None, filename='test.png')
        import matplotlib
        if isinstance(fig, matplotlib.figure.Figure) == False:
            raise AssertionError("return_fig=True: did not return a figure")
        # NOT TESTED: save=True.
