# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import sys
import seaborn as sns
from pathlib import Path
import shutil
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

    def test_extended_function_inputs(self):
        list_of_dfs = methylcheck.samples.postprocessQC._load_data(['docs/test_betas.pkl'], progress_bar=True)
        if list_of_dfs[0].shape != (485512, 6):
            raise AssertionError("._load_data returned wrong size test pkl")


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
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_cumulative_sum_beta_distribution(self, mock):
        df2 = methylcheck.cumulative_sum_beta_distribution(self.df, cutoff=0.7, verbose=False, save=False, silent=True)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_cumulative_sum_beta_distribution_transposed(self, mock):
        df2 = methylcheck.cumulative_sum_beta_distribution(self.df.transpose(), cutoff=0.7, verbose=False, save=False, silent=True)

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_beta_mds_plot(self, mock):
        print("*** MDS 1 ***")
        #df2 = methylcheck.beta_mds_plot(self.df, filter_stdev=2, save=False, verbose=True)
        df2 = methylcheck.beta_mds_plot(self.df, filter_stdev=2, save=False, verbose=False, silent=True)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_beta_mds_plot_transposed(self, mock):
        print("*** MDS 2 ***")
        df2 = methylcheck.beta_mds_plot(self.df.transpose(), filter_stdev=2, save=False, verbose=False, silent=True)
    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_combine_mds(self, mock):
        print("*** MDS 3 ***")
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

    @patch("methylcheck.samples.postprocessQC.plt.show")
    def test_cli_all_plots_silent(self, mock):
        # NOTE: this still shows MDS plots even though silent ON, because verbose ON too?
        testargs = ["__program__", 'qc', '-d', 'docs/test_betas.pkl', '--exclude_all', '--silent', '--verbose']
        with patch.object(sys, 'argv', testargs):
            results = methylcheck.cli.cli_app()

    def test_cli_controls_report(self):
        """ this test does NOT cover poobah and sex prediction, because they require 3 more big files """
        report_folder = 'docs/example_data/controls_test'
        testargs = ["__program__", 'controls', '-d', report_folder, '--pval_off']
        with patch.object(sys, 'argv', testargs):
            results = methylcheck.cli.cli_app()
        outfile = Path(report_folder, 'controls_test_QC_Report.xlsx')
        if not outfile.exists():
            raise FileNotFoundError('controls_test_QC_Report.xlsx not found')
        Path(outfile).unlink()

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
