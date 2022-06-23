from pathlib import Path
import pandas as pd
import logging
import shutil
from tqdm import tqdm
#patching
import unittest
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch
#app
import methylcheck

TESTPATH = 'tests'
PROCESSED = Path('docs/example_data/GSE69852/9247377085')
PROCESSED_5CSV = Path('docs/example_data/GSE69852/9247377093')
PROCESSED_ALL = Path('docs/example_data/GSE69852')
PROCESSED_BATCH = Path('docs/example_data/GSE105018')


def test_qc_plot_load_processed_beta():
    df = methylcheck.load(PROCESSED,'beta_value')

def test_qc_plot_load_processed_m():
    df = methylcheck.load(PROCESSED,'m_value')

def test_load_processed_meth_df():
    df = methylcheck.load(PROCESSED_BATCH, 'meth_df', silent=True, verbose=True)

def test_load_processed_noob_df():
    df = methylcheck.load(PROCESSED_BATCH, 'noob_df', silent=False, verbose=True)

def test_load_processed_meth_df():
    df = methylcheck.load(PROCESSED_BATCH, 'beta_value', silent=True, verbose=False)

def test_load_processed_noob_df():
    df = methylcheck.load(PROCESSED_BATCH, 'm_value', silent=False, verbose=False)

def test_qc_plot_load_processed_meth():
    containers = methylcheck.load(PROCESSED,'meth')
    if type(containers) != list and type(containers[0]._SampleDataContainer__data_frame) != type(pd.DataFrame()):
        raise AssertionError("processed data is not a dictionary of SampleDataContainers")

def test_qc_signal_intensity_containers():
    containers = methylcheck.load(PROCESSED, 'meth')
    methylcheck.qc_signal_intensity(
        data_containers=containers, path=None, meth=None, unmeth=None,
        noob=True, silent=True, verbose=False, plot=False, bad_sample_cutoff=10.5)

def test_qc_signal_intensity_path():
    methylcheck.qc_signal_intensity(
        data_containers=None, path=PROCESSED, meth=None, unmeth=None,
        noob=True, silent=True, verbose=False, plot=False, bad_sample_cutoff=10.5)

class TestQcPlots(unittest.TestCase):
    meth, unmeth = methylcheck.qc_plot._get_data(path=PROCESSED)
    poobah = methylcheck.load(Path(PROCESSED_ALL,'poobah_values.pkl'))

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_meth_unmeth(self, mock):
        methylcheck.qc_signal_intensity(
            data_containers=None, path=None, meth=self.meth, unmeth=self.unmeth,
            noob=True, silent=True, verbose=False, plot=True, bad_sample_cutoff=10.5)

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_meth_unmeth_poobah(self, mock):
        """ failed because meth and unmeth from csv lost sentrix_id in processing, but poobah retained it """
        methylcheck.qc_signal_intensity(
            data_containers=None, path=None, meth=self.meth, unmeth=self.unmeth, poobah=self.poobah,
            noob=True, silent=True, verbose=False, plot=False, bad_sample_cutoff=10.5)

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_meth_unmeth_poobah_palette(self, mock):
        methylcheck.qc_signal_intensity(
            data_containers=None, path=None, meth=self.meth, unmeth=self.unmeth, poobah=self.poobah, palette="Blues",
            noob=True, silent=True, verbose=False, plot=True, bad_sample_cutoff=10.5)

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_path_poobah(self, mock):
        methylcheck.qc_signal_intensity(
            data_containers=None, path=PROCESSED_ALL, meth=None, unmeth=None, poobah=True,
            noob=True, silent=False, verbose=False, plot=True, cutoff_line=True, bad_sample_cutoff=10.5)

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_plots_path_parquet(self, mock):
        PROCESSED_PARQUET = Path('docs/example_data/GSE69852_parquet')
        shutil.copytree(PROCESSED_ALL, PROCESSED_PARQUET,  dirs_exist_ok=True)
        print(f"{PROCESSED_ALL} ... copied to {PROCESSED_PARQUET}")
        for _file in tqdm(PROCESSED_PARQUET.rglob('*'), desc="Converting pickle to parquet"):
            if '.pkl' in _file.suffixes:
                df = pd.read_pickle(_file)
                new_file = Path(f"{_file.stem}.parquet")
                if isinstance(df, pd.DataFrame):
                    df.to_parquet(Path(PROCESSED_PARQUET,new_file))
                elif (isinstance(df, dict) and all([isinstance(sub_df, pd.DataFrame) for dict_key,sub_df in df.items()])):
                    control = pd.concat(df) # creates multiindex; might also apply to mouse_probes.pkl --> parquet
                    (control.reset_index()
                        .rename(columns={'level_0': 'Sentrix_ID', 'level_1': 'IlmnID'})
                        .astype({'IlmnID':str})
                        .to_parquet(Path(PROCESSED_PARQUET,new_file))
                    )
                _file.unlink()

        methylcheck.qc_signal_intensity(
            data_containers=None, path=PROCESSED_PARQUET, meth=None, unmeth=None, poobah=True,
            noob=True, silent=False, verbose=False, plot=True, cutoff_line=True, bad_sample_cutoff=10.5)
        methylcheck.plot_M_vs_U(PROCESSED_PARQUET, silent=True, poobah=True)
        methylcheck.plot_M_vs_U(PROCESSED_PARQUET, silent=True, poobah=False)
        methylcheck.plot_M_vs_U(PROCESSED_PARQUET, silent=True, poobah=True, noob=True)
        methylcheck.plot_M_vs_U(PROCESSED_PARQUET, silent=True, poobah=True, compare=True)
        methylcheck.run_qc(PROCESSED_PARQUET)
        shutil.rmtree(PROCESSED_PARQUET, ignore_errors=True)

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_from_path(self, mock):
        test = methylcheck.qc_signal_intensity(path=Path(TESTPATH), silent=True, verbose=False, plot=False)
        if not (test['cutoffs'][0] == 12.889238234452801 and test['good_samples'] == ['0'] and test['bad_samples'] == [] and test['bad_sample_cutoff'] == 11.5):
            raise AssertionError(f"qc_signal_intensity results don't match tests for sample in {TESTPATH}")

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_from_path(self, mock):
        methylcheck.qc_signal_intensity(
            data_containers=None, path=PROCESSED_ALL,
            silent=True, verbose=False, plot=True)

    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_signal_intensity_from_path_no_poobah(self, mock):
        methylcheck.qc_signal_intensity(path=PROCESSED_ALL, silent=True, poobah=False, palette="twilight")

    @patch("methylcheck.qc_plot.plt.show")
    def test_plot_M_vs_U_from_path_poobah(self, mock):
        methylcheck.plot_M_vs_U(PROCESSED_ALL, silent=True, poobah=True)

    @patch("methylcheck.qc_plot.plt.show")
    def test_plot_M_vs_U_from_path_no_poobah(self, mock):
        methylcheck.plot_M_vs_U(PROCESSED_ALL, silent=True, poobah=False)
        methylcheck.plot_M_vs_U(PROCESSED_ALL, silent=True, poobah=None)

    @patch("methylcheck.qc_plot.plt.show")
    def test_plot_M_vs_U_from_path_poobah_noob(self, mock):
        """ loads noob_meth_values.pkl instead """
        methylcheck.plot_M_vs_U(PROCESSED_ALL, silent=True, poobah=True, noob=True)

    @patch("methylcheck.qc_plot.plt.show")
    def test_plot_M_vs_U_from_path_compare(self, mock):
        import methylprep
        if not (Path(PROCESSED_ALL, f'noob_meth_values.pkl').exists() and Path(PROCESSED_ALL, f'meth_values.pkl').exists()):
            methylprep.run_pipeline(PROCESSED_ALL, save_uncorrected=True, poobah=True, export_poobah=True)
        methylcheck.plot_M_vs_U(PROCESSED_ALL, silent=True, poobah=True, compare=True)

    @patch("methylcheck.qc_plot.plt.show")
    def test_plot_M_vs_U_from_csv(self, mock):
        methylcheck.plot_M_vs_U(PROCESSED_5CSV, silent=True, poobah=False)

    @patch("methylcheck.qc_plot.plt.show")
    def test_plot_M_vs_U_from_csv_poobah(self, mock):
        # this should fail, but with a useful error message. can't find the poobah file
        methylcheck.plot_M_vs_U(PROCESSED_5CSV, silent=True, poobah=True)


class TestRunQc(unittest.TestCase):
    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_plot(self, mock):
        methylcheck.run_qc(PROCESSED_ALL)
