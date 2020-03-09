from pathlib import Path
import pandas as pd
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
PROCESSED_ALL = Path('docs/example_data/GSE69852')


def test_qc_plot_load_processed_beta():
    df = methylcheck.load(PROCESSED,'beta_value')

def test_qc_plot_load_processed_m():
    df = methylcheck.load(PROCESSED,'m_value')

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

def test_qc_signal_intensity_meth_unmeth():
    meth, unmeth = methylcheck.qc_plot._get_data(path=PROCESSED)
    methylcheck.qc_signal_intensity(
        data_containers=None, path=None, meth=meth, unmeth=unmeth,
        noob=True, silent=True, verbose=False, plot=False, bad_sample_cutoff=10.5)

class TestRunQc(unittest.TestCase):
    @patch("methylcheck.qc_plot.plt.show")
    def test_qc_plot(self, mock):
        methylcheck.run_qc(PROCESSED_ALL)
