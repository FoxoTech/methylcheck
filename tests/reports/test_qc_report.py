from pathlib import Path
import pandas as pd
import numpy as np
import logging

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
PROCESSED_450K = Path('docs/example_data/GSE69852') # only 1 sample
PROCESSED_EPIC = Path('docs/example_data/epic') # lacks full data
PROCESSED_MOUSE = Path('docs/example_data/mouse_test')

class TestQCReport():

    def test_qc_run_pipeline(self):
        df = methylcheck.load(PROCESSED_450K)
        methylcheck.run_pipeline(df, exclude_all=True, plot=['all'], silent=True)

    def test_ReportPDF(self):
        import warnings
        warnings.filterwarnings("ignore", message='invalid value encountered')
        myreport = methylcheck.ReportPDF(path=PROCESSED_450K, outpath=PROCESSED_450K)
        myreport.run_qc()
        myreport.pdf.close()
        if Path(PROCESSED_450K,'multipage_pdf.pdf').exists():
            Path(PROCESSED_450K,'multipage_pdf.pdf').unlink()
        else:
            raise FileNotFoundError(Path(PROCESSED_450K,'multipage_pdf.pdf'))
