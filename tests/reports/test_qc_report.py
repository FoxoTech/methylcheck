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
PROCESSED_450K = Path('docs/example_data/GSE69852')
PROCESSED_EPIC = Path('docs/example_data/epic')
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


    def test_dummy(self):
        import warnings
        from sklearn.manifold import MDS
        warnings.filterwarnings("ignore", message='invalid value encountered')
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        #df = methylcheck.load(PROCESSED_450K)
        df = pd.DataFrame(data={'9247377093_R02C01':[0.1,.2,.3,.4,.5,.6,np.nan,np.nan,0.8,0.9,1.0],
        'two':[0.1,.2,.3,.4,.5,np.nan,.6,np.nan,0.8,0.9,0.1]})
        mds = MDS(n_jobs=-1, random_state=1, verbose=1)
        mds_transformed = mds.fit_transform(df.dropna().transpose().values)
