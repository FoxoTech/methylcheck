# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
TESTPATH = 'tests'
#app
import methylcheck


class TestLoadProcessed():
    epic_df = Path(TESTPATH,'test_epic_filter.pkl')

    def test_load_beta_pickle(self):
        df = methylcheck.load(self.epic_df)
        if df.shape != (865859, 1):
            raise AssertionError()

    def test_load_beta_pickle_gzip(self):
        df = methylcheck.load(Path(TESTPATH,'test_epic_filter.pkl.gz'))
        if df.shape != (865859, 1):
            raise AssertionError(df.shape)

    def test_load_beta_csv(self): #includes one .csv and one .csv.gz
        df = methylcheck.load(Path(TESTPATH,'204247480034')) #,'204247480034_R08C01_processed.csv'))
        if df.shape != (230, 2):
            raise AssertionError(df.shape)

    def test_filter_epic(self):
        df = methylcheck.load(Path(TESTPATH,'test_epic_filter.pkl'))
        df_test = methylcheck.exclude_probes(df, ['Zhou2016','McCartney2016'] )
        if df_test.shape != (477231, 1):
            raise AssertionError()

    def test_filter_epic_illumina_sketchy(self):
        df = methylcheck.load(Path(TESTPATH,'test_epic_filter.pkl'))
        df_test = methylcheck.exclude_probes(df, ['illumina'] )
        if df_test.shape != (864869, 1):
            raise AssertionError()

    def test_filter_epic_illumina_sketchy_string(self):
        df = methylcheck.load(Path(TESTPATH,'test_epic_filter.pkl'))
        df_test = methylcheck.exclude_probes(df, 'illumina')
        if df_test.shape != (864869, 1):
            raise AssertionError()

    def test_filter_epic_with_450k_criteria(self):
        df = methylcheck.load(Path(TESTPATH,'test_epic_filter.pkl'))
        try:
            df_test = methylcheck.exclude_probes(df, ['Zhou2016','McCartney2016', 'Chen2013'] )
        except ValueError:
            return # passed
        raise AssertionError("Failed to detect a 450k criteria applied to ECPI data.")

    def test_load_missing_format(self, bad_format='gobbledeegook'):
        with pytest.raises(ValueError):
            df = methylcheck.load(self.epic_df, format=bad_format)

    def test_load_sesame_csv(self):
        # make some random csv files
        import pandas as pd
        import numpy as np
        WORKDIR = Path('tests/sesame/')
        if not Path(WORKDIR).exists():
            Path(WORKDIR).mkdir()

        filenames = ['sesame_testfile_R01C01_123456_calls.csv',
                     'sesame_testfile_R01C02_123456_calls.csv',
                     'sesame_testfile_R02C03_123456_calls.csv',
                    ]
        for filename in filenames:
            test = pd.DataFrame(data={'Probe_ID':list(range(10000)),
                'ind_beta':np.random.uniform(0, 1, 10000),
                'ind_poob':np.random.uniform(0.000000001, 0.06, 10000)})
            test['Probe_ID'] = 'cg000' + test['Probe_ID'].astype(str)
            test.set_index('Probe_ID', inplace=True)
            test = test.round(4)
            test.to_csv(Path(WORKDIR, filename))

        df = methylcheck.load(WORKDIR, format='sesame')
        #print(df.head())
        #print(f"{df.isna().sum(axis=0).mean()} failed probes per sample")
        for filename in filenames:
            Path(WORKDIR, filename).unlink()
        if Path(WORKDIR).exists():
            Path(WORKDIR).rmdir()
