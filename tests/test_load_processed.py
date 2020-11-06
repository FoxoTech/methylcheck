# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
import pandas as pd
TESTPATH = 'tests'
#app
import methylcheck


class TestLoadProcessed():
    epic_df = Path(TESTPATH,'test_epic_filter.pkl')
    test_450k = Path('docs/example_data/GSE69852')
    test_alt_450k = Path('docs/example_data/GSE105018') # not used here

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

    def test_load_meth_df_from_pickle_450k(self):
        dfs = methylcheck.load(self.test_450k, 'meth_df')
        (meth,unmeth) = dfs
        if meth.shape != (485512, 1) or unmeth.shape != (485512, 1):
            raise AssertionError("wrong probe count or sample count returned")
        if isinstance(meth,pd.DataFrame) is False or isinstance(unmeth,pd.DataFrame) is False:
            raise AssertionError(f"error in DataFrames returned: ({meth.shape} vs (485512,1) | {unmeth.shape} vs (485512,1))")

    def test_load_noob_df_from_pickle_450k(self):
        dfs = methylcheck.load(self.test_450k, 'noob_df')
        (meth,unmeth) = dfs
        if meth.shape != (485512, 1) or unmeth.shape != (485512, 1):
            raise AssertionError("wrong probe count or sample count returned")
        if isinstance(meth,pd.DataFrame) is False or isinstance(unmeth,pd.DataFrame) is False:
            raise AssertionError(f"error in DataFrames returned: ({meth.shape} vs (485512,1) | {unmeth.shape} vs (485512,1))")

    def test_load_both_path_doesnt_exist(self):
        try:
            methylcheck.load_both(Path(self.test_450k,'blahblah'))
        except FileNotFoundError as e:
            return
        raise ValueError("load_both didn't catch that the path didn't exist")

    def test_load_both(self):
        (df,meta) = methylcheck.load_both(self.test_450k)

    def test_load_containers_and_container_to_pkl(self):
        containers = methylcheck.load(self.test_450k, 'meth')
        df = methylcheck.container_to_pkl(containers, 'betas', save=False)
        df = methylcheck.container_to_pkl(containers, 'm_value', save=False)
        meth,unmeth = methylcheck.container_to_pkl(containers, 'meth', save=False)
        df = methylcheck.container_to_pkl(containers, 'copy_number', save=False)
        # test data lacks a 'noob_meth' column; can't test this yet.
        #meth,unmeth = methylcheck.container_to_pkl(containers, 'noob', save=False)
