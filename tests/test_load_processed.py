# -*- coding: utf-8 -*-
from pathlib import Path
TESTPATH = 'tests'
#app
import methylcheck


class TestLoadProcessed():

    def test_load_beta_pickle(self):
        df = methylcheck.load(Path(TESTPATH,'test_epic_filter.pkl'))
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
