# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
import pandas as pd
import shutil
import logging
from tqdm import tqdm
TESTPATH = 'tests'
#app
import methylcheck


class TestLoadProcessed():

    epic_df = Path(TESTPATH,'test_epic_filter.pkl')
    epic_df2 = Path(TESTPATH,'test_epic_filter.pkl.gz')
    test_450k = Path('docs/example_data/GSE69852')
    test_alt_450k = Path('docs/example_data/GSE105018')
    test_epic = Path('docs/example_data/epic')
    mouse_test = Path('docs/example_data/mouse_test')

    pytest.fixture()
    def test_load_noob_tuple_pickles(self, caplog):
        workdir = Path(self.mouse_test,'temp')
        if workdir.exists():
            shutil.rmtree(workdir)
        (meth,unmeth) = methylcheck.load(self.mouse_test, format='noob_df', verbose=False, silent=True)

        # (1) scramble, save, and confirm you get errors with duplicate samples
        workdir.mkdir(parents=True, exist_ok=True)
        bad_meth = pd.concat([meth, meth], axis='columns')
        bad_meth.to_pickle(Path(workdir,'noob_meth_values.pkl'))
        unmeth.to_pickle(Path(workdir,'noob_unmeth_values.pkl'))
        (bad_meth,bad_unmeth) = methylcheck.load(workdir, format='noob_df', verbose=False, silent=True)
        errors = [record for record in caplog.get_records('call') if record.levelno >= logging.ERROR]
        relevant_msg = [record.msg for record in errors if 'duplicate sample names' in record.msg]
        if len(relevant_msg) == 0:
            shutil.rmtree(workdir)
            raise AssertionError(f"Did not detect ERROR message when loading noob_meth with duplicate sample names.")
        else:
            print(f"OK: mouse_test (meth, unmeth) tuple with duplicate samples yields ERROR message: {relevant_msg[-1]}")
        shutil.rmtree(workdir)

        # (2) duplicate probes
        workdir.mkdir(parents=True, exist_ok=True)
        bad_meth = pd.concat([meth, meth], axis='index')
        bad_meth.to_pickle(Path(workdir,'noob_meth_values.pkl'))
        unmeth.to_pickle(Path(workdir,'noob_unmeth_values.pkl'))
        (bad_meth,bad_unmeth) = methylcheck.load(workdir, format='noob_df', verbose=False, silent=True)
        errors = [record for record in caplog.get_records('call') if record.levelno >= logging.ERROR]
        relevant_msg = [record.msg for record in errors if 'duplicate probe names' in record.msg]
        if len(relevant_msg) == 0:
            shutil.rmtree(workdir)
            raise AssertionError(f"Did not detect ERROR message when loading noob_meth with duplicate probe names.")
        else:
            print(f"OK: mouse_test (meth, unmeth) tuple with duplicate probes yields ERROR message: {relevant_msg[-1]}")
        shutil.rmtree(workdir)


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
        if meth.shape != (485512, 6) or unmeth.shape != (485512, 6):
            raise AssertionError("wrong probe count or sample count returned")
        if isinstance(meth,pd.DataFrame) is False or isinstance(unmeth,pd.DataFrame) is False:
            raise AssertionError(f"error in DataFrames returned: ({meth.shape} vs (485512,1) | {unmeth.shape} vs (485512,1))")

    def test_load_noob_df_from_pickle_450k(self):
        dfs = methylcheck.load(self.test_450k, 'noob_df')
        (meth,unmeth) = dfs
        if meth.shape != (485512, 6) or unmeth.shape != (485512, 6):
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

    def test_load_both_returns_probes_in_rows(self):
        (df,meta) = methylcheck.load_both(self.test_alt_450k)
        if df.shape[0] < df.shape[1]:
            raise AssertionError("load_both returned probes in columns")

    def test_load_containers_and_container_to_pkl(self):
        containers = methylcheck.load(self.test_450k, 'meth')
        df = methylcheck.container_to_pkl(containers, 'betas', save=False)
        df = methylcheck.container_to_pkl(containers, 'm_value', save=False)
        meth,unmeth = methylcheck.container_to_pkl(containers, 'meth', save=False)
        df = methylcheck.container_to_pkl(containers, 'copy_number', save=False)
        # test data lacks a 'noob_meth' column; can't test this yet.
        #meth,

    def test_load_batch_parquet(self):
        # first, make a temp parquet copy of docs/exampe_data/epic folder:
        meta_shape = (6, 12)
        test_dir = Path('docs/example_data/GSE69852_parquet')
        if test_dir.exists():
            shutil.rmtree(test_dir)
        shutil.copytree(self.test_450k, test_dir)
        files = []
        print(f"... copied to {test_dir}")
        for _file in tqdm(test_dir.rglob('*'), desc="Converting pickle to parquet"):
            if '.pkl' in _file.suffixes:
                df = pd.read_pickle(_file)
                new_file = Path(f"{_file.stem}.parquet")
                if isinstance(df, pd.DataFrame):
                    df.to_parquet(Path(test_dir,new_file))
                elif (isinstance(df, dict) and all([isinstance(sub_df, pd.DataFrame) for dict_key,sub_df in df.items()])):
                    control = pd.concat(df) # creates multiindex; might also apply to mouse_probes.pkl --> parquet
                    (control.reset_index()
                        .rename(columns={'level_0': 'Sentrix_ID', 'level_1': 'IlmnID'})
                        .astype({'IlmnID':str})
                        .to_parquet(Path(test_dir,new_file))
                    )
                files.append(new_file)
                _file.unlink()
        for test_format,dshape in {'beta_value': (485512, 6), 'm_value':(485512, 6), 'meth': list, 'beta_csv': (485577, 6), 'meth_df': tuple, 'noob_df': tuple}.items(): # 'sesame'
            try:
                data, meta = methylcheck.load_both(test_dir, format=test_format, verbose=False, silent=True)
                result = data.shape if isinstance(data, pd.DataFrame) else type(data)
                if meta.shape != meta_shape:
                    raise AssertionError("meta shape does not match expected")
                if result != dshape and result != type(dshape):
                    print('DEBUG', result, dshape, result, type(dshape))
                    raise AssertionError("data shape or data type does not match expected")
                if isinstance(data, list):
                    print('OK', [type(item) for item in data])
                else:
                    print('OK', test_format, result, meta.shape)
            except Exception as e:
                import traceback;print(traceback.format_exc())
                raise Exception(e)
        shutil.rmtree(test_dir)

    def test_load_batch_of_pickles(self):
        # first, make a temp copy of docs/exampe_data/epic folder:
        meta_shape = (6, 12)
        test_dir = Path('docs/example_data/GSE69852_pickle_copy')
        if test_dir.exists():
            shutil.rmtree(test_dir)
        shutil.copytree(self.test_450k, test_dir)
        print(f"... copied to {test_dir}")
        for test_format,dshape in {'beta_value': (485512, 6), 'm_value':(485512, 6), 'meth': list, 'beta_csv': (485577, 6), 'meth_df': tuple, 'noob_df': tuple}.items(): # 'sesame'
            try:
                data, meta = methylcheck.load_both(test_dir, format=test_format, verbose=False, silent=True)
                result = data.shape if isinstance(data, pd.DataFrame) else type(data)
                if meta.shape != meta_shape:
                    raise AssertionError("meta shape does not match expected")
                if result != dshape and result != type(dshape):
                    print('DEBUG', result, dshape, result, type(dshape))
                    raise AssertionError("data shape or data type does not match expected")
                if isinstance(data, list):
                    print('OK', [type(item) for item in data])
                else:
                    print('OK', test_format, result, meta.shape)
            except Exception as e:
                import traceback;print(traceback.format_exc())
                raise Exception(e)
        shutil.rmtree(test_dir, ignore_errors=True)
