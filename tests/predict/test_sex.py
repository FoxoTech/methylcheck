from pathlib import Path
import pandas as pd
import numpy as np
import logging
import random

#patching
import pytest
import unittest
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch

#app
import methylcheck

PROCESSED_450K = Path('docs/example_data/GSE69852') # partial data
MOUSE_TEST = Path('docs/example_data/mouse_test') # partial data


@patch("methylcheck.qc_plot.plt.show")
def test_get_sex_plot_label_compare_with_actual_450k(mock):
    meta = pd.read_pickle(Path(PROCESSED_450K,'sample_sheet_meta_data.pkl'))
    orig_meta = meta.copy() # will restore this after tests complete
    meta['Sex'] = 'M'
    meta.to_pickle(Path(PROCESSED_450K,'sample_sheet_meta_data.pkl'))
    custom_label={i:random.randrange(-10,10) for i in meta['Sample_ID']}
    df = methylcheck.get_sex(PROCESSED_450K, plot=True, custom_label=custom_label)
    print(df)
    meta['Sex'] = meta['Sex'].apply(lambda x: random.choice(['Male','Female']))
    meta.to_pickle(Path(PROCESSED_450K,'sample_sheet_meta_data.pkl'))
    df = methylcheck.get_sex(PROCESSED_450K, plot=True, custom_label=custom_label, median_cutoff=-3, poobah_cutoff=5)
    print(df)
    meta = meta.drop('Sex', axis=1)
    meta['Gender'] = 'F'
    meta['Gender'] = meta['Gender'].apply(lambda x: random.choice(['MALE','FEMALE']))
    meta.to_pickle(Path(PROCESSED_450K,'sample_sheet_meta_data.pkl'))
    df = methylcheck.get_sex(PROCESSED_450K, plot=True, custom_label=custom_label, median_cutoff=-3, poobah_cutoff=5)
    print(df)
    # restore original samplesheet pickle
    orig_meta.to_pickle(Path(PROCESSED_450K,'sample_sheet_meta_data.pkl'))

@patch("methylcheck.qc_plot.plt.show")
def test_get_sex_plot_label_compare_with_actual_mouse(mock):
    """ DOES NOT WORK WITH MOUSE ARRAY YET -- X/Y mapping is off? get nans back """
    meta = pd.read_pickle(Path(MOUSE_TEST,'sample_sheet_meta_data.pkl'))
    # meta has Gender 'M' column already
    custom_label={i:random.randrange(-10,10) for i in meta['Sample_ID']}
    df = methylcheck.get_sex(MOUSE_TEST, plot=True, custom_label=custom_label)
    print(df)

def test_get_actual_sex():
    LOCAL = 'docs/example_data/GSE147391'
    files = ['sample_sheet_meta_data.pkl', 'GSE147391_GPL21145_meta_data.pkl', 'GSE147391_GPL21145_samplesheet.csv']
    for _file in files:
        local_file = Path(LOCAL,_file)
        if '.csv' in local_file.suffixes:
            df = pd.read_csv(local_file)
        if '.pkl' in local_file.suffixes:
            df = pd.read_pickle(local_file)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.set_index('Sample_ID')
        df2 = methylcheck.predict.sex._fetch_actual_sex_from_sample_sheet_meta_data(LOCAL,df)
        print(df2[['gender','actual_sex','sex_matches']], df2.columns)
        assert set(list(df2.columns)) & set(['gender','actual_sex','sex_matches']) == set(['gender','actual_sex','sex_matches'])
        df3 = df.reset_index()
        with pytest.raises(KeyError) as excinfo:
            df4 = methylcheck.predict.sex._fetch_actual_sex_from_sample_sheet_meta_data(LOCAL,df3)
            assert excinfo.value.message == "Could not read actual sex from meta data to compare."

@patch("methylcheck.qc_plot.plt.show")
def test_get_sex_plot_return_fig_labels(mock):
    """ verifies all the options and what they return; cannot check if show() worked. """
    import seaborn
    from io import StringIO
    fig = methylcheck.get_sex(MOUSE_TEST, return_fig=False, plot=True)
    if not isinstance(fig, type(None)):
        raise AssertionError("return_fig was false, but returned something, instead of None")
    fig = methylcheck.get_sex(MOUSE_TEST, return_fig=True, plot=True)
    if not isinstance(fig, seaborn.axisgrid.FacetGrid):
        raise AssertionError("return_fig was True, but did not return figure")
    fig = methylcheck.get_sex(MOUSE_TEST, return_fig=True, plot=False)
    if not isinstance(fig, seaborn.axisgrid.FacetGrid):
        raise AssertionError("return_fig was True, but did not return figure")
    labels = methylcheck.get_sex(MOUSE_TEST, return_fig=False, plot=False, return_labels=True)
    ref_labels = {'204879580038_R01C02': 'A', '204879580038_R02C02': 'B', '204879580038_R03C02': 'C', '204879580038_R04C02': 'D', '204879580038_R05C02': 'E', '204879580038_R06C02': 'F'}
    if labels != ref_labels:
        raise AssertionError("return_labels did not match expected output")
    default = methylcheck.get_sex(MOUSE_TEST)
    ref_df = pd.read_csv(StringIO("""sample\tx_median\ty_median\tpredicted_sex\tX_fail_percent\tY_fail_percent\tactual_sex\tsex_matches
204879580038_R01C02\t12.0\t7.4\t             F\t             2.9\t            28.0\t          M\t           0
204879580038_R02C02\t12.5\t7.8\t             F\t             2.5\t            27.9\t          M\t           0
204879580038_R03C02\t12.1\t7.5\t             F\t             5.4\t            28.2\t          M\t           0
204879580038_R04C02\t12.0\t7.4\t             F\t             5.6\t            28.2\t          M\t           0
204879580038_R05C02\t12.1\t7.6\t             F\t             5.6\t            28.2\t          M\t           0
204879580038_R06C02\t11.9\t7.3\t             F\t             7.1\t            28.3\t          M\t           0"""), sep='\t').set_index('sample')
    if not default[['x_median','y_median']].equals(ref_df[['x_median','y_median']]):
        raise AssertionError("default output (dataframe of predicted sexes) did not match reference data")
