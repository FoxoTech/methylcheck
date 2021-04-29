from pathlib import Path
import pandas as pd
import numpy as np
import logging
import random

#patching
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
