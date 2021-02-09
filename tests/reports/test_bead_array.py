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


class TestBeadArrayControlsReporter(): #unittest.TestCase):
    #r450 = methylcheck.reports.BeadArrayControlsReporter(PROCESSED_450K)
    #r450.process()
    #r450.save()
    epic = methylcheck.reports.BeadArrayControlsReporter(PROCESSED_EPIC)
    epic.process()
    epic.save()
    mouse = methylcheck.reports.BeadArrayControlsReporter(PROCESSED_MOUSE)
    mouse.process()
    mouse.save()

    def test_mouse(self):
        expected_outfile = 'mouse_test_QC_Report.xlsx'
        if not Path(PROCESSED_MOUSE, expected_outfile).exists():
            raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_MOUSE}")
        results = pd.read_excel(Path(PROCESSED_MOUSE, expected_outfile))
        if results.shape != (7,32):
            raise AssertionError("Result file shape differs.")
        if not all(results['Result'].values[1:4] == ['MARGINAL (0.7)', 'MARGINAL (0.64)', 'FAIL (0.53)']):
            raise AssertionError("Values in result column differ.")
        if Path(PROCESSED_MOUSE,expected_outfile).exists():
            Path(PROCESSED_MOUSE,expected_outfile).unlink()

    def test_epic(self):
        expected_outfile = 'epic_QC_Report.xlsx'
        if not Path(PROCESSED_EPIC, expected_outfile).exists():
            raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_EPIC}")
        results = pd.read_excel(Path(PROCESSED_EPIC, expected_outfile))
        if results.shape != (2,30):
            raise AssertionError(f"Result file shape differs: {results.shape} vs (2,30)")
        if not list(results.iloc[1].values) == ['202908430131_R07C01', 0.29, 70.18, 45.5, 41.57, 15.44, 1.78, 1.88, 8.07, 7.22, 12.42, 4.67, 7.07, 2.49, 6.13, 2.83, 7.67, 5.25, 19.46, 6.07, 9.18, 15.88, 495, 1700, 404, 354, 0.89, 0.87, 99.5, 'OK (0.98)']:
            raise AssertionError(f"Values in result column differ: {list(results.iloc[1].values)}")
        if Path(PROCESSED_EPIC,expected_outfile).exists():
            Path(PROCESSED_EPIC,expected_outfile).unlink()

        # next, hide the poobah and run without it
        Path(PROCESSED_EPIC,'poobah_values.pkl').rename(Path(PROCESSED_EPIC,'_poobah_values.pkl'))
        try:
            epic = methylcheck.reports.BeadArrayControlsReporter(PROCESSED_EPIC, pval=False)
            epic.process()
            epic.save()
            results = pd.read_excel(Path(PROCESSED_EPIC, expected_outfile))
            if results.shape != (2,29):
                raise AssertionError(f"Result file shape differs: {results.shape} vs (2,29)")
            if not list(results.iloc[1].values) == ['202908430131_R07C01', 0.29, 70.18, 45.5, 41.57, 15.44, 1.78, 1.88, 8.07, 7.22, 12.42, 4.67, 7.07, 2.49, 6.13, 2.83, 7.67, 5.25, 19.46, 6.07, 9.18, 15.88, 495, 1700, 404, 354, 0.89, 0.87, 'OK (0.98)']:
                raise AssertionError(f"Values in result column differ: {list(results.iloc[1].values)}")
            if Path(PROCESSED_EPIC,expected_outfile).exists():
                Path(PROCESSED_EPIC,expected_outfile).unlink()
        except:
            Path(PROCESSED_EPIC,'_poobah_values.pkl').rename(Path(PROCESSED_EPIC,'poobah_values.pkl'))
        # UNhide the poobah
        Path(PROCESSED_EPIC,'_poobah_values.pkl').rename(Path(PROCESSED_EPIC,'poobah_values.pkl'))


    def test_r450(self):
        """ 450k is tested multiple ways, so best to rerun here"""
        r450 = methylcheck.reports.BeadArrayControlsReporter(PROCESSED_450K)
        r450.process()
        r450.save()
        expected_outfile = 'GSE69852_QC_Report.xlsx'
        if not Path(PROCESSED_450K, expected_outfile).exists():
            raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_450K}")
        results = pd.read_excel(Path(PROCESSED_450K, expected_outfile))
        if results.shape != (2,31):
            raise AssertionError(f"Result file shape differs: {results.shape} vs (2,31)")
        if not all(results['Result'].values[1:1] == ['OK (0.96)']):
            raise AssertionError(f"Values in result column differ: {results['Result'].values[1:1]}")


def test_controls_report_minimal():
    expected_outfile = 'GSE69852_QC_Report.xlsx'
    if Path(PROCESSED_450K,expected_outfile).exists():
        Path(PROCESSED_450K,expected_outfile).unlink()
    methylcheck.controls_report(filepath=PROCESSED_450K)
    if not Path(PROCESSED_450K,expected_outfile).exists():
        raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_450K}")

def test_controls_report_kwargs_legacy():
    expected_outfile = 'GSE69852_QC_Report.xlsx'
    if Path(PROCESSED_450K,expected_outfile).exists():
        Path(PROCESSED_450K,expected_outfile).unlink()
    methylcheck.controls_report(filepath=PROCESSED_450K, legacy=True)
    if not Path(PROCESSED_450K,expected_outfile).exists():
        raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_450K} --legacy")
    results = pd.read_excel(Path(PROCESSED_450K, expected_outfile))
    if results.shape != (1,24):
        raise AssertionError(f"Result file shape differs: {results.shape} vs (1,24)")
    if not list(results.iloc[0].values)[3:] == [0.1,62.8,99.5,51.8,10.9,1.7,1.9,8.4,5.9,20,5.4,7.8,5.9,5.5,3,13,5.9,13.2,7.4,10.5,14.9]:
        raise AssertionError(f"--legacy: Calculated Numbers don't match those stored in test.")

def test_controls_report_kwargs_colorblind_bg_offset():
    expected_outfile = 'GSE69852_QC_Report.xlsx'
    if Path(PROCESSED_450K,expected_outfile).exists():
        Path(PROCESSED_450K,expected_outfile).unlink()
    methylcheck.controls_report(filepath=PROCESSED_450K, legacy=False, colorblind=True, outfilepath=PROCESSED_450K,
        bg_offset=0, roundoff=3, passing=0.5)
    if not Path(PROCESSED_450K,expected_outfile).exists():
        raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_450K}")
    results = pd.read_excel(Path(PROCESSED_450K, expected_outfile))
    if not list(results.iloc[1].values) == ['9247377093_R02C01', 0.671, 62.84, 99.475, 51.826, 10.854, 1.661, 1.894, 1.017, 0.716, 19.962, 0.66, 7.776, 1.97, 5.47, 0.361, 12.98, 5.932, 13.168, 0.902, 10.483, 14.944, 414, 1511, 294, 204, 0.85, 0.88, 99.7, 'M', 'FAIL (0.76)']:
        raise AssertionError(f"--colorblind, outfilepath, bg_offset=0, roundoff=3, passing=0.5: Calculated Numbers don't match those stored in test.")

def test_controls_report_kwargs_no_pval():
    expected_outfile = 'GSE69852_QC_Report.xlsx'
    if Path(PROCESSED_450K,expected_outfile).exists():
        Path(PROCESSED_450K,expected_outfile).unlink()
    methylcheck.controls_report(filepath=PROCESSED_450K, pval=False)
    if not Path(PROCESSED_450K,expected_outfile).exists():
        raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_450K}")
    results = pd.read_excel(Path(PROCESSED_450K, expected_outfile))
    if not list(results.iloc[1].values) == ['9247377093_R02C01', 0.08, 62.84, 99.47, 51.83, 10.85, 1.66, 1.89, 8.39, 5.91, 19.96, 5.44, 7.78, 5.88, 5.47, 2.97, 12.98, 5.93, 13.17, 7.44, 10.48, 14.94, 414, 1511, 294, 204, 0.85, 0.88, 'M', 'OK (0.96)']:
        raise AssertionError(f"--pval=False: Calculated Numbers don't match those stored in test.")

def test_controls_report_kwargs_pval_sig():
    expected_outfile = 'GSE69852_QC_Report.xlsx'
    if Path(PROCESSED_450K,expected_outfile).exists():
        Path(PROCESSED_450K,expected_outfile).unlink()
    methylcheck.controls_report(filepath=PROCESSED_450K, pval=True, pval_sig=0.001)
    if not Path(PROCESSED_450K,expected_outfile).exists():
        raise FileNotFoundError(f"QC Report file missing for folder: {PROCESSED_450K}")
    results = pd.read_excel(Path(PROCESSED_450K, expected_outfile))
    if not list(results.iloc[1].values) == ['9247377093_R02C01', 0.08, 62.84, 99.47, 51.83, 10.85, 1.66, 1.89, 8.39, 5.91, 19.96, 5.44, 7.78, 5.88, 5.47, 2.97, 12.98, 5.93, 13.17, 7.44, 10.48, 14.94, 414, 1511, 294, 204, 0.85, 0.88, 74.1, 'M', 'FAIL (pval)']:
        raise AssertionError(f"--pval=True pval_sign=0.01: Calculated Numbers don't match those stored in test.")
    if Path(PROCESSED_450K,expected_outfile).exists():
        Path(PROCESSED_450K,expected_outfile).unlink()
