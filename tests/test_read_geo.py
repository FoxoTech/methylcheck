# -*- coding: utf-8 -*-
from pathlib import Path
TESTPATH = 'tests'

#app
import methylcheck

class TestReadGeo():

    # these files are small sub-sets of the real GEO files, created to ensure read_geo() can parse various file structures without commiting huge files to the repo.
    unit_test_files = [
        'GSE111165_test.csv',
        'GSE73745_test.txt',
        'GSE46573_test.txt',
        'GSE72354_test.csv',
        'GSE72354_test.csv.gz',
        'GSE72120_test.txt',
        #'GSE133355_test.xlsx', # this one failed with the multiline header in sub-file. so I just truncated the original file
        'GSE133355_processed_test.xlsx', # multiline header here, and extra columns that are not sample betas.
        'GSE61653_test.txt',
        'GSE72556_test.txt', # multiline header
        'GSE138279_test.csv',
        'GSE78874_datSignal_test.csv',
        'GSE116992_GPL21145_signals_test.csv',
        'GSE116924_matrix_signal_intensities_test.csv',
        'GSE110554_signals_test.txt',
        'GSE50660_matrix_processed_test.csv',
        'GSE53045_matrix_signal_GEO_test.txt',
    ]
    unit_test_file_shapes = {
        'GSE111165_test.csv': (200, 101),
        'GSE73745_test.txt': (200, 24),
        'GSE46573_test.txt': (200, 22),
        'GSE72354_test.csv': (200, 34),
        'GSE72354_test.csv.gz': (200, 34),
        'GSE72120_test.txt': (200, 72),
        'GSE133355_processed_test.xlsx': (6735, 44),
        'GSE61653_test.txt': (200, 127),
        'GSE72556_test.txt': (1200, 96),
        'GSE138279_test.csv': (200, 65),
        'GSE78874_datSignal_test.csv': (200, 257),
        'GSE116992_GPL21145_signals_test.csv': (200, 22),
        'GSE116924_matrix_signal_intensities_test.csv': (200, 24),
        'GSE110554_signals_test.txt': (200, 49),
        'GSE50660_matrix_processed_test.csv': (200, 464),
        'GSE53045_matrix_signal_GEO_test.txt': (200, 111),
    }

    def test_read_csv(self):
        files = [file for file in self.unit_test_files if ('.csv' in Path(TESTPATH,file).suffixes)]
        for infile in files:
            df = methylcheck.read_geo(Path(TESTPATH,infile), verbose=False)
            if not hasattr(df,'shape'):
                raise AssertionError(f"[CSV] {infile.name} failed to return a dataframe")
            if df.shape != self.unit_test_file_shapes[Path(infile).name]:
                raise ValueError(f"[CSV] {infile.name} shape did not match ({df.shape} vs {unit_test_file_shapes[Path(infile).name]})")

    def test_read_xlsx(self):
        files = [file for file in self.unit_test_files if Path(TESTPATH,file).suffix == '.xlsx']
        for infile in files:
            df = methylcheck.read_geo(Path(TESTPATH,infile), verbose=False)
            if not hasattr(df,'shape'):
                raise AssertionError(f"[XLSX] {infile} failed to return a dataframe")

    def test_read_txt(self):
        files = [file for file in self.unit_test_files if Path(TESTPATH,file).suffix == '.txt']
        for infile in files:
            df = methylcheck.read_geo(Path(TESTPATH,infile), verbose=False)
            if not hasattr(df,'shape'):
                raise AssertionError(f"[TXT] {infile} failed to return a dataframe")
