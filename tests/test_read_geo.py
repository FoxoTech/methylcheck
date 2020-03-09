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
        'GSE72120_test.txt',
        #'GSE133355_test.xlsx', # this one failed with the multiline header in sub-file. so I just truncated the original file
        'GSE133355_processed_test.xlsx',
        'GSE61653_test.txt',
        'GSE72556_test.txt',
        'GSE138279_test.csv',
    ]

    def test_read_csv(self):
        files = [file for file in self.unit_test_files if Path(TESTPATH,file).suffix == '.csv']
        for infile in files:
            df = methylcheck.read_geo(Path(TESTPATH,infile), verbose=False)
            if not hasattr(df,'shape'):
                raise AssertionError(f"[CSV] {infile} failed to return a dataframe")

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
