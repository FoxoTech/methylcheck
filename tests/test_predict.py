#app
import methylcheck
test_filepath = 'docs/example_data/GSE69852'

def test_get_sex_from_path():
    output = methylcheck.get_sex(test_filepath, verbose=True, plot=False)
    print(output)

def test_get_sex_from_path():
    output = methylcheck.get_sex(test_filepath, array_type='EPIC', verbose=True, plot=False)
    print(output)

def test_get_sex_from_data_containers():
    dfs = methylcheck.load(test_filepath, 'meth')
    output = methylcheck.get_sex(dfs, verbose=True, plot=False)

def test_get_sex_from_meth_unmeth():
    meth, unmeth = methylcheck.qc_plot._get_data(path=test_filepath)
    output = methylcheck.get_sex((meth, unmeth), verbose=True, plot=False)

def test_get_sex_from_path():
    output = methylcheck.get_sex(test_filepath, verbose=True, plot=False)
