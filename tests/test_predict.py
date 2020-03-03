#app
import methylcheck
test_filepath = 'docs/example_data/GSE69852'

def test_get_sex_from_data_containers():
    dfs = methylcheck.load(test_filepath, 'meth')
    output = methylcheck.get_sex(data_containers=dfs, verbose=True, plot=False)

def test_get_sex_from_filepath():
    output = methylcheck.get_sex(test_filepath, verbose=True, plot=False)
