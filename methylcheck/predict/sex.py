import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
#app
import methylcheck # uses .load; get_sex uses methylprep models too and detect_array()

import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_copy_number(meth,unmeth):
    """function to return copy number.
    requires dataframes of methylated and
    unmethylated values. can be raw OR corrected"""
    # minfi R version:
    # log2(getMeth(object) + getUnmeth(object))
    return np.log2(meth+unmeth)


def get_sex(data_source, array_type=None, verbose=False, plot=False, on_lambda=False, median_cutoff= -2, include_probe_failure_percent=True):
    """This will calculate and predict the sex of each sample.

inputs:
=======
    the "data_source" can be any one of:
        path -- to a folder with csv data that contains processed sample data
        path -- to a folder with the 'meth_values.pkl' and 'unmeth_values.pkl' dataframes
        data_containers -- object created from methylprep.run_pipeline() or methylcheck.load(path, 'meth')
        tuple of (meth, unmeth) dataframes
    array_type (string)
        enum: {'27k','450k','epic','epic+','mouse'}
        if not specified, it will load the data from data_source and determine the array for you.
    median_cutoff
        the minimum difference in the medians of X and Y probe copy numbers to assign male or female
        (copied from the minif sex predict function)

while providing a filepath is the easiest way, you can also pass in a data_containers object,
a list of data_containers containing raw meth/unmeth values, instead. This object is produced
by methylprep.run_pipeline, or by using methylcheck.load(filepath, format='meth') and lets you
customize the import if your files were not prepared using methylprep (non-standand CSV columns, for example)

If a `poobah_values.pkl` file can be found in path, the dataframe returned will also include
percent of probes for X and Y chromosomes that failed quality control, and warn the user if any did.
This feature won't work if a containers object is passed in, instead of a path.

Note: ~90% of Y probes should fail if the sample is female. That chromosome is missing.
    """
    allowed_array_types = {'27k','450k','epic','epic+','mouse'}

    try:
        from methylprep.files import Manifest
        from methylprep.models import ArrayType
    except ImportError:
        raise ImportError("This function requires methylprep to be installed (pip3 install `methylprep`)")

    (data_source_type, data_source) = methylcheck.load_processed._data_source_type(data_source)
    # data_source_type is one of {'path', 'container', 'control', 'meth_unmeth_tuple'}
    if data_source_type in ('path'):
        # this will look for saved pickles first, then csvs or parsing the containers (which are both slower)
        # the saved pickles function isn't working for batches yet.
        meth, unmeth = methylcheck.qc_plot._get_data(
            data_containers=None, path=data_source,
            compare=False, noob=False, verbose=False)
        if include_probe_failure_percent == True and Path(data_source,'poobah_values.pkl').exists():
            poobah = pd.read_pickle(Path(data_source,'poobah_values.pkl'))

    elif data_source_type in ('container'):
        # this will look for saved pickles first, then csvs or parsing the containers (which are both slower)
        # the saved pickles function isn't working for batches yet.
        meth, unmeth = methylcheck.qc_plot._get_data(
            data_containers=data_source, path=None,
            compare=False, noob=False, verbose=False)
        poobah = None

    elif data_source_type is 'meth_unmeth_tuple':
        (meth, unmeth) = data_source
        poobah = None

    if len(meth) != len(unmeth):
        raise ValueError(f"WARNING: probe count mismatch: meth {len(meth)} -- unmeth {len(unmeth)}")

    if array_type == None:
        # get list of X any Y probes - using .methylprep_manifest_files (or MANIFEST_DIR_PATH_LAMBDA) and auto-detected array here
        array_type = ArrayType(methylcheck.detect_array(meth, on_lambda=on_lambda))
    elif isinstance(array_type,str):
        if array_type in allowed_array_types:
            array_type = ArrayType(array_type)
        else:
            raise ValueError(f"Your array_type must be one of these: {allowed_array_types} or None.")

    if verbose:
        LOGGER.debug(array_type)
    LOGGER.setLevel(logging.WARNING)
    manifest = Manifest(array_type, on_lambda=on_lambda)._Manifest__data_frame # 'custom', '27k', '450k', 'epic', 'epic+'
    LOGGER.setLevel(logging.INFO)
    x_probes = manifest.index[manifest['CHR']=='X']
    y_probes = manifest.index[manifest['CHR']=='Y']
    if verbose:
        LOGGER.info(f"Found {len(x_probes)} X and {len(y_probes)} Y probes")

    # dataframes of meth and unmeth values for the sex chromosomes
    x_meth = meth[meth.index.isin(x_probes)]
    x_unmeth = unmeth[unmeth.index.isin(x_probes)]

    y_meth = meth[meth.index.isin(y_probes)]
    y_unmeth = unmeth[unmeth.index.isin(y_probes)]

    # create empty dataframe for output
    output = pd.DataFrame(index=[s for s in meth.columns], columns=['x_median','y_median','predicted_sex'])
    # get median values for each sex chromosome for each sample
    x_med = _get_copy_number(x_meth,x_unmeth).median()
    y_med = _get_copy_number(y_meth,y_unmeth).median()

    # populate output dataframe with values
    output['x_median'] = output.index.map(x_med)
    output['y_median'] = output.index.map(y_med)

    # compute difference
    median_difference = output['y_median'] - output['x_median']

    # median cutoff - can be manipulated by user --- default = -2 --- used to predict sex
    sex0 = ['F' if x < median_cutoff else 'M' for x in median_difference]

    # populate dataframe with predicted sex
    output['predicted_sex'] = sex0
    output = output.round(1)

    # if poobah_df exists, calculate percent X and Y probes that failed
    if include_probe_failure_percent == True and isinstance(poobah, pd.DataFrame):
        p_value_cutoff = 0.05
        X_col = []
        Y_col = []
        failed_samples = []
        for column in poobah.columns:
            failed_probe_names = poobah[column][poobah[column] >= p_value_cutoff].index
            failed_x_probe_names = list(set(failed_probe_names) & set(x_probes))
            failed_y_probe_names = list(set(failed_probe_names) & set(y_probes))
            X_percent = round(100*len(failed_x_probe_names)/poobah.index.isin(list(x_probes)).sum(),1)
            Y_percent = round(100*len(failed_y_probe_names)/poobah.index.isin(list(y_probes)).sum(),1)
            X_col.append(X_percent)
            Y_col.append(Y_percent)
            if X_percent > 10:
                failed_samples.append(column)
        output['X_fail_percent'] = X_col #output.index.map(X_col)
        output['Y_fail_percent'] = Y_col #output.index.map(Y_col)
        if failed_samples != []:
            LOGGER.warning(f"{len(failed_samples)} samples had >10% of X probes fail p-value probe detection. Predictions for these may be unreliable:")
            LOGGER.warning(f"{failed_samples}")

    if plot == True:
        sb.scatterplot(output['x_median'], output['y_median'], output['predicted_sex'])
        plt.show()
    return output
