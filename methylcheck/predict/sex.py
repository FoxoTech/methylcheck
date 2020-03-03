import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sb
#app
import methylcheck # uses .load; get_sex uses methylprep models too

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

def get_sex(filepath=None, data_containers=None, array_type=None, verbose=False, plot=False):
    """
    given a path to a folder with data that contains processed sample data,
    calculate and predict the sex of each sample.

while providing a filepath is the easiest way, you can also pass in a data_containers object,
a list of data_containers containing raw meth/unmeth values, instead. This object is produced
by methylprep.run_pipeline, or by using methylcheck.load(filepath, format='meth') and lets you
customize the import if your files were not prepared using methylprep (non-standand CSV columns, for example)
    """
    try:
        from methylprep.files import Manifest
        from methylprep.models import ArrayType
    except ImportError:
        raise ImportError("This function requires methylprep to be installed (pip3 install `methylprep`)")
    if filepath and not data_containers:
        silent = False if verbose is True else False
        data_containers = methylcheck.load(filepath, format='meth', silent=silent)

    # Pull raw M and U values into one dataframe
    if data_containers[0]._SampleDataContainer__data_frame.index.name != 'IlmnID':
        data_containers[0]._SampleDataContainer__data_frame.rename_axis('IlmnID', inplace=True)

    meth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)
    unmeth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)
    for i,c in enumerate(data_containers):
        sample = data_containers[i].sample
        if c._SampleDataContainer__data_frame.index.name != 'IlmnID':
            c._SampleDataContainer__data_frame.rename_axis('IlmnID', inplace=True)
        m = c._SampleDataContainer__data_frame.rename(columns={'meth':sample})
        u = c._SampleDataContainer__data_frame.rename(columns={'unmeth':sample})
        meth = pd.merge(left=meth, right=m[sample], left_on='IlmnID', right_on='IlmnID')
        unmeth = pd.merge(left=unmeth, right=u[sample], left_on='IlmnID', right_on='IlmnID')
    meth = meth.set_index('IlmnID')
    unmeth = unmeth.set_index('IlmnID')

    if len(meth) != len(unmeth):
        raise ValueError(f"WARNING: probe count mismatch: meth {len(meth)} -- unmeth {len(unmeth)}")

    # get list of X any Y probes - using .methylprep_manifest_files and auto-detected array here
    if not array_type:
        probe_count = len(meth)
        if 27000 < probe_count < 30000:
            array_type = ArrayType.ILLUMINA_27K
        elif 30000 < probe_count < 500000:
            array_type = ArrayType.ILLUMINA_450K
        elif 500000 < probe_count < 867000:
            array_type = ArrayType.ILLUMINA_EPIC
        elif 867000 < probe_count < 870000:
            array_type = ArrayType.ILLUMINA_EPIC_PLUS
        else:
            raise ValueError(f"{error} -- are you loading full sample array data, or a subset? The manifest file can only work with full array probe data.")

    if verbose:
        LOGGER.debug(array_type)
    LOGGER.setLevel(logging.DEBUG)
    manifest = Manifest(array_type)._Manifest__data_frame # 'custom', '27k', '450k', 'epic', 'epic+'
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
    difference = output['y_median'] - output['x_median']

    # minfi cutoff - can be manipulated by user
    cutoff = -2

    # use cutoff to predict sex
    sex0 = ['F' if x < -2 else 'M' for x in difference]

    # populate dataframe with predicted sex
    output['predicted_sex'] = sex0
    output = output.round(1)

    if plot == True:
        sb.scatterplot(output['x_median'], output['y_median'], output['predicted_sex'])
        plt.show()
    return output
