import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def get_sex(data_source, array_type=None, verbose=False, plot=False, save=False,
        on_lambda=False, median_cutoff= -2, include_probe_failure_percent=True,
        poobah_cutoff=20, custom_label=None, return_fig=False, return_labels=False):
    """This will calculate and predict the sex of each sample.

inputs:
=======
    the "data_source" can be any one of:
        path -- to a folder with csv data that contains processed sample data
        path -- to a folder with the 'meth_values.pkl' and 'unmeth_values.pkl' dataframes
        path -- to a folder also containing samplesheet pkl and poobah_values.pkl, if you want to compare predicted sex with actual sex.
        data_containers -- object created from methylprep.run_pipeline() or methylcheck.load(path, 'meth')
        tuple of (meth, unmeth) dataframes
    array_type (string)
        enum: {'27k','450k','epic','epic+','mouse'}
        if not specified, it will load the data from data_source and determine the array for you.
    median_cutoff
        the minimum difference in the medians of X and Y probe copy numbers to assign male or female
        (copied from the minfi sex predict function)
    include_probe_failure_percent:
        True: includes poobah percent per sample as column in the output table and on the plot.
        Note: you must supply a 'path' as data_source to include poobah in plots.
    poobah_cutoff
        The maximum percent of sample probes that can fail before the sample fails. Default is 20 (percent)
        Has no effect if `include_probe_failure_percent` is False.
    plot
        True: creates a plot, with option to `save` as image or `return_fig`.
    save
        True: saves the plot, if plot is True
    return_fig
        If True, returns a pyplot figure instead of a dataframe. Default is False.
        Note: return_fig will not show a plot on screen.
    return_labels: (requires plot == True)
        When using poobah_cutoff, the figure only includes A-Z,1...N labels on samples on plot to make it easier to read.
        So to get what sample_ids these labels correspond to, you can rerun the function with return_labels=True and it will
        skip plotting and just return a dictionary with sample_ids and these labels, to embed in a PDF report if you like.
    custom_label:
        Option to provide a dictionary with keys as sample_ids and values as labels to apply to samples.
        e.g. add more data about samples to the multi-dimensional QC plot

while providing a filepath is the easiest way, you can also pass in a data_containers object,
a list of data_containers containing raw meth/unmeth values, instead. This object is produced
by methylprep.run_pipeline, or by using methylcheck.load(filepath, format='meth') and lets you
customize the import if your files were not prepared using methylprep (non-standand CSV columns, for example)

If a `poobah_values.pkl` file can be found in path, the dataframe returned will also include
percent of probes for X and Y chromosomes that failed quality control, and warn the user if any did.
This feature won't work if a containers object or tuple of dataframes is passed in, instead of a path.

Note: ~90% of Y probes should fail if the sample is female. That chromosome is missing."""
    allowed_array_types = {'27k','450k','epic','epic+','mouse'}

    try:
        from methylprep.files import Manifest
        from methylprep.models import ArrayType
    except ImportError:
        raise ImportError("This function requires methylprep to be installed (pip3 install `methylprep`)")

    (data_source_type, data_source) = methylcheck.load_processed._data_source_type(data_source)
    # data_source_type is one of {'path', 'container', 'control', 'meth_unmeth_tuple'}
    poobah=None
    if data_source_type in ('path'):
        # this will look for saved pickles first, then csvs or parsing the containers (which are both slower)
        # the saved pickles function isn't working for batches yet.
        try:
            meth, unmeth = methylcheck.qc_plot._get_data(
                data_containers=None, path=data_source,
                compare=False, noob=False, verbose=False)
        except Exception as e:
            meth, unmeth = methylcheck.qc_plot._get_data(
                data_containers=None, path=data_source,
                compare=False, noob=True, verbose=False)
        if include_probe_failure_percent == True and Path(data_source,'poobah_values.pkl').expanduser().exists():
            poobah = pd.read_pickle(Path(data_source,'poobah_values.pkl').expanduser())

    elif data_source_type in ('container'):
        # this will look for saved pickles first, then csvs or parsing the containers (which are both slower)
        # the saved pickles function isn't working for batches yet.
        meth, unmeth = methylcheck.qc_plot._get_data(
            data_containers=data_source, path=None,
            compare=False, noob=False, verbose=False)

    elif data_source_type == 'meth_unmeth_tuple':
        (meth, unmeth) = data_source

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
    manifest = Manifest(array_type, on_lambda=on_lambda, verbose=verbose)._Manifest__data_frame # 'custom', '27k', '450k', 'epic', 'epic+'
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
    # NOTE for testing: GSE85566/GPL13534 (N=120) has 4 samples that are predicted as wrong sex when using -2, but work at -0.5.

    # populate dataframe with predicted sex
    output['predicted_sex'] = sex0
    output = output.round(1)

    # if poobah_df exists, calculate percent X and Y probes that failed
    sample_failure_percent = {} # % of ALL probes in sample, not just X or Y
    if include_probe_failure_percent == True and isinstance(poobah, pd.DataFrame):
        p_value_cutoff = 0.05
        X_col = []
        Y_col = []
        failed_samples = []
        for column in poobah.columns:
            sample_failure_percent[column] = round(100*len(poobah[column][poobah[column] >= p_value_cutoff].index) / len(poobah.index),1)
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

    if data_source_type in ('path'):
        output = _fetch_actual_sex_from_sample_sheet_meta_data(data_source, output)

    if plot == True:
        fig = _plot_predicted_sex(data=output, # 'x_median', 'y_median', 'predicted_sex', 'X_fail_percent', 'Y_fail_percent'
            sample_failure_percent=sample_failure_percent,
            median_cutoff=median_cutoff,
            include_probe_failure_percent=include_probe_failure_percent,
            verbose=verbose,
            save=save,
            poobah_cutoff=poobah_cutoff,
            custom_label=custom_label,
            data_source_type=data_source_type,
            data_source=data_source,
            return_fig=return_fig,
            return_labels=return_labels,
            )
        if return_labels:
            return fig # these are a lookup dictionary of labels
    if return_fig:
        return fig
    return output


def _plot_predicted_sex(data=pd.DataFrame(),
    sample_failure_percent={},
    median_cutoff= -2,
    include_probe_failure_percent=True,
    verbose=False,
    save=False,
    poobah_cutoff=20, #%
    custom_label=None,
    data_source_type=None,
    data_source=None,
    return_fig=False,
    return_labels=False):
    """
data columns: ['x_median', 'y_median', 'predicted_sex', 'X_fail_percent', 'Y_fail_percent']
- color is sex, pink or blue
- marker circle size will be larger and more faded if poobah values are worse, smaller and darker if low variance. Like a probability cloud.
- sample text is (ID, delta age)
- sex mismatches are X, matched samples are circles (if samplesheet contains actual sex data)
- omits labels for samples that have LOW failure rates, but shows IDs when failed
- adds legend of sketchy samples and labels
- show delta age on labels (using custom column dict)
- unit tests with custom label and without, and check that controls_report still works with this function
- save_fig
- return_labels, returns a lookup dict instead of plot

if there is a "custom_label" dict passed in, such as (actual_age - predicted_age), it simply adds those this label to the marker text labels.
Dicts must match the data DF index.
    """
    if sample_failure_percent != {} and set(sample_failure_percent.keys()) == set(data.index):
        data['sample_failure_percent'] = pd.Series(sample_failure_percent)
    else:
        LOGGER.warning("sample_failure_percent index did not align with output data index")
    #sns.set_theme(style="white")
    show_mismatches = None if 'sex_matches' not in data.columns else "sex_matches"
    if show_mismatches:
        data["sex_matches"] = data["sex_matches"].map({0:"Mismatch", 1:"Match"})
    show_failure = None if 'sample_failure_percent' not in data.columns else "sample_failure_percent"
    sample_sizes = (20, 600)
    if show_failure: # avoid sizing dots with narrow range; gives false impression of bad samples.
        poobah_range = data["sample_failure_percent"].max() - data["sample_failure_percent"].min()
        if poobah_range < poobah_cutoff/2:
            show_failure = None
            sample_sizes = (40,40)

    custom_palette = sns.set_palette(sns.color_palette(['#FE6E89','#0671B7']))
    # if only one sex, make sure male is blue; female is pink
    # if hasattr(output, 'actual_sex') and set(output.actual_sex) == set('M')
    # if first value to be plotted is male, change palette
    if hasattr(data, 'predicted_sex') and list(data.predicted_sex)[0] == 'M':
        custom_palette = sns.set_palette(sns.color_palette(['#0671B7','#FE6E89']))

    fig = sns.relplot(data=data,
        x='x_median',
        y='y_median',
        hue="predicted_sex",
        size=show_failure,
        style=show_mismatches,
        sizes=sample_sizes,
        alpha=.5,
        palette=custom_palette,
        height=8,
        aspect=1.34)
    ax = fig.axes[0,0]
    fig.fig.subplots_adjust(top=.95)
    # for zoomed-in plots with few points close together, set the min scale to be at least 2 units.
    yscale = plt.gca().get_ylim()
    xscale = plt.gca().get_xlim()
    if abs(yscale[1]-yscale[0]) < 2.0:
        ax.set_xlim(xmin=xscale[0]-1, xmax=xscale[1]+1)
        ax.set_ylim(ymin=yscale[0]-1, ymax=yscale[1]+1)
    label_lookup = {index_val: chr(i+65) if (i <= 26) else str(i-26) for i,index_val in enumerate(data.index)}
    for idx,row in data.iterrows():
        if "sample_failure_percent" in row and row['sample_failure_percent'] > poobah_cutoff:
            label = f"{label_lookup[idx]}, {custom_label.get(idx)}" if isinstance(custom_label, dict) and custom_label.get(idx) else label_lookup[idx]
            ax.text(row['x_median'], row['y_median'], label, horizontalalignment='center', fontsize=10, color='darkred')
        else:
            label = f"{custom_label.get(idx)}" if isinstance(custom_label, dict) else None
            if label:
                ax.text(row['x_median']+0.05, row['y_median']+0.05, label, horizontalalignment='center', fontsize=10, color='grey')
    if return_labels:
        plt.close() # release memory
        return label_lookup
    if "sample_failure_percent" in data.columns:
        N_failed = len(data[data['sample_failure_percent'] > poobah_cutoff].index)
        N_total = len(data['sample_failure_percent'].index)
        ax.set_title(f"{N_failed} of {N_total} samples failed poobah, with at least {poobah_cutoff}% of probes failing")
    else:
        ax.set_title(f"Predicted sex based on matching X and Y probes.")
    if save:
        filepath = 'predicted_sexes.png' if data_source_type != 'path' else Path(data_source,'predicted_sexes.png').expanduser()
        plt.savefig(filepath, bbox_inches="tight")
    if return_fig:
        return fig
    plt.show()

def _fetch_actual_sex_from_sample_sheet_meta_data(filepath, output):
    """output is a dataframe with Sample_ID in the index. This adds actual_sex as a column and returns it."""
    # controls_report() does the same thing, and only calls get_sex() with the minimum of data to be fast, because these are already loaded. Just passes in meth/unmeth data
    # Sample sheet should have 'M' or 'F' in column to match predicted sex.

    # merge actual sex into processed output, if available
    file_patterns = {
        'sample_sheet_meta_data.pkl': 'meta',
        '*_meta_data.pkl': 'meta',
        '*samplesheet*.csv': 'meta',
        '*sample_sheet*.csv': 'meta',
    }
    loaded_files = {}
    for file_pattern in file_patterns:
        for filename in Path(filepath).expanduser().rglob(file_pattern):
            if '.pkl' in filename.suffixes:
                loaded_files['meta'] = pd.read_pickle(filename)
                break
            if '.csv' in filename.suffixes:
                loaded_files['meta'] = pd.read_csv(filename)
                break
    if len(loaded_files) == 1:
        # methylprep v1.5.4-6 was creating meta_data files with two Sample_ID columns. Check and fix here:
        # methylcheck 0.7.9 / prep 1.6.0 meta_data lacking Sample_ID when sample_sheet uses alt column names and gets replaced.
        if any(loaded_files['meta'].columns.duplicated()):
            loaded_files['meta'] = loaded_files['meta'].loc[:, ~loaded_files['meta'].columns.duplicated()]
            LOGGER.info("Removed a duplicate Sample_ID column in samplesheet")
        if 'Sample_ID' in loaded_files['meta'].columns:
            loaded_files['meta'] = loaded_files['meta'].set_index('Sample_ID')
        elif 'Sentrix_ID' in loaded_files['meta'].columns and 'Sentrix_Position' in loaded_files['meta'].columns:
            loaded_files['meta']['Sample_ID'] = loaded_files['meta']['Sentrix_ID'].astype(str) + '_' + loaded_files['meta']['Sentrix_Position'].astype(str)
            loaded_files['meta'] = loaded_files['meta'].set_index('Sample_ID')
        else:
            raise ValueError("Your sample sheet must have a Sample_ID column, or (Sentrix_ID and Sentrix_Position) columns.")
        # fixing case of the relevant column
        renamed_column = None
        if ('Gender' in loaded_files['meta'].columns or 'Sex' in loaded_files['meta'].columns):
            if 'Gender' in loaded_files['meta'].columns:
                renamed_column = 'Gender'
            elif 'Sex' in loaded_files['meta'].columns:
                renamed_column = 'Sex'
        else:
            renamed_columns = {col:(col.title() if col.lower() in ('sex','gender') else col) for col in loaded_files['meta'].columns}
            loaded_files['meta'] = loaded_files['meta'].rename(columns=renamed_columns)
            if 'Gender' in renamed_columns.values():
                renamed_column = 'Gender'
            elif 'Sex' in renamed_columns.values():
                renamed_column = 'Sex'
        if renamed_column is not None:
            # next, ensure samplesheet Sex/Gender (Male/Female) are recoded as M/F; controls_report() does NOT do this step, but should.
            sex_values = set(loaded_files['meta'][renamed_column].unique())
            #print('sex_values', sex_values)
            if not sex_values.issubset(set(['M','F'])): # subset, because samples might only contain one sex
                if 'Male' in sex_values or 'Female' in sex_values:
                    loaded_files['meta'][renamed_column] = loaded_files['meta'][renamed_column].map({'Male':'M', 'Female':'F'})
                elif 'male' in sex_values or 'female' in sex_values:
                    loaded_files['meta'][renamed_column] = loaded_files['meta'][renamed_column].map({'male':'M', 'female':'F'})
                elif 'MALE' in sex_values or 'FEMALE' in sex_values:
                    loaded_files['meta'][renamed_column] = loaded_files['meta'][renamed_column].map({'MALE':'M', 'FEMALE':'F'})
                elif 'm' in sex_values or 'f' in sex_values:
                    loaded_files['meta'][renamed_column] = loaded_files['meta'][renamed_column].map({'m':'M', 'f':'F'})
                else:
                    raise ValueError(f"Cannot compare with predicted sex because actual sexes listed in your samplesheet are not understood (expecting M or F): (found {sex_values})")
            output['actual_sex'] = None
            output['sex_matches'] = None
            for row in output.itertuples():
                try:
                    actual_sex = str(loaded_files['meta'].loc[row.Index].get(renamed_column))
                except KeyError:
                    if 'Sample_ID' in output.columns:
                        LOGGER.warning("Sample_ID was another column in your output DataFrame; Set that to the index when you pass it in.")
                    raise KeyError("Could not read actual sex from meta data to compare.")
                if isinstance(actual_sex, pd.Series):
                    LOGGER.warning(f"Multiple samples matched actual sex for {row.Index}, because Sample_ID repeats in sample sheets. Only using first match, so matches may not be accurate.")
                    actual_sex = actual_sex[0]
                if hasattr(row,'predicted_sex'):
                    sex_matches = 1 if actual_sex.upper() == str(row.predicted_sex).upper() else 0
                else:
                    sex_matches = np.nan
                output.loc[row.Index, 'actual_sex'] = actual_sex
                output.loc[row.Index, 'sex_matches'] = sex_matches
        else:
            pass # no Sex/Gender column found in samplesheet
    return output
