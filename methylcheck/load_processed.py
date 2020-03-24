import time
from pathlib import Path
import numpy as np
import re # for sesame filename extraction
import pandas as pd
#app
from .progress_bar import * # context tqdm

import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

__all__ = ['load', 'load_both', 'container_to_pkl']


## modified this from methylprep on 2020-02-20 to allow for data_containers to be returned as option
def load(filepath='.', format='beta_value', file_stem='', verbose=False, silent=False, column_names=None, no_poobah=False, pval_cutoff=0.05):
    """When methylprep processes large datasets, you use the 'batch_size' option to keep memory and file size
    more manageable. Use the `load` helper function to quickly load and combine all of those parts into a single
    data frame of beta-values or m-values.

    Doing this with pandas is about 8 times slower than using numpy in the intermediate step.

    If no arguments are supplied, it will load all files in current directory that have a 'beta_values_X.pkl' pattern.

Arguments:
==========

    filepath:
        Where to look for all the pickle files of processed data.

    format: ('beta_value', 'm_value', 'meth', 'meth_df', 'noob_df')
        This also allows processed.csv file data to be loaded.
        If you need meth and unmeth values, choose 'meth' and
        it will return a data_containers object with the 'meth' and 'unmeth' values,
        exactly like the data_containers object returned by methylprep.run_pipeline.

        If you choose 'meth_df' or 'noob_df' it will load the pickled meth and unmeth dataframes from the
        folder specified.

    file_stem (string):
        By default, methylprep process with batch_size creates a bunch of generically named files, such as
        'beta_values_1.pkl', 'beta_values_2.pkl', 'beta_values_3.pkl', and so on. IF you rename these or provide
        a custom name during processing, provide that name here.
        (i.e. if your pickle file is called 'GSE150999_beta_values_X.pkl', then your file_stem is 'GSE150999_')

    column_names:
        if your csv files contain column names that differ from those expected, you can specify them as a list of strings
        by default it looks for ['noob_meth', 'noob_unmeth'] or ['meth', 'unmeth'] or ['beta_value'] or ['m_value']
        Note: if you csv data has probe names in a column that is not the FIRST column, or is not named "IlmnID",
        you should specify it with column_names and put it first in the list, like ['illumina_id', 'noob_meth', 'noob_umeth'].

    no_poobah:
        if loading from CSVs, and there is a column for probe p-values (the poobah_pval column),
        the default is to filter out probes that fail the p < 0.05 cutoff. if you specify 'no_poobah'=True,
        it will load everything, regardless of p-values.

    pval_cutoff:
        if applying poobah (pvalue probe detection based on poor signal to noise)
        this specifies the threashold for cutoff (0.05 by default)

    verbose:
        outputs more processing messages.

    silent:
        suppresses all processing messages, even warnings.

Use cases and format:
=====================

    format = beta_value
        you have beta_values.pkl file in the path specified and want a dataframe returned
        or you have a bunch of beta_values_1.pkl files in the path and want them merged and returned as one dataframe
        (when using 'batch_size' option in methylprep.run_pipeline() you'll get multiple files saved)
    format = m_value
        you have m_values.pkl file in the path specified and want a dataframe returned
        or you have a bunch of m_values_1.pkl files in the path and want them merged and returned as one dataframe
    format = meth (data_containers)
        you have processed CSV files in the path specified and want a data_container returned
    format = meth (dataframe)
        you have processed CSV files in the path specified and want a dataframe returned
        take the data_containers object returned and run `methylcheck.container_to_pkl(containers, save=True)` function on it.
    format = sesame
        for reading csvs processed using R's sesame package. It has a different format (Probe_ID, ind_beta, ind_negs, ind_poob) per sample.
        Only those probes that pass the p-value cutoff will be included.

TODO:
    - BUG: custom fields cannot auto detect the -pval- column and this isn't supplied in kwargs
    - DONE: meth_df deal with batches of files

    """
    formats = ('beta_value', 'm_value', 'meth', 'meth_df', 'noob_df', 'sesame')
    if format not in formats:
        raise ValueError(f"Check the spelling of your format. Allowed: {formats}")
    if format == 'sesame':
        return load_sesame(
            filepath=filepath,
            format=format,
            file_stem=file_stem,
            verbose=verbose,
            silent=silent,
            column_names=column_names,
            no_poobah=no_poobah,
            pval_cutoff=pval_cutoff
            )
    processed_csv = False # whether to use individual sample files, or beta pkl files.
    total_parts = []

    # bug: total_parts will match 'meth_values' for format='meth' instead of reading csvs.
    # looking for multiple pickled files (beta/m)
    if format in ('beta_value', 'm_value'):
        total_parts = list(Path(filepath).rglob(f'{file_stem}{format}*.pkl'))
    elif format == 'meth_df':
        # this needs to deal with batches too
        test_parts = list([str(file) for file in Path(filepath).rglob(f'{file_stem}*_values*.pkl')])
        meth_dfs = []
        unmeth_dfs = []
        for part in test_parts:
            if 'meth_values' in part and 'noob' not in part:
                meth_dfs.append( pd.read_pickle(part) )
            if 'unmeth_values' in part and 'noob' not in part:
                unmeth_dfs.append( pd.read_pickle(part) )
        if meth_dfs != [] and unmeth_dfs != []:
            tqdm.pandas()
            meth_dfs = pd.concat(meth_dfs, axis='columns', join='inner').progress_apply(lambda x: x)
            unmeth_dfs = pd.concat(unmeth_dfs, axis='columns', join='inner').progress_apply(lambda x: x)
            print(meth_dfs.shape, unmeth_dfs.shape)
            return meth_dfs, unmeth_dfs
    elif format == 'noob_df':
        # this needs to deal with batches too
        test_parts = list([str(file) for file in Path(filepath).rglob(f'{file_stem}*_values*.pkl')])
        meth_dfs = []
        unmeth_dfs = []
        for part in test_parts:
            if 'noob_meth_values' in part:
                meth_dfs.append( pd.read_pickle(part) )
            if 'noob_unmeth_values' in part:
                unmeth_dfs.append( pd.read_pickle(part) )
        if meth_dfs != [] and unmeth_dfs != []:
            tqdm.pandas()
            meth_dfs = pd.concat(meth_dfs, axis='columns', join='inner').progress_apply(lambda x: x)
            unmeth_dfs = pd.concat(unmeth_dfs, axis='columns', join='inner').progress_apply(lambda x: x)
            print(meth_dfs.shape, unmeth_dfs.shape)
            return meth_dfs, unmeth_dfs

    if total_parts == []:
        # part 2: scan for *_processed.csv files in subdirectories and pull out beta values from them.
        total_parts = list(Path(filepath).rglob('*_R0[0-9]C0[0-9][_.]processed.csv'))
        if verbose and not silent:
            print(len(total_parts),'files matched')
        if total_parts != []:
            sample_betas = []
            sample_names = []
            data_containers = [] # only used if format='meth'
            processed_csv = True
            if verbose and not silent:
                LOGGER.info(f"Found {len(total_parts)} processed samples; building a {format} dataframe from them.")
            # loop through files, open each one, find 'beta_value' column of CSV. save and merge.
            # make sure the rows (probes) match up too.
            # verbose most: shows each file on a new line; silent mode: nothing shown; in between mode: tqdm bar (default)
            for part in (tqdm(total_parts, desc='Files', total=len(total_parts)) if (not verbose and not silent) else total_parts):
                try: # first testing the fastest loader
                    # you get a 30% speed up if you only load the columns you need here
                    if column_names is not None:
                        columns = column_names
                    elif format == 'beta_value':
                        columns = ['IlmnID', 'beta_value']
                    elif format == 'm_value':
                        columns = ['IlmnID', 'm_value']
                    elif format == 'meth':
                        columns = ['IlmnID', 'noob_meth', 'noob_unmeth']
                    else:
                        raise ValueError(f"format {format} not supported. fix.")
                    if no_poobah == False:
                        columns.append('poobah_pval')
                    index_column = 'IlmnID' if 'IlmnID' in columns else columns[0] # fragile assumption: first column is index
                    sample = pd.read_csv(part,
                        # this param assigns the index from one column
                        index_col=index_column,
                        # these params speed up reading
                        usecols=columns,
                        dtype={'illumina_id': str, 'IlmnID':str, 'noob_meth':np.float16,
                            'noob_unmeth':np.float16, 'meth':np.float16, 'unmeth':np.float16,
                            'beta_value':np.float16, 'm_value':np.float16, 'poobah_pval':np.float16},
                        engine='c',
                        memory_map=True, # load all into memory at once for faster reading (less IO)
                        #na_filter=False, # disable to speed read, if not expecting NAs
                    )
                    if sample.index.name == 'illumina_id':
                        # ONLY triggered when specifying custom column_names to function, and 'illumina_id' is first on list.
                        # that will auto-set the index to 'illumina_id'
                        sample.rename_axis('IlmnID', inplace=True)
                except ValueError as e:
                    sample = pd.read_csv(part)
                    if 'IlmnID' in sample.columns:
                        sample.set_index('IlmnID', inplace=True)
                    elif 'illumina_id' in sample.columns:
                        sample.set_index('illumina_id', inplace=True)
                        sample.rename_axis('IlmnID', inplace=True)
                    else:
                        # assume first column
                        guess_index = columns[0]
                        sample.set_index(guess_index, inplace=True)
                        sample.rename_axis('IlmnID', inplace=True)

                fname = str(Path(part).name)
                if '_processed.csv' in fname:
                    sample_name = fname.replace('_processed.csv','')
                else:
                    sample_name = ''
                # FUTURE TODO: if sample_sheet or meta_data supplied, fill in with proper sample_names here
                # incorporate .load_both() here

                if 'beta_value' in sample.columns and format == 'beta_value':
                    if no_poobah == False and 'beta_value' in sample.columns and 'poobah_pval' in sample.columns:
                        sample.loc[sample['poobah_pval'] >= pval_cutoff, 'beta_value'] = np.nan

                    col = sample.loc[:, ['beta_value']]
                    col.rename(columns={'beta_value': sample_name}, inplace=True)
                    sample_names.append(sample_name)
                    sample_betas.append(col)
                    if verbose and not silent:
                        print(f'{sample_name}, {col.shape} --> {len(sample_betas)}')
                elif 'm_value' in sample.columns and format == 'm_value':
                    if no_poobah == False and 'm_value' in sample.columns and 'poobah_pval' in sample.columns:
                        sample.loc[sample['poobah_pval'] >= pval_cutoff, 'm_value'] = np.nan

                    col = sample.loc[:, ['m_value']]
                    col.rename(columns={'m_value': sample_name}, inplace=True)
                    sample_names.append(sample_name)
                    sample_betas.append(col)
                    if verbose and not silent:
                        print(f'{sample_name}, {col.shape} --> {len(sample_betas)}')
                elif (column_names is not None and
                    len(column_names) == 1 and
                    column_names[0] in sample.columns and
                    format in ('beta_value', 'm_value')):
                    # HERE: loading beta_value or m_value from csvs using a custom column_names option
                    # BUG::: cannot auto detect the -pval- column and isn't supplied in kwargs

                    if no_poobah == False and 'poobah_pval' in sample.columns:
                        sample.loc[sample['poobah_pval'] >= pval_cutoff, col_name] = np.nan

                    col_name = column_names[0]
                    col = sample.loc[:, [col_name]]
                    col.rename(columns={col_name: sample_name}, inplace=True)
                    sample_names.append(sample_name)
                    sample_betas.append(col)
                    if verbose and not silent:
                        print(f'{sample_name}, {col.shape} --> {len(sample_betas)}')
                elif format == 'meth':
                    # this returns a data_containers object with meth and unmeth values
                    # find a suitable pair of columns to load
                    if column_names is not None and len(column_names) == 2:
                        columns = column_names
                        new_column_names = column_names
                    else:
                        columns = ['noob_meth', 'noob_unmeth']
                        if 'noob_meth' not in sample.columns and 'noob_unmeth' not in sample.columns:
                            if 'meth' not in sample.columns and 'unmeth' not in sample.columns:
                                columns = ['meth', 'unmeth']
                            else:
                                raise ValueError("Did not find meth data in csv (looked for ['noob_meth', 'noob_unmeth', 'meth', 'unmeth'])")
                        # want ouput col names to be 'meth' or 'unmeth' regardless of NOOB part
                        new_column_names = ['meth', 'unmeth']

                    meth = sample.loc[:, [columns[0]]]
                    unmeth = sample.loc[:, [columns[1]]]
                    meth.rename(columns={columns[0]: new_column_names[0]}, inplace=True)
                    unmeth.rename(columns={columns[1]: new_column_names[1]}, inplace=True)
                    sample_names.append(sample_name)
                    data_container = SampleDataContainer(meth, unmeth, sample_name)
                    data_containers.append(data_container)
                    if verbose and not silent:
                        if meth.shape[0] != unmeth.shape[0]:
                            LOGGER.warning(f"{sample_name} probe counts don't match: {meth.shape}|{unmeth.shape}")
                        print(f'{sample_name} --> {len(data_containers)}')
                else:
                    raise Exception(f"unknown format: {format} (allowed options: {formats})")

            # merge and return; dropping any probes that aren't shared across samples.
            tqdm.pandas() # https://stackoverflow.com/questions/56256861/is-it-possible-to-use-tqdm-for-pandas-merge-operation
            ## if you use Jupyter notebooks, you can also use tqdm_notebooks to get a prettier bar. Together with pandas you'd currently need to instantiate it like
            ## from tqdm import tqdm_notebook; tqdm_notebook().pandas(*args, **kwargs) ##
            if format == 'meth':
                if not silent:
                    LOGGER.info('Produced a list of Sample objects (use obj._SampleDataContainer__data_frame to get values)...')
                return data_containers
            print('merging...')
            df = pd.concat(sample_betas, axis='columns', join='inner').progress_apply(lambda x: x)
            return df
        elif not silent:
            LOGGER.warning(f"No pickled files of type ({format}) found in {filepath} (or sub-folders).")
        return # wrapped up _processed.csv and _calls.csv.gz version of load()

    # does this apply only to batches of beta/m-val.pkl files???
    start = time.process_time()
    parts = []
    #for i in range(1,total_parts):
    probes = pd.DataFrame().index
    samples = pd.DataFrame().index
    for file in tqdm(total_parts, total=len(total_parts), desc="Files"):
        if verbose:
            print(file)
        if processed_csv:
            df = pd.read_csv(file, index_col='IlmnID')
        else:
            df = pd.read_pickle(file)

        # ensure probes are in rows.
        if df.shape[0] < df.shape[1]:
            df = df.transpose()
        # getting probes: both files should have probes in rows.
        if len(probes) == 0:
            probes = df.index
            if verbose:
                print(f'Probes: {len(probes)}')
        if processed_csv:
            if format == 'beta_value':
                samples = samples.append(df['beta_value'])
            if format == 'm_value':
                samples = samples.append(df['m_value'])
        else:
            samples = samples.append(df.columns)
        npy = df.to_numpy()
        parts.append(npy)
    npy = np.concatenate(parts, axis=1) # 8x faster with npy vs pandas
    # axis=1 -- assume that appending to rows, not columns. Each part has same columns (probes)
    try:
        df = pd.DataFrame(data=npy, index=samples, columns=probes)
    except:
        df = pd.DataFrame(data=npy, columns=samples, index=probes)
    if not silent:
        LOGGER.info(f'loaded data {df.shape} from {len(total_parts)} pickled files ({round(time.process_time() - start,3)}s)')
    return df


def load_both(filepath='.', format='beta_value', file_stem='', verbose=False, silent=False, column_names=None, rename_samples=False, sample_names='Sample_Name'):
    """Creates and returns TWO objects (data and meta_data) from the given filepath. Confirms sample names match.

    Returns TWO objects (data, meta) as dataframes for analysis.
    If meta_data files are found in multiple folders, it will read them all and try to match to the samples
    in the beta_values pickles by sample ID.

Arguments:
    filepath:
        Where to look for all the pickle files of processed data.

    format:
        'beta_values', 'm_value', or some other custom file pattern.

    file_stem (string):
        By default, methylprep process with batch_size creates a bunch of generically named files, such as
        'beta_values_1.pkl', 'beta_values_2.pkl', 'beta_values_3.pkl', and so on. IF you rename these or provide
        a custom name during processing, provide that name here.
        (i.e. if your pickle file is called 'GSE150999_beta_values_X.pkl', then your file_stem is 'GSE150999_')
    column_names:
        if your processed csv files contain column names that differ from those expected, you can specify them as a list of strings
        by default it looks for ['noob_meth', 'noob_unmeth'] or ['meth', 'unmeth'] or ['beta_value'] or ['m_value']
        Note: if you csv data has probe names in a column that is not the FIRST column, or is not named "IlmnID",
        you should specify it with column_names and put it first in the list, like ['illumina_id', 'noob_meth', 'noob_umeth'].
    rename_samples:
        if your meta_data contains a 'Sample_Name' column, the returned data and meta_data will have
        index and columns renamed to Sample_Names instead of Sample_IDs, respectively.
    sample_name (string):
        the column name to use in meta dataframe for sample names. Assumes 'Sample_Name' if unspecified.

    verbose:
        outputs more processing messages.
    silent:
        suppresses all processing messages, even warnings.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Invalid filepath: {filepath}")
    meta_files = list(Path(filepath).rglob(f'*_meta_data.pkl'))
    multiple_metas = False
    partial_meta = False
    if len(meta_files) > 1:
        LOGGER.info(f"Found several meta_data files; attempting to match each with its respective beta_values files in same folders.")
        multiple_metas = True # this will skip the df-size-match check below.
        ### if multiple_metas, combine them into one dataframe of meta data with
        ### all samples rows and tags in columns
        # Note: this approach assumes that:
        #    (a) the goal is a row-wise concatenation (i.e., axis=0) and
        #    (b) all dataframes share the same column names.
        frames = [pd.read_pickle(pkl) for pkl in meta_files]
        meta_tags = frames[0].columns # assumes all have same columns
        # do all tags match the first file's columns?
        meta_sets = set()
        for frame in frames:
            meta_sets |= set(frame.columns)
        if meta_sets != set(meta_tags):
            LOGGER.warning(f'Columns in sample sheet meta data files does not match for these files and cannot be combined:'
                           f'{[str(i) for i in meta_files]}')
            meta = pd.read_pickle(meta_files[0])
            partial_meta = True
        else:
            meta = pd.concat(frames, axis=0, sort=False)
            # need to check whether there are multiple samples for each sample name. and warn.

    if len(meta_files) == 1:
        meta = pd.read_pickle(meta_files[0])
    elif multiple_metas:
        if partial_meta:
            LOGGER.info("Multiple meta_data found. Only loading the first file.")
        LOGGER.info(f"Loading {len(meta.index)} samples.")
    else:
        LOGGER.info(f"No meta_data found in ({filepath}).")
        meta = pd.DataFrame()

    data_df = load(filepath=filepath,
        format=format,
        file_stem=file_stem,
        verbose=verbose,
        silent=silent,
        column_names=column_names
        )


    ### confirm the Sample_ID in meta matches the columns (or index) in data_df.
    check = False
    if 'Sample_ID' in meta.columns:
        if len(meta['Sample_ID']) == len(data_df.columns) and all(meta['Sample_ID'] == data_df.columns):
            data_df = data_df.transpose() # samples should be in index
            check = True
        elif len(meta['Sample_ID']) == len(data_df.index) and all(meta['Sample_ID'] == data_df.index):
            check = True
        # or maybe the data is there, but mis-ordered? fix now.
        elif set(meta['Sample_ID']) == set(data_df.columns):
            LOGGER.info(f"Transposed data and reordered meta_data so sample ordering matches.")
            data_df = data_df.transpose() # samples should be in index
            # faster to realign the meta_data instead of the probe data
            sample_order = {v:k for k,v in list(enumerate(data_df.index))}
            # add a temporary column for sorting
            meta['__temp_sorter__'] = meta['Sample_ID'].map(sample_order)
            meta.sort_values('__temp_sorter__', inplace=True)
            meta.drop('__temp_sorter__', axis=1, inplace=True)
            check = True
        elif set(meta['Sample_ID']) == set(data_df.index):
            LOGGER.info(f"Reordered sample meta_data to match data.")
            sample_order = {v:k for k,v in list(enumerate(data_df.index))}
            meta['__temp_sorter__'] = meta['Sample_ID'].map(sample_order)
            meta.sort_values('__temp_sorter__', inplace=True)
            meta.drop('__temp_sorter__', axis=1, inplace=True)
            check = True
    else:
        LOGGER.info('Could not check whether samples in data align with meta_data "Sample_ID" column.')
    if check == False:
        LOGGER.warning("Data samples don't align with 'Sample_ID' column in meta_data.")
    else:
        LOGGER.info("meta.Sample_IDs match data.index (OK)")

    # if rename_samples = True, swap out meta_data index and sample_df columns for Sample_Name field in meta_data
    if rename_samples and 'Sample_Name' in meta.columns:
        # orient
        if data_df.shape[1] > data_df.shape[0]:
            data_df = data_df.transpose()
        if ( (len(set(data_df.columns)) < len(data_df.columns)) or
            (len(set(meta.index.tolist())) < len(meta.index.tolist())) ):
            LOGGER.error(f"Some of your sample_ids or sample_names are repeated (duplicates) and this cannot rename your dataframe columns. Returning with Sample_IDs instead.")
            return data_df, meta
        sample_order = {v:k for k,v in list(enumerate(data_df.index))}
        meta_order = {v:k for k,v in list(enumerate(meta.columns))}
        sample_mapping = {v['Sample_ID']: v['Sample_Name'] for k,v in meta.iterrows()}
        # replace data_df sample columns
        #meta.set_index('Sample_Name', inplace=True) --- meta doesn't use its index
        data_df = data_df.rename(columns=sample_mapping)
        LOGGER.info("Renamed data_df.index and meta_df.columns to Sample_Name")

    return data_df, meta


# similar to methylprep.processing.SampleDataContainer but only requires the meth/unmeth and sample_name data
class SampleDataContainer():
    """compatible output to the methylprep class.

    - includes option to store meth, unmeth, beta, and m_value in self._SampleDataContainer__data_frame.
    - meth_col is a df with column called 'meth'.
    - output is a list of DFs with 'meth' and 'unmeth' columns in each __data_frame."""

    # this part is conserved from methylprep SampleDataContainer.
    __data_frame = None # == self._SampleDataContainer__data_frame

    def __init__(self, meth_col, unmeth_col, sample_name):
        self.sample = sample_name
        #self.methylated = meth_col --- save space by not double-storing this
        #self.unmethylated = unmeth_col
        self.__data_frame = meth_col.join(
            unmeth_col,
            lsuffix='_meth', # Suffix to use from left frame’s overlapping/redundant columns.
            rsuffix='_unmeth',
        )

        # reduce to float32 during processing.
        self.__data_frame = self.__data_frame.astype('float32')
        self.__data_frame = self.__data_frame.round(3)


# was previously named 'container_to_beta'
def container_to_pkl(containers, output='betas', save=True):
    """ simple helper function to convert a list of SampleDataContainer objects to a df and pickle it.

options:
========
    save (True|False)
        whether to save the data to disk, in the current directory
    output ('betas'|'m_value'|'meth'|'noob'|'copy_number')
        reads processed CSVs and consolidates into a single dataframe,
        with samples in columns and probes in rows:
        betas -- saves 'beta_values.pkl'
        m_value -- saves 'm_values.pkl'
        meth -- saves uncorrected 'meth_values.pkl' and 'unmeth_values.pkl'
        noob -- saves 'meth_noob_values.pkl' and 'unmeth_noob_values.pkl'
        copy_number -- saves 'copy_number_values.pkl'

example:
========
    this is for loading a bunch of processed csv files into containers, then into betas
    ```
    import methylcheck as m
    files = '/Volumes/202761400007'
    containers = m.load(files, 'meth')
    df = m.load_processed.container_to_beta(containers)
    ```
    """
    def calculate_beta_value(methylated_noob, unmethylated_noob, offset=100):
        """ the ratio of (methylated_intensity / total_intensity)
        where total_intensity is (meth + unmeth + 100) -- to give a score in range of 0 to 1.0"""
        methylated = max(methylated_noob, 0)
        unmethylated = max(unmethylated_noob, 0)
        total_intensity = methylated + unmethylated + offset
        with np.errstate(all='raise'):
            intensity_ratio = np.true_divide(methylated, total_intensity)
        return intensity_ratio

    def calculate_m_value(methylated_noob, unmethylated_noob, offset=1):
        """ the log(base 2) (1+meth / (1+unmeth_ intensities (with an offset to avoid divide-by-zero-errors)"""
        methylated = methylated_noob + offset
        unmethylated = unmethylated_noob + offset
        with np.errstate(all='raise'):
            intensity_ratio = np.true_divide(methylated, unmethylated)
        return np.log2(intensity_ratio)

    def calculate_copy_number(methylated_noob, unmethylated_noob):
        """ the log(base 2) of the combined (meth + unmeth AKA green and red) intensities """
        total_intensity = methylated_noob + unmethylated_noob
        copy_number = np.log2(total_intensity)
        return copy_number

    def export_meth(methylated, unmethylated):
        return

    def export_noob(methylated_noob, unmethylated_noob):
        return

    functions = {
        'betas': calculate_beta_value,
        'm_value': calculate_m_value,
        'copy_number': calculate_copy_number,
        'meth': export_meth,
        'noob': export_noob,
    }
    vectorized_func = np.vectorize(functions[output])

    df = pd.DataFrame(index=containers[0]._SampleDataContainer__data_frame.index)
    df2 = pd.DataFrame(index=containers[0]._SampleDataContainer__data_frame.index)
    for i,obj in enumerate(containers):
        sample = containers[i].sample
        if output in ('betas','m_value','copy_number'):
            # .values returns the np-array for a 50X increase in speed
            meth = obj._SampleDataContainer__data_frame['meth'].values
            unmeth = obj._SampleDataContainer__data_frame['unmeth'].values
            # adds one column with betas for this sample to the growing DF.
            df[sample] = vectorized_func(meth, unmeth)
        elif output == 'meth':
            meth = obj._SampleDataContainer__data_frame['meth'].values
            unmeth = obj._SampleDataContainer__data_frame['unmeth'].values
            df[sample] = meth
            df2[sample] = unmeth
        elif output == 'noob':
            meth = obj._SampleDataContainer__data_frame['noob_meth'].values
            unmeth = obj._SampleDataContainer__data_frame['noob_unmeth'].values
            df[sample] = meth
            df2[sample] = unmeth

    filenames = {
        'betas': 'beta_values.pkl',
        'm_value': 'm_values.pkl',
        'copy_number': 'copy_number_values.pkl',
        'meth': ['meth_values.pkl', 'unmeth_values.pkl'],
        'noob': ['meth_noob_values.pkl','unmeth_noob_values.pkl'],
    }
    if save:
        files = filenames[output]
        if type(files) is list:
            df.to_pickle(files[0]) # meth
            df2.to_pickle(files[1]) # unmeth
            print(f"wrote {files[0]}, {files[1]}")
        else:
            df.to_pickle(filenames[output])
            print(f"wrote {filenames[output]}")
    if type(filenames[output]) is list:
        return df, df2
    else:
        return df


def _data_source_type(data_source):
    """ determines the type of data_source (path, containers, or meth/unmeth from pickles).
    returns a pair: (data_source_type, object)
    where data_source_type is one of {'path', 'container', 'control', 'meth_unmeth_tuple'} """
    data_source_type = type(data_source)
    dummy_container = SampleDataContainer(pd.DataFrame(), pd.DataFrame(), 'test')

    if data_source_type in [type(Path()), str]:
        # filepath
        path = Path(data_source)
        return ('path', path)

    elif (data_source_type is dict and
        all([type(df) is type(pd.DataFrame()) for df in data_source.values()]) ):
        # control_probes are a dict of dataframes
        return ('control',data_source)

    elif (data_source_type is list and
        type(data_source[0]) is type(dummy_container) and
        hasattr(data_source[0],'_SampleDataContainer__data_frame') and
        all([type(df._SampleDataContainer__data_frame) is type(pd.DataFrame()) for df in data_source]) ):
        # containers are a list of SampleDataContainer objects
        return ('container', data_source)

    elif (data_source_type is tuple and
        len(data_source) == 2 and
        type(data_source[0]) is type(pd.DataFrame()) and
        type(data_source[1]) is type(pd.DataFrame()) ):
        return ('meth_unmeth_tuple', data_source)

    else:
        raise ValueError("Unknown data structure.")


def load_sesame(filepath='.',
    format='sesame',
    file_stem='',
    verbose=False,
    silent=False,
    column_names=None,
    no_poobah=False,
    pval_cutoff=0.05,
    # these are not passed in from .load() but can be overridden if calling this directly.
    index_column='Probe_ID',
    beta_column='ind_beta',
    poobah_column='ind_poob'
    ):
    """ called within .load() for loading sesame-processed samples in csvs (optionally gzipped).
    returns a dataframe of betas, with failing probes filtered out (unless overridden). """
    SESAME_RGLOB_PATTERN = '*_R0[0-9]C0[0-9]*_calls.csv*'
    SESAME_FILENAME_REGEX = '(.*_R0[0-9]C0[0-9]).*_calls\.csv(\.gz)?'

    # files are Sentrix_ID ... manifest code ... _calls.csv.gz
    total_parts = list(Path(filepath).rglob(SESAME_RGLOB_PATTERN))
    if verbose and not silent:
        print(len(total_parts),'files matched')
    if total_parts != []:
        sample_betas = []
        sample_names = []
        columns = [index_column, beta_column, poobah_column]
        for part in (tqdm(total_parts, desc='Files', total=len(total_parts)) if (not verbose and not silent) else total_parts):
            sample = pd.read_csv(part,
                # this param assigns the index from one column
                index_col=index_column,
                # usecols param speeds up reading
                usecols=columns,
                # compression='infer' by default; works for .gz files
                dtype={index_column: str,
                    beta_column: np.float16,
                    poobah_column: np.float16
                    },
                engine='c', #fastest
                memory_map=True, # load all into memory at once for faster reading (less IO)
                #na_filter=False, # disable to speed read, if not expecting NAs
            )
            if no_poobah == False:
                # remove all failed probes by replacing with NaN before building DF.
                sample.loc[sample[poobah_column] >= pval_cutoff, beta_column] = np.nan

            fname = str(Path(part).name)
            pattern = re.match(SESAME_FILENAME_REGEX, fname, re.I)
            if pattern and len(pattern.groups()) >= 1:
                sample_name = pattern.groups(1)
            else:
                sample_name = ''

            col = sample.loc[:, [beta_column]]
            col.rename(columns={'beta_value': sample_name}, inplace=True)
            sample_names.append(sample_name)
            sample_betas.append(col)
            if verbose and not silent:
                print(f'{sample_name}, {col.shape} --> {len(sample_betas)}')
        if len(sample_betas) > 30:
            tqdm.pandas()
            df = pd.concat(sample_betas, axis='columns', join='inner').progress_apply(lambda x: x)
        else:
            df = pd.concat(sample_betas, axis='columns', join='inner')
        return df
    else:
        print("No sesame files found.")
