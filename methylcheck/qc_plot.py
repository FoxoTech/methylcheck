import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path
import logging
#app
import methylcheck
from .progress_bar import *

LOGGER = logging.getLogger(__name__)

__all__ = ['qc_signal_intensity', 'plot_M_vs_U', 'plot_controls']

def qc_signal_intensity(data_containers=None, path=None, noob=True, silent=False, verbose=False, plot=True, bad_sample_cutoff=10.5):
    """
input:
    PATH to csv files processed using methylprep
        these have "noob_meth" and "noob_unmeth" columns per sample file this function can use.
        if you want it to processed data uncorrected data.

    data_containers = run_pipeline(data_dir = '../../junk-drawer-analyses/geneseek_data/archive_32_blood/',
                                       save_uncorrected=True,
                                       sample_sheet_filepath='../../junk-drawer-analyses/geneseek_data/archive_32_blood/SampleSheet.csv')

optional params:
    bad_sample_cutoff (default 10.5): set the cutoff for determining good vs bad samples, based on signal intensities of meth and unmeth fluorescence channels. 10.5 was borrowed from minfi's internal defaults.
    noob: use noob-corrected meth/unmeth values
    verbose: additional messages
    plot: if True (default), shows a plot. if False, this function returns the median values per sample of meth and unmeth probes.
    compare: if the processed data contains both noob and uncorrected values, it will plot both in different colors

this will draw a diagonal line on plots

FIX:
    doesn't return both types of data if using compare and not plotting
    doesn't give good error message for compare

    """
    if not path and not data_containers:
        print("You must specify a path to methylprep processed data files or provide a data_containers object as input.")
        return
    meth, unmeth = _get_data(data_containers, path, compare=False, noob=noob, verbose=verbose)

    # Plotting
    medians = _make_qc_df(meth,unmeth)
    cutoffs = (medians.mMed.values + medians.uMed.values)/2
    bad_samples = medians.index[cutoffs < bad_sample_cutoff]

    # flex the x and y axes depending on the data
    min_x = int(min(medians.mMed))
    max_x = max(medians.mMed) + 1
    min_y = int(min(medians.uMed))
    max_y = max(medians.uMed) + 1

    if not plot:
        return {
            'medians': medians,
            'cutoffs': cutoffs,
            'good_samples': [str(s) for s in medians.index[cutoffs >= bad_sample_cutoff]],
            'bad_samples': [str(s) for s in bad_samples],
            'bad_sample_cutoff': bad_sample_cutoff,
        }
    # set up figure
    fig,ax = plt.subplots(figsize=(10,10))
    plt.grid()
    plt.title('M versus U plot')
    plt.xlabel('Meth Median Intensity (log2)')
    plt.ylabel('Unmeth Median Intensity (log2)')
    # bad values
    plt.scatter(x='mMed',y='uMed',data=medians[medians.index.isin(bad_samples)],label='Bad Samples',c='red')
    # good values
    plt.scatter(x='mMed',y='uMed',data=medians[~medians.index.isin(bad_samples)],label="Good Samples",c='black')
    plt.xlim([min_x,max_x])
    plt.ylim([min_y,max_y])
    # cutoff line
    x = np.linspace(6,14)
    y = -1*x+(2*bad_sample_cutoff)
    plt.plot(x, y, '--', lw=1, color='black', alpha=0.75, label='Cutoff')
    # legend
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # display plot
    plt.show()
    # print list of bad samples for user
    print('List of Bad Samples')
    print([str(s) for s in bad_samples])


def _make_qc_df(meth,unmeth):
    """Function takes meth and unmeth dataframes,
    returns a single dataframe with log2 medians for
    m and u values"""
    mmed = pd.DataFrame(np.log2(meth.median(axis=0)),columns=['mMed'])
    umed = pd.DataFrame(np.log2(unmeth.median(axis=0)),columns=['uMed'])
    qc = pd.merge(left=mmed,
               right=umed,
               left_on=mmed.index,
               right_on=umed.index,
               how='inner').set_index('key_0',drop=True)
    del qc.index.name
    return qc


def _get_data(data_containers=None, path=None, compare=False, noob=True, verbose=True):
    """ internal function that loads data from object or path and returns 2 or 4 dataframes """
    # NOTE: not a flexible function because it returns 0, 2, or 4 objects depending on inputs.
    # NOTE: this requires that data_containers label the index 'IlmnID' for each sample
    if data_containers:
        # Pull M and U values
        meth = pd.DataFrame(index=data_containers[0]._SampleDataContainer__data_frame.index)
        unmeth = pd.DataFrame(index=data_containers[0]._SampleDataContainer__data_frame.index)

        for i,c in enumerate(data_containers):
            sample = data_containers[i].sample
            m = c._SampleDataContainer__data_frame.rename(columns={'meth':sample})
            u = c._SampleDataContainer__data_frame.rename(columns={'unmeth':sample})
            meth = pd.merge(left=meth,right=m[sample],left_on='IlmnID',right_on='IlmnID',)
            unmeth = pd.merge(left=unmeth,right=u[sample],left_on='IlmnID',right_on='IlmnID')
    elif path:
        n = 'noob_' if noob else ''
        sample_filenames = []
        csvs = []
        files_found = False
        for file in tqdm(Path(path).expanduser().rglob('*_processed.csv'), desc='Loading files', total=len(list(Path(path).expanduser().rglob('*_processed.csv')))):
            this = pd.read_csv(file)
            files_found = True
            if f'{n}meth' in this.columns and f'{n}unmeth' in this.columns:
                csvs.append(this)
                sample_filenames.append(str(file.stem).replace('_processed',''))
        # note, this doesn't give a clear error message if using compare and missing uncorrected data.
        if verbose and len(csvs) > 0:
            print(f"{len(csvs)} processed samples found.")

        if csvs != []:
            meth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 0: csvs[0][f'{n}meth']})
            unmeth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 0: csvs[0][f'{n}unmeth']})
            meth.set_index('IlmnID', inplace=True)
            unmeth.set_index('IlmnID', inplace=True)
            if compare:
                n2 = '' if noob else 'noob_'
                _meth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 0: csvs[0][f'{n2}meth']})
                _unmeth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 0: csvs[0][f'{n2}unmeth']})
                _meth.set_index('IlmnID', inplace=True)
                _unmeth.set_index('IlmnID', inplace=True)
            for idx, sample in tqdm(enumerate(csvs[1:],1), desc='Samples', total=len(csvs)):
                # columns are meth, unmeth OR noob_meth, noob_unmeth, AND IlmnID
                meth = pd.merge(left=meth, right=sample[f'{n}meth'], left_on='IlmnID', right_on=sample['IlmnID'])
                meth = meth.rename(columns={f'{n}meth': sample_filenames[idx]})
                unmeth = pd.merge(left=unmeth, right=sample[f'{n}unmeth'], left_on='IlmnID', right_on=sample['IlmnID'])
                unmeth = unmeth.rename(columns={f'{n}unmeth': sample_filenames[idx]})
                if compare:
                    _meth = pd.merge(left=_meth, right=sample[f'{n2}meth'], left_on='IlmnID', right_on=sample['IlmnID'])
                    _meth = _meth.rename(columns={f'{n2}meth': sample_filenames[idx]})
                    _unmeth = pd.merge(left=_unmeth, right=sample[f'{n2}unmeth'], left_on='IlmnID', right_on=sample['IlmnID'])
                    _unmeth = _unmeth.rename(columns={f'{n2}unmeth': sample_filenames[idx]})
        else:
            print(f"{len(csvs)} processed samples found in {path} using NOOB: {noob}.")
            if files_found:
                data_columns = "NOOB meth/unmeth" if noob else "non-NOOB-corrected meth/unmeth"
                print(f"processed files found, but did not contain the right data ({data_columns})")
            return
    if compare:
        return meth, unmeth, _meth, _unmeth
    return meth, unmeth


def plot_M_vs_U(data_containers_or_path=None, noob=True, silent=False, verbose=False, plot=True, compare=False):
    """
input:
    PATH to csv files processed using methylprep
        these have "noob_meth" and "noob_unmeth" columns per sample file this function can use.
        if you want it to processed data uncorrected data.
    OR
    data_containers = run_pipeline(data_dir = '../../junk-drawer-analyses/geneseek_data/archive_32_blood/',
                                       save_uncorrected=True,
                                       sample_sheet_filepath='../../junk-drawer-analyses/geneseek_data/archive_32_blood/SampleSheet.csv')

optional params:
    noob: use noob-corrected meth/unmeth values
    verbose: additional messages
    plot: if True (default), shows a plot. if False, this function returns the median values per sample of meth and unmeth probes.
    compare: if the processed data contains both noob and uncorrected values, it will plot both in different colors

this will draw a diagonal line on plots

FIX:
    doesn't return both types of data if using compare and not plotting
    doesn't give good error message for compare

    """
    if type(data_containers_or_path) == type(Path()):
        path = data_containers_or_path
        data_containers = None
    else:
        path = None
        data_containers = data_containers_or_path

    if not path and not data_containers:
        print("You must specify a path to methylprep processed data files or provide a data_containers object as input.")
        return
    try:
        if compare:
            meth, unmeth, _meth, _unmeth = _get_data(data_containers, path, compare=compare)
        else:
            meth, unmeth = _get_data(data_containers, path, compare=compare)
    except Exception as e:
        print(e)
        print("No processed data found.")
        return

    if plot:
        # plot it
        fig,ax = plt.subplots(figsize=(10,10))
        plt.grid()
        this = sb.scatterplot(x=meth.median(),y=unmeth.median(),s=75)
        if compare:
            # combine both and reference each
            if not silent:
                print(f'Blue: {"noob" if noob else "uncorrected"}')
            sb.scatterplot(x=_meth.median(),y=_unmeth.median(),s=75)
        plt.title('M versus U plot')
        plt.xlabel('Median Methylated Intensity')
        plt.ylabel('Median Unmethylated Intensity')

        # add diagonal line
        line = {'y': this.axes.get_ylim(), 'x': this.axes.get_xlim()}
        sx = []
        sy = []
        for i in range(1000):
            sx.append(line['x'][0] + i/1000*(line['x'][1] - line['x'][0]))
            sy.append(line['y'][0] + i/1000*(line['y'][1] - line['y'][0]))
        sb.scatterplot(x=sx, y=sy, s=3)
    else:
        return {'meth_median': meth.median(), 'unmeth_median': unmeth.median()}


def plot_beta_by_type(beta_df, probe_type='I'):
    """plotBetasByType (adopted from genome studio p. 43)

Plot the overall density distribution of beta values and the density distributions of the Infinium I or II probe types
1 distribution plot; user defines type (I or II infinium)

    Doesn't work with 27k arrays because they are all of the same type, Infinium Type I.
    """
    probe_types = ('I', 'II') # 'SnpI', 'Control' are in manifest, but not in the processed data
    if probe_type not in probe_types:
        raise ValueError(f"Please specify an Infinium probe_type: ({probe_types}) to plot")
    # get manifest data from .methylprep_manifest_files
    from methylprep.files.manifests import MANIFEST_DIR_PATH, ARRAY_TYPE_MANIFEST_FILENAMES, Manifest
    from methylprep.models.arrays import ArrayType

    # orient
    if beta_df.shape[1] > beta_df.shape[0]:
        beta_df = beta_df.transpose() # probes should be in rows.
    # THIS WILL need to be updated if new arrays are added.
    #array_type = ArrayType.from_probe_count() --- this one doesn't match the probe counts we see
    lookup = {
        '450k': ArrayType.ILLUMINA_450K,
        'epic': ArrayType.ILLUMINA_EPIC}
    array_type = 'epic' if len(beta_df) > 500000 else '450k'
    array_type = lookup[array_type]
    man_path = Path(MANIFEST_DIR_PATH).expanduser()
    man_filename = ARRAY_TYPE_MANIFEST_FILENAMES[array_type]
    man_filepath = Path(man_path, man_filename)
    if Path.exists(man_filepath):
        manifest = Manifest(array_type, man_filepath)
    else:
        print("Error: manifest file not found.")
        return
    # II, I, SnpI, Control
    # merge reference col, filter probes, them remove ref col
    orig_shape = beta_df.shape
    mapper = manifest._Manifest__data_frame['probe_type']
    beta_df = beta_df.merge(mapper, right_index=True, left_index=True)
    subset = beta_df[beta_df['probe_type'] == probe_type]
    subset = subset.drop('probe_type', axis='columns')
    print(f'Found {subset.shape[0]} type {probe_type} probes.')
    methylcheck.sample_plot(subset)


def plot_controls(path=None, subset='all'):
    """
    - uses control probe values to plot staining controls (available with the `--save_control` methylprep process option)

options:
========
    subset ('staining' | 'negative' | 'hybridization' | 'extension' | 'bisulfite' |
            'non-polymorphic' | 'target-removal' | 'specificity' | 'all'):
    'all' will plot every control function (default)
    """
    subset_options = {'staining', 'negative', 'hybridization', 'extension', 'bisulfite', 'non-polymorphic', 'target-removal', 'specificity', 'all'}
    if subset not in subset_options:
        raise ValueError(f"Choose one of these options for plot type: {subset_options}")
    if not path:
        print("You must specify a path to the control probes processed data file or folder (available with the `--save_control` methylprep process option).")
        return
    try:
        if path.is_dir():
            control = pd.read_pickle(Path(path, 'control_probes.pkl'))
        elif path.is_file():
            control = pd.read_pickle(path) # allows for any arbitrary filename to be used, so long as structure is same, and it is a pickle.
    except Exception as e: # cannot unpack NoneType
        print(e)
        print("No data.")
        return

    # Create empty dataframes for red and green negative controls
    control_R = pd.DataFrame(list(control.values())[0][['Control_Type','Color','Extended_Type']])
    control_G = pd.DataFrame(list(control.values())[0][['Control_Type','Color','Extended_Type']])
    # convert the list of DFs into one DF for each red and green channel
    for sample,c in control.items():
        # drop SNPS from control DF using Control_Type column.
        c = c[c['Control_Type'].notna() == True]
        df_red = c[['Extended_Type','Mean_Value_Red']].rename(columns={'Mean_Value_Red':sample})
        df_green = c[['Extended_Type','Mean_Value_Green']].rename(columns={'Mean_Value_Green':sample})
        control_R = pd.merge(left=control_R,right=df_red,on=['Extended_Type'])
        control_G = pd.merge(left=control_G,right=df_green,on=['Extended_Type'])

    if subset in ('staining','all'):
        stain_red = control_R[control_R['Control_Type']=='STAINING'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        stain_green = control_G[control_G['Control_Type']=='STAINING'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(stain_green.Extended_Type, stain_green.Color))
        stain_green = stain_green.drop(columns=['Color']).set_index('Extended_Type')
        stain_red = stain_red.drop(columns=['Color']).set_index('Extended_Type')
        stain_red = stain_red.T
        stain_green = stain_green.T
        _qc_plotter(stain_red, stain_green, color_dict, ymax=50000, title='Staining')

    if subset in ('negative','all'):
        neg_red = control_R[control_R['Control_Type']=='NEGATIVE'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        neg_green= control_G[control_G['Control_Type']=='NEGATIVE'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(neg_green.Extended_Type, neg_green.Color))
        neg_green = neg_green.drop(columns=['Color']).set_index('Extended_Type')
        neg_red = neg_red.drop(columns=['Color']).set_index('Extended_Type')
        neg_red = neg_red.T
        neg_green = neg_green.T
        # GenomeStudio appears to only do the first 16
        # Maybe user should be able to select which they want to see
        # There is a total of 600, which is too many to plot at once
        list_of_negative_controls_to_plot = ['Negative 1','Negative 2','Negative 3','Negative 4','Negative 5',
                                             'Negative 6','Negative 7','Negative 8','Negative 9','Negative 10',
                                             'Negative 11','Negative 12','Negative 13','Negative 14','Negative 15',
                                             'Negative 16']
        _qc_plotter(neg_red, neg_green, color_dict, columns=list_of_negative_controls_to_plot, title='Negative')

    if subset in ('hybridization','all'):
        hyb_red   = control_R[control_R['Control_Type']=='HYBRIDIZATION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        hyb_green = control_G[control_G['Control_Type']=='HYBRIDIZATION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(hyb_green.Extended_Type, hyb_green.Color))
        hyb_green = hyb_green.drop(columns=['Color']).set_index('Extended_Type')
        hyb_red = hyb_red.drop(columns=['Color']).set_index('Extended_Type')
        hyb_red = hyb_red.T
        hyb_green = hyb_green.T
        _qc_plotter(hyb_red, hyb_green, color_dict, ymax=35000, title='Hybridization')

    if subset in ('extension','all'):
        ext_red   = control_R[control_R['Control_Type']=='EXTENSION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        ext_green = control_G[control_G['Control_Type']=='EXTENSION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(ext_green.Extended_Type, ext_green.Color))
        ext_green = ext_green.drop(columns=['Color']).set_index('Extended_Type')
        ext_red = ext_red.drop(columns=['Color']).set_index('Extended_Type')
        ext_red = ext_red.T
        ext_green = ext_green.T
        _qc_plotter(ext_red, ext_green, color_dict, ymax=50000, title='Extension')

    if subset in ('bisulfite','all'):
        bci_red   = control_R[control_R['Control_Type']=='BISULFITE CONVERSION I'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        bci_green = control_G[control_G['Control_Type']=='BISULFITE CONVERSION I'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(bci_green.Extended_Type, bci_green.Color))
        bci_green = bci_green.drop(columns=['Color']).set_index('Extended_Type')
        bci_red = bci_red.drop(columns=['Color']).set_index('Extended_Type')
        bci_red = bci_red.T
        bci_green = bci_green.T
        _qc_plotter(bci_red, bci_green, color_dict, ymax=30000, title='Bisulfite Conversion')

    if subset in ('non-polymorphic','all'):
        np_red = control_R[control_R['Control_Type']=='NON-POLYMORPHIC'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        np_green = control_G[control_G['Control_Type']=='NON-POLYMORPHIC'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(np_green.Extended_Type, np_green.Color))
        np_green = np_green.drop(columns=['Color']).set_index('Extended_Type')
        np_red = np_red.drop(columns=['Color']).set_index('Extended_Type')
        np_red = np_red.T
        np_green = np_green.T
        _qc_plotter(np_red, np_green, color_dict, ymax=20000, title='Non-polymorphic')

    if subset in ('target-removal','all'):
        tar_red = control_R[control_R['Control_Type']=='TARGET REMOVAL'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        tar_green = control_G[control_G['Control_Type']=='TARGET REMOVAL'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(tar_green.Extended_Type, tar_green.Color))
        tar_green = tar_green.drop(columns=['Color']).set_index('Extended_Type')
        tar_red = tar_red.drop(columns=['Color']).set_index('Extended_Type')
        tar_red = tar_red.T
        tar_green = tar_green.T
        _qc_plotter(tar_red, tar_green, color_dict, ymax=2000, title='Target Removal')

    if subset in ('specificity','all'):
        spec_red = control_R[control_R['Control_Type']=='SPECIFICITY I'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        spec_green = control_G[control_G['Control_Type']=='SPECIFICITY I'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(spec_green.Extended_Type, spec_green.Color))
        spec_green = spec_green.drop(columns=['Color']).set_index('Extended_Type')
        spec_red = spec_red.drop(columns=['Color']).set_index('Extended_Type')
        spec_red = spec_red.T
        spec_green = spec_green.T
        _qc_plotter(spec_red, spec_green, color_dict, ymax=30000, title='Specificity (Type I)')


def _qc_plotter(stain_red, stain_green, color_dict={}, columns=None, ymax=None,
        title=''):
    """ draft generic plotting function for all the genome studio QC functions.
    used by plot_staining_controls()

options:
========
    required: stain_red and stain_green
        have red/green values in columns and probe characteristics in rows (transposed from pickle format).
    color_dict
        can be passed in: defines which color to make each value in the index.
    columns
        list of columns(probes) in stain_red and stain_green to plot (if ommitted it plots everything)."""
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,10))
    plt.tight_layout(w_pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    ax1.grid(axis='both')
    ax2.grid(axis='both')
    title = title + ' ' if title != '' else title
    ax1.set_title(f'{title}Green')
    ax2.set_title(f'{title}Red')

    if columns != None:
        stain_red = stain_red.loc[:, columns]
        stain_green = stain_green.loc[:, columns]
    for c in stain_red.columns:
        ax1.plot(stain_green.index,
                 c,
                 data=stain_green, label=c,
                 color=color_dict[c], linewidth=0, marker='o')

        ax2.plot(stain_red.index,
                 c,
                 data=stain_red, label=c,
                 color=color_dict[c], linewidth=0, marker='o')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if ymax != None:
        ax1.set_ylim([0,ymax])
        ax2.set_ylim([0,ymax])

    plt.show()
