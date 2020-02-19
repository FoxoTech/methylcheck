import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path
import logging
#app
from .progress_bar import *

LOGGER = logging.getLogger(__name__)

__all__ = ['qc_signal_intensity', 'plot_M_vs_U']

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
    if data_containers:
        # Pull M and U values
        meth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)
        unmeth = pd.DataFrame(data_containers[0]._SampleDataContainer__data_frame.index)

        for i,c in enumerate(data_containers):
            sample = data_containers[i].sample
            m = c._SampleDataContainer__data_frame.rename(columns={'meth':sample})
            u = c._SampleDataContainer__data_frame.rename(columns={'unmeth':sample})
            meth = pd.merge(left=meth,right=m[sample],left_on='IlmnID',right_on='IlmnID',)
            unmeth = pd.merge(left=unmeth,right=u[sample],left_on='IlmnID',right_on='IlmnID')
    elif path:
        n = 'noob_' if noob else ''
        csvs = []
        for file in Path(path).expanduser().rglob('*processed.csv'):
            this = pd.read_csv(file)
            if f'{n}meth' in this.columns and f'{n}unmeth' in this.columns:
                csvs.append(this)
        # note, this doesn't give a clear error message if using compare and missing uncorrected data.
        if verbose:
            print(f"{len(csvs)} processed samples found.")
        if csvs != []:
            meth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 1: csvs[0][f'{n}meth']})
            unmeth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 1: csvs[0][f'{n}unmeth']})
            if compare:
                n2 = '' if noob else 'noob_'
                _meth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 1: csvs[0][f'{n2}meth']})
                _unmeth = pd.DataFrame({'IlmnID': csvs[0]['IlmnID'], 1: csvs[0][f'{n2}unmeth']})
            for idx, sample in tqdm(enumerate(csvs[1:],2), desc='Samples', total=len(csvs)):
                # columns are meth, unmeth OR noob_meth, noob_unmeth, AND IlmnID
                meth = pd.merge(left=meth, right=sample[f'{n}meth'], left_on='IlmnID', right_on=sample['IlmnID'])
                meth = meth.rename(columns={f'{n}meth': idx})
                unmeth = pd.merge(left=unmeth, right=sample[f'{n}unmeth'], left_on='IlmnID', right_on=sample['IlmnID'])
                unmeth = unmeth.rename(columns={f'{n}unmeth': idx})
                if compare:
                    _meth = pd.merge(left=_meth, right=sample[f'{n2}meth'], left_on='IlmnID', right_on=sample['IlmnID'])
                    _meth = _meth.rename(columns={f'{n2}meth': idx})
                    _unmeth = pd.merge(left=_unmeth, right=sample[f'{n2}unmeth'], left_on='IlmnID', right_on=sample['IlmnID'])
                    _unmeth = _unmeth.rename(columns={f'{n2}unmeth': idx})
        else:
            print("No processed data found.")
            return
    if compare:
        return meth, unmeth, _meth, _unmeth
    return meth, unmeth


def plot_M_vs_U(data_containers=None, path=None, noob=True, silent=False, verbose=False, plot=True, compare=False):
    """
input:
    PATH to csv files processed using methylprep
        these have "noob_meth" and "noob_unmeth" columns per sample file this function can use.
        if you want it to processed data uncorrected data.

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
    if not path and not data_containers:
        print("You must specify a path to methylprep processed data files or provide a data_containers object as input.")
        return
    try:
        if compare:
            meth, unmeth, _meth, _unmeth = _get_data(data_containers, path, compare=compare)
        else:
            meth, unmeth = _get_data(data_containers, path, compare=compare)
    except Exception as e:
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


def plot_staining_controls(path):
    """
    -- Looks at df to determine which manifest file to load.
    -- collects control probe names from manifest
    -- uses uncorrected sample values to plot staining controls

    (the methylated signal is measured in the green channel and the unmethylated signal in the red channel)
    so red=unmeth and grn=meth
    """
    # get control probes list from manifest

    # make data_containers from path
    meth, unmeth = _get_data(data_containers=None, path=path, compare=False, noob=False, verbose=True)
    # rows are probes
    from methylprep.models import ArrayType, Channel
    from methylprep.files.manifests import Manifest

    if sum([len(m) for m in meth])/len(meth) > 500000:
        array_type = ArrayType('epic')
    else:
        array_type = ArrayType('450k') # control probes are same for epic+

    manifest = Manifest(array_type)
    control_probes = manifest.control_data_frame
    probe_index = meth.index

    data_containers = []
    for idx, m in enumerate(meth):
        container = pd.DataFrame(data=[m], index=probe_index, columns=['meth'])

        def get_fg_controls(self, manifest, channel): # red or grn
            #channel_means = self.get_channel_means(channel)
            if channel is Channel.GREEN:
                return self.green_idat.probe_means
            return self.red_idat.probe_means

            return inner_join_data(control_probes, channel_means)

    # Create empty dataframes for red and green negative controls
    negctlsR = pd.DataFrame(data_containers[0].ctrl_red[['Control_Type','Color','Extended_Type']])
    negctlsG = pd.DataFrame(data_containers[0].ctrl_green[['Control_Type','Color','Extended_Type']])

    # Fill red and green dataframes
    for i,c in enumerate(data_containers):
        sample = str(data_containers[i].sample)
        dfR = c.ctrl_red
        dfR = dfR[['Extended_Type','mean_value']].rename(columns={'mean_value':sample})
        dfG = c.ctrl_green
        dfG = dfG[['Extended_Type','mean_value']].rename(columns={'mean_value':sample})
        negctlsR = pd.merge(left=negctlsR,right=dfR,on=['Extended_Type'])
        negctlsG = pd.merge(left=negctlsG,right=dfG,on=['Extended_Type'])

    # reformat data frames for plotting ease
    stain_green = negctlsG[negctlsG['Control_Type']=='STAINING'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
    stain_red = negctlsR[negctlsR['Control_Type']=='STAINING'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
    color_dict  = dict(zip(stain_green.Extended_Type, stain_green.Color))
    stain_green = stain_green.drop(columns=['Color']).set_index('Extended_Type')
    stain_red = stain_red.drop(columns=['Color']).set_index('Extended_Type')
    stain_red = stain_red.T
    stain_green = stain_green.T


def _qc_plotter():
    """ draft generic plotting function for all the genome studio QC functions """
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,10))
    plt.tight_layout(w_pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    ax1.grid(axis='both')
    ax2.grid(axis='both')
    ax1.set_title('Staining Green')
    ax2.set_title('Staining Red')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim([0,50000])
    ax2.set_ylim([0,50000])

    plt.show()
