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

__all__ = ['run_qc', 'plot_beta_by_type', 'qc_signal_intensity', 'plot_M_vs_U', 'plot_controls']

def run_qc(path):
    """Generates all QC plots for a dataset in the path provided.
    if `process --all` was used to create control probes and raw values for QC,
    because it uses four output files:

    - beta_values.pkl
    - control_probes.pkl
    - meth_values.pkl or noob_meth_values.pkl
    - unmeth_values.pkl or noob_unmeth_values.pkl

    output is all to screen, so best to use in a jupyter notebook.
    If you prefer output in a PDF, use ReportPDF instead.

    Note: this will only look in the path folder; it doesn't do a recursive search for matching files.
    """
    try:
        beta_df = pd.read_pickle(Path(path,'beta_values.pkl'))
        controls = pd.read_pickle(Path(path,'control_probes.pkl'))
        if Path(path,'meth_values.pkl').exists() and Path(path,'unmeth_values.pkl').exists():
            meth_df = pd.read_pickle(Path(path,'meth_values.pkl'))
            unmeth_df = pd.read_pickle(Path(path,'unmeth_values.pkl'))
        else:
            meth_df = pd.read_pickle(Path(path,'noob_meth_values.pkl'))
            unmeth_df = pd.read_pickle(Path(path,'noob_unmeth_values.pkl'))
        if Path(path,'poobah_values.pkl').exists():
            poobah = pd.read_pickle(Path(path,'poobah_values.pkl'))
        else:
            poobah = None
    except FileNotFoundError:
        if not Path(path).exists():
            raise FileNotFoundError("Invalid path")
        elif not Path(path).is_dir():
            raise FileNotFoundError("Path is not a directory.")
        raise FileNotFoundError("Files missing. run_qc() only works if you used `methylprep process --all` option to produce beta_values, control_probes, meth_values, and unmeth_values files.")
    # needs meth_df, unmeth_df, controls, and beta_df
    # if passing in a path, it will auto-search for poobah. but if meth/unmeth passed in, you must explicitly tell it to look.
    plot_M_vs_U(meth=meth_df, unmeth=unmeth_df, poobah=poobah)
    qc_signal_intensity(meth=meth_df, unmeth=unmeth_df, poobah=poobah)
    plot_controls(controls, 'all')
    plot_beta_by_type(beta_df, 'all')


def qc_signal_intensity(data_containers=None, path=None, meth=None, unmeth=None, poobah=None, palette=None,
    noob=True, silent=False, verbose=False, plot=True, bad_sample_cutoff=10.5, return_fig=False):
    """Suggests sample outliers based on methylated and unmethylated signal intensity.

input (one of these):
=====================
    path
        to csv files processed using methylprep
        these have "noob_meth" and "noob_unmeth" columns per sample file this function can use.
        if you want it to processed data uncorrected data.

    data_containers
        output from the methylprep.run_pipeline() command when run in a script or notebook.
        you can also recreate the list of datacontainers using methylcheck.load(<filepath>,'meth')

    (meth and unmeth)
        if you chose `process --all` you can load the raw intensities like this, and pass them in:
        meth = pd.read_pickle('meth_values.pkl')
        unmeth = pd.read_pickle('unmeth_values.pkl')
        THIS will run the fastest.
    (meth and unmeth and poobah)
        if poobah=None (default): Does nothing
        if poobah=False: suppresses this color
        if poobah=dataframe: color-codes samples according to percent probe failure range,
            but only if you pass in meth and unmeth dataframes too, not data_containers object.

optional params:
================
    bad_sample_cutoff (default 10.5): set the cutoff for determining good vs bad samples, based on signal intensities of meth and unmeth fluorescence channels. 10.5 was borrowed from minfi's internal defaults.
    noob: use noob-corrected meth/unmeth values
    verbose: additional messages
    plot: if True (default), shows a plot. if False, this function returns the median values per sample of meth and unmeth probes.
    return_fig (False default), if True, and plot is True, returns a figure object instead of showing plot.
    compare: if the processed data contains both noob and uncorrected values, it will plot both in different colors
    palette: if using poobah to color code, you can specify a Seaborn palette to use.

this will draw a diagonal line on plots

FIX:
    doesn't return both types of data if using compare and not plotting
    doesn't give good error message for compare
    """
    if not path and not data_containers and type(meth) is type(None) and type(unmeth) is type(None):
        print("You must specify a path to methylprep processed data files or provide a data_containers object as input.")
        return
    if type(meth) is type(None) and type(unmeth) is type(None):
        meth, unmeth = _get_data(data_containers=data_containers, path=path, compare=False, noob=noob, verbose=verbose)
    if (path is not None and not isinstance(poobah, pd.DataFrame)
        and not isinstance(poobah, type(None))
        and verbose and not silent):
        if poobah in (False,None):
            pass # unless poobah IS a dataframe below, nothing happens. None/False suppress this
        else:
            if 'poobah_values.pkl' in [i.name for i in list(path.rglob('poobah_values.pkl'))]:
                poobah = pd.read_pickle(list(path.rglob('poobah_values.pkl'))[0])
            else:
                LOGGER.info("Cannot load poobah_values.pkl file.")

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
    plt.xlabel('Meth Median Intensity (log2)')
    plt.ylabel('Unmeth Median Intensity (log2)')
    if not isinstance(poobah, pd.DataFrame):
        plt.title('M versus U plot')
        # bad values
        plt.scatter(x='mMed',y='uMed',data=medians[medians.index.isin(bad_samples)],label='Bad Samples',c='red')
        # good values
        plt.scatter(x='mMed',y='uMed',data=medians[~medians.index.isin(bad_samples)],label="Good Samples",c='black')
    elif isinstance(poobah, pd.DataFrame):
        plt.title('M versus U plot: Colors are the percent of probe failures per sample')
        if poobah.isna().sum().sum() > 0:
            LOGGER.warning("Your poobah_values.pkl file contains missing values; color coding will be inaccurate.")
        percent_failures = round(100*( poobah[poobah > 0.05].count() / poobah.count() ),1)
        percent_failures = percent_failures.rename('probe_failure_(%)')
        # Series.where will replace the stuff that is False, so you have to negate it.
        percent_failures_hues = percent_failures.where(~percent_failures.between(0,5), 0)
        percent_failures_hues.where(~percent_failures_hues.between(5,10), 1, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(10,15), 2, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(15,20), 3, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(20,25), 4, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(25,30), 5, inplace=True)
        percent_failures_hues.where(~(percent_failures_hues > 30), 6, inplace=True)
        percent_failures_hues = percent_failures_hues.astype(int)
        #sizes = percent_failures_hues.copy()
        percent_failures_hues = percent_failures_hues.replace({0:'0 to 5', 1:'5 to 10', 2:'10 to 15', 3:'15 to 20', 4:'20 to 25', 5:'25 to 30', 6:'>30'})
        legend_order = ['0 to 5','5 to 10','10 to 15','15 to 20','20 to 25','25 to 30','>30']
        qc = pd.merge(left=medians,
           right=percent_failures_hues,
           left_on=medians.index,
           right_on=percent_failures_hues.index,
           how='inner')
        hues_palette = sb.color_palette("twilight", n_colors=7, desat=0.8) if palette is None else sb.color_palette(palette, n_colors=7, desat=0.8)
        this = sb.scatterplot(data=qc, x="mMed", y="uMed", hue="probe_failure_(%)",
            palette=hues_palette, hue_order=legend_order, legend="full") # size="size"
    else:
        raise NotImplementedError("poobah color coding is not implemented with 'compare' option")

    plt.xlim([min_x,max_x])
    plt.ylim([min_y,max_y])
    # cutoff line
    x = np.linspace(6,14)
    y = -1*x+(2*bad_sample_cutoff)
    plt.plot(x, y, '--', lw=1, color='black', alpha=0.75, label='Cutoff')
    # legend
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # display plot
    if return_fig:
        return fig
    plt.show()
    # print list of bad samples for user
    if len(bad_samples) > 0:
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
    #del qc.index.name
    qc.index.name = None
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
        # first try to load from disk
        if Path(path, 'meth_values.pkl').exists() and Path(path,'unmeth_values.pkl').exists():
            meth = pd.read_pickle(Path(path, 'meth_values.pkl'))
            unmeth = pd.read_pickle(Path(path, 'unmeth_values.pkl'))
            return meth, unmeth
        else:
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


def plot_M_vs_U(data_containers_or_path=None, meth=None, unmeth=None, poobah=None,
    noob=True, silent=False, verbose=False, plot=True, compare=False, return_fig=False, palette=None):
    """plot methylated vs unmethylated probe intensities

input (choose one of these):
============================
    PATH to csv files processed using methylprep
        these have "noob_meth" and "noob_unmeth" columns per sample file this function can use.
        if you want it to processed data uncorrected data.
        (If there is a poobah_values.pkl file in this PATH, it will use the file to color code points)

    data_containers = run_pipeline(data_dir = 'somepath',
                                       save_uncorrected=True,
                                       sample_sheet_filepath='samplesheet.csv')
        you can also recreate the list of datacontainers using methylcheck.load(<filepath>,'meth')


    (meth and unmeth)
        if you chose `process --all` you can load the raw intensities like this, and pass them in:
        meth = pd.read_pickle('meth_values.pkl')
        unmeth = pd.read_pickle('unmeth_values.pkl')
        THIS will run the fastest.

    poobah
        filepath: You may supply the file path to the p-value detection dataframe. If supplied, it will color
        code points on the plot.
        False: set poobah to False to suppress this coloring.
        None (default): if there is a poobah_values.pkl file in your path, it will use it.

optional params:
    noob: use noob-corrected meth/unmeth values
    verbose: additional messages
    plot: if True (default), shows a plot. if False, this function returns the median values per sample of meth and unmeth probes.
    return_fig: (False default), if True (and plot is true), returns the figure object instead of showing it.
    compare:
        if the processed data contains both noob and uncorrected values, it will plot both in different colors
        the compare option will not work with using the 'meth' and 'unmeth' inputs, only with path or data_containers.

this will draw a diagonal line on plots.
    the cutoff line is based on the X-Y scale of the plot, which depends on the range of intensity values in your data set.

FIX:
    doesn't return both types of data if using compare and not plotting
    doesn't give good error message for compare

    """
    try:
        if Path(data_containers_or_path).exists(): # if passing in a valid string, this should work.
            path = Path(data_containers_or_path)
        else:
            path = None
    except:
        path = None # but fails if passing in a data_containers object

    if type(data_containers_or_path) == type(Path()): #this only recognizes a Path object
        path = data_containers_or_path
        data_containers = None
    else:
        path = None
        data_containers = data_containers_or_path # by process of exclusion, this must be an object

    if not path and not data_containers and type(meth) is None and type(unmeth) is None:
        print("You must specify a path to methylprep processed data files or provide a data_containers object as input.")
        return

    # 2. load meth + unmeth from path
    elif isinstance(meth,type(None)) and isinstance(unmeth,type(None)): #type(meth) is None and type(unmeth) is None:
        try:
            if compare:
                meth, unmeth, _meth, _unmeth = _get_data(data_containers, path, compare=compare)
            else:
                meth, unmeth = _get_data(data_containers, path, compare=compare)
        except Exception as e:
            print(e)
            print("No processed data found.")
            return

    # 2. load poobah_df if exists
    if isinstance(poobah,pd.DataFrame):
        poobah_df = poobah
        poobah = True
    else:
        poobah_df = None
        if poobah is not False and isinstance(path, Path) and 'poobah_values.pkl' in [i.name for i in list(path.rglob('poobah_values.pkl'))]:
            poobah_df = pd.read_pickle(list(path.rglob('poobah_values.pkl'))[0])
            poobah=True
        else:
            if poobah_df is None: # didn't find a poobah file to load
                LOGGER.warning("Did not find a poobah_values.pkl file; unable to color-code plot.")
                poobah = False #user may have set this to True or None, but changing params to fit data.
    if verbose and not silent and isinstance(poobah_df,pd.DataFrame):
        LOGGER.info("Using poobah_values.pkl")

    #palette options to pass in: "CMRmap" "flare" "twilight" "Blues"
    hues_palette = sb.color_palette("twilight", n_colors=7, desat=0.8) if palette is None else sb.color_palette(palette, n_colors=7, desat=0.8)

    if poobah is not False and isinstance(poobah_df, pd.DataFrame) and not compare:
        if poobah_df.isna().sum().sum() > 0:
            LOGGER.warning("Your poobah_values.pkl file contains missing values; color coding will be inaccurate.")
        percent_failures = round(100*( poobah_df[poobah_df > 0.05].count() / poobah_df.count() ),1)
        percent_failures = percent_failures.rename('probe_failure (%)')
        meth_med = meth.median()
        unmeth_med = unmeth.median()
        # Series.where will replace the stuff that is False, so you have to negate it.
        percent_failures_hues = percent_failures.where(~percent_failures.between(0,5), 0)
        percent_failures_hues.where(~percent_failures_hues.between(5,10), 1, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(10,15), 2, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(15,20), 3, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(20,25), 4, inplace=True)
        percent_failures_hues.where(~percent_failures_hues.between(25,30), 5, inplace=True)
        percent_failures_hues.where(~(percent_failures_hues > 30), 6, inplace=True)
        percent_failures_hues = percent_failures_hues.astype(int)
        #sizes = percent_failures_hues.copy()
        percent_failures_hues = percent_failures_hues.replace({0:'0 to 5', 1:'5 to 10', 2:'10 to 15', 3:'15 to 20', 4:'20 to 25', 5:'25 to 30', 6:'>30'})
        legend_order = ['0 to 5','5 to 10','10 to 15','15 to 20','20 to 25','25 to 30','>30']
        df = pd.concat([
            meth_med.rename('meth'),
            unmeth_med.rename('unmeth'),
            percent_failures_hues],
            #sizes.rename('size')],
            axis=1)

    if plot:
        # plot it
        fig,ax = plt.subplots(figsize=(10,10))
        plt.grid()
        if poobah and not compare:
            this = sb.scatterplot(data=df, x="meth", y="unmeth", hue="probe_failure (%)",
                palette=hues_palette, hue_order=legend_order, legend="full") # size="size"
        elif not poobah and not compare:
            this = sb.scatterplot(x=meth.median(),y=unmeth.median(),s=75)
        elif compare:
            this = sb.scatterplot(x=meth.median(),y=unmeth.median(),s=75)
            # combine both and reference each
            if not silent:
                print(f'Blue: {"noob" if noob else "uncorrected"}')
            sb.scatterplot(x=_meth.median(),y=_unmeth.median(),s=75)
        if poobah:
            plt.title('M versus U plot: Colors are the percent of probe failures per sample')
        else:
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
        if return_fig:
            sns_scatterplot = sb.scatterplot(x=sx, y=sy, s=3)
            return sns_scatterplot.get_figure()
        else:
            sb.scatterplot(x=sx, y=sy, s=3)
    else:
        return {'meth_median': meth.median(), 'unmeth_median': unmeth.median()}


def plot_beta_by_type(beta_df, probe_type='all', return_fig=False, silent=False, on_lambda=False):
    """compare betas for type I and II probes -- (adopted from genome studio plotBetasByType(), p. 43)

Plot the overall density distribution of beta values and the density distributions of the Infinium I or II probe types
1 distribution plot; user defines type (I or II infinium)

    Doesn't work with 27k arrays because they are all of the same type, Infinium Type I.

options:
    return_fig: (default False) if True, returns a list of figure objects instead of showing plots.
    """
    mouse_probe_types = ['cg','ch']
    probe_types = ['I', 'II', 'IR', 'IG', 'all'] # 'SnpI', 'Control' are in manifest, but not in the processed data
    if probe_type not in probe_types + mouse_probe_types:
        raise ValueError(f"Please specify an Infinium probe_type: ({probe_types}) to plot or, if mouse array, one of these ({mouse_probe_types}) or 'all'.")

    # orient
    if beta_df.shape[1] > beta_df.shape[0]:
        beta_df = beta_df.transpose() # probes should be in rows.
    array_type, man_filepath = methylcheck.detect_array(beta_df, returns='filepath', on_lambda=on_lambda)
    # note that 'array_type' can look like string 'mouse' but only str(array_type) will match the string 'mouse'

    if Path.exists(man_filepath):
        try:
            from methylprep import Manifest, ArrayType
        except ImportError:
            raise ImportError("this required methylprep")

        manifest = Manifest(ArrayType(array_type), man_filepath, on_lambda=on_lambda)
    else:
        raise FileNotFoundError("manifest file not found.")

    # merge reference col, filter probes, them remove ref col(s)
    orig_shape = beta_df.shape
    # II, I, IR, IG, Control
    mapper = manifest.data_frame.loc[:, ['probe_type','Color_Channel']]
    beta_df = beta_df.merge(mapper, right_index=True, left_index=True)

    figs = []
    if probe_type in ('I', 'all'):
        subset = beta_df[beta_df['probe_type'] == 'I']
        subset = subset.drop('probe_type', axis='columns')
        subset = subset.drop('Color_Channel', axis='columns')
        if return_fig:
            figs.append( methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type I probes', return_fig=True, silent=silent) )
        else:
            print(f'Found {subset.shape[0]} type I probes.')
            methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type I probes', silent=silent)
    if probe_type in ('II', 'all'):
        subset = beta_df[beta_df['probe_type'] == 'II']
        subset = subset.drop('probe_type', axis='columns')
        subset = subset.drop('Color_Channel', axis='columns')
        if return_fig:
            figs.append( methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type II probes', return_fig=True, silent=silent) )
        else:
            print(f'Found {subset.shape[0]} type II probes.')
            methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type II probes', silent=silent)
    if probe_type in ('IR', 'all'):
        subset = beta_df[(beta_df['probe_type'] == 'I') & (beta_df['Color_Channel'] == 'Red')]
        subset = subset.drop('probe_type', axis='columns')
        subset = subset.drop('Color_Channel', axis='columns')
        if return_fig:
            figs.append( methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type I Red (IR) probes', return_fig=True, silent=silent) )
        else:
            print(f'Found {subset.shape[0]} type I Red (IR) probes.')
            methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type I Red (IR) probes', silent=silent)
    if probe_type in ('IG', 'all'):
        subset = beta_df[(beta_df['probe_type'] == 'I') & (beta_df['Color_Channel'] == 'Grn')]
        subset = subset.drop('probe_type', axis='columns')
        subset = subset.drop('Color_Channel', axis='columns')
        if return_fig:
            figs.append( methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type I Green (IG) probes', return_fig=True, silent=silent) )
        else:
            print(f'Found {subset.shape[0]} type I Green (IG) probes.')
            methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} type I Green (IG) probes', silent=silent)
    if str(array_type) != 'mouse':
        if return_fig:
            return figs
        return

    ############ MOUSE ONLY ################
    # TODO: control probe types #
    # 'probe_type' are I, II, IR, IG and Probe_Type (mouse only) are 'cg','ch' | 'mu','rp','rs' are on control or mouse probes
    mapper = manifest.data_frame.loc[:, ['Probe_Type']]
    beta_df = beta_df.merge(mapper, right_index=True, left_index=True)

    if probe_type in ('cg','ch'):
        subset = beta_df[beta_df['Probe_Type'] == probe_type]
        subset = subset.drop('probe_type', axis='columns')
        subset = subset.drop('Color_Channel', axis='columns')
        subset = subset.drop('Probe_Type', axis='columns')
        if return_fig:
            figs.append( methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} {probe_type} probes', return_fig=True, silent=silent) )
        else:
            methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} {probe_type} probes', silent=silent)
    if probe_type == 'all':
        for mouse_probe_type in mouse_probe_types:
            subset = beta_df[beta_df['Probe_Type'] == mouse_probe_type]
            subset = subset.drop('probe_type', axis='columns')
            subset = subset.drop('Color_Channel', axis='columns')
            subset = subset.drop('Probe_Type', axis='columns')
            if return_fig:
                figs.append( methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} {mouse_probe_type} probes', return_fig=True, silent=silent) )
            else:
                methylcheck.beta_density_plot(subset, plot_title=f'{subset.shape[0]} {mouse_probe_type} probes', silent=silent)

    if return_fig:
        return figs

def plot_controls(path=None, subset='all', return_fig=False):
    """internal array QC controls (available with the `--save_control` or `--all` methylprep process option)


input:
======
    path
        can either be a path to the file, or a path to the folder containing a file called 'control_probes.pkl',
        or it can be the dictionary of control dataframes in `control_probes.pkl`.

options:
========
    subset ('staining' | 'negative' | 'hybridization' | 'extension' | 'bisulfite' |
            'non-polymorphic' | 'target-removal' | 'specificity' | 'all'):
    'all' will plot every control function (default)

    return_fig (False)
        if True, returns a list of matplotlib.pyplot figure objects INSTEAD of showing then. Used in QC ReportPDF.

    if there are more than 30 samples, plots will not have sample names on x-axis.
    """
    subset_options = {'staining', 'negative', 'hybridization', 'extension', 'bisulfite', 'non-polymorphic', 'target-removal', 'specificity', 'all'}
    if subset not in subset_options:
        raise ValueError(f"Choose one of these options for plot type: {subset_options}")
    if not path:
        print("You must specify a path to the control probes processed data file or folder (available with the `--save_control` methylprep process option).")
        return
    try:
        # detect a dict of dataframes (control_probes.pkl) object
        if type(path) is dict and all([type(df) is type(pd.DataFrame()) for df in path.values()]):
            control = path
            path = None
        else:
            path = Path(path)
            if path.is_dir():
                control = pd.read_pickle(Path(path, 'control_probes.pkl'))
            elif path.is_file():
                control = pd.read_pickle(path) # allows for any arbitrary filename to be used, so long as structure is same, and it is a pickle.
    except Exception as e: # cannot unpack NoneType
        print(e)
        print("No data.")
        return

    mouse = True if list(control.values())[0].shape[0] == 473 else False # vs 694 controls for epic.
    plotx = 'show' if len(list(control.keys())) <= 30 else None
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

    figs = []
    if subset in ('staining','all'):
        stain_red = control_R[control_R['Control_Type']=='STAINING'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        stain_green = control_G[control_G['Control_Type']=='STAINING'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(stain_green.Extended_Type, stain_green.Color))
        color_dict.update({k: (v if v != '-99' else 'gold') for k,v in color_dict.items()})
        stain_green = stain_green.drop(columns=['Color']).set_index('Extended_Type')
        stain_red = stain_red.drop(columns=['Color']).set_index('Extended_Type')
        stain_red = stain_red.T
        stain_green = stain_green.T
        if stain_red.shape[1] == 0 or stain_green.shape[1] == 0:
            LOGGER.info("No staining probes found")
        else:
            fig = _qc_plotter(stain_red, stain_green, color_dict, ymax=60000, title='Staining', return_fig=return_fig)
            if fig:
                figs.append(fig)

    if subset in ('negative','all'):
        if mouse:
            # mouse manifest defines control probes in TWO columns, just to be annoying.
            neg_red   = control_R[(control_R['Control_Type'] == 'NEGATIVE') & (control_R['Extended_Type'].str.startswith('neg_'))].copy().drop(columns=['Control_Type']).reset_index(drop=True)
            neg_green = control_G[(control_G['Control_Type'] == 'NEGATIVE') & (control_G['Extended_Type'].str.startswith('neg_'))].copy().drop(columns=['Control_Type']).reset_index(drop=True)
            neg_mouse_probe_names = list(neg_red.Extended_Type.values)
        else:
            neg_red   = control_R[control_R['Control_Type']=='NEGATIVE'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
            neg_green = control_G[control_G['Control_Type']=='NEGATIVE'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(neg_green.Extended_Type, neg_green.Color))
        color_dict.update({k: (v if v != '-99' else 'Black') for k,v in color_dict.items()})
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
        # UPDATE: picking a smattering of probes that are in both EPIC and EPIC+
        list_of_negative_controls_to_plot = ['Negative 1','Negative 142','Negative 3','Negative 4','Negative 5',
                                             'Negative 6','Negative 7','Negative 8','Negative 119','Negative 10',
                                             'Negative 484','Negative 12','Negative 13','Negative 144','Negative 151',
                                             'Negative 166']
        probes_to_plot = list_of_negative_controls_to_plot
        if mouse:
            probes_to_plot = neg_mouse_probe_names[:36] # plot the first 36
        dynamic_controls = [c for c in probes_to_plot if c in neg_red.columns and c in neg_green.columns]
        dynamic_ymax = max([max(neg_red[dynamic_controls].max(axis=0)), max(neg_green[dynamic_controls].max(axis=0))])
        dynamic_ymax = dynamic_ymax + int(0.1*dynamic_ymax)
        fig = _qc_plotter(neg_red, neg_green, color_dict, columns=probes_to_plot, ymax=dynamic_ymax, xticks=plotx, title='Negative', return_fig=return_fig)
        if fig:
            figs.append(fig)

    if subset in ('hybridization','all'):
        hyb_red   = control_R[control_R['Control_Type']=='HYBRIDIZATION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        hyb_green = control_G[control_G['Control_Type']=='HYBRIDIZATION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(hyb_green.Extended_Type, hyb_green.Color))
        hyb_green = hyb_green.drop(columns=['Color']).set_index('Extended_Type')
        hyb_red = hyb_red.drop(columns=['Color']).set_index('Extended_Type')
        hyb_red = hyb_red.T
        hyb_green = hyb_green.T
        fig = _qc_plotter(hyb_red, hyb_green, color_dict, ymax=35000, xticks=plotx, title='Hybridization', return_fig=return_fig)
        if fig:
            figs.append(fig)

    if subset in ('extension','all'):
        ext_red   = control_R[control_R['Control_Type']=='EXTENSION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        ext_green = control_G[control_G['Control_Type']=='EXTENSION'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(ext_green.Extended_Type, ext_green.Color))
        ext_green = ext_green.drop(columns=['Color']).set_index('Extended_Type')
        ext_red = ext_red.drop(columns=['Color']).set_index('Extended_Type')
        ext_red = ext_red.T
        ext_green = ext_green.T
        if ext_red.shape[1] == 0 or ext_green.shape[1] == 0:
            LOGGER.info("No extension probes found")
        else:
            fig = _qc_plotter(ext_red, ext_green, color_dict, ymax=50000, xticks=plotx, title='Extension', return_fig=return_fig)
            if fig:
                figs.append(fig)

    if subset in ('bisulfite','all'):
        bci_red   = control_R[control_R['Control_Type'].isin(['BISULFITE CONVERSION I','BISULFITE CONVERSION II'])].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        bci_green = control_G[control_G['Control_Type'].isin(['BISULFITE CONVERSION I','BISULFITE CONVERSION II'])].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(bci_green.Extended_Type, bci_green.Color))
        color_dict.update({k: (v if v != 'Both' else 'seagreen') for k,v in color_dict.items()}) # mouse has Both; others don't
        bci_green = bci_green.drop(columns=['Color']).set_index('Extended_Type')
        bci_red = bci_red.drop(columns=['Color']).set_index('Extended_Type')
        bci_red = bci_red.T
        bci_green = bci_green.T
        fig = _qc_plotter(bci_red, bci_green, color_dict, ymax=30000, xticks=plotx, title='Bisulfite Conversion', return_fig=return_fig)
        if fig:
            figs.append(fig)

    if subset in ('non-polymorphic','all'):
        np_red = control_R[control_R['Control_Type']=='NON-POLYMORPHIC'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        np_green = control_G[control_G['Control_Type']=='NON-POLYMORPHIC'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(np_green.Extended_Type, np_green.Color))
        color_dict.update({k: (v if v != '-99' else 'Black') for k,v in color_dict.items()})
        np_green = np_green.drop(columns=['Color']).set_index('Extended_Type')
        np_red = np_red.drop(columns=['Color']).set_index('Extended_Type')
        np_red = np_red.T
        np_green = np_green.T
        if np_red.shape[1] == 0 or np_green.shape[1] == 0:
            LOGGER.info("No non-polymorphic probes found")
        else:
            fig = _qc_plotter(np_red, np_green, color_dict, ymax=30000, xticks=plotx, title='Non-polymorphic', return_fig=return_fig)
            if fig:
                figs.append(fig)

    if subset in ('target-removal','all'):
        tar_red = control_R[control_R['Control_Type']=='TARGET REMOVAL'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        tar_green = control_G[control_G['Control_Type']=='TARGET REMOVAL'].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(tar_green.Extended_Type, tar_green.Color))
        tar_green = tar_green.drop(columns=['Color']).set_index('Extended_Type')
        tar_red = tar_red.drop(columns=['Color']).set_index('Extended_Type')
        tar_red = tar_red.T
        tar_green = tar_green.T
        if tar_red.shape[1] == 0 or tar_green.shape[1] == 0:
            LOGGER.info("No target-removal probes found")
        else:
            fig = _qc_plotter(tar_red, tar_green, color_dict, ymax=2000, title='Target Removal', return_fig=return_fig)
            if fig:
                figs.append(fig)

    if subset in ('specificity','all'):
        spec_red = control_R[control_R['Control_Type'].isin(['SPECIFICITY I','SPECIFICITY II'])].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        spec_green = control_G[control_G['Control_Type'].isin(['SPECIFICITY I','SPECIFICITY II'])].copy().drop(columns=['Control_Type']).reset_index(drop=True)
        color_dict  = dict(zip(spec_green.Extended_Type, spec_green.Color))
        spec_green = spec_green.drop(columns=['Color']).set_index('Extended_Type')
        spec_red = spec_red.drop(columns=['Color']).set_index('Extended_Type')
        spec_red = spec_red.T
        spec_green = spec_green.T
        fig = _qc_plotter(spec_red, spec_green, color_dict, ymax=30000, xticks=plotx, title='Specificity (Type I)', return_fig=return_fig)
        if fig:
            figs.append(fig)

    if return_fig and figs != []:
        return figs


def _qc_plotter(stain_red, stain_green, color_dict=None, columns=None, ymax=None, xticks='show',
        title='', return_fig=False):
    """ draft generic plotting function for all the genome studio QC functions.
    used by plot_staining_controls()

options:
========
    required: stain_red and stain_green
        contains: red/green values in columns and probe characteristics in rows (transposed from control_probes.pkl format).
    color_dict
        {value: color-code} dictionary passed in to define which color to make each value in the index.
    ymax
        if defined, constrains the plot y-max values. Used to standardize view of each probe type within normal ranges.
        any probe values that fall outside this range generate warnings.
    columns
        list of columns(probes) in stain_red and stain_green to plot (if ommitted it plots everything).
    return_fig (False)
        if True, returns the figure object instead of showing plot

todo:
=====
    add a batch option that splits large datasets into multiple charts, so labels are readable on x-axis.
        """
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,8)) # was (12,10)
    plt.tight_layout(w_pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    ax1.grid(axis='both')
    ax2.grid(axis='both')
    title = title + ' ' if title != '' else title
    ax1.set_title(f'{title}Green')
    ax2.set_title(f'{title}Red')
    if color_dict is None:
        color_dict = {}

    # DEBUG: control probes contain '-99 in the Color column. Breaks plot.' But resolved by plot_controls() now.
    if '-99' in color_dict.values():
        missing_colors = {k:v for k,v in color_dict.items() if v == '-99'}
        LOGGER.warning(f"{title} has invalid colors: {missing_colors}")
        color_dict.update({k:'Black' for k,v in missing_colors.items()})

    if columns != None:
        # TODO: ensure all columns in list are in stain_red/green first.
        # failed with Barnes idats_part3 missing some probes
        if (set(columns) - set(stain_red.columns) != set() or
        set(columns) - set(stain_green.columns) != set()):
            cols_removed = [c for c in columns if c not in stain_red or c not in stain_green]
            columns = [c for c in columns if c in stain_red and c in stain_green]
            LOGGER.warning(f'These probes were expected but missing from the {title}data: ({", ".join(cols_removed)})')
        stain_red = stain_red.loc[:, columns]
        stain_green = stain_green.loc[:, columns]
    for c in stain_red.columns:
        if ymax is not None and (stain_red[c] > ymax).any():
            LOGGER.warning(f'Some Red {c} values exceed chart maximum and are not shown.')
        if ymax is not None and (stain_green[c] > ymax).any():
            LOGGER.warning(f'Some Green {c} values exceed chart maximum and are not shown.')
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
    if xticks != 'show':
        #plt.xticks([]) # hide
        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
    if return_fig:
        return fig
    plt.show()
