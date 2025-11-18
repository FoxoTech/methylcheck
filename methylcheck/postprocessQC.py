# -*- coding: utf-8 -*-
# built ins
import logging
import os
import datetime

# package
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import MDS
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def mean_beta_plot(df, verbose=False, save=False, silent=False):
    """Returns a plot of the average beta values for all probes in a batch of samples.

    Input (df):
        - a dataframe with probes in rows and sample_ids in columns.
        - to get this formatted import, use `methylprep.consolidate_values_for_sheet()`,
        as this will return a matrix of beta-values for a batch of samples (by default)."""
    if df.shape[0] < df.shape[1]:
        ## ensure probes in rows and samples in cols
        if verbose:
            print("Your data needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
        df = df.copy().transpose() # don't overwrite the original

    data = df.copy(deep=True)
    data['mean'] = data.mean(numeric_only=True, axis=1)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.distplot(data['mean'], hist=False, rug=False, ax=ax, axlabel='beta')
    plt.title('Mean Beta Plot')
    plt.grid()
    plt.xlabel('Mean Beta')
    if save:
        plt.savefig('mean_beta.png')
    if not silent:
        plt.show()
    else:
        plt.close(fig)


def beta_density_plot(df, verbose=False, save=False, silent=False):
    """Returns a plot of beta values for each sample in a batch of samples as a separate line.
    Y-axis values is the count (of what? intensity? normalized?).
    X-axis values are beta values (0 to 1) for a single samples

    Input (df):
        - a dataframe with probes in rows and sample_ids in columns.
        - to get this formatted import, use `methylprep.consolidate_values_for_sheet()`,
        as this will return a matrix of beta-values for a batch of samples (by default).

    Returns:
        None"""
    if df.shape[0] < df.shape[1]:
        ## ensure probes in rows and samples in cols
        if verbose:
            print("Your data needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
        df = df.copy().transpose() # don't overwrite the original

    fig, ax = plt.subplots(figsize=(12, 9))

    for col in df.columns:
        if col != 'Name': # probe name
            if len(df.columns) <= 30:
                sns.distplot(
                    df[col], hist=False, rug=False,
                    label=col, ax=ax, axlabel='beta')
            else:
                sns.distplot(
                    df[col], hist=False, rug=False, axlabel='beta')

    if len(df.columns) <= 30:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #else:
    #    ax.get_legend().set_visible(False)
    #    print('suppressing legend')

    plt.title('Beta Density Plot')
    plt.grid()
    plt.xlabel('Beta values')
    if save:
        plt.savefig('beta.png')
    if not silent:
        plt.show()
    else:
        plt.clf()
        plt.cla()
        plt.close()


def cumulative_sum_beta_distribution(df, cutoff=0.7, verbose=False, save=False, silent=False):
    """ attempts to filter outlier samples based on the cumulative area under the curve
    exceeding a reasonable value (cutoff).

    Inputs:
        DataFrame -- wide format (probes in columns, samples in rows)
        cutoff (default 0.7)
        silent -- suppresses figure, so justs returns transformed data if False.
        if save==True: saves figure to disk.

    Returns:
        dataframe with subjects removed that exceed cutoff value."""
    # ensure probes in colums, samples in rows
    if df.shape[1] < df.shape[0]:
        df = df.copy().transpose() # don't overwrite the original
        if verbose:
            print("Your data needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")

    good_samples = []
    outliers = []
    if not silent:
        print("Calculating area under curve for each sample.")
    fig, ax = plt.subplots(figsize=(12, 9))
    # first, check if probes aren't consistent, to avoid a crash here
    df = drop_nan_probes(df, silent=silent, verbose=verbose)

    # if silent is True, tqdm will not show process bar.
    for subject_num, (row, subject_id) in tqdm(enumerate(zip(df.values,
                                                             df.index)), disable=silent):
        hist_vals = np.histogram(row, bins=10)[0]
        hist_vals = hist_vals / np.sum(hist_vals)
        cumulative_sum = np.cumsum(hist_vals)
        if cumulative_sum[5] < cutoff:
            good_samples.append(subject_num)
            sns.distplot(row, hist=False, norm_hist=False)
        else:
            outliers.append(subject_id) # drop uses ids, not numbers.

    plt.title('Cumulative Sum Beta Distribution (filtered at {0})'.format(cutoff))
    plt.grid()
    if len(df.columns) <= 30:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        ax.legend_ = None
    if save:
        plt.savefig('cum_beta.png')
    if not silent:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return df.drop(outliers, axis=0)


def beta_mds_plot(df, filter_stdev=1.5, verbose=True, save=False, silent=False, multi_params={'draw_box':True}):
    """
    1 needs to read the manifest file for the array, or at least a list of probe names to exclude/include.
        manifest_file = pd.read_csv('/Users/nrigby/GitHub/stp-prelim-analysis/working_data/CombinedManifestEPIC.manifest.CoreColumns.csv')[['IlmnID', 'CHR']]
        probe_names_no_sex_probes = manifest_file.loc[manifest_file['CHR'].apply(lambda x: x not in ['X', 'Y', np.nan]), 'IlmnID'].values
        probe_names_sex_probes = manifest_file.loc[manifest_file['CHR'].apply(lambda x: x in ['X', 'Y']), 'IlmnID'].values

    df_no_sex_probes = df[probe_names_no_sex_probes]
    df_no_sex_probes.head()

    Arguments
    ---------
    df
        dataframe of beta values for a batch of samples (rows are probes; cols are samples)
    filter_stdev
        a value (unit: standard deviations) between 0 and 3 (typically) that represents
        the fraction of samples to include, based on the standard deviation of this batch of samples.
        So using the default value of 1.5 means that all samples whose MDS-transformed beta sort_values
        are within +/- 1.5 standard deviations of the average beta are retained in the data returned.
    multi_params
        is a dict, passed into this function from a multi-compare-MDS wrapper function, containing:
        {return_plot_obj=True,
        fig=None,
        ax=None,
        draw_box=False,
        xy_lim=None,
        color_num=0,
        PSF=1.2 -- plot scale factor (margin beyond points to display)}

    Options
    --------
    silent
        if running from command line in an automated process, you can run in `silent` mode to suppress any user interaction.
        In this case, whatever `filter_stdev` you assign is the final value, and a file will be processed with that param.
        Silent also suppresses plots (images) from being generated. only files are returned.

    returns
    -------
        returns a filtered dataframe.
        if `return_plot_obj` is True, it returns the plot, for making overlays in methylize.

    requires
    --------
        pandas, numpy, pyplot, sklearn.manifold.MDS """

    # ensure "long format": probes in rows and samples in cols. This is how methylprep returns data.
    if df.shape[1] < df.shape[0]:
        ## methylcheck needs probes in rows and samples in cols. but MDS needs a wide matrix.
        df = df.copy().transpose() # don't overwrite the original
        if verbose:
            print("Your data needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
    if verbose == True:
        print(df.shape)
        df.head()
        LOGGER.info('DataFrame has shape: {0}'.format(df.shape))
        print("Making sure that probes are in columns (the second number should be larger than the first).")
        LOGGER.info("Making sure that probes are in columns (the second number should be larger than the first).")
        # before running this, you'd typically exclude probes.
        print("Starting MDS fit_transform. this may take a while.")
        LOGGER.info("Starting MDS fit_transform. this may take a while.")

    #df = drop_nan_probes(df, silent=silent, verbose=verbose)

    # CHECK for missing probe values NaN
    missing_probe_counts = df.isna().sum()
    total_missing_probes = sum([i for i in missing_probe_counts])/len(df) # sum / columns
    if sum([i for i in missing_probe_counts]) > 0:
        df = df.dropna()
        if verbose == True:
            print("We found {0} probe(s) were missing values and removed them from MDS calculations.".format(total_missing_probes))
        if silent == False:
            LOGGER.info("{0} probe(s) were missing values removed from MDS calculations.".format(total_missing_probes))

    mds = MDS(n_jobs=-1, random_state=1, verbose=1)
    #n_jobs=-1 means "use all processors"
    mds_transformed = mds.fit_transform(df.values)
    # pass in df.values (a np.array) instead of a dataframe, as it processes much faster.
    # old range is used for plotting, with some margin on outside for chart readability

    # plot_scale_fator -- an empirical number to stretch the plot out and show clusters more easily.
    if 'PSF' in multi_params:
        PSF = multi_params.get('PSF')
    else:
        PSF = 2

    if df.shape[0] < 40:
        DOTSIZE = 16
    elif 40 < df.shape[0] < 60:
        DOTSIZE = 14
    elif 40 < df.shape[0] < 60:
        DOTSIZE = 12
    elif 60 < df.shape[0] < 80:
        DOTSIZE = 10
    elif 80 < df.shape[0] < 100:
        DOTSIZE = 8
    elif 100 < df.shape[0] < 300:
        DOTSIZE = 7
    else:
        DOTSIZE = 5
    old_X_range = [min(mds_transformed[:, 0]), max(mds_transformed[:, 0])]
    old_Y_range = [min(mds_transformed[:, 1]), max(mds_transformed[:, 1])]
    #old_X_range = [old_X_range[0] - PSF*old_X_range[0], old_X_range[1] + PSF*old_X_range[1]]
    #old_Y_range = [old_Y_range[0] - PSF*old_Y_range[0], old_Y_range[1] + PSF*old_Y_range[1]]
    x_std, y_std = np.std(mds_transformed,axis=0)
    x_avg, y_avg = np.mean(mds_transformed,axis=0)

    adj = filter_stdev #(1.5)

    if verbose == True:
        print("""You can now remove outliers based on their transformed beta values
 falling outside a range, defined by the sample standard deviation.""")

    while True:
        df_indexes_to_retain = []
        df_indexes_to_exclude = []
        # xy_lim is a manual override preset boundary cutoff range for MDS filtering.
        if multi_params.get('xy_lim'):
            xy_lim = multi_params['xy_lim']
            if type(xy_lim) in (list,tuple): # ((xmin, xmax), (ymin, ymax))
                minX = xy_lim[0][0]
                maxX = xy_lim[0][1]
                minY = xy_lim[1][0]
                maxY = xy_lim[1][1]
            else:
                raise ValueError ("xy_lim in multi_params must be a list or tuple")
        else:
            minX = round(x_avg - adj*x_std)
            maxX = round(x_avg + adj*x_std)
            minY = round(y_avg - adj*y_std)
            maxY = round(y_avg + adj*y_std)

        if verbose == True:
            print('Your acceptable value range: x=({0} to {1}), y=({2} to {3}).'.format(
                minX, maxX,
                minY, maxY
            ))
        # md2 are the dots that fall inside the cutoff.
        md2 = []
        for idx,row in enumerate(mds_transformed):
            if minX <= row[0] <= maxX and minY <= row[1] <= maxY:
                md2.append(row)
                df_indexes_to_retain.append(idx)
            else:
                df_indexes_to_exclude.append(idx)
            #pandas style: mds2 = mds_transformed[mds_transformed[:, 0] == class_number[:, :2]

        # this is a np array, not a df. Remove all dots that are retained from the "exluded" data set (mds_transformed))
        mds_transformed = np.delete(mds_transformed, [df_indexes_to_retain], axis=0)
        md2 = np.array(md2)

        color_num = multi_params.get('color_num',0)
        # ADD TO EXISTING FIG... up to 24 colors (repeated 3x for 72 total)
        COLORSET = dict(enumerate(['xkcd:blue', 'xkcd:green', 'xkcd:coral', 'xkcd:lightblue', 'xkcd:magenta', 'xkcd:goldenrod', 'xkcd:plum', 'xkcd:beige',
                                   'xkcd:orange', 'xkcd:orchid', 'xkcd:silver', 'xkcd:purple', 'xkcd:pink', 'xkcd:teal', 'xkcd:tomato', 'xkcd:yellow',
                                   'xkcd:olive', 'xkcd:lavender', 'xkcd:indigo', 'xkcd:black', 'xkcd:azure', 'xkcd:brown', 'xkcd:aquamarine', 'xkcd:darkblue',

                                   'xkcd:blue', 'xkcd:green', 'xkcd:coral', 'xkcd:lightblue', 'xkcd:magenta', 'xkcd:goldenrod', 'xkcd:plum', 'xkcd:beige',
                                   'xkcd:orange', 'xkcd:orchid', 'xkcd:silver', 'xkcd:purple', 'xkcd:pink', 'xkcd:teal', 'xkcd:tomato', 'xkcd:yellow',
                                   'xkcd:olive', 'xkcd:lavender', 'xkcd:indigo', 'xkcd:black', 'xkcd:azure', 'xkcd:brown', 'xkcd:aquamarine', 'xkcd:darkblue',

                                   'xkcd:blue', 'xkcd:green', 'xkcd:coral', 'xkcd:lightblue', 'xkcd:magenta', 'xkcd:goldenrod', 'xkcd:plum', 'xkcd:beige',
                                   'xkcd:orange', 'xkcd:orchid', 'xkcd:silver', 'xkcd:purple', 'xkcd:pink', 'xkcd:teal', 'xkcd:tomato', 'xkcd:yellow',
                                   'xkcd:olive', 'xkcd:lavender', 'xkcd:indigo', 'xkcd:black', 'xkcd:azure', 'xkcd:brown', 'xkcd:aquamarine', 'xkcd:darkblue']))

        if multi_params.get('fig') == None:
            fig = plt.figure(figsize=(12, 9))
            plt.title('MDS Plot of betas from methylation data')
            plt.grid() # adds fig.axes implied
        else:
            fig = multi_params.get('fig')

        if multi_params.get('ax') == None and fig.axes != []:
            if verbose:
                print('axes', multi_params.get('ax'), fig.axes, 'assigned to ax.')
            ax = fig.axes[0] # appears to get implicitly created when plot.grid() runs
        elif multi_params.get('ax') == None:
            # passing the 'ax' (fig.axes[0]) object back in will avoid the plotlib warning.
            ax = fig.add_subplot(1,1,1)
        else:
            ax = multi_params.get('ax')

        ax.scatter(md2[:, 0], md2[:, 1], s=DOTSIZE, c=COLORSET.get(color_num,'black')) # RETAINED
        ax.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=DOTSIZE, c='xkcd:ivory', edgecolor='black', linewidth='0.2',) # EXCLUDED

        x_range_min = PSF*old_X_range[0] if PSF*old_X_range[0] < minX else PSF*minX
        x_range_max = PSF*old_X_range[1] if PSF*old_X_range[1] > maxX else PSF*maxX
        y_range_min = PSF*old_Y_range[0] if PSF*old_Y_range[0] < minY else PSF*minY
        y_range_max = PSF*old_Y_range[1] if PSF*old_Y_range[1] > maxY else PSF*maxY
        ax.set_xlim([x_range_min, x_range_max])
        ax.set_ylim([y_range_min, y_range_max])
        #print(int(x_range_min), int(x_range_max), int(y_range_min), int(y_range_max))

        if multi_params.get('draw_box') == True:
            ax.vlines([minX, maxX], minY, maxY, color=COLORSET.get(color_num,'red'), linestyle=':')
            ax.hlines([minY, maxY], minX, maxX, color=COLORSET.get(color_num,'red'), linestyle=':')

        if multi_params.get('return_plot_obj') == True:
            return fig, ax, df_indexes_to_retain

        if silent == True:
            # take the original dataframe (df) and remove samples that are outside the sample thresholds, returning a new dataframe
            df.drop(df.index[df_indexes_to_exclude], inplace=True)
            image_name = df.index.name or 'beta_mds_n={0}_p={1}'.format(len(df.index), len(df.columns)) # np.size(df,0), np.size(md2,1)
            outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
            plt.savefig(outfile)
            LOGGER.info("Saved {0}".format(outfile))
            plt.close(fig)
            # returning DataFrame in original structure: rows are probes; cols are samples.
            return df  # may need to transpose this first.
        else:
            plt.show()
            plt.close(fig)

        ########## BEGIN INTERACTIVE MODE #############
        print("Original samples {0} vs filtered {1}".format(mds_transformed.shape, md2.shape))
        print('Your scale factor was: {0}'.format(adj))
        adj = input("Enter new scale factor, <enter> to accept and save:")
        if adj == '':
            break
        try:
            adj = float(adj)
        except ValueError:
            print("Not a valid number. Type a number with a decimal value, or Press <enter> to quit.")
            continue

    prev_df = len(df)
    if save:
        # save file. return dataframe.
        fig = plt.figure(figsize=(12, 9))
        plt.title('MDS Plot of betas from methylation data')
        plt.grid() # adds fig.axes implied
        ax = fig.add_subplot(1,1,1)
        ax.scatter(md2[:, 0], md2[:, 1], s=DOTSIZE, c=COLORSET.get(color_num,'black')) # RETAINED
        ax.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=DOTSIZE, c='xkcd:ivory', edgecolor='black', linewidth='0.2',) # EXCLUDED
        ax.set_xlim(old_X_range)
        ax.set_ylim(old_Y_range)
        # take the original dataframe (df) and remove samples that are outside the sample thresholds, returning a new dataframe
        df_out = df.drop(df.index[df_indexes_to_exclude]) # inplace=True will modify the original DF outside this function.

        # UNRESOLVED BUG.
        # was getting 1069 samples back from this; expected 1076. discrepancy is because
        # pre_df_excl = len(df.index[df_indexes_to_exclude])
        # unique_df_excl = len(set(df.index[df_indexes_to_exclude]))
        # print(pre_df_excl, unique_df_excl)

        image_name = df.index.name or 'beta_mds_n={0}_p={1}'.format(len(df.index), len(df.columns)) # np.size(df,0), np.size(md2,1)
        outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
        plt.savefig(outfile)
        plt.close(fig) # avoids displaying plot again in jupyter.
        if verbose:
            print("Saved {0}".format(outfile))
            LOGGER.info("Saved {0}".format(outfile))
    else:
        df_out = df
    # returning DataFrame in original structure: rows are probes; cols are samples.
    return df_out #, df_indexes_to_exclude  # may need to transpose this first.


def mean_beta_compare(df1, df2, save=False, verbose=False, silent=False):
    """Use this function to compare two dataframes, pre-vs-post filtering and removal of outliers.

    silent: suppresses figure, so no output unless save==True too."""
    if df1.shape[0] < df1.shape[1]:
        ## ensure probes in rows and samples in cols
        if verbose:
            print("Your first data set needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
        df1 = df1.copy().transpose() # don't overwrite the original
    if df2.shape[0] < df2.shape[1]:
        ## ensure probes in rows and samples in cols
        if verbose:
            print("Your second data set needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
        df2 = df2.copy().transpose() # don't overwrite the original

    data1 = df1.copy(deep=True)
    data1['mean'] = data1.mean(numeric_only=True, axis=1)
    data2 = df2.copy(deep=True)
    data2['mean'] = data2.mean(numeric_only=True, axis=1)

    fig, ax = plt.subplots(figsize=(12, 9))
    line1 = sns.distplot(data1['mean'], hist=False, rug=False, ax=ax, axlabel='beta', color='xkcd:blue')
    line2 = sns.distplot(data2['mean'], hist=False, rug=False, color='xkcd:green')
    plt.title('Mean Beta Plot (Compare pre (blue) vs post (green) filtering)')
    plt.grid()
    plt.xlabel('Mean Beta')
    #plt.legend([line1, line2], ['pre','post'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save:
        plt.savefig('mean_beta_compare.png')
    if not silent:
        plt.show()
    else:
        plt.close('all')


def combine_mds(*args, **kwargs):
    """
    combine (or segment) datasets how it works
    ------------------------------------------
    Use this function on multiple dataframes to combine datasets, or to visualize
    parts of the same dataset in separate colors. It is a wrapper of `methylcheck.beta_mds_plot()` and applies
    multidimensional scaling to cluster similar samples based on patterns in probe values, as well as identify
    possible outlier samples (and exclude them).

    - combine datasets,
    - run MDS,
    - see how each dataset (or subset) overlaps with the others on a plot,
    - exclude outlier samples based on a composite cutoff box (the average bounds of the component data sets)
    - calculate the percent of data excluded from the group

    inputs
    ------
        - *args: pass in any number of pandas dataframes, and it will combine them into one mds plot.
        - alternatively, you may pass in a list of filepaths as strings, and it will attempt to load these files as pickles.
        but they must be pickles of pandas dataframes

    optional keyword arguments
    --------------------------
    - silent: (default False)
        (automated processing mode)
        if True, suppresses most information and avoids prompting user for anything.
        silent mode processes data but doesn't show the plot.
    - save: (default False)
        if True, saves the plot png to disk.
    - verbose: (default False)
        if True, prints extra debug information to screen or logger.

    analysis parameters
    -------------------
    - filter_stdev (how broadly should you retain samples? units are standard deviations, defaulting to 1.5 STDEV.)
    if you increase this number, fewer outlier samples will be removed.

    returns
    ------
        - TODO: one dataframe of the retained samples, cutoff box is avg of datasets
        ~~nothing returned (currently)~~
        - TODO: each dataset's results as a transformed file
        - default: list of samples retained or excluded
        - option: a list of pyplot subplot objects
    """

    # check if already dataframes, or are they strings?
    list_of_dfs = list(args)
    if any([not isinstance(item, pd.DataFrame) for item in list_of_dfs]):
        if set([type(item) for item in list_of_dfs]) == {str}:
            try:
                list_of_dfs = _load_data(list_of_dfs)
            except Exception as e:
                raise FileNotFoundError ("Either your files don't exist, or they are not pickled: {0}".format(e))
    # kwargs
    save = kwargs.get('save', False)
    silent = kwargs.get('silent', False)
    verbose = kwargs.get('verbose', True)
    filter_stdev = kwargs.get('filter_stdev', 1.5)
    output_format = kwargs.get('output')
    PRINT = print if verbose else _noprint

    # data to combine
    dfs = pd.DataFrame()
    # OPTIONAL OUTPUT: TRACK each source df's samples for plot color coding prior to merging
    # i.e. was this sample included or excluded at end?
    # maybe use a simple class here to track the data as it is being manipulated?
    sample_source = {}
    frame_transposed = {}
    # subplots: possibly useful - list of subplot objects within the figure, and their metadata.
    subplots = []
    # track x/y ranges to adjust plot area later
    xy_lims = []

    # PROCESS MDS
    fig = None
    ax = None
    for idx, df in enumerate(list_of_dfs):
        transposed = False
        if df.shape[1] > df.shape[0]: # put probes in rows
            df = df.transpose()
            transposed = True
        for sample in df.columns:
            if sample in sample_source:
                print("WARNING: sample names are not unique across data sets")
            sample_source[sample] = idx
        frame_transposed[idx] = transposed

        # first, PLOT separate MDS, each with its own box.
        try:
            fig,ax,df_indexes_to_retain = beta_mds_plot(df, filter_stdev=filter_stdev, save=save, verbose=verbose, silent=silent,
                              multi_params={'return_plot_obj':True,
                                            'fig':fig,
                                            'ax':ax,
                                            'color_num':idx,
                                            'draw_box':True,
                                            'PSF':1.2,
                                            })
            subplots.append(fig)
            xy_lims.append( (fig.axes[0].get_xlim(), fig.axes[0].get_ylim()) ) # (x_range, y_range)
        except Exception as e:
            PRINT(e)

    #set max range
    x_range_min = min(item[0][0] for item in xy_lims)
    x_range_max = max(item[0][1] for item in xy_lims)
    y_range_min = min(item[1][0] for item in xy_lims)
    y_range_max = max(item[1][1] for item in xy_lims)
    fig.axes[0].set_xlim([x_range_min, x_range_max])
    fig.axes[0].set_ylim([y_range_min, y_range_max])
    #PRINT('full chart range', int(x_range_min), int(x_range_max), int(y_range_min), int(y_range_max))
    if not silent:
        plt.show()
    plt.close('all')

    # PART 2 - calculate the average MDS QC score
    #1 run the loop. calc avg MDS boundary box for group.
    #2 rerun the loop with this as the specified boundary.
    #3 calculate percent retained per dataset
    #4 overall QC score is avg percent retained per dataset.
    avg_x_range_min = int(sum(item[0][0] for item in xy_lims)/len(xy_lims))
    avg_x_range_max = int(sum(item[0][1] for item in xy_lims)/len(xy_lims))
    avg_y_range_min = int(sum(item[1][0] for item in xy_lims)/len(xy_lims))
    avg_y_range_max = int(sum(item[1][1] for item in xy_lims)/len(xy_lims))
    xy_lim = ((avg_x_range_min, avg_x_range_max), (avg_y_range_min, avg_y_range_max))
    PRINT('Average MDS window coordinate range: (x: {0}, y:{1}'.format(xy_lim[0], xy_lim[1]))
    fig = None
    ax = None
    transformed_dfs = []
    for idx, df in enumerate(list_of_dfs):
        if df.shape[1] > df.shape[0]: # put probes in rows
            df = df.transpose()
        fig,ax,df_indexes_to_retain = beta_mds_plot(df,
            filter_stdev=filter_stdev, save=save, verbose=verbose, silent=silent,
            multi_params={
                'return_plot_obj':True,
                'fig':fig,
                'ax':ax,
                'color_num':idx,
                'draw_box':True,
                'xy_lim':xy_lim,
                'PSF':1.2,
            })
        try:
            if frame_transposed[idx]:
                # this data (df) was transposed from file-orientation, so samples are in rows now; probes in columns
                df_transformed = df.iloc[df_indexes_to_retain, :] # samples in rows
                print(df.shape, df_transformed.shape)
            else:
                # iloc: first list is row index; 2nd is column index
                df_transformed = df.iloc[:, df_indexes_to_retain] # samples in columns
                print(df.shape, df_transformed.shape)
            transformed_dfs.append(df_transformed)
        except:
            import pdb;pdb.set_trace()
    fig.axes[0].set_xlim([x_range_min, x_range_max])
    fig.axes[0].set_ylim([y_range_min, y_range_max])

    # https://stackoverflow.com/questions/32213889/get-positions-of-points-in-pathcollection-created-by-scatter
    #print(len(fig.axes[0].collections)) # collections has data in every 4th list item.
    # items in the collections after the main data set are the excluded points
    all_coords = []
    retained = []
    excluded = []
    retained_sample_dfs = []
    for i in range(0, 4*len(list_of_dfs), 1):
        DD = fig.axes[0].collections[i]
        DD.set_offset_position('data')
        #print(DD.get_offsets())
        all_coords.extend(DD.get_offsets().tolist())
        #print(i, len(all_coords))
        if i % 4 == 0: # 0, 4, 8 -- this is the first data set applied to plot. (x,y plot coords)
            retained.extend( DD.get_offsets().tolist() )
            # go from plot sample x,y to idx of samples in original dfs.
        if i % 4 == 1: # 1, 5, 9, etc -- this is the second data set applied to plot.
            excluded.extend( DD.get_offsets().tolist() )
    if verbose:
        PRINT('{0} % retained overall ({1} out of {2} samples)'.format(
            round(100*len(retained) / (len(retained) + len(excluded))),
            len(retained), len(retained) + len(excluded) ))
    if not silent:
        plt.show()
    plt.close('all')

    # TODO: output_format
    return transformed_dfs


def _load_data(filepaths):
    dfs = []
    for ff in filepaths:
        df = pd.read_pickle(ff)
        dfs.append(df)
    return dfs


def _noprint(*messages):
    """ a helper function to suppress print() if not verbose mode. """
    pass


def drop_nan_probes(df, silent=False, verbose=False):
    """ accounts for df shape (probes in rows or cols) so dropna() will work.

    the method used inside MDS may be faster, but doesn't tell you which probes were dropped."""
    ### histogram can't have NAN values -- so need to exclude before running, or warn user.
    # from https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null -- returns a slimmer df of col/rows with NAN.
    dfnan = df[df.isnull().any(axis=1)][df.columns[df.isnull().any()]]
    if len(dfnan) > 0 and df.shape[0] > df.shape[1]: # a list of probe names that contain nan.
        #probes in rows
        pre_shape = df.shape
        df = df.dropna()
        note = "(probes,samples)"
        if not silent:
            LOGGER.info(f"dropping probe(s) that are missing a value (for this calculation): {dfnan}")
            LOGGER.info(f"retained {df.shape} {note} from the original {pre_shape} {note}.")
        if verbose:
            print("We found {0} probe(s) were missing values and removed them from calculations.".format(len(dfnan)))
    elif len(dfnan.columns) > 0 and df.shape[1] > df.shape[0]:
        pre_shape = df.shape
        df = df.dropna(axis='columns')
        note = "(samples,probes)"
        if not silent:
            LOGGER.info(f"dropping probe(s) that are missing a value (for this calculation): {dfnan.columns}")
            LOGGER.info(f"retained {df.shape} {note} from the original {pre_shape} {note}.")
        if verbose:
            print("We found {0} probe(s) were missing values and removed them from calculations.".format(len(dfnan.columns)))
    return df
