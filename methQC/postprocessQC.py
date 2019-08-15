# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import MDS
from tqdm import tqdm
import datetime

LOGGER = logging.getLogger(__name__)

def mean_beta_plot(df, verbose=False, save=False, silent=False):
    """Returns a plot of the average beta values for all probes in a batch of samples.

    Input (df):
        - a dataframe with probes in rows and sample_ids in columns.
        - to get this formatted import, use `methpype.consolidate_values_for_sheet()`,
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
        - to get this formatted import, use `methpype.consolidate_values_for_sheet()`,
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
    print(len(df.columns))

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


def beta_mds_plot(df, filter_stdev=1.5, verbose=True, save=False, silent=False, return_plot_obj=False):
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
        pandas, numpy, pyplot, sklearn.manifold.MDS"""
    # ensure "long format": probes in rows and samples in cols. This is how methpype returns data.
    if df.shape[1] < df.shape[0]:
        ## methQC needs probes in rows and samples in cols. but MDS needs a wide matrix.
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
    PSF = 2 # plot_scale_fator -- an empirical number to stretch the plot out and show clusters more easily.
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
    ##########
    if verbose == True:
        print("""You can now remove outliers based on their transformed beta values
 falling outside a range, defined by the sample standard deviation.""")
    while True:
        df_indexes_to_retain = []
        df_indexes_to_exclude = []
        minX = round(x_avg - adj*x_std)
        maxX = round(x_avg + adj*x_std)
        minY = round(y_avg - adj*y_std)
        maxY = round(y_avg + adj*y_std)
        if verbose == True:
            print('Your acceptable value range: x=({0} to {1}), y=({2} to {3}).'.format(
                minX, maxX,
                minY, maxY
            ))
        md2 = [] # samples that fit within the cutoff box.
        for idx,row in enumerate(mds_transformed):
            if minX <= row[0] <= maxX and minY <= row[1] <= maxY:
                md2.append(row)
                df_indexes_to_retain.append(idx)
            else:
                df_indexes_to_exclude.append(idx)
            #pandas style: mds2 = mds_transformed[mds_transformed[:, 0] == class_number[:, :2]
        # this is a np array, not a df. Removing all dots that are retained from the "exluded" data set (mds_transformed)
        mds_transformed = np.delete(mds_transformed, [df_indexes_to_retain], axis=0)
        md2 = np.array(md2)
        fig = plt.figure(figsize=(12, 9))
        plt.title('MDS Plot of betas from methylation data')
        plt.grid()

        plt.scatter(md2[:, 0], md2[:, 1], s=DOTSIZE, c='blue')
        plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=5, c='red')
        # note/bug: the red dots are merely being overwritten by blue dots

        x_range_min = PSF*old_X_range[0] if PSF*old_X_range[0] < minX else PSF*minX
        x_range_max = PSF*old_X_range[1] if PSF*old_X_range[1] > maxX else PSF*maxX
        y_range_min = PSF*old_Y_range[0] if PSF*old_Y_range[0] < minY else PSF*minY
        y_range_max = PSF*old_Y_range[1] if PSF*old_Y_range[1] > maxY else PSF*maxY
        plt.xlim([x_range_min, x_range_max])
        plt.ylim([y_range_min, y_range_max])
        plt.vlines([minX, maxX], minY, maxY, color='red', linestyle=':')
        plt.hlines([minY, maxY], minX, maxX, color='red', linestyle=':')
        #line = mlines.Line2D([minX, minX], [minY, minY], color='red', linestyle='--')
        #ax = fig.add_subplot(111)
        #ax.add_line(line)

        if return_plot_obj == True:
            return plt
        elif silent == True:
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

    # save file. return dataframe.
    fig = plt.figure(figsize=(12, 9))
    plt.title('MDS Plot of betas from methylation data')
    plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=5)
    plt.scatter(md2[:, 0], md2[:, 1], s=5, c='red')
    plt.xlim(old_X_range)
    plt.ylim(old_Y_range)
    plt.xlabel('X')
    plt.ylabel('Y')
    # take the original dataframe (df) and remove samples that are outside the sample thresholds, returning a new dataframe
    df_out = df.drop(df.index[df_indexes_to_exclude]) # inplace=True will modify the original DF outside this function.

    # UNRESOLVED BUG.
    # was getting 1069 samples back from this; expected 1076. discrepancy is because
    # pre_df_excl = len(df.index[df_indexes_to_exclude])
    # unique_df_excl = len(set(df.index[df_indexes_to_exclude]))
    # print(pre_df_excl, unique_df_excl)

    prev_df = len(df)
    if save:
        image_name = df.index.name or 'beta_mds_n={0}_p={1}'.format(len(df.index), len(df.columns)) # np.size(df,0), np.size(md2,1)
        outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
        plt.savefig(outfile)
        if verbose:
            print("Saved {0}".format(outfile))
            LOGGER.info("Saved {0}".format(outfile))
    plt.close(fig) # avoids displaying plot again in jupyter.
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
    line1 = sns.distplot(data1['mean'], hist=False, rug=False, ax=ax, axlabel='beta')
    line2 = sns.distplot(data2['mean'], hist=False, rug=False)
    plt.title('Mean Beta Plot (Compare pre vs post filtering)')
    plt.grid()
    plt.xlabel('Mean Beta')
    #plt.legend([line1, line2], ['pre','post'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save:
        plt.savefig('mean_beta_compare.png')
    if not silent:
        plt.show()
    else:
        plt.close(fig)

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
