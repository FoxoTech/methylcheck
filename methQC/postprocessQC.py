# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
import numpy as np
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def mean_beta_plot(df):
    """Returns a plot of the average beta values for all probes in a batch of samples.

    Input (df):
        - a dataframe with probes in rows and sample_ids in columns.
        - to get this formatted import, use `methpype.consolidate_values_for_sheet()`,
        as this will return a matrix of beta-values for a batch of samples (by default)."""
    data = df.copy(deep=True)
    data['mean'] = data.mean(numeric_only=True, axis=1)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.distplot(data['mean'], hist=False, rug=False, ax=ax, axlabel='beta')
    plt.title('Mean Beta Plot')
    plt.xlabel('Mean Beta')
    plt.ylabel('Count')
    plt.show()


def beta_density_plot(df):
    """Returns a plot of beta values for each sample in a batch of samples as a separate line.
    Y-axis values is the count (of what? intensity? normalized?).
    X-axis values are beta values (0 to 1) for a single samples

    Input (df):
        - a dataframe with probes in rows and sample_ids in columns.
        - to get this formatted import, use `methpype.consolidate_values_for_sheet()`,
        as this will return a matrix of beta-values for a batch of samples (by default).

    Returns:
        None"""
    fig, ax = plt.subplots(figsize=(12, 9))
    for col in df.columns:
        if col != 'Name':
            sns.distplot(
                df[col], hist=False, rug=False,
                label=col, ax=ax, axlabel='beta')
    if len(df.columns) <= 30:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Beta Density Plot')
    plt.grid()
    plt.xlabel('Beta values')
    plt.ylabel('Count')
    plt.show()


def cumulative_sum_beta_distribution(df, cutoff=0.7, plot=True):
    """ attempts to filter outlier samples based on the cumulative area under the curve
    exceeding a reasonable value (cutoff).

    Inputs:
        DataFrame
        cutoff (default 0.7)
        plot (default True) -- show plot, or just return transformed data if False.

    Returns:
        dataframe with subjects removed that exceed cutoff value."""
    good_samples = []
    outliers = []
    print("Calculating area under curve for each sample.")
    for subject_num, (row, subject_id) in tqdm(enumerate(zip(df.values,
                                                             df.index))):
        hist_vals = np.histogram(row, bins=10)[0]
        hist_vals = hist_vals / np.sum(hist_vals)
        cumulative_sum = np.cumsum(hist_vals)
        if cumulative_sum[5] < cutoff:
            good_samples.append(subject_num)
            ax = sb.distplot(row, hist=False, norm_hist=False)
        else:
            outliers.append(subject_id) # drop uses ids, not numbers.
    if plot == True:
        if len(df.index) <= 30:
            ax.legend_.remove()
        plt.figure(figsize=(12, 9))
        plt.title('Beta Distributions (filtered by {0})'.format(cutoff))
        plt.xlabel('Beta values')
        plt.ylabel('Count')
        plt.grid()
        plt.show()
    return df.drop(outliers, axis=0)


def _importCoefHannum():
    """Imports Hannum Coefficients into dataframe"""
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(
        basepath, "data_files", "datCoefHannum.csv"))
    datCoefHannum = pd.read_csv(filepath)
    return datCoefHannum


def DNA_mAge_Hannum(dat0):
    """Calculates DNAmAge for each sample

    Parameters
    ----------
    dat0: dataframe
        Dataframe containing beta values

    Returns
    -------
    dat2: dataframe
        Dataframe containing calculated values for
        each sample

    sample data (dat0) has probes in columns, and samples in rows.
    https://stackoverflow.com/questions/39827897/get-dataframe-columns-from-a-list-using-isin
    """
    datCoefHannum = _importCoefHannum()
    dat1 = dat0[dat0.columns.difference(  # was: dat0['CGidentifier'].isin(...)
        datCoefHannum.Marker.values)].copy(deep=True)
    dat1.sort_values(by='CGidentifier', inplace=True)
    datCoefHannum.sort_values(by='Marker', inplace=True)
    dat2 = pd.DataFrame(
        index=[s for s in dat1.columns if s != 'CGidentifier'], columns=['DNAmAgeHannum'])
    for sample in dat2.index:
        values = np.multiply(dat1[sample], datCoefHannum['Coefficient'])
        num_missing = values.isna().sum()
        if num_missing > 30:
            dat2.loc[sample, 'DNAmAgeHannum'] = np.nan
        else:
            dat2.loc[sample, 'DNAmAgeHannum'] = np.nansum(values)
    return dat2



def beta_mds_plot(df, filter_stdev=1.5, verbose=True, silent=False):
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

    requires
    --------
        pandas, numpy, pyplot, sklearn.manifold.MDS"""
    import datetime
    #the test notebook begins by importing npy and building the dataframe
    #data = np.load('working_data/stp-blood-betavals.npy')
    #probes = np.load('working_data/stp-blood-betavals-probe-names.npy')
    #samples = np.load('working_data/stp-blood-betavals-subject-ids.npy')
    # methQC needs probes in rows and samples in cols. This is how methpype returns data.
    #otherwise --> data = data.transpose()
    if verbose == True:
        print(df.shape)
        df.head()
        LOGGER.info('DataFrame has shape: {0}'.format(df.shape))
        print("Making sure that probes are in columns (the second number should be larger than the first).")
        LOGGER.info("Making sure that probes are in columns (the second number should be larger than the first).")
        if df.shape[1] < df.shape[0]:
            ## methQC needs probes in rows and samples in cols. but MDS needs a wide matrix.
            print("Your data needed to be transposed (df = df.transpose()) before you run this.")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
            df = df.copy().transpose() # don't overwrite the original
        # before running this, you'd typically exclude probes.
        print("Starting MDS fit_transform. this may take a while.")
        LOGGER.info("Starting MDS fit_transform. this may take a while.")

    mds = MDS(n_jobs=-1, random_state=1, verbose=1)
    #n_jobs=-1 means "use all processors"
    mds_transformed = mds.fit_transform(df.values)
    # pass in df.values (a np.array) instead of a dataframe, as it processes much faster.
    old_X_range = [min(mds_transformed[:, 0]), max(mds_transformed[:, 0])]
    old_Y_range = [min(mds_transformed[:, 1]), max(mds_transformed[:, 1])]
    x_std, y_std = np.std(mds_transformed,axis=0)
    x_avg, y_avg = np.mean(mds_transformed,axis=0)

    adj = filter_stdev #(1.5)
    ##########
    if verbose == True:
        print("""You can now remove outliers based on their transformed beta values
 falling outside a range, defined by the sample standard deviation.""")
    while True:
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
        md2 = []
        for idx,row in enumerate(mds_transformed):
            if minX <= row[0] <= maxX and minY <= row[1] <= maxY:
                md2.append(row)
            else:
                df_indexes_to_exclude.append(idx)
            #pandas style: mds2 = mds_transformed[mds_transformed[:, 0] == class_number[:, :2]
        md2 = np.array(md2)
        plt.figure(figsize=(12, 9))
        plt.title('MDS Plot of betas from methylation data')
        plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=5)
        plt.scatter(md2[:, 0], md2[:, 1], s=5, c='red')
        plt.xlim(old_X_range)
        plt.ylim(old_Y_range)
        plt.xlabel('X')
        plt.ylabel('Y')

        if silent == True:
            # take the original dataframe (df) and remove samples that are outside the sample thresholds, returning a new dataframe
            df.drop(df.index[df_indexes_to_exclude], inplace=True)
            image_name = df.index.name or 'meth_n={0}_p={1}'.format(len(df.index), len(df.columns)) # np.size(df,0), np.size(md2,1)
            outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
            plt.savefig(outfile)
            LOGGER.info("Saved {0}".format(outfile))
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
    image_name = df.index.name or 'meth_n={0}_p={1}'.format(len(df_out.index), len(df_out.columns)) # np.size(df,0), np.size(md2,1)
    outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
    plt.savefig(outfile)
    plt.close(fig) # avoids displaying plot again in jupyter.
    print('DEBUG {0} {1} = {2} | pre {3} | post {4}'.format(len(md2), len(df_indexes_to_exclude), len(md2) + len(df_indexes_to_exclude),
        prev_df,
        len(df_out)
     ))
    if verbose == True:
        print("Saved {0}".format(outfile))
        LOGGER.info("Saved {0}".format(outfile))
    # returning DataFrame in original structure: rows are probes; cols are samples.
    return df_out, df_indexes_to_exclude  # may need to transpose this first.
