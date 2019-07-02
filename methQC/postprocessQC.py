# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def mean_beta_plot(df):
    """Returns a plot of the average beta values for all probes in a batch of samples.

    Input (df):
        - a dataframe with probes in rows and sample_ids in columns.
        - to get this formatted import, use `methpype.consolidate_values_for_sheet()`,
        as this will return a matrix of beta-values for a batch of samples (by default)."""
    data = df.copy(deep=True)
    data['mean'] = data.mean(numeric_only=True, axis=1)
    fig, ax = plt.subplots(figsize=(10, 10))
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
            sb.distplot(row, hist=False, norm_hist=False)
        else:
            outliers.append(subject_id) # drop uses ids, not numbers.
    if plot == True:
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

def DNAmAgeHannumFunction(dat0):
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

    """
    datCoefHannum = _importCoefHannum()
    dat1 = dat0[dat0['CGidentifier'].isin(
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

def beta_mds_plot():
    """
    manifest_file = pd.read_csv('/Users/nrigby/GitHub/stp-prelim-analysis/working_data/CombinedManifestEPIC.manifest.CoreColumns.csv')[['IlmnID', 'CHR']]
    probe_names_no_sex_probes = manifest_file.loc[manifest_file['CHR'].apply(lambda x: x not in ['X', 'Y', np.nan]), 'IlmnID'].values
    probe_names_sex_probes = manifest_file.loc[manifest_file['CHR'].apply(lambda x: x in ['X', 'Y']), 'IlmnID'].values

    df_no_sex_probes = df[probe_names_no_sex_probes]
    df_no_sex_probes.head()
    """
    pass
