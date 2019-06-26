# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _importCoefHannum():
    """Imports Hannum Coefficients into dataframe"""
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(
        basepath, "methQC", "data_files", "datCoefHannum.csv"))
    datCoefHannum = pd.read_csv(filepath)
    return datCoefHannum


def meanBetaPlot(df):
    data = df.copy(deep=True)
    data['mean'] = data.mean(numeric_only=True, axis=1)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(data['mean'], hist=False, rug=False, ax=ax, axlabel='beta')
    plt.title('Mean Beta Plot')
    plt.xlabel('Mean Beta')
    plt.ylabel('Count')
    plt.show()


def betaDensityPlot(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    for col in df.columns:
        if col != 'Name':
            sns.distplot(
                df[col], hist=False, rug=False,
                label=col, ax=ax, axlabel='beta')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Beta Density Plot')
    plt.xlabel('Beta')
    plt.ylabel('Count')
    plt.show()


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
