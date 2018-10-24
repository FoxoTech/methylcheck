# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt


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
