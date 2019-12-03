# Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from umap import UMAP  #  pip3 install umap-learn
from hdbscan import HDBSCAN
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from collections import defaultdict
import pickle
import logging

def in_notebook():
    if hasattr(__builtins__,'__IPYTHON__'):
        return True #globals()[notebook_mod] = __import__(notebook_mod)
    else:
        return False #globals()[console_mod] = __import__(console_mod)

if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
# app
from .postprocessQC import sample_plot

LOGGER = logging.getLogger(__name__)
# notes
# source: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html


def find_clusters(df, **kwargs):
    """
required:
    df -- a dataframe of samples X beta values or m_values.
kwargs:
    bins=100
    plot=True
    n_neighbors=5
    min_distance=0.0
    min_cluster_size=7
    alpha (0 to 1.0) --- if specified, the plot will show points larger and overlapping, with cluster centers darker.
    """
    params = ['bins','plot','n_neighbors','min_dist','min_cluster_size','reduce','alpha']
    if set(kwargs) - set(params) != set():
        raise KeyError(f"These parameters: {set(kwargs) - set(params)} are not recognized.")
    bins=kwargs.get('bins', 100)
    plot=kwargs.get('plot', True)
    n_neighbors = kwargs.get('n_neighbors', 5)
    min_dist= kwargs.get('min_dist', 0.0)
    min_cluster_size = kwargs.get('min_cluster_size', 7)
    reduce = None if 'reduce' not in kwargs else kwargs.get('reduce', 0.1)
    alpha = kwargs.get('alpha',1.0) # if 'alpha' in kwargs else None

    hist_df = reduce_df_as_histogram(df, bins=bins)
    clusterable_embedding = umap_fit_transform(hist_df, plot=plot, n_neighbors=n_neighbors, min_dist=min_dist)
    if alpha == 1.0:
        sample_cluster_labels = hdbscan_fit_predict(clusterable_embedding, plot=plot, min_cluster_size=min_cluster_size, figsize=(12,12), fontsize=12, bins=bins)
    else:
        sample_cluster_labels = hdbscan_fit_strength(clusterable_embedding, plot=plot, min_cluster_size=min_cluster_size, figsize=(12,12), fontsize=12, bins=bins, alpha=alpha)
    clusters_beta_distribution(df, sample_cluster_labels, reduce=reduce)


def reduce_df_as_histogram(df, bins=100):
    """ data reduction -- computes a 100-bin histogram of beta values per sample.
    The HDBscan clustering algorithm works on the SHAPE of the beta/m_value distribution, and is much faster
    than running on the raw probe data. Sample clustering is based on shape of beta distribution, not the
    patterns in probe values.
    """
    # first, confirm samples are in rows, the first dimenion of df.shape
    if df.shape[0] > df.shape[1]:
        df = df.transpose() # now samples are in rows (index).
    # now convert/reduce it: same number of samples in index, but only "bins" number of columns.
    hist_df = pd.DataFrame(index=df.index, columns=range(0, bins))
    for probe_vals, sample_id in tqdm(zip(df.values, df.index), total=df.shape[0], desc='Reducing probe data'):
        hist_vals = np.histogram(probe_vals, bins=bins)[0]
        hist_vals = hist_vals / np.sum(hist_vals)
        hist_df.loc[sample_id,] = hist_vals
    return hist_df


def umap_fit_transform(df, plot=True, n_neighbors=5, min_dist=0.0):
    """ reduces the histogram to a clusterable_embedding matrix for HDBSCAN.
    output is a numpy array with two columns (x,y) for each sample. The histogram is converted to a point in 2-dimensional space. """
    clusterable_embedding = UMAP(
        random_state=4,
        n_neighbors=n_neighbors,
        min_dist=min_dist).fit_transform(
        df.values)
    if plot:
        plt.figure(figsize=(12, 9))
        plt.title('UMAP transformed data'.format(cluster))
        plt.scatter(
            clusterable_embedding[:, 0],
            clusterable_embedding[:, 1],
            s=10) # dot size
    return clusterable_embedding


def hdbscan_fit_predict(df, plot=True, min_cluster_size=7, figsize=(12,12), fontsize=16, bins=100, alpha=1.0):
    """ input: the clusterable embedding numpy-array from umap_fit_transforms output.
    ADD: color schemes for output as kwarg. """
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(df)
    clustered = (labels >= 0)

    if plot:
        sns.set(style='white', rc={'figure.figsize':figsize})
        fig, ax = plt.subplots()
        plt.title(f'UMAP + HDBSCAN clustering of {bins} bins')
        scatter = plt.scatter(x=df[clustered, 0],
                             y=df[clustered, 1],
                             c=labels[clustered],
                             s=10,
                             cmap='Spectral',
                             label=labels,
                             alpha=1.0)

    # text summary / report
    labels_used = set()
    for i, txt in enumerate(labels[clustered]):
        if plot and txt not in labels_used:
            labels_used.add(txt)
            ax.annotate(
                txt,
                (df[clustered, 0][i], df[clustered, 1][i]),
                fontsize=fontsize)
    if not plot:
        print(f"{len(set(labels))} labels; {min_cluster_size} min samples per cluster")
    # return a map of each sample's cluster
    return labels


def hdbscan_fit_strength(df, plot=True, min_cluster_size=12, figsize=(12,12), fontsize=16, bins=100, alpha=0.25):
    """ input: the clusterable embedding numpy-array from umap_fit_transforms output.
    Provides a fuzzier plot, showing how strongly each sample fits within its cluster.
    ADD: color schemes for output as kwarg. """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size).fit(df)
    color_palette = sns.color_palette('deep', 8) + sns.color_palette('dark', 8) + sns.color_palette('muted', 8) + sns.color_palette('pastel', 8) + sns.color_palette('bright', 8) + sns.color_palette('colorblind', 8) + sns.color_palette('hls', 8)
    # TODO: add more colors beyond 56
    if len(set(clusterer.labels_)) > 56:
        raise ValueError(f"This cannot show more than 56 clusters, and you have {len(clusterer.labels_)}. Use hdbscan_fit_predict() instead.")
    cluster_colors = [color_palette[x] if x >= -1
                      else (0.5, 0.5, 0.5) # outliers appear in light grey
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    if plot:
        sns.set(style='white', rc={'figure.figsize':figsize})
        fig, ax = plt.subplots()
        plt.title(f'UMAP + HDBSCAN clustering of {bins} bins')

    # next, find the "best" sample in each cluster and label it.
    best_samples = {} # cluster: sample-index
    cluster_cohesion = defaultdict(list) # cluster: [members' scores] --> avg membership score of samples
    for sample_xy, member_score, cluster in zip(df, clusterer.probabilities_, clusterer.labels_):
        cluster_cohesion[cluster].append(member_score)
        if member_score == 1:
            best_samples[tuple(sample_xy)] = cluster # lookup by xy later
    for cluster, members in cluster_cohesion.copy().items():
        cluster_cohesion[cluster] = round(np.average(members),3)
    # it turns out, avg member scores are not correlated with noise / signal for meth samples.
    # print(sorted(cluster_cohesion.items(), key=lambda x:x[1], reverse=True))

    # label clusters
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(df) # redundant. is this necessary?
    clustered = (labels >= -1)
    labels_used = set()
    for i, cluster in enumerate(labels[clustered]):
        this_xy = tuple((df[clustered, 0][i], df[clustered, 1][i]))
        if plot and cluster not in labels_used and this_xy in best_samples:
            labels_used.add(cluster)
            ax.annotate(
                cluster,
                this_xy,
                fontsize=fontsize)
    if plot:
        plt.scatter(*df.T, s=80, linewidth=0, c=cluster_member_colors, alpha=alpha)
    return labels


def clusters_beta_distribution(beta_df, labels, reduce=0.1):
    """ plots a beta_density_plot for each individual cluster of samples.
    labels is a np array of 1s and 0s for whether sample is in the cluster or not. """
    if beta_df.shape[0] > beta_df.shape[1]:
        # PROBES in columns (values, 1); SAMPLES in rows (index, 0)
        # this df orientation for plotting is MUCH faster.
        beta_df = beta_df.transpose()

    if reduce != None:
        if not isinstance(reduce, (int,float)):
            try:
                reduce = float(reduce)
            except Exception as e:
                raise ValueError("reduce must be a float between 0 and 1.0")
        sample_probes = np.random.choice(beta_df.shape[1], int(reduce*beta_df.shape[1]))

    for cluster in set(labels): #tqdm(set(labels), total=len(set(labels)), desc='Plotting clusters'):
        if reduce:
            sub = beta_df.iloc[:, sample_probes] # subset of probes for plotting (index) - same for all clusters
        else:
            sub = beta_df
        sub = sub.loc[labels==cluster] # subset of samples (rows) per cluster
        if not in_notebook():
            LOGGER.info("Cluster: {}; {} samples".format(cluster, sub.shape[0]))
        else:
            print("Cluster: {}; {} samples".format(cluster, sub.shape[0]))
        plt.figure(figsize=(12, 9))
        if cluster == -1:
            cluster = '-1 (outliers)'
        plt.title('Beta Distributions, cluster {} ({}/{} samples)'.format(cluster, sub.shape[0], beta_df.shape[0]))
        plt.xlabel('Beta values')
        plt.grid()
        #for probes, sample_id in zip(sub.values, sub.index):
        for probes in sub.values:
            sns.distplot(probes, hist=False, norm_hist=False)
        plt.show()


def make_model(df, filestem, **kwargs):
    """
required:
    df -- a dataframe of samples X beta values or m_values.
kwargs:
    bins=100
    plot=True
    n_neighbors=5
    min_distance=0.0
    min_cluster_size=7
    alpha (0 to 1.0) --- if specified, the plot will show points larger and overlapping, with cluster centers darker.
    """
    params = ['bins','n_neighbors','min_dist','min_cluster_size','reduce','alpha']
    if set(kwargs) - set(params) != set():
        raise KeyError(f"These parameters: {set(kwargs) - set(params)} are not recognized.")
    bins=kwargs.get('bins', 100)
    n_neighbors = kwargs.get('n_neighbors', 5)
    min_dist= kwargs.get('min_dist', 0.0)
    min_cluster_size = kwargs.get('min_cluster_size', 7)
    reduce = None if 'reduce' not in kwargs else kwargs.get('reduce', 0.1)
    alpha = kwargs.get('alpha',1.0) # if 'alpha' in kwargs else None

    hist_df = reduce_df_as_histogram(df, bins=bins)
    # this is a dataframe of the histogram values
    clusterable_embedding = umap_fit_transform(hist_df, plot=False, n_neighbors=n_neighbors, min_dist=min_dist)
    # this is an array of x,y coords for each sample
    sample_cluster_labels = hdbscan_fit_predict(clusterable_embedding, plot=plot, min_cluster_size=min_cluster_size, figsize=(3,3), fontsize=10, bins=bins)
    # this is a 1D-array (list) of cluster numbers for each sample
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True).fit(clusterable_embedding)

    with open(f'{filestem}_model.pkl','wb') as f:
        pickle.dump(clusterer, f)
    with open(f'{filestem}_model_hist.pkl','wb') as f:
        pickle.dump(hist_df, f) # needed for the model_combine_predict() method.
    print(f"{filestem}_model.pkl, {filestem}_model_hist.pkl saved.")


def model_predict(df, tissue=None, clusterer=None, array=None, figsize=(12,12), fontsize=24):
    """ Predict (estimate) whether samples are good or bad, based on a loaded trained model.

required:
    clusterer (hdbscan model)
    df (a dataframe of beta values to classify)
todo:
    add: tissue [blood, saliva] as parameter, to load different models.
    add: version [for model version to specify].
    add: array type for model
    NOTE: only works with bins=100, because all the standard models have 100 bins.

to use:
    first, load the "clusterer" model you want, or specify a tissue and array type.
    with open('stp1_blood_cluster_model.pkl','rb') as f:
        clusterer = pickle.load(f)
    """
    from hdbscan import approximate_predict
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if not clusterer and tissue == 'blood':
        clusterer = pd.read_pickle('stp1_blood_cluster_model.pkl')
    if not clusterer and tissue == 'saliva':
        clusterer = pd.read_pickle('stp1_saliva_cluster_model.pkl')

    model_clusterable_embedding = clusterer._raw_data # == prediction_data_.raw_data
    pal = color_palette = sns.color_palette('deep', 8) + sns.color_palette('dark', 8) + sns.color_palette('muted', 8) + sns.color_palette('pastel', 8) + sns.color_palette('bright', 8) + sns.color_palette('colorblind', 8) + sns.color_palette('hls', 8)
    colors = [sns.desaturate(pal[col], max(sat - 0.3,0.2))
              for col, sat in
              zip(clusterer.labels_,
                  clusterer.probabilities_)]

    #transform
    test_hist_df = reduce_df_as_histogram(df)
    test_clusterable_embedding = umap_fit_transform(test_hist_df, plot=False)
    #fit/compare
    test_labels, strengths = approximate_predict(clusterer, test_clusterable_embedding)
    test_colors = [pal[col] if col >= 0 else (0.5, 0.5, 0.5) for col in test_labels]

    sns.set(style='white', rc={'figure.figsize':figsize})
    fig, ax = plt.subplots()
    plt.title(f"HDBSCAN predicted clusters: Dots with dark circles are your samples, superimposed on the model's samples")
    plt.scatter(*model_clusterable_embedding.T, s=80, linewidth=0,  c=colors, alpha=0.05)
    plt.scatter(*test_clusterable_embedding.T,  s=80, linewidths=2, c=test_colors, edgecolors='k', alpha=0.25)

    # label clusters
    labels = clusterer.labels_ # a list of cluster numbers; order matches clusterer.probabilities_
    best_samples = {label:tuple(dat) for label,prob,dat in zip(clusterer.labels_, clusterer.probabilities_, clusterer._raw_data) if prob == 1.0}
    clustered = (labels >= 0)
    labels_used = set()
    for i, cluster_label in enumerate(clusterer.labels_):
        this_xy = best_samples[cluster_label]
        if cluster_label not in labels_used: # and this_xy in best_samples:
            labels_used.add(cluster_label)
            ax.annotate(
                cluster_label,
                this_xy,
                fontsize=fontsize)


def model_combine_predict(df, tissue='blood', min_cluster_size=12, reduce=0.05, figsize=(12,12), fontsize=24):
    """this version of model_predict() will combine your test data (df, beta values) with
    a preloaded dataset of tissue-specific beta density plots. IT should lead to a model stable classification model.

    plot: uses a modified version of hdbscan_fit_strength() that allows code to specify which dots are samples, vs ref data"""

    if tissue == 'blood':
        ref_hist_df = pd.read_pickle('stp1_blood_cluster_model_hist.pkl')
    if tissue == 'saliva':
        ref_hist_df = pd.read_pickle('stp1_saliva_cluster_model_hist.pkl')

    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from hdbscan import HDBSCAN

    # transform
    test_hist_df = reduce_df_as_histogram(df)
    # combine
    if ref_hist_df.shape[1] == test_hist_df.shape[1]:
        df_both = pd.concat([ref_hist_df, test_hist_df])
        # same index cutoff between ref and test data
        test_index_start = ref_hist_df.shape[0] + 1
    else:
        raise ValueError(f"Model ref data for {tissue} does not have same number of points as test data {ref_hist_df.shape[1]}.")
    #fit/compare
    clusterable_embedding = umap_fit_transform(df_both, plot=False)

    # uses a modified version of hdbscan_fit_strength() that allows user to specify which dots are samples, vs ref data
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size).fit(df_both)
    model_cluster_labels = clusterer.labels_[:test_index_start]
    model_cluster_probabilities = clusterer.probabilities_[:test_index_start]
    model_cluster_embedding = clusterable_embedding[:test_index_start] #clusterer._raw_data[:test_index_start]
    sample_cluster_labels = clusterer.labels_[test_index_start:]
    sample_cluster_probabilities = clusterer.probabilities_[test_index_start:]
    sample_cluster_raw = clusterer._raw_data[test_index_start:]
    sample_cluster_embedding = clusterable_embedding[test_index_start:] #clusterer._raw_data[test_index_start:]

    pal = color_palette = sns.color_palette('deep', 8) + sns.color_palette('dark', 8) + sns.color_palette('muted', 8) + sns.color_palette('pastel', 8) + sns.color_palette('bright', 8) + sns.color_palette('colorblind', 8) + sns.color_palette('hls', 8)
    model_colors = [sns.desaturate(pal[col], max(sat - 0.3,0.2))
                    for col, sat in
                    zip(model_cluster_labels,
                        model_cluster_probabilities)]
    sample_colors = [sns.desaturate(pal[col], sat)
                     for col, sat in
                     zip(sample_cluster_labels,
                         sample_cluster_probabilities)]
    sns.set(style='white', rc={'figure.figsize':figsize})
    fig, ax = plt.subplots()
    plt.title(f"HDBSCAN predicted clusters: Dots with dark circles are your samples, superimposed on the model's samples")
    plt.scatter(*model_cluster_embedding.T, s=80, linewidth=0,  c=model_colors, alpha=0.05)
    plt.scatter(*sample_cluster_embedding.T,  s=80, linewidths=2, c=sample_colors, edgecolors='k', alpha=0.25)

    # label clusters
    labels_used = set()
    for label,prob,emb_xy in zip(sample_cluster_labels, sample_cluster_probabilities, sample_cluster_embedding):
        if label not in labels_used and prob > 0.6:
            labels_used.add(label)
            ax.annotate(
                label,
                emb_xy,
                fontsize=fontsize)
    if set(sample_cluster_labels) - labels_used != {-1}:
        print(set(sample_cluster_labels) - labels_used, 'unlabeled clusters')

    # only plotting the new data here
    clusters_beta_distribution(df, sample_cluster_labels, reduce=reduce)
