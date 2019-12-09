# Lib
from logging import NullHandler, getLogger
# App
from .cli import detect_array
from .filters import exclude_sex_control_probes, list_problem_probes, exclude_probes
from .postprocessQC import (
    mean_beta_plot, beta_density_plot, beta_mds_plot, drop_nan_probes,
    cumulative_sum_beta_distribution, mean_beta_compare, combine_mds
    )
from .hdbscan_clustering import (
    find_clusters,
    hdbscan_fit_predict,
    hdbscan_fit_strength,
    make_model,
    model_predict,
    model_combine_predict,
    reduce_df_as_histogram,
    umap_fit_transform,
    )

getLogger(__name__).addHandler(NullHandler())


__all__ = [
    'beta_density_plot',
    'beta_mds_plot',
    'combine_mds',
    'cumulative_sum_beta_distribution',
    'detect_array',
    'drop_nan_probes',
    'exclude_probes',
    'exclude_sex_control_probes',
    'find_clusters',
    'hdbscan_fit_predict',
    'hdbscan_fit_strength',
    'list_problem_probes',
    'mean_beta_plot',
    'mean_beta_compare',
    'sample_plot',
    'make_model',
    'model_predict',
    'model_combine_predict',
    'reduce_df_as_histogram',
    'umap_fit_transform',
    'hdbscan_clustering'
]
