# Lib
from logging import NullHandler, getLogger
# App
from .cli import detect_array
from .postprocessQC import (
    mean_beta_plot, beta_density_plot, beta_mds_plot, drop_nan_probes,
    cumulative_sum_beta_distribution, mean_beta_compare, combine_mds)
from .filters import exclude_sex_control_probes, list_problem_probes, exclude_probes

getLogger(__name__).addHandler(NullHandler())


__all__ = [
    'beta_density_plot',
    'beta_mds_plot',
    'cumulative_sum_beta_distribution',
    'detect_array',
    'drop_nan_probes',
    'exclude_probes',
    'exclude_sex_control_probes',
    'list_problem_probes',
    'mean_beta_plot',
    'mean_beta_compare',
    'combine_mds'
]
