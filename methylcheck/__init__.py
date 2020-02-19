# Lib
from logging import NullHandler, getLogger
# App
from .cli import detect_array
from .probes.filters import (
    exclude_sex_control_probes,
    list_problem_probes,
    exclude_probes,
    drop_nan_probes,
    problem_probe_reasons,
    )
from .samples.postprocessQC import (
    mean_beta_plot, beta_density_plot, beta_mds_plot,
    cumulative_sum_beta_distribution, mean_beta_compare, combine_mds,
    sample_plot,
    )
from .qc_plot import (
    plot_M_vs_U,
    qc_signal_intensity,
    )

try:
    import methylprep
    load = methylprep.load
    load_both = methylprep.load_both
    del methylprep
except ImportError as error:
    pass # these functions are not available otherwise.

from .read_geo_processed import read_geo

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
    'list_problem_probes',
    'mean_beta_plot',
    'mean_beta_compare',
    'read_geo'
    'sample_plot',
    'qc_signal_intensity',
    'plot_M_vs_U',
]

if "load" in dir():
    __all__.append("load")
if "load_both" in dir():
    __all__.append("load_both")
