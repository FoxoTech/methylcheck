# Lib
from logging import NullHandler, getLogger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=PendingDeprecationWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
# App
from .cli import detect_array

from .probes.filters import (
    exclude_sex_control_probes,
    list_problem_probes,
    exclude_probes,
    drop_nan_probes,
    problem_probe_reasons,
    )
from .reports import *
from .samples.postprocessQC import (
    mean_beta_plot, beta_density_plot, beta_mds_plot,
    cumulative_sum_beta_distribution, mean_beta_compare, combine_mds,
    sample_plot,
    )
from .samples.assign_groups import assign, plot_assigned_groups

from .predict.sex import get_sex

from .qc_plot import (
    run_qc,
    plot_beta_by_type,
    plot_M_vs_U,
    qc_signal_intensity,
    plot_controls,
    bis_conversion_control,
    )

from .load_processed import load, load_both, container_to_pkl
from .read_geo_processed import read_geo, detect_header_pattern
from .version import __version__

getLogger(__name__).addHandler(NullHandler())

__all__ = [
    'assign',
    'plot_assigned_groups',
    'ControlsReporter',
    'beta_density_plot',
    'beta_mds_plot',
    'bis_conversion_control',
    'controls_report',
    'combine_mds',
    'cumulative_sum_beta_distribution',
    'container_to_pkl',
    'detect_array',
    'detect_header_pattern',
    'drop_nan_probes',
    'exclude_probes',
    'exclude_sex_control_probes',
    'get_sex',
    'list_problem_probes',
    'load',
    'load_both',
    'mean_beta_plot',
    'mean_beta_compare',
    'read_geo',
    'run_pipeline',
    'run_qc',
    'sample_plot',
    'qc_signal_intensity',
    'plot_beta_by_type',
    'plot_M_vs_U',
    'plot_controls',
    'ReportPDF',
]
