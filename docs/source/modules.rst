.. _modules:

API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   methylcheck.cli
   methylcheck.run_pipeline
   methylcheck.run_qc

   methylcheck.read_geo
   methylcheck.load
   methylcheck.load_both

   methylcheck.qc_signal_intensity
   methylcheck.plot_M_vs_U
   methylcheck.plot_controls
   methylcheck.plot_beta_by_type

   methylcheck.probes
   methylcheck.list_problem_probes
   methylcheck.exclude_probes
   methylcheck.exclude_sex_control_probes
   methylcheck.drop_nan_probes

   methylcheck.samples
   methylcheck.sample_plot
   methylcheck.beta_density_plot
   methylcheck.mean_beta_plot
   methylcheck.mean_beta_compare
   methylcheck.beta_mds_plot
   methylcheck.combine_mds
   methylcheck.cumulative_sum_beta_distribution

   methylcheck.predict
   methylcheck.get_sex
   methylcheck.assign

QC pipeline functions
---------------------

.. automodule:: methylcheck.qc_report
    :members:
    :undoc-members:
    :show-inheritance:


filtering probes
----------------

.. automodule:: methylcheck.probes
    :members:


plotting functions
------------------

.. automodule:: methylcheck.samples
    :members:


sex prediction
--------------

.. automodule:: methylcheck.predict
    :members:

`ReportPDF` Report Builder class
--------------------------------

.. automodule:: methylcheck.qc_report.ReportPDF
    :members:
