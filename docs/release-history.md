# Release History

## v0.7.0
- Illumina Mouse Array Support
- Complete rewrite of documentation
- qc_signal_intensity and plot_M_vs_U have additional options, including superimposing poobah (percent probe failures per sample) on the plot coloring.
- .load will work on control_probes.pkl and mouse_probes.pkl files (with alt structure: dictionary of dataframe)
- .sample_plot uses "best" legend positioning now, because it was not fitting on screen with prev settings.

## v0.6.4
- get_sex() function returns a dataframe that also includes percent of X and Y probes that failed p-value-probe detection, as an indication of whether the predicted sex is reliable.
- better unit test coverage of predictions, load, load_both, and container_to_pkl functions
- fixed bug in load( 'meth_df')

## v0.6.3
- fixed bug in detect_array() where it labeled EPIC+ as EPIC

## v0.6.2
- minor fixes to load() and read_geo()
- exclude_probes() accepts problem_probes criteria as alternate way to specify probes.
    - Exclude probes from a df of samples. Use list_problem_probes() to obtain a list of probes (or pass in the names of 'Criteria' from problem probes), then pass that in as a probe_list along with the dataframe of beta values (array)
- load_processed now has a --no_filter=False option that will remove probes that failed p-value detection, if passing in beta_values.pkl and poobah_values.pkl files.
- load() now handles gzipped files the same way (so .pkl.gz or .csv.gz OK as file or folder inputs)
- seaborn v0.10 --> v0.11 deprecrated the distplot() function that was used heavily. So now this employs kdeplot() in its place, with similar results.

## v0.6.1
- exposed more beta_density_plot parameters, so it can be used to make a QC plot (highlighting one
or several samples within a larger batch, and graying out the others in the plot).

## v0.6.0
- improved read_geo() function, for downloading GEO methylation data sets and parsing meta_data from projects.
- change org name from lifeepigenetics to FoxoBioScience on Github.

## v0.5.9
- qc_plot bug fixes -99

## v0.5.7
- -99 bug in negative controls fixed

## v0.5.4
- tweaking custom-tables in ReportPDF

## v0.5.2
- ReportPDF.run_qc() supports on_lambda, and any functions that require .methylprep_manifest_files can be set to look for manifests in /tmp using on_lambda=True

## v0.5.1
- sklearn now optional for MDS

## v0.5.0
- adds kwargs to functions for silent processing returning figure objects, and a report_pdf class that can run QC and generate a PDF report.
- added __version__
- p-value probe detection
- hdbscan clustering functions
- more QC methods testing

## v0.4.0
- more tests, smart about df orientation, and re-organized files
- added read_geo() for processed datafiles, and unit tests for it. Works with txt,csv,xlsx,pkl files
- read_geo() docs
- debugged filters.list_problem_probes:
- updated the docs to have correct spelling for refs/reasons.
- added a function that lets you see more detail on the probes and reasons/pubs criteria
- added more genome studio QC functions,
  - improved .load function (but not consolidated through methyl-suite yet)
  - function .assign() for manually categorizing samples
  - unit testing on the predict.sex function
  - get_sex() prediction
- consolidated data loading for functions and uses fastest option
