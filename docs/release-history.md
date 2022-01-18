# Release History

## v0.8.1
- .load gives clearer error when loading beta values from CSVs ('beta_csv') if probe names are not unique,
and returns a list of series for each sample when indeces fail to merge (pandas.concat)
- .beta_mds_plot() can now suppress the interactive portion and still display plots, using `silent=True` and `plot=True`
(`plot` is a new kwarg, and defaults to `True`). Previously `silent` mode would suppress both prompts and plot display.
Change in behavior: `silent` mode will not disable plotting. Must also include `plot=False` for that.

## v0.8.0
- Fixed bug in `.load` that requires `tqdm` >= 4.61.2
- Added more detailed error message on `.load`; it cannot load and merge two meth/unmeth dataframes with redundant probe names.

## v0.7.9
- `ReportPDF` accepts 'poobah_colormap' kwarg to feed in beta_mds_plot colormap.
- `ReportPDF` custom tables: You can insert your custom table on the first page by specifying 'order_after' == None.
- `beta_mds_plot` `palette` can now be any matplotlib colormap name. Defaults to 'magma' if not specified. The palette
is only used to color-code poobah failure rates, if the poobah file path is specified.
- `beta_mds_plot` new kwarg `extend_poobah_range`: Default (True) shows 7 colors for poobah failure rates. If False, will show only 5.

## v0.7.6
- Reading IDATs loading bar didn't work correctly, showed up after loading.
- Fixed error/logging messages:
  - exclude_sex_control_probes() said 916 control probes were removed, then said “it appears your sample had no control probes”
  - Erroneous message about missing values in poobah file: "color coding may be inaccurate."
  - Filtering probes info message said there were N samples when it meant probes.
  - methylprep.download.build_composite_dataset() Process time was negative.
- Target Removal and Staining graphs in plot_controls() had unreadable X-axis sample names. Labels are suppressed when
  showing more than 30 samples.
- methylcheck.detect_array() sometimes returned array types in wrong case. All functions expect lowercase array types now.
  - resolves exclude_sex_control_probes bugs.
- run_qc() and get_sex() did not recognize poobah_values.pkl on MacOS when using "~" in the filepath.
- methylcheck.problem_probe_reasons() lists probes matching any/all criteria when passing in no arguments, as documented
- get_sex() understands samplesheet ‘m’ and ‘f’ when not capitalized now.
- Load_both: always returns dataframe with probes in rows now, like .load() does.
- plot_M_vs_U now loads the noob_meth_values.pkl files if noob=True and files are found; otherwise it uses whatever meth/unmeth data is available.
- Methylcheck.qc_plot.qc_signal_intensity returns a dictionary of data about good/bad samples based on signal intensity.
  Previously it was only returning this if 'plot' was False.
- controls_report() bug fixed: methylprep was producing samplesheet meta data pickles that contained Sample_ID twice,
  because the GEO series_matrix files had this data appear twice. This broke the report, but this case is caught and avoided now.
  controls_report() will recognize a wider array of samplesheet filenames now; anything with 'samplesheet' or 'meta_data' in the filename.


## v0.7.5
- added 'methylcheck report' CLI option to create a ReportPDF
- updated documentation
- minor bug fixes in read_geo()
  - qc_plot() now handles mouse probe type differently
  - handles importing from multiple pandas versions correctly
  - read_geo can open series_matrix.txt files now

## v0.7.4
- fixed big where csv data_files were not included in pypi

## v0.7.3
- Improved ReportPDF custom tables option
  - if fields are too long, it will truncate them or auto scale the font size smaller to fit on page.

## v0.7.2
- added GCT score to controls_report() used in the ReportPDF class.
- ReportPDF changes
  - uses noob_meth/unmeth instead of raw, uncorrected meth/unmeth values for GCT and U vs M plot
  - inverted poobah table to report percent passing (instead of failing) probes per sample
  - this changed input from 'poobah_max_percent' (default 5%) to 'poobah_min_percent', (default 80%)
  - M_vs_U not included by default, because redundant with qc_signal_intensity
  - M_vs_U compare=True now labels each sample and has legend, so you can see effect of NOOB+dye correction on batch
  - added poobah color-coding to MDS plot
- get_sex improved plotting
  - will read poobah data and size sample points according to percent of failed probes
  - save plots, or return fig, and more options now

## v0.7.1
- Added a controls_report() function that creates a spreadsheet summary of control probe performance.
- New unit test coverage. Note that because methylprep v1.4.0 changes processing, the results will change slightly
    to match `sesame` instead of `minfi`, with nonlinear-dye-bias correction and infer-type-I-probe-switching.
- changed org name from FoxoBioScience to FoxoTech

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
- changed org name from life-epigenetics to FoxoBioScience on Github.

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
