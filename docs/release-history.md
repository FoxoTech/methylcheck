# Release History

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
