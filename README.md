methylcheck is a Python-based package for filtering and visualizing Illumina methylation array data. The focus is on quality control.

[![Readthedocs](https://readthedocs.com/projects/life-epigenetics-methylcheck/badge/?version=latest)](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CircleCI](https://circleci.com/gh/FOXOBioScience/methylcheck.svg?style=shield&circle-token=58a514d3924fcfe0287c109d2323b7f697956ec9)](https://circleci.com/gh/FOXOBioScience/methylcheck) [![Build status](https://ci.appveyor.com/api/projects/status/j15lpvjg1q9u2y17?svg=true)](https://ci.appveyor.com/project/life_epigenetics/methQC) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/aedf5c223e39415180ff35153b2bad89)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=FOXOBioScience/methylcheck&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/FOXOBioScience/methylcheck/badge.svg?t=OVL45Q)](https://coveralls.io/github/FOXOBioScience/methylcheck) ![PyPI-Downloads](https://img.shields.io/pypi/dm/methylcheck.svg?label=pypi%20downloads&logo=PyPI&logoColor=white) <img src="https://raw.githubusercontent.com/FOXOBioScience/methylcheck/feature/mouse/docs/python3.6.png" height="50">

![methylprep snapshots](https://raw.githubusercontent.com/FOXOBioScience/methylcheck/master/docs/methylcheck_overview.png "methylcheck snapshots")

## Methylcheck Package

This package contains high-level APIs for filtering processed data from local files. 'High-level' means that the details are abstracted away, and functions are designed to work with a minimum of knowledge and specification required. But you can always override the "smart" defaults with custom settings if things don't work. Before starting you must first process a set of `idat` files with the associated `methylprep` package or download processed data from the NIH GEO database. Refer to [methylprep](https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/index.html) for instructions on this step.

![methylprep functions](https://raw.githubusercontent.com/FOXOBioScience/methylcheck/master/docs/methylcheck_functions.png)

## Installation

This package is available in PyPi.
`pip install methylcheck` or `pip3 install methylcheck` if your OS defaults to python2x. This package only works in python3.6+.

## Importing your data

Methylcheck is designed to accept the output from the `methylprep` package. If you have a bunch of `idat` samples, `methylprep` will return a single pickled pandas dataframe containing all the beta values for probes.

Load your data in a Jupyter Notebook like this:

```python
mydata = pandas.read_pickle('beta_values.pkl')
```
For `pandas` you must specify the file name.

If you used `methylprep process` with the `--all` option, there's a convenience function in `methylcheck` (methylcheck.load) to open methylation files in a variety of formats.

```python
import methycheck
df = methylcheck.load('<path to folder with methylprep output>')
# or
df,meta = methylcheck.load_both('<path to folder with methylprep output>')
```

#### meta data
`.load_both()` also conveniently loads a dataframe of all meta data associated with the samples. If you are using public GEO data, it will load that collection's meta data. Some analysis functions require specifying which samples are part of a treatment group (vs control) and the `meta` dataframe object can be used for this.

#### csv
If you point to a folder with processed CSVs, this will load and combine these output files and return a dataframe:

 ```python
 import methycheck
 df = methylcheck.load('docs/example_data/GSE29852/9247377093/')
 ```

#### raw data
You can also use `.load()` to read processed files in these formats:
```
('beta_value', 'm_value', 'meth', 'meth_df', 'noob_df', 'sesame')
```
Specify these using the `format=...` parameter.

#### sesame
It will also load methylation data files processed using `R`'s `sesame` package:
```
df = methylcheck.load(<filename>, format='sesame')
```

For more, check out our [examples of loading data into `methylcheck`](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/docs/demo_qc_functions.html)

### GEO (`idat`)

Alternatively, you can download and process public GEO datasets. If you have a gzip of `idat` files, process them first with `methylprep`, or use the `methylprep download -i <GEO_ID>` option to download and process public data.

In general, the best way to import data is to use `methylprep` and run
```python
import methylprep
methylprep.run_pipeline(data_folder, betas=True)

# or from the command line:
python -m methylprep process -d <filepath to idats> --all
```

collect the `beta_values.pkl` file it returns/saves to disk, and load that in a Jupyter notebook.

### GEO (processed data in `csv, txt, xlsx` formats)
If idats are not available on GEO, you can imported the processed tabular data using `methylcheck.read_geo`. This will convert methylation/unmethylated signal intensities to beta values by default, returning a Pandas dataframe with samples in columns and probes in rows. As of version 0.6, it recognizes six different ways authors have organized their data, but does not handle these cases yet:

- Combining two files containing methylated and unmethylated values for a set of samples
- Reading GSM12345-tbl-1.txt type files (found in GSExxxx_family.tar.gz packages)

To obtain probe `beta` values for samples or methylated/unmethylated signal intensities, download the file locally and run:
```python
import methylcheck
df = methylcheck.read_geo(filepath)
```
If you include `verbose=True` it will explain what it is doing. You can also use `test_only=True` to ensure the parser will work without loading the whole file (which can be several gigabytes in size). Test mode will only return the first 200 probes, but should parse the header and detect the file structure if it can.

From there, each data transformation is a single line of code using Pandas DataFrames. `methylcheck` will keep track of the data format/structures for you, and you can visualize the effect of each filter as you go. You can also export images of your charts for publication.

Refer to the Jupyter notebooks on readthedocs for examples of filtering probes from a batch of samples, removing outlier samples, and generating plots of data.

## Quality Control (QC) reports

#### run_qc()
The simplest way to generate a set of plots about the quality of your probes/array is to run this function in a Jupyter notebook:

```python
import methylcheck
methylcheck.run_qc('<path to your methylprep processed files>')
```

`run_qc()` is adapted from Illumina's Genome Studio QC functions.

#### run_pipeline()
A second, more customizable quality control pipeline is the `methylcheck.run_pipeline()` function. `run_pipeline()` wraps `run_qc()` but adds several sample outlier detection tools. One method, multi-dimensional scaling, is interactive, and allows you to identify samples within your batch that you can statistically reject as outliers. Note that `methylprep process` automatically removes probes that fail the poobah p-value detection limit test by default; `run_pipeline()` examines where samples with lots of unreliable probes should be disregarded entirely.

### ReportPDF
There is also a `methylcheck.qc_report.ReportPDF` class that allows you to build your own QC report and save it to PDF. This is most useful if you process multiple batches of data and want to create a standardized, easy-to-read PDF report about the quality of samples in each batch.

## Other functions

`methylcheck` provides functions to
- predict the sex of samples (`.get_sex`)
- detect probes that differ between two sets of samples within a batch (`.diff_meth_probes`)
- remove sex-chromosome-linked probes and control probes
- remove "sketchy" probes, deemed unreliable by researchers
- filter sample outliers based on multi-dimensional scaling
- combine datasets for analysis
- plot sample beta or m-value distributions, or raw uncorrected probe channel intensities

## Authors

Parts of this package were ported from `minfi`, an `R` package, and extended/developed by the team at Foxo Bioscience, who maintains it. You can write to `info@FOXOBioScience.com` to give feedback, ask for help, or suggest improvements. For bugs, report issues on our [github repo](https://github.com/FOXOBioScience/methylcheck) page.
