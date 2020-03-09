methylcheck is a Python-based package for filtering and visualizing Illumina methylation array data. The focus is on quality control.

[![Readthedocs](https://readthedocs.com/projects/life-epigenetics-methylcheck/badge/?version=latest)](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CircleCI](https://circleci.com/gh/LifeEGX/methylcheck.svg?style=shield&circle-token=58a514d3924fcfe0287c109d2323b7f697956ec9)](https://circleci.com/gh/LifeEGX/methylcheck) [![Build status](https://ci.appveyor.com/api/projects/status/j15lpvjg1q9u2y17?svg=true)](https://ci.appveyor.com/project/life_epigenetics/methQC) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/aedf5c223e39415180ff35153b2bad89)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LifeEGX/methylcheck&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/LifeEGX/methylcheck/badge.svg?t=OVL45Q)](https://coveralls.io/github/LifeEGX/methylcheck)

![methylprep snapshots](https://raw.githubusercontent.com/LifeEGX/methylcheck/master/docs/methylcheck_overview.png "methylcheck snapshots")

## methylcheck Package

This package contains high-level APIs for filtering processed data from local files. 'High-level' means that the details are abstracted away, and functions are designed to work with a minimum of knowledge and specification required. But you can always override the "smart" defaults with custom settings if things don't work. Before starting you must first download processed data from the NIH GEO database or process a set of `idat` files with `methylprep`. Refer to [methylprep](https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/index.html) for instructions on this step.

## Installation

This package is available in PyPi.
`pip install methylcheck` or `pip3 install methylcheck` if your OS defaults to python2x. This package only works in python3.6+.

## Importing your data

Methylcheck is designed to accept the output from the `methylprep` package. If you have a bunch of `idat` samples, `methylprep` will return a single pickled pandas dataframe containing all the beta values for probes.

Load your data in a Jupyter Notebook like this:

```python
mydata = pandas.read_pickle('beta_values.pkl')
```

If you processed a large batch of samples using the `batch_size` option in `methylprep process`, there's a convenience function in `methylcheck` (methylcheck.load) that will load and combine a bunch of output files in the same folder:

```python
import methylize
df = methylcheck.load('<path to folder with methylprep output>')
# or
df,meta = methylcheck.load_both('<path to folder with methylprep output>')
```

This conveniently loads a dataframe of all meta data associated with the samples, if you are using public GEO data. Some analysis functions require specifying which samples are part of a treatment group (vs control) and the `meta` dataframe object can be used for this.

For more, check out our [examples of loading data into `methylcheck`](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/docs/demo_qc_functions.html)

### GEO

Alternatively, you can import public GEO datasets directly, if they are processed data containing either probe `beta` values for samples or methylated/unmethylated signal intensities. If you have `idat` files, process them first with `methylprep`, or use the `methylprep download -i <GEO_ID>` option to download and process public data.

In general, the best way to import data is to use `methylprep` and run
```python
run_pipeline(data_folder, betas=True)

# or from the command line:
python -m methylprep process -d <filepath to idats> --all
```

collect the `beta_values.pkl` file it returns/saves to disk, and load that in a Jupyter notebook. From there, each data transformation is a single line of code using Pandas DataFrames. `methylcheck` will keep track of the data format/structures for you, and you can visualize the effect of each filter as you go. You can also export images of your charts for publication.

Refer to the Jupyter notebooks on readthedocs for examples of filtering probes from a batch of samples, removing outlier samples, and generating plots of data.

## Quality Control (QC)

The simplest way to generate a battery of plots about your data is to run this function in a Jupyter notebook:

```python
import methylcheck
methylcheck.run_qc('<path to your methylprep processed files>')
```

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

Parts of this package were ported from `minfi`, an `R` package, and extended/developed by the team at Foxo Bioscience, who maintains it. You can write to `info@LifeEgx.com` to give feedback, ask for help, or suggest improvements. For bugs, report issues on our [github repo](https://github.com/lifeEGX/methylcheck) page.
