methylcheck is a Python-based package for filtering and visualizing Illumina methylation array data. The focus is on quality control. View on [ReadTheDocs.](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/)

[![Readthedocs](https://readthedocs.com/projects/life-epigenetics-methylcheck/badge/?version=latest)](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![CircleCI](https://circleci.com/gh/FoxoTech/methylcheck.svg?style=shield)](https://circleci.com/gh/FoxoTech/methylcheck)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/aedf5c223e39415180ff35153b2bad89)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=FoxoTech/methylcheck&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/FoxoTech/methylcheck/badge.svg?t=OVL45Q)](https://coveralls.io/github/FoxoTech/methylcheck) ![PyPI-Downloads](https://img.shields.io/pypi/dm/methylcheck.svg?label=pypi%20downloads&logo=PyPI&logoColor=white) <img src="https://raw.githubusercontent.com/FoxoTech/methylcheck/feature/mouse/docs/python3.6.png" height="50">

![methylcheck snapshots](https://raw.githubusercontent.com/FoxoTech/methylcheck/master/docs/methylcheck_overview.png "methylcheck snapshots")

## methylcheck is part of the methylsuite

`methylcheck` is part of the [methylsuite](https://pypi.org/project/methylsuite/) of python packages that provide functions to process and analyze DNA methylation data from Illumina's Infinium arrays (27k, 450k, and EPIC, as well as mouse arrays). This package is focused on quality control for processed methylation data.

`methylcheck` functions are designed to work with a minimum of knowledge and specification required. But you can always override the "smart" defaults with custom settings if the default settings don't work for your data. The entire `methylsuite` is designed in this format: to offer ease of use while still maintaining flexibility for customization as needed.


## Methylsuite package components

You should install all three components, as they work together. The parts include:

- `methylprep`: for processing `idat` files or downloading GEO datasets from NIH. Processing steps include
   - infer type-I channel switch
   - NOOB (normal-exponential convolution on out-of-band probe data)
   - poobah (p-value with out-of-band array hybridization, for filtering low signal-to-noise probes)
   - qualityMask (to exclude historically less reliable probes)
   - nonlinear dye bias correction (AKA signal quantile normalization between red/green channels across a sample)
   - calculate beta-value, m-value, or copy-number matrix
   - large batch memory management, by splitting it up into smaller batches during processing

- `methylcheck`: (this package) for quality control (QC) and analysis, including
   - functions for filtering out unreliable probes, based on the published literature
      - Note that `methylprep process` will exclude a set of unreliable probes by default. You can disable that using the --no_quality_mask option from CLI.
   - sample outlier detection
   - array level QC plots of staining, bisulfite conversion, hybridization, extension, negative, non-polymorphic, target removal, and specificity
   - spreadsheet summary of control probe performance
   - data visualization functions based on `seaborn` and `matplotlib` graphic libraries.
   - predict sex of human samples from probes
   - interactive method for assigning samples to groups, based on array data, in a Jupyter notebook

- `methylize` provides more analysis and interpretation functions
   - differentially methylated probe statistics (between treatment and control samples)
   - differentially methylated regions, with gene annotation from the UCSC Human Genome Browser
   - volcano plots (to identify probes that are the most different)
   - manhattan plots (to identify clusters of probes associated with genomic regions that are different)

## Installation

`methylcheck` maintains configuration files for your Python package manager of choice: [pipenv](https://pipenv.readthedocs.io/en/latest/) or [pip](https://pip.pypa.io/en/stable/). Conda install is coming soon.

```shell
>>> pip install methylcheck
```

or if you want to install all three packages at once (recommended):
```shell
>>> pip install methylsuite
```

## Tutorials and Guides

If you are new to DNA methylation analysis, we recommend reading through this [introduction](https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/introduction/introduction.md) from the `methylprep` documentation. Otherwise, you are ready to use `methylcheck` to:

- [load processed methylation data](docs/loading-data.ipynb)
- [filter unreliable probes from your data](docs/filtering-probes.ipynb)
- [run array-level quality control reports](docs/quality-control-example.ipynb)
- [detect outlier samples](docs/mds-example.ipynb)
- [predict the sex of human samples](docs/quality-control-example.ipynb#predicting-sex)
