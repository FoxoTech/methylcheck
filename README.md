methylcheck is a Python-based package for filtering and visualizing Illumina methylation array data. The focus is on quality control.

[![Readthedocs](https://readthedocs.com/projects/life-epigenetics-methylcheck/badge/?version=latest)](https://life-epigenetics-methylcheck.readthedocs-hosted.com/en/latest/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CircleCI](https://circleci.com/gh/LifeEGX/methylcheck.svg?style=shield&circle-token=58a514d3924fcfe0287c109d2323b7f697956ec9)](https://circleci.com/gh/LifeEGX/methylcheck) [![Build status](https://ci.appveyor.com/api/projects/status/j15lpvjg1q9u2y17?svg=true)](https://ci.appveyor.com/project/life_epigenetics/methQC) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/aedf5c223e39415180ff35153b2bad89)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LifeEGX/methylcheck&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/LifeEGX/methylcheck/badge.svg?t=OVL45Q)](https://coveralls.io/github/LifeEGX/methylcheck)

## methylcheck Package

Methylcheck is designed to accept the output from the Methylprep package. It contains both high-level APIs for processing data from local files and low-level functionality allowing you to customize the flow of data and how it is processed.

## Installation

This package is available in PyPi.
`pip install methylcheck`

## How to use it

In general, the best way to import data is to use `methylprep` and run `run_pipeline(data_folder, betas=True)`, collect the beta_values.pkl file it returns/saves to disk, and load that in a Jupyter notebook with methylcheck. From there, each data transformation is a single line of code using Panadas DataFrames. `methylcheck` will keep track of the data format/structures for you, and you can visualize the effect of each filter as you go. You can also export images of your charts for publication.

Refer to the Jupyter notebooks on readthedocs for examples of filtering probes from a batch of samples, removing outlier samples, and generating plots of data.

## Authors

Parts of this package were ported from `minfi`, a `R` package, and extended/developed by the team at Life Epigenetics, who maintains it. You can write to `info@LifeEgx.com` to give feedback, ask for help, or suggest improvements. For bugs, report issues on our github repo page.
