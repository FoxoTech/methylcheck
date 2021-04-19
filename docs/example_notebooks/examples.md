.. _examples:

# Examples of methylcheck used for quality control (QC), control probe analysis, sample outlier filtering, and sex-checks

Examples:

- [Filtering and Quality Control (QC)](methylprep-methylcheck-qc-example.ipynb)
- [QC with sex check](qc-sex-check.ipynb)
- [QC and outlier detection with Multidimensional Scaling (MDS)](interactive-mds-filtering.ipynb)
- [Reading and combining GEO processed data](demo_read_geo_processed.ipynb)
- [Merging meta data into beta values DataFrame](demo_using_matched_meta_data.ipynb)
- [Methylcheck QC from command line interface (CLI)](methylcheck-cli.md)

Examples from older versions (pre version 0.7):

- [Older Example of Processing Samples](demo-methylprep-to-methylcheck-example.ipynb)
- [Another QC example](another-methylcheck-qc-example.ipynb)

## Plots

### `methylcheck.plot_M_vs_U`

This scatterplot compares unmethylated median intensity on Y-axis with methylated median intensity on X-axis. Passing `compare=True` into the function allows you to see the effect of all processing corrections (NOOB, non-linear-dye bias, infer-type-I-channel-probe-switch, etc) on the median intensities. In the example below, the orange distribution (corrected intensities) is more linear and has a different slope than the blue one (raw intensities), which should theoretically mean that these values [better approximate][1] the true epigenetic state of the underlying DNA. A list of the processing steps employed (under default methylprep processing) are listed in the [reference][1].

[1]: https://pubmed.ncbi.nlm.nih.gov/27924034/

As of v0.7.2, `compare` defaults to True in the ReportPDF class, but you can turn it off.

![plot_M_vs_U with compare option](https://raw.githubusercontent.com/FoxoTech/methylcheck/master/docs/plot_M_vs_U_compare_dye_noob_to_raw.png)
