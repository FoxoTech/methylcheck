# `methylcheck` command line interface (CLI)

In mose cases you will want to use ``methylcheck`` in a python Jupyter Notebook environment.
But if you really want, the is a basic command line interface that runs ``methylcheck.run_qc()``.

Efficient and reliable quality control is important. The **methylcheck** package (part of the methyl-suite along with `methylprep`) can be used to perform quality control and interactively visualize processed samples, either using the command line or a Jupyter Notebook. If you are only interesed in using a Jupyter Notebook for quality control, skip to the [next section](#JN).

**methylcheck** features one CLI command where various arguments dictate how the program runs.
Users must specify at least two arguements, `-d` followed by the path of the data file to load and (optionally) `-a` followed by the array type of that data file.
By default, all quality control plots are run. For each plot, a PNG image is shown on the screen.
For detailed information about each plot, see the section: [Methylcheck in Jupyter Notebook](https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/docs/methylprep_tutorial.html#methylcheck-in-jupyter-notebook).

Here we use a data frame created from the GSE69852 samples provided with **methylprep** produced by first running `python3 -m methylprep -v process -d "docs/example_data/GSE69852/" --betas`.

```bash
$ python3 -m methylcheck -d beta_values.pkl -a '450k'
```

Mean Beta Plot

![Fig.1](tutorial_figs/fig1.png)

Beta Density Plot

![Fig.2](tutorial_figs/fig2.png)

```python
Calculating area under curve for each sample.
6it [00:00,  9.52it/s]
```

![Fig.3](tutorial_figs/fig3.png)

MDS Plot (outlier detection)

![Fig.4](tutorial_figs/fig4.png)

```python
Original samples (6, 2) vs filtered (6, 2)
Your scale factor was: 1.5
Enter new scale factor, <enter> to accept and save:
```

To specify a specific plot, include the `-p` switch followed by the desired plot chosen from the following: `mean_beta_plot`, `beta_density_plot`, `cumulative_sum_beta_distribution`, `beta_mds_plot`, or `all` (all of which are covered in detail in the section: [Jupyter Notebook](https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/docs/methylprep_tutorial.html#methylcheck-in-jupyter-notebook)). Note that while all plot functions have beta in the title, they are also used to plot M value data frames.

```bash
$ python3 -m methylcheck -d beta_values.pkl -a '450k' -p mean_beta_plot
```

![Fig.5](tutorial_figs/fig5.png)

Users can also specify which probes should be removed. To exclude sex probes, control probes, or probes that have been identified as problematic, provide the `--exclude_sex`, `--exclude_control`, or `--exclude_probes` arguments respectively. To remove all of the aforementioned probes, use `--exclude_all`.

```bash
$ python3 -m methylcheck -d beta_values.pkl -a '450k' -p mean_beta_plot --exclude_sex
```

![Fig.6](tutorial_figs/fig6.png)

Here, we add the `--verbose` flag to get additional information about `methylcheck` as it runs, which can be utilized for every plot.

```bash
$ python3 -m methylcheck -d beta_values.pkl --verbose -a '450k' -p mean_beta_plot --exclude_all
Array 450k: Removed 11648 sex linked probes and 916 internal control probes from 6 samples. 473864 probes remaining.
Discrepancy between number of probes to exclude (12564) and number actually removed (11648): 916
It appears that your sample had no control probes, or that the control probe names didn't match the manifest (450k).
Of 473864 probes, 334500 matched, yielding 139364 probes after filtering.
```

![Fig.7](tutorial_figs/fig7.png)

For all plots a PNG image is shown on the screen. To save this image to disk, include `--save`. We also use the `--silent` flag here to supress the PNG image from being shown on the screen (which also suppresses progress bars from being displayed).

```bash
$ python3 -m methylcheck -d beta_values.pkl -a '450k' -p mean_beta_plot --save --silent
```

<a name="JN"></a>
