# BeadArray Controls Reporter

Methylcheck provides a python clone of Illumina's Windows software, BeadArray Controls Reporter. This generates
a color-coded excel document showing any irregularities with any sample's array processing.

![](https://raw.githubusercontent.com/FoxoTech/methylcheck/master/docs/example_controls_report.png)

You can run it from the command line like this:

```
python3 -m methylcheck controls -d <path to processed data>
```

Or in a Jupyter notebook like this:

```python
import methylcheck

methylcheck.controls_report('<path to processed data>')

```

### Available options

Argument | Default | Description
--- | --- | ---
data_dir | path **REQUIRED** | Specify where to find `methylprep` processed output files.
outfilepath | `None` | If not specified, QC_Report.xlsx is saved in the same folder with the control probes file.
colorblind  | `False` | Set to True to enabled pass/fail colors that work better for colorblind users.
bg_offset | 3000 | Illumina's software add 3000 to the baseline/background values as an offset. This function does the same, unless you change this value. Changing this will affect whether some tests pass the cutff_adjust thresholds or not.
cutoff_adjust | 1.0 | A value of 1.0 will replicate Illumina's definition of pass/fail cutoff values for tests. You can increase this number to apply more stringent benchmarks for what constitutes good data, typically in the range of 1.0 to 3.0 (as a 1X to 3X cutoff multiplier).
legacy | `False` | `True`: changes XLSX column names to match BeadArray output file exactly. So columns are TitleCase instead of readable text; drops the formulas row;  splits sample column into two SentrixID columns. Rounds all numbers to 1 decimal instead of 2. Use this if you have automation that uses the Illumina legacy format. This will also exclude p-value poobah column and the result (interpretation) column from report.
roundoff | 2 | Option to adjust the level of rounding in report output numbers to this many decimal places.
pval | `True` | `True`: Loads poobah_values.pkl and adds a column with percent of probes that failed per sample.
pval_sig | 0.05 | pval significance level to define passing/failing probes
passing | 0.7 | Option to specify the fraction (from 0 to 1.0) of tests that must pass for sample to pass.  Note that this calculation also weights failures by how badly the miss the threshold value, so if any tests is a massive failure, the sample fails. Also, if the pval(poobah) pickle file is available in the folder, and over 20% of probes fail, the sample fails regardless of other tests.

### How `result` OK|FAIL|MARGINAL column is calculated

- Each test result for each sample appears in a column, along with the PASS/FAIL criteria shown at the top. All calculated values are designed so that low scores are poor quality, higher scores are better. The `result` column tallies the percentage of other columns that pass and if at least 70% of tests pass the minimum threshold, the sample passes overall with an OK status.
- If less than 100% of tests pass, the OK may have a number beside it between 0 and 1, reflecting the fraction of tests that passed for that sample.
- If less than 70% of tests pass, you'll see a FAIL or MARGINAL result, depending on how close to the minimum it got.
- Separately, if the poobah_values.pkl file is available, and >20% of probes failed p-value detection (based on fluorescence intensity), the sample fails. IF this file is absent, this test is ignored.
- The predicted sex for each sample is shown, and, if the sample sheet contains a "Sex" or "Gender" column, it will match the predicted sex to the reported sex and flag any mismatches. This, however, does not affect the `result` column.
- You can adjust the pass/fail criteria using `passing` and `pval_sig` parameters in command line or in function kwargs.
