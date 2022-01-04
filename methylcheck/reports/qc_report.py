import pandas as pd
from pathlib import Path
from io import StringIO
import re
import sys
import logging
import math
LOGGER = logging.getLogger(__name__)
# app
import methylcheck

__all__ = ['run_pipeline', 'ReportPDF']

def run_pipeline(df, **kwargs):
    """Run a variety of probe and sample filters in tandem, then plot results

by specifying all of your options at once, instead of running every part of methylcheck
in piacemeal fashion.

this is analogous to using the methylcheck CLI, but for notebooks/scripts

required:
    df: (required)
        - data as a DataFrame of beta values (or DataFrame of m_values)
        - sample names in columns and probes in rows

parameters:
    verbose: (True/False)
        default: False -- shows extra info about processing if True
    silent: (True/False)
        default: False -- suppresses all warnings/info

    exclude_sex:
        filters out probes on sex-chromosomes
    exclude_control:
        filters out illumina control probes
    exclude_all:
        filters out the most probes (sex-linked, control, and all sketchy-listed probes from papers)
    exclude: (list of strings, shorthand references to papers with sketchy probes to exclude)

        If the array is 450K the publications may include:
            ``'Chen2013'
            'Price2013'
            'Zhou2016'
            'Naeem2014'
            'DacaRoszak2015'``
        If the array is EPIC the publications may include:
            ``'Zhou2016'
            'McCartney2016'``
        or these reasons:
            ``'Polymorphism'
            'CrossHybridization'
            'BaseColorChange'
            'RepeatSequenceElements'``
        or use ``'exclude_all'``:
            to do maximum filtering, including all of these papers' lists.

    plot: (list of strings)
        ['mean_beta_plot', 'beta_density_plot', 'cumulative_sum_beta_distribution', 'beta_mds_plot', 'all']
        if 'all', then all of these plots will be generated. if omitted, no plots are created.
    save_plots: (True|False)
        default: False
    export (True|False):
        default: False -- will export the filtered df as a pkl file if True

note:
    this pipeline cannot also apply the array-level methylcheck.run_qc() function
    because that relies on additional probe information that may not be present. Everything
    in this pipeline applies to a dataframe of beta or m-values for a set of samples.

returns:
    a filtered dataframe object
    """
    # verify kwargs are expected strings
    param_list ={
    'exclude': ['Chen2013',
                'Price2013',
                'Zhou2016',
                'Naeem2014'
                'DacaRoszak2015',
                'Zhou2016',
                'McCartney2016',
                'Polymorphism',
                'CrossHybridization',
                'BaseColorChange',
                'RepeatSequenceElements'],
    'verbose': [True, False],
    'silent': [True, False],
    'exclude_sex': [True, False],
    'exclude_control': [True, False],
    'exclude_all': [True, False],
    'plot': ['mean_beta_plot', 'beta_density_plot', 'cumulative_sum_beta_distribution', 'beta_mds_plot', 'all'],
    'save_plots': [True, False],
    'export': [True, False],
    'run_qc': [True, False],
    'on_lambda': [True, False],
    }
    for k,v in kwargs.items():
        if k not in param_list:
            raise KeyError(f"{k} not an option among {list(param_list.keys())}")
        possible_values = param_list[k]
        if type(v) is list:
            for item in v:
                if item not in possible_values:
                    raise ValueError(f"{item} not a valid option for {k}: {possible_values}")
        else:
            if v not in possible_values:
                raise ValueError(f"{item} not a valid option for {k}: {possible_values}")
    # now we know all inputs are among the allowed/expected params.

    # set detail level of messages
    if kwargs.get('verbose') == True:
        logging.basicConfig(level=logging.INFO)
    else:
        kwargs['verbose'] = False
        logging.basicConfig(level=logging.WARNING)
    if kwargs.get('silent') == True:
        logging.basicConfig(level=logging.ERROR)
    else:
        kwargs['silent'] = False

    # assign some default pipeline params, if not specified by user
    default_params = {
        'verbose': False,
        'silent': False,
        'exclude_all': True,
        'plot': 'all',
        'save_plots': False,
        'export': False,
        'on_lambda': False}
    for param,setting in default_params.items():
        if kwargs.get(param) is None:
            kwargs[param] = setting

    # determine array type
    array_type = methylcheck.detect_array(df, on_lambda=kwargs.get('on_lambda'))

    # apply some filters
    if kwargs.get('exclude_all'):
        try:
            df = methylcheck.exclude_sex_control_probes(df, array_type, verbose=kwargs.get('verbose'))
            sketchy_probe_list = methylcheck.list_problem_probes(array_type)
            df = methylcheck.exclude_probes(df, sketchy_probe_list)
        except ValueError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if "probe list" in str(exc_value) or "probe exclusion lists" in str(exc_value):
                # keep df the same instead of filtering.
                LOGGER.info(exc_value)
            else:
                raise Exception(exc_value)
    elif kwargs.get('exclude'):
        # could be a list or a string
        if type(kwargs['exclude']) is not list:
            kwargs['exclude'] = [kwargs['exclude']]
        try:
            sketchy_probe_list = methylcheck.list_problem_probes(array_type, criteria=kwargs['exclude'])
            df = methylcheck.exclude_probes(df, sketchy_probe_list)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if "probe list" in str(exc_value) or "probe exclusion lists" in str(exc_value):
                LOGGER.info(exc_value)
            else:
                raise Exception(exc_value)

    # apply some more filters
    if kwargs.get('exclude_sex') and kwargs.get('exclude_control'):
        df = methylcheck.exclude_sex_control_probes(df, array_type, verbose=kwargs.get('verbose'))
    elif kwargs.get('exclude_sex'):
        df = methylcheck.exclude_sex_control_probes(df, array_type, no_sex=True, no_control=False, verbose=kwargs.get('verbose'))
    elif kwargs.get('exclude_control'):
        df = methylcheck.exclude_sex_control_probes(df, array_type, no_sex=False, no_control=True, verbose=kwargs.get('verbose'))

    # plot stuff
    # run calculations and plots
    if 'all' in kwargs.get('plot'):
        methylcheck.mean_beta_plot(df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
        methylcheck.beta_density_plot(df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
        wide_df = df.copy().transpose()
        methylcheck.cumulative_sum_beta_distribution(wide_df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
        pre_mds = wide_df.shape[0]
        df = methylcheck.beta_mds_plot(wide_df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
        LOGGER.info(f"MDS filtering dropped {pre_mds - df.shape[1]} samples from dataframe.")
    else:
        if 'mean_beta_plot' in args.plot:
            methylcheck.mean_beta_plot(df, verbose=kwargs['verbose'], save=kwargs.get('save_plots'), silent=kwargs['silent'])
        if 'beta_density_plot' in args.plot:
            methylcheck.beta_density_plot(df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
        if 'cumulative_sum_beta_distribution' in args.plot:
            wide_df = df.copy().transpose()
            methylcheck.cumulative_sum_beta_distribution(wide_df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
        if 'beta_mds_plot' in args.plot:
            wide_df = df.copy().transpose()
            pre_mds = wide_df.shape[0]
            df = methylcheck.beta_mds_plot(wide_df, verbose=kwargs['verbose'], save=kwargs['save_plots'], silent=kwargs['silent'])
            LOGGER.info(f"MDS filtering dropped {pre_mds - df.shape[1]} samples from dataframe.")
            # also has filter_stdev params to pass in.
    if kwargs.get('export') == True:
        outfile = 'beta_values_filtered.pkl' # in long format for faster loading
        df.to_pickle(outfile)
        LOGGER.info(f'Saved {outfile}')
    return df


def detection_poobah(poobah_df, pval_cutoff=0.05):
    """Given a dataframe of p-values with sample_ids in columns and probes in index,
    calculates the percent PASSING probes per sample. Part of QC report.

    v0.7.2: if 'gct_score' is enabled, add column for GCT % per sample.
    v0.7.2: changed from percent failing to percent passing"""
    out = {}
    for sample_id in poobah_df.columns:
        total = poobah_df.shape[0]
        passing = poobah_df[sample_id][poobah_df[sample_id] <= pval_cutoff].count()
        percent_passing = round(100*passing/total,1)
        out[sample_id] = percent_passing
    return out


class ReportPDF:
    """ReportPDF allows you to build custom QC reports.

To use:

- First, initialize the report and pass in kwargs, like ``myReport = ReportPDF(**kwargs)``
- Next, run ```myReport.run_qc()`` to fill it in.
- Third, you must run ``myReport.pdf.close()`` after ``run_qc()`` to save the file to disk.
- You can supply kwargs to specify which QC plots to include
  - and supply a list of chart names to control the order of objects in the report.
  - if you pass in 'order' in kwargs, any page you omit in the order will be omitted from the final report.
  - You may pass in a custom table to override one of the built-in pages.
- include 'path' with the path to your processed pickle files.
- include an optional 'outpath' for where to save the pdf report.

kwargs:

- processing params
  - filename
  - poobah_min_percent (e.g. at least 80% of probes must pass for sample to pass)
  - pval_cutoff (e.g. set alpha at 0.05)
  - outpath
  - path
  - runme: Default is not to actually generate all the parts of PDF with report.run_qc() then report.pdf.close(), but setting this to True will do everything at once.
- front page text
  - title
  - author
  - subject
  - keywords
- if 'debug=True' is in kwargs,
  - then it will return a report without any parts that failed.
- tests
  - poobah: includes a table with each sample and percent of probes that passed p-value signal detection
  - gct: includes GCT scores (bisulfite conversion completeness) in poobah table
  - mds: performs multidimensional scaling to identify and report on sample outliers
- plots
  - beta_density_plot
  - M_vs_U (default False)
  - M_vs_U_compare (default False) -- shows the effect of all processing steps vs raw intensity
  - qc_signal_intensity
  - controls (A battery of probe performance plots)
  - probe_types
- customizing plots
  - poobah_colormap (pass in the matplotlib colormap name to override the meta_mds default colormap)
    This also overrides the default colormap used in M_vs_U plot.
  - extend_poobah_range (Default: True will show 7 colors for poobah failure range on beta_mds_plot, max 30%; False will show only 5, max 20%)
  - cutoff_line -- False to disable cutoff line on qc_signal_intensity and M_vs_U plots
  - appendix_fontsize (default 12 point) -- specify an int for other fontsize

custom tables:

pass in arbitrary data using kwarg ``custom_tables`` as list of dictionaries with this structure:

```python
custom_tables=[
{
'title': "some title, optional", # NOTE: chart titles must be unique!
'col_names': ["<list of strings>"],
'row_names': ["<list of strings, optional>"],
'data': ["<list of lists, with order matching col_names>"],
'order_after': "string name of the plot this should come after. It cannot appear first in list.",
'font_size': <can be None, int, 'auto' (shrink to page), or 'truncate' (chop of long values to make fit)>
},
{"<...second table here...>"}
]
```

If 'order_after' is None, the custom table will be inserted at the beginning of the report.
If there are multiple custom tables and all have 'order_after' set to None, the first table
in the list gets inserted, then the next one, sequentially, so that the last table inserted
will be the first table to appear.

Pre-processing pipeline:

Probe-level (w/explanations of suggested exclusions)
    - Links to recommended probe exclusion lists/files/papers
    - Background subtraction and normalization ('noob')
    - Detection p-value ('neg' vs 'oob')
    - Dye-bias correction (from SeSAMe)

Sample-level (w/explanations of suggested exclusions)
    Detection p-value (% failed probes)
    - custom detection (% failed, of those in a user-defined-list supplied to function)
    MDS

Suggested for customer to do on their own
    - Sex check
    - Age check
    - SNP check
    """
    # larger font
    # based on 16pt with 0.1 (10% of page) margins around it: use 80, 26, 16
    # MAXWIDTH = 80 # based on 16pt with 0.1 (10% of page) margins around it
    # MAXLINES = 26
    # FONTSIZE = 16
    # ORIGIN = (0.1, 0.1) # achored on page in lower left
    #
    # normal font -- for 12pt font: use 100 x 44 lines
    MAXWIDTH = 100
    MAXLINES = 44
    FONTSIZE = 12
    ORIGIN = (0.1, 0.05) # achored on page in lower left

    def __init__(self, **kwargs):
        # https://stackoverflow.com/questions/8187082/how-can-you-set-class-attributes-from-variable-arguments-kwargs-in-python
        self.__dict__.update(kwargs)
        self.debug = True if self.__dict__.get('debug') == True else False
        self.__dict__['poobah_min_percent'] = self.__dict__.get('poobah_min_percent', 80)
        self.__dict__['pval_cutoff'] = self.__dict__.get('pval_cutoff', 0.05)
        self.errors = self.open_error_buffer()

        #SHELVING the PDF_PART, since it worked but not neeeded right now
        from matplotlib.backends.backend_pdf import PdfPages
        self.outfile = Path(self.__dict__.get('outpath','.'), self.__dict__.get('filename', 'multipage_pdf.pdf'))
        self.pdf = PdfPages(self.outfile)
        import matplotlib.pyplot as plt
        self.plt = plt
        import textwrap
        self.textwrap = textwrap
        import datetime
        self.today = str(datetime.date.today())
        self.on_lambda = self.__dict__.get('on_lambda', False)
        d = self.pdf.infodict()
        if any(kwarg in ('title','author','subject','keywords') for kwarg in kwargs):
            # set the file's metadata via the PdfPages object:
            d['Title'] = self.__dict__.get('title','')
            d['Author'] = self.__dict__.get('author','')
            d['Subject'] = self.__dict__.get('subject','')
            d['Keywords'] = self.__dict__.get('keywords','')
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

        if 'order' in self.__dict__:
            self.order = self.__dict__['order']
        else:
            self.order = ['beta_density_plot', 'detection_poobah', 'predict_sex', 'mds', 'auto_qc',
            'qc_signal_intensity', 'M_vs_U_compare', 'M_vs_U', 'controls', 'probe_types']

        self.tests = {
            'detection_poobah': self.__dict__.get('poobah',True),
            'mds': self.__dict__.get('mds',True), # part of detection_poobah table
            'auto_qc': self.__dict__.get('auto_qc',True), # algorithmic failure prediction, NOT IMPLEMENTED YET #
            # v0.7.3 adds bis-conversion completeness, the GCT score table --- https://rdrr.io/bioc/sesame/src/R/sesame.R
            'gct_score': self.__dict__.get('gct',True), # part of detection_poobah table
        }
        self.plots = {
            'beta_density_plot': self.__dict__.get('beta_density_plot', True),
            'qc_signal_intensity': self.__dict__.get('qc_signal_intensity', True),
            'M_vs_U_compare': self.__dict__.get('M_vs_U_compare', False),
            'M_vs_U': self.__dict__.get('M_vs_U', False),
            'predict_sex': self.__dict__.get('predict_sex', False),
            'controls': self.__dict__.get('controls', True),
            'probe_types': self.__dict__.get('probe_types', True),
        }

        self.custom = {}
        if 'custom_tables' in self.__dict__:
            self.parse_custom_tables(self.__dict__['custom_tables'])
            LOGGER.info(f"Found custom_tables and inserted into order: {self.order}")
            #LOGGER.info(self.custom)

        self.poobah_colormap = kwargs.get('poobah_colormap', None)
        self.extend_poobah_range = kwargs.get('extend_poobah_range', True)
        self.cutoff_line = kwargs.get('cutoff_line', True)
        self.appendix_fontsize = kwargs.get('appendix_fontsize', None) # reverts to self.FONTSIZE if blank

        if self.__dict__.get('runme') == True:
            self.run_qc()
            self.pdf.close()

    def parse_custom_tables(self, tables):
        """tables is a list of {
        'title': "some title, optional",
        'col_names': [list of strings],
        'row_names': [list of strings, optional],
        'data': [list of lists, with order matching col_names],
        'order_after': string name of the plot this should come after. It cannot appear first in list.
        }"""
        for table in tables:
            required_attributes = {'title', 'data', 'order_after', 'col_names'}
            for i in required_attributes:
                if i not in table:
                    raise KeyError("Your custom table must contain these keys: {required_attributes} (row_names is optional)")
            #1 place order -- refer to this later in self.custom
            if table['order_after'] == None: # key must be present and set to None
                _index = 0 # insert the table at start of report.
            else:
                try:
                    _index = self.order.index(table['order_after']) + 1
                except ValueError:
                    raise ValueError(f"Your custom table's 'order_after' label is not in this list of chart objects, so could not be ordered: {self.order}")
            self.order.insert(_index, table['title'])
            self.custom[table['title']] = table

            if not isinstance(table['col_names'], list):
                raise TypeError(f"Your custom table 'col_names' must be a list.")
            if table.get('row_names') and not isinstance(table['row_names'], list):
                raise TypeError(f"Your custom table 'row_names' must be a list.")
            if not isinstance(table['data'], list):
                raise TypeError(f"Your custom table 'data' must be a list of lists, matching the number of columns listed in col_names.")
                if not (isinstance(table['data'][0],list) and len(table['data'][0]) == len(table['col_names'])):
                    raise TypeError(f"Your custom table 'data' must be a list of lists, matching the number of columns listed in col_names.")
            # passes! later, the table will be looked-up by title name and fed into chart in order.

    def open_error_buffer(self):
        """ preparing a stream of log messages to add to report appendix. """
        appendix = logging.getLogger()
        appendix.setLevel(logging.INFO)
        errors = StringIO()
        error_handler = logging.StreamHandler(errors)
        # %(module)s.%(funcName)s() ... %Y-%m-%d
        formatter = logging.Formatter('^^^%(asctime)s - %(levelname)s - %(message)s', "%H:%M:%S")
        error_handler.setFormatter(formatter)
        appendix.addHandler(error_handler)
        return errors

    def run_qc(self):
        # load all the data from self.path; make S3-compatible too.
        path = self.__dict__.get('path','')
        if self.tests['detection_poobah'] is True and 'detection_poobah' in self.order:
            try:
                poobah_df = pd.read_pickle(Path(path,'poobah_values.pkl')) # off by default, process with --export_poobah
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not load pickle file: 'poobah_values.pkl'. Add 'poobah=False' to inputs to turn this off.")
        if self.tests['gct_score'] is True and self.tests['detection_poobah'] is False:
            LOGGER.warning(f"Cannot include GCT score unless poobah table is also enabled.")
        try:
            beta_df = pd.read_pickle(Path(path,'beta_values.pkl'))
            if Path(path,'noob_meth_values.pkl').exists() and Path(path,'noob_unmeth_values.pkl').exists():
                meth_df = pd.read_pickle(Path(path,'noob_meth_values.pkl'))
                unmeth_df = pd.read_pickle(Path(path,'noob_unmeth_values.pkl'))
            else:
                meth_df = pd.read_pickle(Path(path,'meth_values.pkl'))
                unmeth_df = pd.read_pickle(Path(path,'unmeth_values.pkl'))
            control_dict_of_dfs = pd.read_pickle(Path(path,'control_probes.pkl'))
            # for M_vs_U_compare only, need both
            if self.__dict__.get('M_vs_U_compare') == True:
                if not (Path(path,'noob_meth_values.pkl').exists() and
                    Path(path,'noob_unmeth_values.pkl').exists() and
                    Path(path,'meth_values.pkl').exists() and
                    Path(path,'unmeth_values.pkl').exists()):
                    LOGGER.error("Not all meth/unmeth pickle files found for M_vs_U compare plot. Skipping this plot.")
                    self.__dict__.pop('M_vs_U_compare')
            LOGGER.info("Data loaded")
        except FileNotFoundError:
            raise FileNotFoundError("Could not load pickle files")

        if 'mds' in self.tests and len(beta_df.columns) > 1:
            # some things must be calculated ahead of time, because used twice
            poobah_path = Path(path,'poobah_values.pkl') if 'detection_poobah' in self.order else None
            beta_mds_fig, ax, df_indexes_to_retain = methylcheck.beta_mds_plot(beta_df, silent=True, multi_params={'return_plot_obj':True, 'draw_box':True},
                poobah=poobah_path, palette=self.poobah_colormap, extend_poobah_range=self.extend_poobah_range)
            mds_passing = [sample_id for idx,sample_id in enumerate(beta_df.columns) if idx in df_indexes_to_retain]
            include_mds = True
        else:
            beta_mds_fig = None
            mds_passing = []
            include_mds = False

        for part in self.order:
            if part in self.tests and self.tests[part] == True:
                try:
                    if part == 'detection_poobah':
                        min_allowed = self.__dict__.get('poobah_min_percent')
                        sample_percent_failed_probes_dict = detection_poobah(poobah_df, pval_cutoff=self.__dict__['pval_cutoff'])
                        if self.tests['gct_score']:
                            sample_gct_percent_dict = methylcheck.bis_conversion_control(meth_df)
                        else:
                            sample_gct_percent_dict = {}
                        sample_poobah_failures = {sample_id: ("pass" if percent > min_allowed else "fail") for sample_id,percent in sample_percent_failed_probes_dict.items()}
                        #LOGGER.info(f"Poobah: {len([k for k in sample_poobah_failures.values() if k =='fail'])} failure(s) out of {len(sample_poobah_failures)} samples.")

                        list_of_lists = []
                        col_names=['Sample_ID', 'Percent', 'Pass/Fail']
                        if self.tests['gct_score']:
                            col_names=['Sample_ID', 'GCT score', 'Percent', 'Pass/Fail']
                        if include_mds:
                            col_names=['Sample_ID', 'MDS', 'Percent', 'Pass/Fail']
                        if self.tests['gct_score'] and include_mds:
                            col_names=['Sample_ID', 'GCT score', 'MDS', 'Passing Probes (%)', 'Pass/Fail']

                        for sample_id,percent in sample_percent_failed_probes_dict.items():
                            if self.tests['gct_score'] and sample_id in sample_gct_percent_dict:
                                gct_score = sample_gct_percent_dict[sample_id]
                            else:
                                gct_score = None
                            mds_pass_fail = 'Pass' if sample_id in mds_passing else 'Fail'

                            if gct_score is not None and include_mds:
                                row = [sample_id, gct_score, mds_pass_fail, percent, sample_poobah_failures[sample_id]]
                            elif gct_score is not None and not include_mds:
                                row = [sample_id, gct_score, percent, sample_poobah_failures[sample_id]]
                            elif gct_score is None and include_mds:
                                row = [sample_id, mds_pass_fail, percent, sample_poobah_failures[sample_id]]
                            elif gct_score is None and not include_mds:
                                row = [sample_id, percent, sample_poobah_failures[sample_id]]

                            list_of_lists.append(row)
                        self.to_table(list_of_lists, col_names=col_names,
                            row_names=None, add_title='Sample probe quality testing')

                    if part == 'mds' and len(beta_df.columns) > 1:
                        LOGGER.info("Beta MDS Plot")
                        # ax and df_to_retain are not used, but could go into a qc chart
                        self.pdf.savefig(beta_mds_fig)
                        self.plt.close()
                        #pretty_table = Table(titles=['Sample_ID', 'MDS Pass/Fail'])

                except Exception as e:
                    if self.debug:
                        LOGGER.error(f"Could not process {part}; {e}")
                        continue
                    else:
                        raise Exception(f"Could not process {part}; {e}")

            elif part in self.plots and self.plots[part] == True:
                try:
                    if part == 'beta_density_plot':
                        LOGGER.info("Beta Density Plot")
                        fig = methylcheck.beta_density_plot(beta_df, save=False, silent=True, verbose=False, reduce=0.1, plot_title=None, ymax=None, return_fig=True)
                        self.pdf.savefig(fig)
                        self.plt.close()
                    elif part == 'M_vs_U':
                        LOGGER.info(f"M_vs_U plot")
                        fig = methylcheck.plot_M_vs_U(meth=meth_df, unmeth=unmeth_df, noob=True, silent=False, verbose=True, plot=True,
                            compare=False, return_fig=True, poobah=poobah_df, palette=self.poobah_colormap, cutoff_line=self.cutoff_line)
                        self.pdf.savefig(fig)
                        self.plt.close()
                        # if not plotting, it will return dict with meth median and unmeth median.
                    elif part == 'M_vs_U_compare':
                        LOGGER.info(f"M_vs_U_compare plot")
                        fig = methylcheck.plot_M_vs_U(data_containers_or_path=path, compare=True, silent=False, verbose=True, plot=True, return_fig=True)
                        self.pdf.savefig(fig)
                        self.plt.close()
                        # if not plotting, it will return dict with meth median and unmeth median.
                    elif part == 'qc_signal_intensity':
                        LOGGER.info(f"QC signal intensity plot")
                        fig = methylcheck.qc_signal_intensity(meth=meth_df, unmeth=unmeth_df, silent=True, return_fig=True, poobah=poobah_df, cutoff_line=self.cutoff_line, palette=self.poobah_colormap)
                        self.pdf.savefig(fig)
                        self.plt.close()
                    elif part == 'controls':
                        LOGGER.info(f"Control probes")
                        list_of_figs = methylcheck.plot_controls(control_dict_of_dfs, 'all', return_fig=True)
                        for fig in list_of_figs:
                            self.pdf.savefig(figure=fig, bbox_inches='tight')
                        self.plt.close('all')
                    elif part == 'probe_types':
                        LOGGER.info(f"Betas by probe type")
                        list_of_figs = methylcheck.plot_beta_by_type(beta_df, 'all', return_fig=True, silent=True, on_lambda=self.on_lambda)
                        for fig in list_of_figs:
                            self.pdf.savefig(figure=fig, bbox_inches='tight')
                        self.plt.close('all')
                    elif part == 'predict_sex':
                        # get_sex() handles missing data issues; no need to validate files exist
                        include_probe_failure_percent = True if (self.tests['detection_poobah'] is True and 'detection_poobah' in self.order) else False
                        poobah_cutoff = (100 - self.__dict__.get('poobah_min_percent',80))
                        fig = methylcheck.get_sex(path, plot=True, save=False, on_lambda=self.on_lambda,
                            include_probe_failure_percent=include_probe_failure_percent, poobah_cutoff=poobah_cutoff,
                            return_fig=True) # custom_label={dict of values for samples keyed to sample IDs}
                        self.pdf.savefig(fig.fig) # seaborn relplots embed the matplotlib fig within them
                        self.plt.close()
                except Exception as e:
                    if self.debug:
                        LOGGER.error(f"Could not process {part}; {e}")
                        continue
                    else:
                        raise Exception(f"Could not process {part}; {e}")

            elif part in self.custom:
                table = self.custom[part]
                self.to_table(table['data'], col_names=table['col_names'],
                    row_names=table.get('row_names'), add_title=table['title'],
                    font_size=table.get('font_size',None))


        # and finally, appendix of log messages
        appendix_msgs = self.errors.getvalue()
        appendix_msgs = appendix_msgs.split('^^^')
        # drop the trailing \n at end of each message, if present, but preserve within-line \n line breaks
        appendix_msgs = [(msg[:-2] if msg[-2:] == '\n' else msg) for msg in appendix_msgs]
        appendix_msgs.insert(0, 'APPENDIX I: PROCESSING MESSAGES')
        total_rows = len(appendix_msgs)
        last_row = 0
        while last_row <= total_rows:
            rows_per_page = 26
            # pretest whether length will fit on a pgae
            while True:
                para_list = appendix_msgs[last_row : (last_row + rows_per_page)]
                para_lengths = [len(i.split('\n')) for i in para_list]
                extra_line_wraps = [max(len(self.textwrap.wrap(para, width=self.MAXWIDTH)) -1, 0) for para in para_list]
                #print((sum(para_lengths) + len(para_lengths) + sum(extra_line_wraps)))
                if (sum(para_lengths) + len(para_lengths) + sum(extra_line_wraps)) > self.MAXLINES:
                    rows_per_page -= 2
                    #print('---',rows_per_page)
                else:
                    break
                if rows_per_page <= 0:
                    break
            if rows_per_page <= 0:
                LOGGER.error("couldn't write log messages to appendix")
                break
            self.page_of_paragraphs(para_list, self.pdf, line_height='single', fontsize=self.appendix_fontsize)
            last_row += rows_per_page
        self.errors.close()


    def exec_summary(self):
        """QC exec summary
    	sample_name/ID
    	probe % failures
    	probe_failure pass
    	auto-qc result (only if present in kwargs passed in, otherwise omitted)
    	MDS pass
    	signal intensity pass
    	if any fails, fail it (so overall pass)
    table 2: meta
    	array type (detect from data)
    	number of samples (from data)
    	processing pipeline version number (passed in)
    	date processed (passed in)
    	avg probe failure rate
    	percent of samples that failed
    	any failures from 'control probes'
    	   reqs a way to capture warnings of data-off-chart
        """
        exec_summary_1 = []
        exec_summary_2 = [] # list of lists
        exec_summary_samples = [] # temp storage lookup
        pass

    def page_of_text(self, text, pdf, fontsize=None):
        """text is a single big string of text, with whitespace for line breaks.
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.text.html (0,0) is lower left; (1,1) is upper right """
        #print([len(i.split('\n')) for i in [text]])
        firstPage = self.plt.figure(figsize=(11,8.5))
        firstPage.clf()
        wrapped_txt = self.textwrap.fill(text, width=self.MAXWIDTH)
        #wrapped_txt_list = textwrap.wrap(txt, width=80)
        # next, identify paragraph breaks here as strings of writespace, and wrap each part separately? then move onto page.
        #firstPage.text(0.5,0.5,txt, size=12, ha="left") # transform=firstPage.transFigure
        fontsize = self.FONTSIZE if fontsize is None else fontsize
        firstPage.text(self.ORIGIN[0], self.ORIGIN[1], wrapped_txt, size=fontsize)
        pdf.savefig()
        self.plt.close()


    def page_of_paragraphs(self, para_list, pdf, line_height='double', fontsize=None):
        """ https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.text.html (0,0) is lower left; (1,1) is upper right.

        This version estimates the size of each paragraph and moves the origin downward accordingly.
        Thsis is tricky because the anchors are lower left, not upper left.

        It is ok if a paragraph contains whitespace line breaks, OR each paragraph is one long line to be wrapped here.
        Also - if a paragraph wraps, this accounts for it in total lines count, so everything fits on a page.
        The default fontsize is 12 if not specified."""
        #1 - finding the start points for each paragraph in page, counting backwards, so first paragraph is at top of page.
        para_lengths = [len(i.split('\n')) for i in para_list]
        extra_line_wraps = [max(len(self.textwrap.wrap(para, width=self.MAXWIDTH)) -1, 0) for para in para_list]
        if (sum(para_lengths) + len(para_lengths) + sum(extra_line_wraps)) > self.MAXLINES:
            raise ValueError(f"Text lines {(sum(para_lengths) + len(para_lengths) + sum(extra_line_wraps))} > {self.MAXLINES}, the max lines per PDF page allowed.")
        #print(para_lengths) -- ok if a paragraph contains whitespace line breaks, OR each para is one long line to be wrapped here.
        # assume each line height is 0.1/26 fraction of total page height

        # paragraph spacing
        if line_height == 'double':
            line_height = round((1 - 2*self.ORIGIN[1])/self.MAXLINES,3)
        elif line_height == 'single':
            line_height = round(0.5*(1 - 2*self.ORIGIN[1])/self.MAXLINES,3)
        else:
            pass
            # else, line_height is whatever int/float you pass in, for custom spacing. (as fraction of page, so use 0.01 to 0.03)

        paragraph_spacing = line_height # 1 line

        firstPage = self.plt.figure(figsize=(11,8.5))
        firstPage.clf()
        current_y_position = (1.0 - self.ORIGIN[1]) # working down page by subtracting
        extra_lines = [] # if something doesn't fit on page, include this in error message
        for nth, para in enumerate(para_list):
            # each para is assumed to be one paragraph of text with no whitespace formatting.
            wrapped_txt = self.textwrap.fill(para, width=self.MAXWIDTH)
            wrapped_addtl_line_count = max(len(self.textwrap.wrap(para, width=self.MAXWIDTH)) -1, 0)
            x_margin = self.ORIGIN[0]
            current_y_position -= (line_height * len(para.split('\n')))
            current_y_position -= (line_height * wrapped_addtl_line_count)
            if nth > 0:
                current_y_position -= paragraph_spacing
            if current_y_position < self.ORIGIN[1]:
                extra_lines.append(wrapped_txt)
                #print(current_y_position, wrapped_addtl_line_count, wrapped_txt)
            else:
                fontsize = self.FONTSIZE if fontsize is None else fontsize
                firstPage.text(x_margin, current_y_position, wrapped_txt, size=fontsize)
        if extra_lines != []:
            print("WARNING: DID NOT FIT")
            for extra_line in extra_lines:
                print(extra_line)
                print("")
        pdf.savefig()
        self.plt.close()

    def to_table(self, list_of_lists, col_names, row_names=None, add_title='', font_size='auto'):
        """
        - embeds a table in a PDF page.
        - attempts to split long tables into multiple pages.
        - should warn if table is too wide to fit.
        - font_size:
          - auto: let matplotlib figure out the best size. This breaks if one column or value is soooooooper long.
          - truncate: IN FUTURE, chop off each value that is too long, to force it to fit on a page. Data loss.
          - INT: set the font for this table at this size (for control freaks)
          - None: default -- use font size 12."""

        # stringify numbers; replace None
        for idx, sub in enumerate(list_of_lists):
            list_of_lists[idx] = [('' if isinstance(i,type(None)) else str(i)) for i in sub]

        rows_per_page = 42 # appears to work for standard default PDF page size and font size.
        pages = math.ceil(len(list_of_lists)/rows_per_page)
        #paginate table
        for page in range(pages):
            #LOGGER.debug(f'page, cols: {len(col_names)}, rows: {len(list_of_lists)}')
            truncate = False
            if isinstance(font_size, str) and font_size.lower() == 'truncate':
                # overwrites list_of_lists so must happen before table is created
                truncate = True
                truncate_font_size = 12
                try:
                    max_field_length = 24
                    # default max length = 24 chars; if more than 4 columns, reduce by ((N-4)/20)*80 chars
                    if 4 < len(col_names) < 10:
                        max_field_length = int((24 - ((len(col_names)-4)/20) * 80))
                        if max_field_length <= 4: # 9 col
                            truncate_font_size = 6
                            max_field_length += 8
                        elif max_field_length <= 8: # 8 col
                            truncate_font_size = 7
                            max_field_length += 3
                        elif max_field_length <= 12: # 7 col
                            truncate_font_size = 7
                            max_field_length += 1
                        elif max_field_length <= 16: # 6 col
                            truncate_font_size = 7
                            #max_field_length += 1
                        elif max_field_length <= 20: # 5 col
                            truncate_font_size = 7
                            #max_field_length += 0
                        elif max_field_length <= 24: # 4 col
                            truncate_font_size = 8
                            #max_field_length += 0
                    elif len(col_names) >= 10:
                        LOGGER.error(f"Could not truncate long fields in custom table because there were 10 or more columns.")
                        truncate = False
                    if truncate is True:
                        for row_num, row in enumerate(list_of_lists.copy()):
                            for field_num,field in enumerate(row):
                                if len(str(field)) > max_field_length:
                                    list_of_lists[row_num][field_num] = str(field)[:max_field_length] + '..'
                                    if self.debug:
                                        LOGGER.info(f"{field} truncated to {field[:max_field_length]}..")
                        for field_num,field in enumerate(col_names):
                            if len(str(field)) > max_field_length:
                                col_names[field_num] = str(field)[:max_field_length] + '..'
                                if self.debug:
                                    LOGGER.info(f"column name {field} truncated to {field[:max_field_length]}..")
                except Exception as e:
                    truncate = False
                    LOGGER.error(f"Could not truncate long fields in custom table: {e}")
            page_data = list_of_lists[(page * rows_per_page) : (page + 1) * rows_per_page]
            fig, ax = self.plt.subplots(1, figsize=(8,10.5))
            ax.axis('off')
            matplotlib_table = ax.table(
                cellText=page_data,
                colLabels=col_names,
                loc='center', # if len(page_data) >= rows_per_page else 'top',
                edges='open',
                colLoc='right',
                )

            if font_size is None:
                matplotlib_table.auto_set_font_size(False)
                matplotlib_table.set_fontsize(12)
            elif isinstance(font_size, int):
                matplotlib_table.auto_set_font_size(False)
                matplotlib_table.set_fontsize(font_size)
            elif ( (isinstance(font_size, str) and font_size.lower() == 'auto') or
                   (isinstance(font_size, str) and font_size.lower() == 'truncate' and truncate is False) ): #default shrinks to fit page
                matplotlib_table.auto_set_font_size(True)
                scale_ratio = 1.2 # use smaller page margins
                matplotlib_table.scale(scale_ratio, scale_ratio)
            elif truncate is True:
                matplotlib_table.auto_set_column_width(col=list(range(len(col_names)))) # Provide integer list of columns to adjust
                matplotlib_table.auto_set_font_size(False)
                matplotlib_table.set_fontsize(truncate_font_size)
                scale_ratio = 1.2
                matplotlib_table.scale(scale_ratio, scale_ratio)
            if page == 0:
                self.plt.title(add_title, y=1.1) #pad=20) # -- placement is off
            #fig.set_size_inches(8,10.5)
            self.pdf.savefig(fig)
            self.plt.close()
