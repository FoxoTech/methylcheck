# Lib
import argparse
from argparse import RawTextHelpFormatter
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
# App
from .probes.filters import list_problem_probes, exclude_probes, exclude_sex_control_probes
from .samples.postprocessQC import mean_beta_plot, beta_density_plot, cumulative_sum_beta_distribution, beta_mds_plot
from .reports import controls_report, ReportPDF

LOGGER = logging.getLogger(__name__)

class DefaultParser(argparse.ArgumentParser):
    def error(self, message):
        self._print_message('[Error]:\n')
        self._print_message(f'{message}\n\n')
        self.print_help()
        self.exit(status=2)


def detect_array(df, returns='name', on_lambda=False):
    """Determines array type using number of probes columns in df. Returns array string.
    Note: this is different from methylprep.models.arrays.ArrayType.from_probe_count, which looks at idat files.

    returns (name | filepath)
        default is 'name' -- returns a string
        if 'filepath', this also returns the filepath to the array, using ArrayType and
        methylprep.files.manifests ARRAY_TYPE_MANIFEST_FILENAMES.
    on_lambda (True | False)
        looks for manifest files in /tmp instead of ~/.methylprep_manifest_files

    returns one of: {27k, 450k, epic, epic+, mouse}
    """

    if returns == 'filepath':
        # get manifest data from .methylprep_manifest_files
        try:
            from methylprep.files.manifests import MANIFEST_DIR_PATH, MANIFEST_DIR_PATH_LAMBDA, ARRAY_TYPE_MANIFEST_FILENAMES, ARRAY_FILENAME
            from methylprep.models.arrays import ArrayType
        except ImportError:
            raise ImportError("this function requires `methylprep` be installed (to read manifest array files).")

        def get_filename(array_name):
            if on_lambda:
                man_path = Path(MANIFEST_DIR_PATH_LAMBDA).expanduser()
            else:
                man_path = Path(MANIFEST_DIR_PATH).expanduser()
            man_filename = ARRAY_FILENAME[array_name]
            man_filepath = Path(man_path, man_filename)
            return man_filepath

    # shape: should be wide, with more columns than rows. The larger dimension is the probe count.
    if df.shape[0] > df.shape[1]:
        # WARNING: this will need to be transposed later.
        col_count = (df.shape[0]) #does the index count in shape? assuming it doesn't.
    else:
        col_count = (df.shape[1])
    if 26000 <= col_count <= 28000:
        return '27k' if returns == 'name' else (ArrayType('27k'), get_filename('27k'))
    elif 440000 <= col_count <= 490000: # actual: 485512
        return '450k' if returns == 'name' else (ArrayType('450k'), get_filename('450k'))
    elif 868001 <= col_count <= 869335: # actual: 868578
        return 'epic+' if returns == 'name' else (ArrayType('epic+'), get_filename('epic+'))
    elif 860000 <= col_count <= 868000: # actual: 865860
        return 'epic' if returns == 'name' else (ArrayType('epic'), get_filename('epic'))
    elif 200000 <= col_count <= 300000: # actual: 236685 C20, 262812 v2, 284762 V3-V4, MM285 293220 --- testing, changed max from 270k to 290k for v1.4.6
        return 'mouse' if returns == 'name' else (ArrayType('mouse'), get_filename('mouse'))
    else:
        raise ValueError(f'Unsupported Illumina array type. Your data file contains {col_count} rows for probes.')


def build_parser():
    parser = DefaultParser(
        prog='methylcheck',
        description="""Visualization and quality control functions for methylation, using Illumina IDAT files processed with methylprep or sesame.

Usage
-----
To run the full battery of tests:
    `python -m methylcheck qc -d <data_file> -p all --exclude_all`
or for a standard QC REPORT
    `python -m methylcheck controls -d <data_dir>`

Commands
--------
qc
    - exclude_sex -- remove sex-chromosome-linked probes from a batch of samples.
    - exclude_control -- remove Illumina control probes  from a batch of samples.
    - exclude_probes -- remove less reliable probes, based on lists of problem probes from literature.
    look at the function help for details on how you fine tune the exclusion list.
    - plot -- various kinds of plots to reveal sample quality.
    - array_type -- QC requires the type of array.
    - verbose

controls
    - data_dir -- folder where control_probes.pkl and other methylprep outputs are found. [required]
    - outfilepath=None
        optional: specify a different folder to save XLSX file output
    - bg_offset=3000
        Illumina's defaults add 3000 to control probe intensities. You can adjust that here.
    - colorblind=False
        Set to True if you'd prefer an alternative color-coding to the
        red/yellow/green traffic light for pass/fail in XLSX file.
    - cutoff_adjust=1.0
        A value of 1.0 will mirror Illumina's recommended pass/fail thresholds. Setting it higher
        results in tests that are MORE conservative. For example, setting it to 2 or 3
        will increase the criteria 2X or 3X, respectively. Fractional values are OK.
        """,
        formatter_class=RawTextHelpFormatter,
    )

    #parser.add_argument(
    #    '-v', '--verbose',
    #    help='Provide more detailed processing messages.',
    #    action='store_true',
    #    default=False,
    #)

    parser.add_argument(
        '-d', '--debug',
        help='Provide VERY detailed processing messages.',
        action='store_true',
        default=False,
    )

    subparsers = parser.add_subparsers(dest='command', required=True)
    #subparsers.required = True # this is a python3.4-3.7 bug; cannot specify in the call above.

    qc_parser = subparsers.add_parser('qc', help="Apply filters to exclude probes based on academic refs, or sex-linked, or control probes")
    qc_parser.set_defaults(func=cli_qc)

    controls_parser = subparsers.add_parser('controls', help='Generate an XLSX QC report of control probe performance')
    controls_parser.set_defaults(func=cli_controls_report)

    report_parser = subparsers.add_parser('report', help="Run methylcheck's ReportPDF pipeline and generate a PDF.")
    report_parser.set_defaults(func=cli_ReportPDF)

    #-- when this has subparsers, use -- args = parser.parse_known_args(sys.argv[1:])
    parsed_args, func_args = parser.parse_known_args(sys.argv[1:])
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif 'verbose' in func_args:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    parsed_args.func(func_args)
    return parser


def cli_qc(cmd_args):
    parser = DefaultParser(
        prog='methylcheck qc',
        description='Run the QC pipeline on files in a given beta values file.',
    )

    parser.add_argument(
        '-d', '--data_file',
        required=True,
        type=Path,
        help="""path to a file containing sample matrix of beta or m_values. \
You can create this output from `methylprep` using the --betas flag.""",
    )

    parser.add_argument(
        '-v', '--verbose',
        help='Provide more detailed processing messages.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '-a', '--array_type',
        choices=['27k','450k','EPIC','EPIC+'],
        required=False,
        help='Type of array being processed. If omitted, methylcheck will autodetect based on shape of data.',
    )

    parser.add_argument(
        '--exclude_sex',
        help='filters out probes on sex-chromosomes',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--exclude_control',
        help='filters out illumina control probes',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--exclude_probes',
        help='This will exclude all problem probes, based on several lists from academic publications. \
If you want fine-tuning-control, use the exclude_probes() function in a jupyter notebook instead. \
Also see methylcheck.list_problem_probes for more details.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--exclude_all',
        help='This will exclude all problem probes, sex-linked probes, and control probes at once.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '-p', '--plot',
        nargs='*', # -p type1 type2 ... zero or more plots.
        choices=['mean_beta_plot', 'beta_density_plot',
            'cumulative_sum_beta_distribution', 'beta_mds_plot', 'all'],
        help='Select which plots to generate. Note that if you omit this, the default setting will run all plots at once. `-p all`',
        default='all',
    )

    parser.add_argument(
        '-s', '--save',
        help='By default, each plot will only appear on screen, but you can save them as png files with this option.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--silent',
        help='Suppress plots and requestss user inputs; for unit testing or automated pipelines.',
        action='store_true',
        default=False,
    )

    #### run_pipeline happens here ###
    #set logging/verbose
    args = parser.parse_args(cmd_args) #sys.argv[1:])

    # load the data file
    if not Path(args.data_file).is_file():
        raise ValueError("Could not find your data_file.")
    if args.data_file.suffix == '.npy':
        npy = np.load(Path(args.data_file))
        df = pd.DataFrame(npy)
    elif args.data_file.suffix == '.pkl':
        df = pd.read_pickle(args.data_file)
    else:
        raise FileNotFoundError("Could not find/read your data file. Must be .pkl or .npy file.")
    # methylprep data will be long format, with samples in columns and probes in rows. MDS transposes this.'

    # determine array type
    if args.array_type is None:
        args.array_type = detect_array(df)

    # apply some filters
    if args.exclude_all:
        df = exclude_sex_control_probes(df, args.array_type, verbose=args.verbose)
        sketchy_probe_list = list_problem_probes(args.array_type)
        #print(len(sketchy_probe_list),'sketchy probes',sketchy_probe_list[:10])
        df = exclude_probes(df, sketchy_probe_list)

    elif args.exclude_sex and args.exclude_control:
        df = exclude_sex_control_probes(df, args.array_type, verbose=args.verbose)
    elif args.exclude_sex:
        df = exclude_sex_control_probes(df, args.array_type, no_sex=True, no_control=False, verbose=args.verbose)
    elif args.exclude_control:
        df = exclude_sex_control_probes(df, args.array_type, no_sex=False, no_control=True, verbose=args.verbose)

    if args.exclude_probes and not args.exclude_all:
        sketchy_probe_list = list_problem_probes(args.array_type)
        #print(type(sketchy_probe_list),len(sketchy_probe_list))
        #print(df.shape, list(df.index)[:20])
        df = exclude_probes(df, sketchy_probe_list)

    # run calculations and plots
    if 'all' in args.plot:
        mean_beta_plot(df, verbose=args.verbose, save=args.save, silent=args.silent)
        beta_density_plot(df, verbose=args.verbose, save=args.save, silent=args.silent)
        wide_df = df.copy().transpose()
        cumulative_sum_beta_distribution(wide_df, verbose=args.verbose, save=args.save, silent=args.silent)
        beta_mds_plot(wide_df, verbose=args.verbose, save=args.save, silent=args.silent)
    else:
        if 'mean_beta_plot' in args.plot:
            mean_beta_plot(df, verbose=args.verbose, save=args.save, silent=args.silent)
        if 'beta_density_plot' in args.plot:
            beta_density_plot(df, verbose=args.verbose, save=args.save, silent=args.silent)
        if 'cumulative_sum_beta_distribution' in args.plot:
            wide_df = df.copy().transpose()
            cumulative_sum_beta_distribution(wide_df, verbose=args.verbose, save=args.save, silent=args.silent)
        if 'beta_mds_plot' in args.plot:
            df_filtered = beta_mds_plot(df, verbose=args.verbose, save=args.save, silent=args.silent)
            # also has silent and filter_stdev params to pass in.

def cli_controls_report(cmd_args):
    parser = DefaultParser(
        prog='methylcheck controls',
        description='Run the controls_report QC pipeline on files in a given folder.',
    )

    parser.add_argument(
        '-d', '--data_dir',
        required=True,
        type=Path,
        help="""Path where input files are located.""",
    )

    parser.add_argument(
        '-o', '--out_dir',
        required=False,
        type=Path,
        help="""Path where output files will be saved.""",
    )
    parser.add_argument(
        '--bg_offset',
        help='Adjust the baseline background offset value. Default is 3000.',
        type=int,
        default=3000,
    )

    parser.add_argument(
        '--colorblind',
        help='Use an alternate color palette in output XLSX file.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--cutoff_adjust',
        help='Adjust the pass/fail cutoff. 1.0 uses the standard limits. 2.0 will increase minimum acceptable ratio (more conservative / lower tolerance)',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--pval_off',
        help='Flag to disable use of probe failures (pvalue) in report. Note that Illumina Legacy compatability mode does not include pval.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--pval_sig',
        help='p-value failure significance threshold (default <0.05). Ignore if not using pval.',
        type=float,
        default=0.05,
    )

    parser.add_argument(
        '-l', '--legacy',
        help='Change XLSX columns to exclude sex check and overall sample quality score, and match the format of Illumina quality control output files.',
        action='store_true',
        default=False,
    )

    args = parser.parse_args(cmd_args) #sys.argv[1:])
    return controls_report(
        filepath=args.data_dir,
        outfilepath=args.out_dir,
        bg_offset=args.bg_offset,
        colorblind=args.colorblind,
        cutoff_adjust=args.cutoff_adjust,
        legacy=args.legacy,
        pval=(not args.pval_off), # needed to invert kwarg, because default is to run it, and flag is only to disable it
        pval_sig=args.pval_sig)


def cli_ReportPDF(cmd_args):
    parser = DefaultParser(
        prog='methylcheck report',
        description='Create a battery of control probe performance plots, saved as a PDF.',
    )

    parser.add_argument(
        '-d', '--data_dir',
        required=False,
        type=Path,
        default='.',
        help="""Path where input files are located. Default is current folder if omitted.""",
    )

    parser.add_argument(
        '-o', '--out_dir',
        required=False,
        type=Path,
        help="""Path where output files will be saved. Defaults to same folder as 'data_dir'.""",
    )

    parser.add_argument(
        '-f', '--filename',
        required=False,
        default='multipage_pdf.pdf',
        help="""Filename of the PDF generated. Defaults to 'multipage_pdf.pdf' if omitted.""",
    )


    parser.add_argument(
        '--pval_min_percent',
        help='Set the minimum percent of probes that must pass pOOBAH for a sample to be retained. Default is 80%.',
        type=int,
        default=80,
    )

    parser.add_argument(
        '--pval_sig',
        help='p-value failure significance level. Default alpha is 0.05.',
        type=float,
        default=0.05,
    )

    args = parser.parse_args(cmd_args)
    if args.out_dir is None:
        args.out_dir = args.data_dir
    return ReportPDF(
        path=args.data_dir,
        outpath=args.out_dir,
        filename=args.filename,
        pval_min_percent=args.pval_min_percent,
        pval_cutoff=args.pval_sig,
        runme=True) # <--- this autocompletes report and closes it at end of __init__ in one step.


def cli_app():
    build_parser()
