# Lib
import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
# App
from .filters import list_problem_probes, exclude_probes, exclude_sex_control_probes
from .postprocessQC import mean_beta_plot, beta_density_plot, cumulative_sum_beta_distribution, DNA_mAge_Hannum, beta_mds_plot

class DefaultParser(argparse.ArgumentParser):
    def error(self, message):
        self._print_message('[Error]:\n')
        self._print_message(f'{message}\n\n')
        self.print_help()
        self.exit(status=2)


def cli_parser():
    parser = DefaultParser(
        prog='methQC',
        description="""Transformation and visualization tool for methylation data from Illumina IDAT files.

        Usage: python -m methQC -d <datafile> --verbose [commands]

        Convenience command, to run the full battery of tests: ptype methQC -d <data> -p all --exclude_all

        Commands
        --------

        exclude_sex -- remove sex-chromosome-linked probes from a batch of samples.

        exclude_control -- remove Illumina control probes  from a batch of samples.

        exclude_probes -- remove less reliable probes, based on lists of problem probes from literature.
        look at the function help for details on how you fine tune the exclusion list.

        plot -- various kinds of plots to reveal sample quality.

        array_type -- QC requires the type of array.""",
    )

    parser.add_argument(
        '-v', '--verbose',
        help='Provide more detailed processing messages.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '-d', '--data_file',
        required=True,
        type=Path,
        help="""path to a file containing sample matrix of beta or m_values. \
You can create this output from `methpype` using the --betas flag.""",
    )

    parser.add_argument(
        '-a', '--array_type',
        choices=['27k','450k','EPIC','EPIC+'],
        required=True,
        help='Type of array being processed.',
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
Also see methQC.list_problem_probes for more details.',
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
            'cumulative_sum_beta_distribution', 'DNA_mAge_Hannum', 'beta_mds_plot', 'all'],
        help='Select which plots to generate. Note that you omit this, the default setting will run all plots at once. `-p all`',
        default='all',
    )

    # pipeline_run is here...
    #set logging/verbose
    args = parser.parse_args(sys.argv[1:])
    #-- when this has subparsers, use -- args = parser.parse_known_args(sys.argv[1:])
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # load the data file
    if not Path(args.data_file).is_file():
        raise ValueError("Could not find your data_file.")
    npy = np.load(Path(args.data_file))
    df = pd.DataFrame(npy)

    # apply some filters

    if args.exclude_all:
        df = exclude_sex_control_probes(df, args.array_type, verbose=args.verbose)
        sketchy_probe_list = list_problem_probes(args.array_type)
        print(type(sketchy_probe_list),len(sketchy_probe_list))
        df = exclude_probes(df, sketchy_probe_list)

    elif args.exclude_sex and args.exclude_control:
        df = exclude_sex_control_probes(df, args.array_type, verbose=args.verbose)
    elif args.exclude_sex:
        df = exclude_sex_control_probes(df, args.array_type, no_sex=True, no_control=False, verbose=args.verbose)
    elif args.exclude_control:
        df = exclude_sex_control_probes(df, args.array_type, no_sex=False, no_control=True, verbose=args.verbose)

    if args.exclude_probes and not args.exclude_all:
        sketchy_probe_list = list_problem_probes(args.array_type)
        print(type(sketchy_probe_list),len(sketchy_probe_list))
        df = exclude_probes(df, sketchy_probe_list)

    if 'all' in args.plot:
        mean_beta_plot(df)
        beta_density_plot(df)
        test_df = df.copy().transpose()
        cumulative_sum_beta_distribution(test_df)
        test, excl = beta_mds_plot(df, verbose=args.verbose)
        transformed = DNA_mAge_Hannum(df)
        return parser

    if 'beta_mds_plot' in args.plot:
        test, excl = beta_mds_plot(df, verbose=args.verbose)
        # also has silent and filter_stdev params to pass in.

    if 'mean_beta_plot' in args.plot:
        mean_beta_plot(df)

    if 'beta_density_plot' in args.plot:
        beta_density_plot(df)

    if 'cumulative_sum_beta_distribution' in args.plot:
        test_df = df.copy().transpose()
        cumulative_sum_beta_distribution(test_df)

    if 'DNA_mAge_Hannum' in args.plot:
        transformed = DNA_mAge_Hannum(df)

    return parser
