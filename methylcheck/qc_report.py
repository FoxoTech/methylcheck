import pandas as pd
import logging
LOGGER = logging.getLogger(__name__)
# app
import methylcheck

def run_pipeline(df, **kwargs):
    """ lets you run a variety of probe and sample filters in tandem, then plot results
    by specifying all of your options at once, instead of running every part of methylcheck
    in piacemeal fashion.

    this is analogous to using the methylcheck CLI, but for notebooks/scripts

required:
    df -- data as a dataframe of beta values, sample names in columns and probes in rows

options:
    verbose (True/False)
        default: False -- shows extra info about processing if True
    silent (True/False)
        default: False -- suppresses all warnings/info

    exclude_sex
        filters out probes on sex-chromosomes
    exclude_control
        filters out illumina control probes
    exclude_all
        filters out the most probes (sex-linked, control, and all sketchy-listed probes from papers)
    exclude (list of strings, shorthand references to papers with sketchy probes to exclude)
        If the array is 450K the publications may include:
            'Chen2013'
            'Price2013'
            'Zhou2016'
            'Naeem2014'
            'DacaRoszak2015'
        If the array is EPIC the publications may include:
            'Zhou2016'
            'McCartney2016'
        or these reasons:
            'Polymorphism'
            'CrossHybridization'
            'BaseColorChange'
            'RepeatSequenceElements'
        or use 'exclude_all' to do maximum filtering, including these papers

    plot (list of strings)
        ['mean_beta_plot', 'beta_density_plot', 'cumulative_sum_beta_distribution', 'beta_mds_plot', 'all']
        if 'all', then all of these plots will be generated. if omitted, no plots are created.
    save_plots (True|False)
        default: False

    export (True|False)
        default: False -- will export the filtered df as a pkl file if True

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
    'export': [True, False]
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
    if kwargs.get('save_plots') is None:
        kwargs['save_plots'] = False
    # determine array type
    array_type = methylcheck.detect_array(df)

    # apply some filters
    if kwargs.get('exclude_all'):
        df = methylcheck.exclude_sex_control_probes(df, array_type, verbose=kwargs.get('verbose'))
        sketchy_probe_list = methylcheck.list_problem_probes(array_type)
        df = methylcheck.exclude_probes(df, sketchy_probe_list)
    elif kwargs.get('exclude'):
        # could be a list or a string
        if type(kwargs['exclude']) is not list:
            kwargs['exclude'] = [kwargs['exclude']]
        sketchy_probe_list = methylcheck.list_problem_probes(array_type, criteria=kwargs['exclude'])
        df = methylcheck.exclude_probes(df, sketchy_probe_list)

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
