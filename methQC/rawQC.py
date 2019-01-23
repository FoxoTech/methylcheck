# -*- coding: utf-8 -*-
import pkg_resources
import pandas as pd


def _import_probe_filter_list(array):
    """Function to identify array type and import the
    appropriate probe exclusion dataframe"""

    if array == 'IlluminaHumanMethylation450k':
        path = 'data_files/450k_polymorphic_crossRxtve_probes.csv.gz'
        filepath = pkg_resources.resource_filename(__name__, path)
        filter_options = pd.read_csv(filepath)
    elif array == 'IlluminaHumanMethylationEPIC':
        path = 'data_files/EPIC_polymorphic_crossRxtve_probes.csv.gz'
        filepath = pkg_resources.resource_filename(__name__, path)
        filter_options = pd.read_csv(filepath)
    else:
        raise ValueError(
            "Did not recognize array type. Please specify 'IlluminaHumanMethylation450k' or 'IlluminaHumanMethylationEPIC'.")
        return
    return filter_options


def createProbeExclusionList(array, pubs=None, criteria=None, custom_list=None):
    """Function to create list of probes to exclude from downstream processes.
    By default, all probes that have been noted in the literature to have
    polymorphisms, cross-hybridization, repeat sequence elements and base color
    changes are included in the exclusion list. Setting any of the args for these
    reasons or publications (described below) to False will result in these probes
    NOT being included in the final exclusion list.to User also has ability to
    add custom list of probes to include in final returned list.

    Parameters
    ----------
    array: dataframe
        Dataframe containing beta values

    pubs: list
        List of the publications to use when excluding probes.
        If the array is 450K the publications may include:
            'Chen2013'
            'Price2013'
            'Zhou2016'
            'Naeem2014'
            'DacaRoszak2015'
        If the array is EPIC the publications may include:
            'Zhou2016'
            'McCarthey2016'
        If no publication list is specified, probes from 
        all publications will be added to the exclusion list.
        If more than one publication is specified, all probes
        from all publications in the list will be added to the
        exclusion list.

    criteria: list
        List of the criteria to use when excluding probes.
        List may contain the following exculsion criteria:
            'Polymorphism'
            'CrossHybridizing'
            'BaseColorChange'
            'RepeatSequenceElements'
        If no criteria list is specified, all critera will be
        excluded. If more than one criteria is specified,
        all probes meeting any of the listed criteria will be 
        added to the exclusion list.

    custom_list: list, default None
        User provided list of probes to add to probe
        exclusion list

    Returns
    -------
    probe_exclusion_list: list
        List containing probe identifiers to be excluded

    """
    probe_dataframe = _import_probe_filter_list(array)
    args = [Polymorphism, CrossHybridization,
            BaseColorChange, RepeatSequenceElements,
            Chen2013, Price2013, Zhou2016, Naeem2014,
            DacaRoszak2015, McCartney2016]
    if custom_list is None:
        custom_list = []
    if False not in args:
        probe_exclusion_list = list(set(probe_dataframe['Probe'].values))
        probe_exclusion_list = list(set(probe_exclusion_list + custom_list))
    else:
        probe_exclusion_list = []
        if Polymorphism is True:
            df = probe_dataframe[probe_dataframe['Reason'] == 'Polymorphism']
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list
        if CrossHybridization is True:
            df = probe_dataframe[probe_dataframe['Reason']
                                 == 'CrossHybridization']
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list
        if BaseColorChange is True:
            df = probe_dataframe[probe_dataframe['Reason']
                                 == 'BaseColorChange']
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list
        if RepeatSequenceElements is True:
            df = probe_dataframe[probe_dataframe['Reason']
                                 == 'RepeatSequenceElements']
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list
        if Chen2013 is True:
            if array == 'IlluminaHumanMethylationEPIC':
                raise ValueError(
                    "Citation Chen2013 does not exist for 'IlluminaHumanMethylationEPIC'.")
                return
            else:
                df = probe_dataframe[probe_dataframe['ShortCitation']
                                     == 'Chen2013']
                probe_exclusion_list = list(
                    set(df['Probe'].values)) + probe_exclusion_list
        if Price2013 is True:
            if array == 'IlluminaHumanMethylationEPIC':
                raise ValueError(
                    "Citation Price2013 does not exist for 'IlluminaHumanMethylationEPIC'.")
                return
            else:
                df = probe_dataframe[probe_dataframe['ShortCitation']
                                     == 'Price2013']
                probe_exclusion_list = list(
                    set(df['Probe'].values)) + probe_exclusion_list
        if Naeem2014 is True:
            if array == 'IlluminaHumanMethylationEPIC':
                raise ValueError(
                    "Citation Naeem2014 does not exist for 'IlluminaHumanMethylationEPIC'.")
                return
            else:
                df = probe_dataframe[probe_dataframe['ShortCitation']
                                     == 'Naeem2014']
                probe_exclusion_list = list(
                    set(df['Probe'].values)) + probe_exclusion_list
        if DacaRoszak2015 is True:
            if array == 'IlluminaHumanMethylationEPIC':
                raise ValueError(
                    "Citation DacaRoszak2015 does not exist for 'IlluminaHumanMethylationEPIC'.")
                return
            else:
                df = probe_dataframe[probe_dataframe['ShortCitation']
                                     == 'DacaRoszak2015']
                probe_exclusion_list = list(
                    set(df['Probe'].values)) + probe_exclusion_list
        if Zhou2016 is True:
            df = probe_dataframe[probe_dataframe['ShortCitation']
                                 == 'Zhou2016']
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list
        if McCartney2016 is True:
            if array == 'IlluminaHumanMethylation450k':
                raise ValueError(
                    "Citation McCartney2016 does not exist for 'IlluminaHumanMethylationEPIC'.")
                return
            else:
                df = probe_dataframe[probe_dataframe['ShortCitation']
                                     == 'McCartney2016']
                probe_exclusion_list = list(
                    set(df['Probe'].values)) + probe_exclusion_list
    probe_exclusion_list = list(set(probe_exclusion_list + custom_list))
    return probe_exclusion_list
