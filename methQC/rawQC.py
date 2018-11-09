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


def createProbeExclusionList(array, Polymorphism=True, CrossHybridization=True,
                             BaseColorChange=True, RepeatSequenceElements=True,
                             Chen2013=True, Price2013=True, Zhou2016=True,
                             Naeem2014=True, DacaRoszak2015=True,
                             McCartney2016=True, custom_list=None):
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
