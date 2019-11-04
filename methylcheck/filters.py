# -*- coding: utf-8 -*-
import pkg_resources
import numpy as np
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


def _import_probe_exclusion_list(array, type):
    """Returns a list of probe_ids based on type of array and type of probes to be excluded (sex or control).

    |array | sex_linked | controls|
    |------|------------|---------|
    |EPIC+ | 20137      | 755     |
    |EPIC  | 19627      | 695     |
    |450k  | 11648      | 916     |
    |27k   |            |         |

    Parameters
    ----------
    array -- [EPIC, EPIC+, 450k, 27k] or ['IlluminaHumanMethylationEPIC', 'IlluminaHumanMethylationEPIC+', 'IlluminaHumanMethylation450k']
    type -- [sex, control]"""

    if array in ('IlluminaHumanMethylation450k','450k'):
        path = 'data_files/450k_{0}.npy'.format(type)
    elif array in ('IlluminaHumanMethylationEPIC','EPIC'):
        path = 'data_files/EPIC_{0}.npy'.format(type)
    elif array in ('IlluminaHumanMethylationEPIC+','EPIC+'):
        path = 'data_files/EPIC+_{0}.npy'.format(type)
    elif array =='27k':
        raise NotImplementedError("We can't read 27k arrays yet.")
    else:
        raise ValueError("""Did not recognize array type. Please specify one of
            'IlluminaHumanMethylation450k', 'IlluminaHumanMethylationEPIC', '450k', 'EPIC', 'EPIC+', '27k'.""")
    filepath = pkg_resources.resource_filename(__name__, path)
    probe_list = np.load(filepath, allow_pickle=True)
    return probe_list


def exclude_sex_control_probes(df, array, no_sex=True, no_control=True, verbose=False):
    """Function to exclude probes from an array, and return a filtered array.

    Parameters
    ----------
    df: dataframe of beta values or m-values.
    array: type of array used.
        {'27k', '450k', 'EPIC'} or {'IlluminaHumanMethylation27k','IlluminaHumanMethylation450k','IlluminaHumanMethylationEPIC',}

    Optional Arguments
    ------------------
    `no_sex`: bool
        (default True)
        if True, will remove all probes that target X and Y chromosome locations,
        as they are sex specific -- and lead to multiple clusters when trying to detect and remove outliers (noisy data).

    `no_control`: bool
        (default True)
        if True, removes Illumina's internal control probes.

    `verbose`: bool
        (default False)
        reports out on number of probes removed.

    Returns
    -------
        a dataframe with samples removed."""
    translate = {
        'IlluminaHumanMethylationEPIC':'EPIC',
        'IlluminaHumanMethylation450k':'450k',
        'IlluminaHumanMethylation27k':'27k',
         }
    if array in translate:
        array = translate[array]
    exclude_sex = _import_probe_exclusion_list(array, 'sex') if no_sex == True else []
    exclude_control = _import_probe_exclusion_list(array, 'control') if no_control == True else []
    exclude = list(exclude_sex)+list(exclude_control)

    # first check shape of df; probes are in the longer axis.
    FLIPPED = False
    if len(df.columns) > len(df.index):
        df = df.transpose()
        FLIPPED = True # use this to reverse at end, so returned in original orientation.

    # next, actually remove probes from all samples matching these lists.
    filtered = df.drop(exclude, errors='ignore')

    # errors = ignore: if a probe is missing from df, or present multiple times, drop what you can.
    if verbose == True:
        print("""Array {4}: Removed {0} sex linked probes and {1} internal control probes \
from {2} samples. {3} probes remaining.""".format(
        len(exclude_sex),
        len(exclude_control),
        len(filtered.columns),
        len(filtered),
        array
        ))
        discrepancy = (len(exclude_sex) + len(exclude_control)) - (len(df) - len(filtered))
        if discrepancy != 0:
            print("Discrepancy between number of probes to exclude ({0}) and number actually removed ({1}): {2}".format(
                len(exclude_sex) + len(exclude_control),
                len(df) - len(filtered),
                discrepancy
            ))
        if len(exclude_control) == discrepancy:
            print("It appears that your sample had no control probes, or that the control probe names didn't match the manifest ({0}).".format(array))
        else:
            print("This happens when probes are present multiple times in array, or the manifest doesn't match the array ({0}).".format(array))
    # reverse the FLIP
    if FLIPPED:
        filtered = filtered.transpose()
    return filtered

def exclude_probes(array, probe_list):
    """
How to:
    use list_problem_probes to obtain a list of probes, then pass that in as a probe_list along with the dataframe of beta values (array)

Resolves a problem whereby probe lists have basic names, but samples have additional meta data added.
Example:

probe list
    ['cg24168924', 'cg15886294', 'cg05943251', 'cg05579622', 'cg01797553', 'cg14885690', 'cg12490816', 'cg02631583', 'cg17361593', 'cg15000031', 'cg21515494', 'cg17219246', 'cg10838001', 'cg13913475', 'cg00492169', 'cg20352786', 'cg05932698', 'cg06736139', 'cg08333283', 'cg10010298', 'cg25984048', 'cg27287823', 'cg19269713', 'cg12456833', 'cg26161708', 'cg04984052', 'cg00033806', 'cg23255774', 'cg10717379', 'cg00880984', 'cg01818617', 'cg18563133', 'cg15895341', 'cg08155050', 'cg06820286', 'cg04325909', 'cg15094920', 'cg08037129', 'cg11161730', 'cg06044537', 'cg11936560', 'cg12404870', 'cg12670496', 'cg01473643', 'cg08605930', 'cg16553354', 'cg22175254', 'cg22966295', 'cg07346931', 'cg06234741']
sample probe names
    Index(['cg00000029_II_F_C_rep1_EPIC', 'cg00000103_II_F_C_rep1_EPIC', 'cg00000109_II_F_C_rep1_EPIC', 'cg00000155_II_F_C_rep1_EPIC',
   'cg00000158_II_F_C_rep1_EPIC', 'cg00000165_II_R_C_rep1_EPIC', 'cg00000221_II_R_C_rep1_EPIC', 'cg00000236_II_R_C_rep1_EPIC',
   ...
   'ch.9.98957343R_II_R_O_rep1_EPIC', 'ch.9.98959675F_II_F_O_rep1_EPIC', 'ch.9.98989607R_II_R_O_rep1_EPIC', 'ch.9.991104F_II_F_O_rep1_EPIC']

This chops off anything after the first underscore, and compares with probe_list to see if percent match increases.
It then drops probes from array that match probe_list, at least partially.

ADDED: checking whether array.index is string or int type. Regardless, this should work and not alter the original index."""

    # 1 - check shape of array, to ensure probes are the index/values
    ARRAY = array.index
    if len(array.index) < len(array.columns):
        ARRAY = array.columns

    pre_overlap = len(set(ARRAY) & set(probe_list))

    # probe_list is always a list of strings. COERCING to strings here for matching.
    array_probes = [str(probe).split('_')[0] for probe in list(ARRAY)]
    post_overlap = len(set(array_probes) & set(probe_list))

    if pre_overlap != post_overlap:
        print("matching probes: {0} vs {1} after name fix, yielding {2} probes.".format(pre_overlap, post_overlap, len(array)-post_overlap))
    else:
        print("Of {0} probes, {1} matched, yielding {2} probes after filtering.".format(len(array), post_overlap, len(array)-post_overlap))
    if post_overlap >= pre_overlap:
        # match which probes to drop from array.
        array_probes_lookup = {str(probe).split('_')[0]: probe for probe in list(ARRAY)}
        # get a list of unmodified array_probe_names to drop, based on matching the overlapping list with the modified probe names.
        exclude = [array_probes_lookup[probe] for probe in (set(array_probes) & set(probe_list))]
        return array.drop(exclude, errors='ignore')
    print("Nothing removed.")
    return array

def list_problem_probes(array, criteria=None, custom_list=None):
    """Function to create a list of probes to exclude from downstream processes.
    By default, all probes that have been noted in the literature to have
    polymorphisms, cross-hybridization, repeat sequence elements and base color
    changes are included in the DEFAULT exclusion list.

    - You can customize the exclusion list by passing in either publication shortnames or criteria into the function.
    - you can combine pubs and reasons into the same list of exclusion criteria.
    - if a publication doesn't match your array type, it will raise an error and tell you.

    Including any of these labels in pubs (publications) or criteria (described below)
    will result in these probes NOT being included in the final exclusion list.

    User also has ability to add custom list of probes to include in final returned list.

    Parameters
    ----------
    array: string
        name for type of array used
        'IlluminaHumanMethylationEPIC', 'IlluminaHumanMethylation450k'
        This shorthand names are also okay:
        'EPIC','EPIC+','450k','27k'

    criteria: list
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

    criteria: lists
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
        User-provided list of probes to be excluded.
        These probe names have to match the probe names in your data exactly.

    Returns
    -------
    probe_exclusion_list: list
        List containing probe identifiers to be excluded"""

    translate = {
        'EPIC+': 'IlluminaHumanMethylationEPIC',
        'EPIC': 'IlluminaHumanMethylationEPIC',
        '450k': 'IlluminaHumanMethylation450k',
        '27k': 'IlluminaHumanMethylation27k',
         }

    probe_pubs_translate = {
        'Price2013': 'Price_etal_2013',
        'Chen2013': 'Chen_etal_2013',
        'Naeem2014': 'Naeem_etal_2014',
        'DacaRoszak2015': 'Daca-Roszak_etal_2015',
        'McCartney2016': 'McCartney_etal_2016',
        'Zhou2016': 'Zhou_etal_2016',
        }
    if array in translate:
        array = translate[array]
    probe_dataframe = _import_probe_filter_list(array)
    """
    I had no idea how this works, so I replaced it. MM July 2019.
    -- syntax error when I ran it: these vars are not defined.
    args = [Polymorphism, CrossHybridization,
            BaseColorChange, RepeatSequenceElements,
            Chen2013, Price2013, Zhou2016, Naeem2014,
            DacaRoszak2015, McCartney2016]"""
    if not custom_list:
        custom_list = []
    if not criteria:
        probe_exclusion_list = list(set(probe_dataframe['Probe'].values))
        probe_exclusion_list = list(set(probe_exclusion_list + custom_list))
        return probe_exclusion_list
    else:
        probe_exclusion_list = []

        # rename criteria to match probe_dataframe names, or pass through.
        criteria = [probe_pubs_translate.get(crit, crit) for crit in criteria]

        reasons = ['Polymorphism', 'CrossHybridization', 'BaseColorChange', 'RepeatSequenceElements']
        for reason in criteria:
            if reason not in reasons:
                continue
            df = probe_dataframe[probe_dataframe['Reason'] == reason]
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list

        epic_pubs = ['Zhou_etal_2016', 'McCartney_etal_2016']
        fourfifty_pubs = ['Chen_etal_2013', 'Price_etal_2013', 'Naeem_etal_2014', 'Daca-Roszak_etal_2015']
        for pub in criteria:
            if pub not in (epic_pubs + fourfifty_pubs):
                continue

            if pub in fourfifty_pubs and array == 'IlluminaHumanMethylationEPIC':
                raise ValueError(
                    "Citation {0} does not exist for '{1}'.".format(pub, array))

            if pub in epic_pubs and array == 'IlluminaHumanMethylation450k':
                raise ValueError(
                    "Citation {0} does not exist for '{1}'.".format(pub, array))

            df = probe_dataframe[probe_dataframe['ShortCitation'] == pub]
            probe_exclusion_list = list(
                set(df['Probe'].values)) + probe_exclusion_list

        probe_exclusion_list = list(set(probe_exclusion_list + custom_list))
    return probe_exclusion_list
