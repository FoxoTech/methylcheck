# -*- coding: utf-8 -*-
try:
    from importlib import resources # py3.7+
except ImportError:
    import pkg_resources #py < 3.7

import numpy as np
import pandas as pd
# app
import methylcheck

import logging
# whilst debugging only: this is on.
#logging.basicConfig(level=logging.DEBUG) # ROOT logger
LOGGER = logging.getLogger(__name__)
#LOGGER.setLevel(logging.DEBUG)
pkg_namespace = 'methylcheck.data_files'

try:
    with resources.path(pkg_namespace, 'illumina_sketchy_probes_996.npy') as probe_filepath:
        illumina_sketchy_probes = np.load(probe_filepath)
except:
    probe_filepath = pkg_resources.resource_filename(pkg_namespace, 'illumina_sketchy_probes_996.npy')
    illumina_sketchy_probes = np.load(probe_filepath)
# "If the first 8 numbers of Sentrix_ID (i.e. xxxxxxxx0001) are greater or equal to 20422033,
# then the BeadChip originates from production batches using the new manufacturing process."
new_manufacturing_cutoff_id = 20422033

## TODO for unit testing:
## no coverage of 27k stuff
## drop_nan_probes

def _import_probe_filter_list(array):
    """Function to identify array type and import the
    appropriate probe exclusion dataframe"""
    if array in ('IlluminaHumanMethylation450k', '450k', '450K'):
        filename = '450k_polymorphic_crossRxtve_probes.csv.gz'
    elif array in ('IlluminaHumanMethylationEPIC', 'EPIC', 'EPIC+', 'epic', 'epic+'):
        filename = 'EPIC_polymorphic_crossRxtve_probes.csv.gz'
    elif array in ('IlluminaHumanMethylation27k','27k'):
        raise ValueError("27K has no problem probe lists available")
    elif array in ('mouse', 'MOUSE', 'IlluminaMouse'):
        raise ValueError("No probe exclusion lists available for mouse arrays at this time.")
    else:
        raise ValueError(f"Did not recognize array type {array}. Please specify 'IlluminaHumanMethylation450k' or 'IlluminaHumanMethylationEPIC'.")
        return
    try:
        with resources.path(pkg_namespace, filename) as probe_filepath:
            filter_options = pd.read_csv(probe_filepath)
    except:
        probe_filepath = pkg_resources.resource_filename(pkg_namespace, PROBE_FILE)
        filter_options = pd.read_csv(probe_filepath)
    return filter_options

def _import_probe_exclusion_list(array, _type):
    """Returns a list of probe_ids based on type of array and type of probes to be excluded (sex or control).

    |array | sex_linked | controls|
    |------|------------|---------|
    |mouse |            | 695     |
    |epic+ | 20137      | 755     |
    |epic  | 19627      | 695     |
    |450k  | 11648      | 916     |
    |27k   |            |         |

    Parameters
    ----------
    array -- [EPIC, EPIC+, 450k, 27k] or ['IlluminaHumanMethylationEPIC', 'IlluminaHumanMethylationEPIC+', 'IlluminaHumanMethylation450k']
    type -- [sex, control]"""

    if array in ('IlluminaHumanMethylation450k','450k','450K'):
        filename = '450k_{0}.npy'.format(_type)
    elif array in ('IlluminaHumanMethylationEPIC','EPIC','epic'):
        filename = 'EPIC_{0}.npy'.format(_type)
    elif array in ('IlluminaHumanMethylationEPIC+','EPIC+','epic+','EPICPLUS','EPIC_PLUS'):
        filename = 'EPIC+_{0}.npy'.format(_type)
    elif array in ('27k','27K'):
        raise NotImplementedError("No probe lists available for 27k arrays.")
    elif array in ('MOUSE','mouse'):
        LOGGER.info("No probe exclusion lists available for mouse arrays at this time.")
        return []
    else:
        raise ValueError("""Did not recognize array type. Please specify one of
            'IlluminaHumanMethylation450k', 'IlluminaHumanMethylationEPIC', '450k', 'EPIC', 'EPIC+', '27k', 'mouse'.""")
    with resources.path(pkg_namespace, filename) as filepath:
        probe_list = np.load(filepath, allow_pickle=True)
    return probe_list


def exclude_sex_control_probes(df, array, no_sex=True, no_control=True, verbose=False):
    """Exclude probes from an array, and return a filtered array.

    Parameters
    ----------
    df: dataframe of beta values or m-values.
    array: type of array used.
        {'27k', '450k', 'EPIC', 'EPICPLUS', 'MOUSE'} or {'IlluminaHumanMethylation27k','IlluminaHumanMethylation450k','IlluminaHumanMethylationEPIC', 'mouse'}
        or {'27k', '450k', 'epic', 'epic+', 'mouse'}

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
        'IlluminaHumanMethylationEPIC':'epic',
        'IlluminaHumanMethylation450k':'450k',
        'IlluminaHumanMethylation27k':'27k',
        'MOUSE':'mouse',
        'EPIC':'epic',
        'EPICPLUS':'epic+',
        'EPIC_PLUS': 'epic+',
        '450K': '450k',
        '27K': '27k',
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

    control_probes_removed = df[ df.index.isin(exclude_control) ].index
    sex_probes_removed = df[ df.index.isin(exclude_sex) ].index
    # next, actually remove probes from all samples matching these lists.
    filtered = df.drop(exclude, errors='ignore')
    # errors = ignore: if a probe is missing from df, or present multiple times, drop what you can.
    if verbose == True:
        if len(control_probes_removed) > 0:
            # control probes are typically not part of the dataset in methylprep
            print(f"""{array}: Removed {len(sex_probes_removed)} sex-linked probes and {len(control_probes_removed)} control probes from {len(filtered.columns)} samples. {len(filtered)} probes remaining.""")
        else:
            print(f"""{array}: Removed {len(sex_probes_removed)} sex-linked probes from {len(filtered.columns)} samples. {len(filtered)} probes remaining.""")
    # reverse the FLIP
    if FLIPPED:
        filtered = filtered.transpose()
    return filtered


def exclude_probes(df, probe_list):
    """Exclude probes from a dataframe of sample beta values. Use list_problem_probes() to obtain a list of probes (or pass in the names of 'Criteria' from problem probes), then pass that in as a probe_list along with the dataframe of beta values (array)

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

ADDED: checking whether array.index is string or int type. Regardless, this should work and not alter the original index.
ADDED v0.6.4: pass in a string like 'illumina' or 'McCartney2016' and it will fetch the list for you.

ref: https://bioconductor.org/packages/devel/bioc/vignettes/sesame/inst/doc/sesame.html#howwhy-probes-are-masked
SESAME probe exclusion lists were pulled using these R commands:
    EPIC_Zhou = sesameDataGet('EPIC.probeInfo')$mask # 104454 probes
    HM450_Zhou <- sesameDataGet('HM450.probeInfo'))$mask # 65144 probes
"""
    # 1 - check shape of array, to ensure probes are the index/values
    ARRAY = df.index
    if len(df.index) < len(df.columns):
        ARRAY = df.columns

    # copied from list_problem_probes()
    all_criteria = ['Polymorphism', 'CrossHybridization',
        'BaseColorChange', 'RepeatSequenceElements', 'illumina',
        'Chen2013', 'Price2013', 'Zhou2016', 'Naeem2014', 'DacaRoszak2015',
        'McCartney2016', 'Sesame'
        ]
    # 2 - check if probe_list is a list of probes, or a string
    if isinstance(probe_list,str):
        array_type = methylcheck.detect_array(df)
        probe_list = list_problem_probes(array_type.upper(), [probe_list])
    elif isinstance(probe_list, list) and set(probe_list).issubset(set(all_criteria)):
        array_type = methylcheck.detect_array(df)
        probe_list = list_problem_probes(array_type.upper(), probe_list)
    elif isinstance(probe_list,list) and all([isinstance(item,str) for item in probe_list]):
        # a list of probes
        pass

    #OLD elif isinstance(probe_list,list) and isinstance(probe_list[0],str) and len(probe_list) < 100:
    #    # detecting names of lists, like 'Zuul2106'; is a short list but not correct

    pre_overlap = len(set(ARRAY) & set(probe_list))

    # probe_list is always a list of strings. COERCING to strings here for matching.
    array_probes = [str(probe).split('_')[0] for probe in list(ARRAY)]
    post_overlap = len(set(array_probes) & set(probe_list))

    if pre_overlap != post_overlap:
        print(f"matching probes: {pre_overlap} vs {post_overlap} after name fix, yielding {len(df)-post_overlap} probes.")
    else:
        print(f"Of {len(df.index)} probes, {post_overlap} matched, yielding {len(df.index)-post_overlap} probes after filtering.")
    if post_overlap >= pre_overlap:
        # match which probes to drop from array.
        array_probes_lookup = {str(probe).split('_')[0]: probe for probe in list(ARRAY)}
        # get a list of unmodified array_probe_names to drop, based on matching the overlapping list with the modified probe names.
        exclude = [array_probes_lookup[probe] for probe in (set(array_probes) & set(probe_list))]
        return df.drop(exclude, errors='ignore')
    print("Nothing removed.")
    return df


def problem_probe_reasons(array, criteria=None):
    """Returns a dataframe of probes to be excluded, based on recommendations from the literature.
    Mouse and 27k arrays are not supported.

    array: string
        name for type of array used
        'IlluminaHumanMethylationEPIC', 'IlluminaHumanMethylation450k'
        This shorthand names are also okay:
            {'EPIC','EPIC+','450k','27k','MOUSE', 'mouse', 'epic', '450k'}

    criteria: list
        List of the publications to use when excluding probes.
        If the array is 450K the publications may include:
            'Chen2013'
            'Price2013'
            'Zhou2016'
            'Naeem2014'
            'DacaRoszak2015'
            'Sesame' -- uses the default mask imported from sesame
        If the array is EPIC the publications may include:
            'Zhou2016'
            'McCartney2016'
            'Sesame' -- uses the default mask imported from sesame
        or these reasons:
            'Polymorphism'
            'CrossHybridization'
            'BaseColorChange'
            'RepeatSequenceElements'
        If no publication list is specified, probes from
        all publications will be added to the exclusion list.
        If more than one publication is specified, all probes
        from all publications in the list will be added to the
        exclusion list.

    returns: dataframe
        this returns a dataframe showing how each probe in the list
        is categorized for exclusion (based on criteria: reasons and paper-refs). This output
        is not suitable for other functions that just expect a list of probe names.
    """
    all_criteria = ['Polymorphism', 'CrossHybridization',
        'BaseColorChange', 'RepeatSequenceElements',
        'Chen2013', 'Price2013', 'Zhou2016', 'Naeem2014', 'DacaRoszak2015',
        'Zhou2016', 'McCartney2016', 'Sesame'
        ]
    if criteria != None and set(criteria) - set(all_criteria) != set():
        unrecognized = set(criteria) - set(all_criteria)
        raise ValueError(f"Unrecognized criteria: ({unrecognized})\n One of these are allowed: {all_criteria}")
    translate = {
        'EPIC+': 'IlluminaHumanMethylationEPIC',
        'EPIC': 'IlluminaHumanMethylationEPIC',
        '450K': 'IlluminaHumanMethylation450k',
        '27K': 'IlluminaHumanMethylation27k',
        'MOUSE':'mouse',
         }
    translate.update({k.lower():v for k,v in translate.items()})
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

    if criteria is None:
        criteria = all_criteria
    sesame = pd.Series()
    if 'Sesame' in criteria:
        if array in ('IlluminaHumanMethylation450k', '450k', '450K'):
            filename = '450k_Sesame.txt.gz'
        elif array in ('IlluminaHumanMethylationEPIC', 'EPIC', 'EPIC+', 'epic', 'epic+'):
            filename = 'EPIC_Sesame.txt.gz'
        try:
            with resources.path(pkg_namespace, filename) as this_filepath:
                sesame = pd.read_csv(this_filepath, header=None)[0] # Series / list
        except:
            this_filepath = pkg_resources.resource_filename(pkg_namespace, filename)
            sesame = pd.read_csv(this_filepath, header=None)[0] # Series / list
    if type(criteria) == str:
        criteria = [criteria]

    criteria = [probe_pubs_translate.get(crit, crit) for crit in criteria]
    reasons = [reason for reason in criteria if reason in ['Polymorphism', 'CrossHybridization', 'BaseColorChange', 'RepeatSequenceElements']]
    pubs = [ref for ref in criteria if ref in probe_pubs_translate.values()]
    filtered_probes = probe_dataframe[ (probe_dataframe['Reason'].isin(reasons)) | (probe_dataframe['ShortCitation'].isin(pubs)) | (probe_dataframe['Probe'].isin(sesame)) ]
    return filtered_probes



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

Parameters:

    ``array``: string
        name for type of array used
        'IlluminaHumanMethylationEPIC', 'IlluminaHumanMethylation450k'
        This shorthand names are also okay:
            ``{'EPIC','EPIC+','450k','27k','MOUSE'}``

    ``criteria``: list
        List of the publications to use when excluding probes.
        If the array is 450K the publications may include:
        ``'Chen2013'
        'Price2013'
        'Zhou2016'
        'Naeem2014'
        'DacaRoszak2015'``

        If the array is EPIC the publications may include:
        ``'Zhou2016'
        'McCartney2016'``

        If array is EPIC or EPIC+, specifying ``'illumina'`` will remove 998
        probes the manufacturer has recommended be excluded. The defects only affected a small number of EPIC arrays produced.

        If no publication list is specified, probes from
        all publications will be added to the exclusion list.
        If more than one publication is specified, all probes
        from all publications in the list will be added to the
        exclusion list.

    ``criteria``: lists
        List of the criteria to use when excluding probes.
        List may contain the following exculsion criteria:
        ``'Polymorphism'
            'CrossHybridization'
            'BaseColorChange'
            'RepeatSequenceElements'
            'illumina'``

        If no criteria list is specified, all critera will be
        excluded. If more than one criteria is specified,
        all probes meeting any of the listed criteria will be
        added to the exclusion list.

    ``custom_list``: list, default None
        User-provided list of probes to be excluded.
        These probe names have to match the probe names in your data exactly.

Returns:

    probe_exclusion_list: list
        List containing probe identifiers to be excluded
    or probe_exclusion_dataframe: dataframe
        DataFrame containing probe names as index and reason | paper_reference as columns

If you supply no criteria (default), then maximum filtering occurs:

- EPIC will have 389050 probes removed
- 450k arrays will have 341057 probes removed

Reason lists for 450k and probes removed:

- Daca-Roszak_etal_2015 (96427)
- Chen_etal_2013 (445389)
- Naeem_etal_2014 (146590)
- Price_etal_2013 (284476)
- Zhou_etal_2016 (184302)
- Polymorphism (796290)
- CrossHybridization (211330)
- BaseColorChange (359)
- RepeatSequenceElements (149205)

Reason lists for epic and probes removed:

- McCartney_etal_2016 (384537)
- Zhou_etal_2016 (293870)
- CrossHybridization (173793)
- Polymorphism (504208)
- BaseColorChange (406)
    """
    all_criteria = ['Polymorphism', 'CrossHybridization',
        'BaseColorChange', 'RepeatSequenceElements', 'illumina',
        'Chen2013', 'Price2013', 'Zhou2016', 'Naeem2014', 'DacaRoszak2015',
        'Zhou2016', 'McCartney2016'
        ]
    if criteria != None and set(criteria) - set(all_criteria) != set():
        unrecognized = set(criteria) - set(all_criteria)
        raise ValueError(f"Unrecognized criteria: ({unrecognized})\n One these are allowed: {all_criteria}")

    translate = {
        'EPIC+': 'IlluminaHumanMethylationEPIC',
        'EPIC': 'IlluminaHumanMethylationEPIC',
        '450k': 'IlluminaHumanMethylation450k',
        '27k': 'IlluminaHumanMethylation27k',
        'MOUSE': 'MOUSE',
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
    [Polymorphism, CrossHybridization,
    BaseColorChange, RepeatSequenceElements,
    Chen2013, Price2013, Zhou2016, Naeem2014,
    DacaRoszak2015, McCartney2016]"""

    if not custom_list:
        custom_list = []

    if criteria and 'illumina' in criteria and array == 'IlluminaHumanMethylationEPIC':
        custom_list.extend(illumina_sketchy_probes)

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
            LOGGER.debug(f'{reason} --> {df.shape}')

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
            LOGGER.debug(f'{pub} --> {df.shape[0]}')

        probe_exclusion_list = list(set(probe_exclusion_list + custom_list))
    #LOGGER.debug(f'final probe list: {len(probe_exclusion_list)}')
    return probe_exclusion_list


def drop_nan_probes(df, silent=False, verbose=False):
    """ accounts for df shape (probes in rows or cols) so dropna() will work.

    the method used inside MDS may be faster, but doesn't tell you which probes were dropped."""
    ### histogram can't have NAN values -- so need to exclude before running, or warn user.
    # from https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null -- returns a slimmer df of col/rows with NAN.
    dfnan = df[df.isnull().any(axis=1)][df.columns[df.isnull().any()]]
    if len(dfnan) > 0 and df.shape[0] > df.shape[1]: # a list of probe names that contain nan.
        #probes in rows
        pre_shape = df.shape
        df = df.dropna()
        note = "(probes,samples)"
        if not silent and verbose and len(dfnan) < 200:
            LOGGER.info(f"Dropped {len(dfnan)} probe(s) that are missing for this calculation: {dfnan}")
            LOGGER.info(f"Retained {df.shape} {note} from the original {pre_shape} {note}.")
        elif not silent and verbose and len(dfnan) >= 200:
            LOGGER.info(f"Dropped {len(dfnan)} probes; retained {df.shape} {note} from the original {pre_shape} {note}.")
    elif len(dfnan.columns) > 0 and df.shape[1] > df.shape[0]:
        pre_shape = df.shape
        df = df.dropna(axis='columns')
        note = "(samples,probes)"
        if not silent and verbose and len(dfnan.columns) < 200:
            LOGGER.info(f"Dropped {len(dfnan.columns)} probe(s) that are missing for this calculation: {dfnan.columns}")
            LOGGER.info(f"Retained {df.shape} {note} from the original {pre_shape} {note}.")
        elif not silent and verbose and len(dfnan.columns) >= 200:
            LOGGER.info(f"Dropped {len(dfnan.columns)} probes; retained {df.shape} {note} from the original {pre_shape} {note}.")
    return df
