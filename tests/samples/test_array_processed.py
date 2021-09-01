import pandas as pd
from pathlib import Path
import methylcheck
import methylprep
filepath = Path('docs/example_data/mouse/')

def test_compare_control_probes_with_manifest():
    """ this should be a QC function, but not yet. showing here the number of probes in data,
    compared to probes in manifest """
    methylprep.run_pipeline(filepath, make_sample_sheet=True)
    mouse_man = methylprep.Manifest(methylprep.ArrayType('mouse'))
    #epic_man = methylprep.Manifest(methylprep.ArrayType('epic'))
    control = pd.read_pickle(Path(filepath, 'control_probes.pkl'))
    mouse_probes = pd.read_pickle(Path(filepath, 'mouse_probes.pkl'))
    if mouse_probes['204879580038_R06C02'].shape != (32753,8):
        raise AssertionError(f"mouse_probe count: {mouse_probes['204879580038_R06C02'].shape} should be (32753,8)")
    control_R = pd.DataFrame(list(control.values())[0][['Control_Type','Color','Extended_Type']])
    mouse_man.control_data_frame.Control_Type.value_counts()
    mouse_con = mouse_man.control_data_frame['Control_Type'].value_counts()
    sample_con = control_R['Control_Type'].value_counts()
    print(mouse_con)
    print(sample_con)
    #prev version <v0.7.5: assert(dict(mouse_con)['NEGATIVE'] == 1484 and dict(sample_con)['NEGATIVE'] == 179)
    if not (dict(mouse_con)['NEGATIVE'] == 411 and dict(sample_con)['NEGATIVE'] == 411):
        raise AssertionError(f"{dict(mouse_con)['NEGATIVE']} != 411 and/or {dict(sample_con)['NEGATIVE']} != 411")
    """
    NEGATIVE                   1484
BISULFITE CONVERSION II      88
BISULFITE CONVERSION I       84
NORM_T                       61
NORM_C                       61
SPECIFICITY I                49
SPECIFICITY II               43
NORM_A                       32
NORM_G                       32
NON-POLYMORPHIC              21
HYBRIDIZATION                 7
EXTENSION                     2
RESTORATION                   1
TARGET REMOVAL                1

(sample)
NEGATIVE                   179
NORM_T                      24
NORM_C                      22
NORM_A                      10
NORM_G                       9
BISULFITE CONVERSION I       3
NON-POLYMORPHIC              3
SPECIFICITY I                3
BISULFITE CONVERSION II      2
HYBRIDIZATION                2
SPECIFICITY II               2
RESTORATION                  1

>>> mouse_man.control_data_frame.Control_Type.value_counts()
NEGATIVE                   411
NORM_C                      58
NORM_T                      58
NORM_G                      27
NORM_A                      27
SPECIFICITY I               12
BISULFITE CONVERSION I      10
NON-POLYMORPHIC              9
STAINING                     6
EXTENSION                    4
BISULFITE CONVERSION II      4
SPECIFICITY II               3
HYBRIDIZATION                3
TARGET REMOVAL               2
RESTORATION                  1

    # making a list of mouse probes that overlap with epic probes
    epic_names = [row[0] for row in manifest.data_frame.index.str.split('_') if row[0].startswith('cg')]
    epic_cg = [row[0] for row in epic_man.data_frame.index.str.split('_') if row[0].startswith('cg')]
    print(len(epic_names), len(epic_cg), len( set(epic_names) & set(epic_cg) ))
    #print(len( set(probes['Probe']) & set(epic_names)) )
    """
    for _file in Path(filepath).rglob('*'):
        if '.idat' in _file.suffixes or _file.name in ('mouse_probes.pkl','control_probes.pkl'):
            # other tests need these
            continue
        _file.unlink()
