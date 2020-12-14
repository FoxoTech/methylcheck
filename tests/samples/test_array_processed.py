import pandas as pd
from pathlib import Path
import methylcheck
import methylprep

def test_compare_control_probes_with_manifest():
    """ this should be a QC function, but not yet. showing here the number of probes in data,
    compared to probes in manifest """
    mouse_man = methylprep.Manifest(methylprep.ArrayType('mouse'))
    epic_man = methylprep.Manifest(methylprep.ArrayType('epic'))
    filepath = Path('docs/example_data/mouse/')
    control = pd.read_pickle(Path(filepath, 'control_probes.pkl'))
    control_R = pd.DataFrame(list(control.values())[0][['Control_Type','Color','Extended_Type']])
    mouse_man.control_data_frame.Control_Type.value_counts()
    mouse_con = mouse_man.control_data_frame['Control_Type'].value_counts()
    sample_con = control_R['Control_Type'].value_counts()
    print(mouse_con)
    print(sample_con)
    assert(dict(mouse_con)['NEGATIVE'] == 1484 and dict(sample_con)['NEGATIVE'] == 179)
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

    # making a list of mouse probes that overlap with epic probes
    epic_names = [row[0] for row in manifest.data_frame.index.str.split('_') if row[0].startswith('cg')]
    epic_cg = [row[0] for row in epic_man.data_frame.index.str.split('_') if row[0].startswith('cg')]
    print(len(epic_names), len(epic_cg), len( set(epic_names) & set(epic_cg) ))
    #print(len( set(probes['Probe']) & set(epic_names)) )
    """
