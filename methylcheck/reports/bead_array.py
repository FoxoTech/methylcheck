#io
from pathlib import Path
from io import StringIO
import logging

# calcs
import math
import numpy as np
import pandas as pd

# app
import methylcheck

LOGGER = logging.getLogger(__name__)

__all__ = ['BeadArrayControlsReporter', 'controls_report']


class BeadArrayControlsReporter():
    """This will load all the methylprep control and raw output data,
    then perform all the calculations listed in
    https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium_hd_methylation/beadarray-controls-reporter-user-guide-1000000004009-00.pdf
    and then produce a tidy color-coded XLSX file with results.

    this function is analogous to methylcheck.plot_controls() except that the output if a color-coded excel sheet instead of charts
    """
    input_filenames = {
        'control_probes.pkl': 'control',
    }
        # does NOT use m_values, meth, unmeth, or any of these yet
        #'sample_sheet_meta_data.pkl': 'samples',
        #'poobah_values.pkl': 'poobah',
        #'noob_meth_values.pkl': 'noob_meth',
        #'noob_unmeth_values.pkl': 'noob_unmeth',
        #'beta_values.pkl': 'betas',

    def __init__(self, filepath, outfilepath=None, bg_offset=3000, colorblind=False, cutoff_adjust=1.0):
        # if outfilepath is not provided, saves to the same folder where the pickled dataframes are located.
        if not outfilepath:
            self.outfilepath = filepath
        else:
            self.outfilepath = outfilepath
        for filename in Path(filepath).rglob('*.pkl'):
            if filename.name in self.input_filenames.keys():
                #vars()[self.input_filenames[filename.name]] = pd.read_pickle(filename)
                setattr(self, self.input_filenames[filename.name], pd.read_pickle(filename))
        if not hasattr(self,'control'):
            raise FileNotFoundError(f"Could not locate control_probes.pkl file in {filepath}")
        self.bg_offset=bg_offset
        self.report = [] # will convert to DF after collecting data; faster; pd.DataFrame(columns=self.report_columns)
        self.formulas = {} # col: formula as string/note
        self.data = {} # sample: {col: <colname>, val: ___, pass: ...} for coloring boxes
        if colorblind:
            self.cpass = '#EDA247'
            self.cmid =  '#FFDD71'
            self.cfail = '#57C4AD'
        else:
            self.cpass = '#F26C64'
            self.cmid =  '#FFDD71'
            self.cfail = '#69B764'
        self.cut = cutoff_adjust

    def process_sample(self, sample, con):
        """ process() will run this throug all samples, since structure of control data is a dict of DFs
        bg_offset = Background correction offset.
        Default value: 3000
            (applies to all background calculations, indicated with (bkg +x).)

NEGATIVE control probes are used as the baseline for p-val calculations.
    (but NEGATIVE and NORM_ not used in BeadArray)

see also https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium_hd_methylation/infinium-hd-methylation-guide-15019519-01.pdf
for list of expected intensity per type

MOUSE conversions (or proxy)
baseline_G Extension -- missing -- use NEGATIVE Hairpin probes as proxy
    -- maybe take average of NORM_A/NORM_T (Green) as proxy for background?
baseline_R Extension -- missing -- NORM_C + NORM_G Red proxy

BIS I II OK
SPEC OK
RESTORATION OK
non-poly OK
hyb OK HIGH = 3_HIGH_MM_50.1_1, mid/low? 90_YEAST_3MM_50.1_1
within NATIVE
        non_specific
        GT mismatch
NO staining found
target removal

        """
        # to get a list of these probes, use
        # con[(~con['Control_Type'].isna()) & (~con['Control_Type'].isin(['NEGATIVE','NORM_A','NORM_T','NORM_C','NORM_G']))][['Control_Type','Extended_Type','Color']]

        # baseline = (Extension Green highest A or T intensity) + offset
        mouse = False
        try:
            baseline_G = max([con[con['Extended_Type'] == 'Extension (A)']['Mean_Value_Green'].values[0], con[con['Extended_Type'] == 'Extension (T)']['Mean_Value_Green'].values[0] ]) + self.bg_offset
            baseline_R = max([con[con['Extended_Type'] == 'Extension (C)']['Mean_Value_Red'].values[0], con[con['Extended_Type'] == 'Extension (G)']['Mean_Value_Red'].values[0] ]) + self.bg_offset
        except: # assume mouse
            mouse = True
            baseline_G = con[con['Extended_Type'] == 'T_Hairpin2.1_1']['Mean_Value_Green'].values[0] + self.bg_offset
            baseline_R = con[con['Extended_Type'] == 'G_Hairpin2.1_1']['Mean_Value_Red'].values[0] + self.bg_offset

        # ("Green"/(bkg+x)) > 0* | restoration_green is Green Channel Intensity/Background.
        self.restoration_green = round( con[con['Extended_Type'].isin(['Restore','neg_ALDOB_3915-4004_1'])]['Mean_Value_Green'].values[0] / baseline_G ,2)

        if mouse:
            self.staining_green = np.nan
            self.staining_red = np.nan
        else:
            # (Biotin High/Biotin Bkg) > 5
            self.staining_green = round( con[con['Extended_Type'] == 'Biotin (High)']['Mean_Value_Green'].values[0] / con[con['Extended_Type'] == 'Biotin (Bkg)']['Mean_Value_Green'].values[0] ,2)
            # (DNP High > DNP Bkg) > 5
            self.staining_red = round( con[con['Extended_Type'] == 'DNP (High)']['Mean_Value_Red'].values[0] / con[con['Extended_Type'] == 'DNP (Bkg)']['Mean_Value_Red'].values[0] ,2)

        if mouse:
            self.extension_green = round( con[con['Extended_Type'] == 'G_Hairpin2.1_1']['Mean_Value_Green'].values[0] / con[con['Extended_Type'] == 'T_Hairpin2.1_1']['Mean_Value_Green'].values[0] ,2)
            self.extension_red =   round( con[con['Extended_Type'] == 'T_Hairpin2.1_1']['Mean_Value_Red'].values[0] / con[con['Extended_Type'] == 'G_Hairpin2.1_1']['Mean_Value_Red'].values[0] ,2)
        else:
            # GREEN min(C or G)/max(A or T) > 5
            self.extension_green = round( min( con[con['Extended_Type'] == 'Extension (C)']['Mean_Value_Green'].values[0], con[con['Extended_Type'] == 'Extension (G)']['Mean_Value_Green'].values[0]) / max( con[con['Extended_Type'] == 'Extension (A)']['Mean_Value_Green'].values[0], con[con['Extended_Type'] == 'Extension (T)']['Mean_Value_Green'].values[0]) ,2)
            # RED max(C or G)/min(A or T) > 5
            self.extension_red = round( min( con[con['Extended_Type'] == 'Extension (A)']['Mean_Value_Red'].values[0], con[con['Extended_Type'] == 'Extension (T)']['Mean_Value_Red'].values[0]) / max( con[con['Extended_Type'] == 'Extension (C)']['Mean_Value_Red'].values[0], con[con['Extended_Type'] == 'Extension (G)']['Mean_Value_Red'].values[0]) ,2)

        if mouse:
            # Hyb (High/Med)
            self.hybridization_green_A = round( con[con['Extended_Type']=='3_HIGH_MM_50.1_1']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='90_YEAST_3MM_50.1_1']['Mean_Value_Green'].values[0] ,2)
            self.hybridization_green_B = np.nan
        else:
            # Hyb (High/Med) > 1
            self.hybridization_green_A = round( con[con['Extended_Type']=='Hyb (High)']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='Hyb (Medium)']['Mean_Value_Green'].values[0] ,2)
            # Hyb (Med/Low) > 1
            self.hybridization_green_B = round( con[con['Extended_Type']=='Hyb (Medium)']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='Hyb (Low)']['Mean_Value_Green'].values[0] ,2)
            # Hyb (High > Med > Low)
            #self.hybridization_green_C = round( con[con['Extended_Type']=='Hyb (High)']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='Hyb (Medium)']['Mean_Value_Green'].values[0] ,2)

        if mouse:
            self.target_removal_green_1 = np.nan
            self.target_removal_green_2 = np.nan
        else:
            # Target ctrl 1 ≤ bkg
            self.target_removal_green_1 = round( baseline_G / con[con['Extended_Type'] == 'Target Removal 1']['Mean_Value_Green'].values[0] ,2)
            # Target ctrl 2 ≤ bkg
            self.target_removal_green_2 = round( baseline_G / con[con['Extended_Type'] == 'Target Removal 2']['Mean_Value_Green'].values[0], 2)

        # con[con['Extended_Type']=='BS_Conversion_I_24_1']['Mean_Value_Green'].values[0],
        if mouse:
            # BS_Conversion_I_54_1, BS_Conversion_I_55_1, BS_Conversion_I_24_1,
            # BS_Conversion_II_5_1, BS_Conversion_II_21_1, BS_Conversion_I_17_1, BS_Conversion_I_72_1
            # higher: 54, 17
            # lower: 24, 55, 72
            self.bisulfite_conversion_I_green_CU = round(
                con[con['Extended_Type']=='BS_Conversion_I_54_1']['Mean_Value_Green'].values[0] /
                con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Green'].values[0], 2)
            self.bisulfite_conversion_I_green_bkg_U = round( baseline_G / con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Green'].values[0], 2)
            self.bisulfite_conversion_I_red_CU = round(
                con[con['Extended_Type']=='BS_Conversion_I_54_1']['Mean_Value_Red'].values[0] /
                con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Red'].values[0], 2)
            self.bisulfite_conversion_I_red_bkg_U = round( baseline_R / con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Red'].values[0], 2)
            self.bisulfite_conversion_II_C_red_ratio = round( np.mean([
                con[con['Extended_Type']=='BS_Conversion_II_5_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS_Conversion_II_21_1']['Mean_Value_Red'].values[0]
                ]) / np.mean([
                con[con['Extended_Type']=='BS_Conversion_II_5_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS_Conversion_II_21_1']['Mean_Value_Green'].values[0]
                ]), 2)
            self.bisulfite_conversion_II_C_green_bkg = round( baseline_G / np.mean([
                con[con['Extended_Type']=='BS_Conversion_II_5_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS_Conversion_II_21_1']['Mean_Value_Green'].values[0]
                ]), 2)
        else:
            # META NOTE: BS Conversion I-C1 is "I C1" in 450k. U1 also differs.
            # BS min(C1, 2, or 3) / BS max(U1, 2, 3) > 1
            self.bisulfite_conversion_I_green_CU = round( min([
                    con[con['Extended_Type'].isin(['BS Conversion I-C1','BS Conversion I C1'])]['Mean_Value_Green'].values[0],
                    con[con['Extended_Type']=='BS Conversion I-C2']['Mean_Value_Green'].values[0],
                    con[con['Extended_Type']=='BS Conversion I-C3']['Mean_Value_Green'].values[0]
                ]) / max([
                    con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Green'].values[0],
                    con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Green'].values[0],
                    con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Green'].values[0]
                ]), 2)
            # Bisulfite Conversion I Green U ≤ bkg | ((bkg + x)/U) > 1
            self.bisulfite_conversion_I_green_bkg_U = round( baseline_G / max([con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Green'].values[0], con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Green'].values[0], con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Green'].values[0], con[con['Extended_Type']=='BS Conversion I-U4']['Mean_Value_Green'].values[0], con[con['Extended_Type']=='BS Conversion I-U5']['Mean_Value_Green'].values[0]]) ,2)

            # BS RED min(C1, 2, or 3) / BS max(U1, 2, 3) > 1
            self.bisulfite_conversion_I_red_CU = ( round(
                min([con[con['Extended_Type'].isin(['BS Conversion I-C1','BS Conversion I C1'])]['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-C2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-C3']['Mean_Value_Red'].values[0]]) /
                max([con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Red'].values[0]])
                ,2) )
            # Bisulfite Conversion I Red U ≤ bkg | ((bkg + x)/U) > 1
            self.bisulfite_conversion_I_red_bkg_U = round( baseline_R / max([con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Red'].values[0], con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Red'].values[0], con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Red'].values[0], con[con['Extended_Type']=='BS Conversion I-U4']['Mean_Value_Red'].values[0], con[con['Extended_Type']=='BS Conversion I-U5']['Mean_Value_Red'].values[0]]) ,2)

            #### USING THE MEAN HERE, not min or max like other tests, because PDF was unclear.
            # Bisulfite Conversion II C Red > C Green
            self.bisulfite_conversion_II_C_red_ratio = round( np.mean([
                con[con['Extended_Type']=='BS Conversion II-1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion II-2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion II-3']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion II-4']['Mean_Value_Red'].values[0]
                ]) / np.mean([
                con[con['Extended_Type']=='BS Conversion II-1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-3']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-4']['Mean_Value_Green'].values[0]
                ]), 2)
            # BiSulfite Conversion II C green ≤ bkg
            self.bisulfite_conversion_II_C_green_bkg = round( baseline_G / np.mean([
                con[con['Extended_Type']=='BS Conversion II-1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-3']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-4']['Mean_Value_Green'].values[0]
                ]), 2)

        if mouse:
            # Non_Specific_I_11_1, Non_Specific_I_24_1, Non_Specific_I_3_1,
            # Non_Specific_II_1_1, Non_Specific_II_17_1, GT_mismatch_ATG2_12259-12348_1
            # Non_Specific_I_30_1, Non_Specific_I_20_1, Non_Specific_I_9_1,
            # Non_Specific_I_15_1, Non_Specific_I_14_1, Non_Specific_I_42_1
            self.specificity_I_green = round(
                con[con['Extended_Type']=='Non_Specific_I_42_1']['Mean_Value_Green'].values[0]
                / max(
                con[con['Extended_Type']=='Non_Specific_I_24_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Non_Specific_I_11_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Non_Specific_I_3_1']['Mean_Value_Green'].values[0],
                ),2)
            self.specificity_I_red = round(
                con[con['Extended_Type']=='Non_Specific_I_42_1']['Mean_Value_Red'].values[0]
                / max([
                con[con['Extended_Type']=='Non_Specific_I_24_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Non_Specific_I_11_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Non_Specific_I_3_1']['Mean_Value_Red'].values[0],
                ]),2)
            self.specificity_II = round( np.mean([
                con[con['Extended_Type']=='Non_Specific_II_17_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Non_Specific_II_1_1']['Mean_Value_Red'].values[0],
                ]) / np.mean([
                con[con['Extended_Type']=='Non_Specific_II_17_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Non_Specific_II_1_1']['Mean_Value_Green'].values[0],
                ]),2)
            self.specificity_II_bkg= round( baseline_G / np.mean([
                con[con['Extended_Type']=='Non_Specific_II_17_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Non_Specific_II_1_1']['Mean_Value_Green'].values[0],
                ]),2)
        else:
            # Specificity I Green (min(PM)/max(MM)) > 1
            self.specificity_I_green = round( min(
                con[con['Extended_Type']=='GT Mismatch 1 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 2 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 3 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 4 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 5 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 6 (PM)']['Mean_Value_Green'].values[0],
                ) / max(
                con[con['Extended_Type']=='GT Mismatch 1 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 2 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 3 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 4 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 5 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 6 (MM)']['Mean_Value_Green'].values[0],
                ),2)

            # Specificity I Red (min(PM)/max(MM)) > 1
            self.specificity_I_red = round( min(
                con[con['Extended_Type']=='GT Mismatch 1 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 2 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 3 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 4 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 5 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 6 (PM)']['Mean_Value_Red'].values[0],
                ) / max(
                con[con['Extended_Type']=='GT Mismatch 1 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 2 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 3 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 4 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 5 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 6 (MM)']['Mean_Value_Red'].values[0],
                ),2)

            # Specificity 1, Specificity 2, Specificity 3
            # Specificity II (S Red/ S Green) > 1
            self.specificity_II = round( np.mean([
                con[con['Extended_Type']=='Specificity 1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Specificity 2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Specificity 3']['Mean_Value_Red'].values[0],
                ]) / np.mean([
                con[con['Extended_Type']=='Specificity 1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 3']['Mean_Value_Green'].values[0],
                ]),2)

            # Specificity II (background/ Spec Green) > 1
            self.specificity_II_bkg = round( baseline_G / np.mean([
                con[con['Extended_Type']=='Specificity 1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 3']['Mean_Value_Green'].values[0],
                ]),2)

        if mouse:
            self.nonpolymorphic_green_lowCG_highAT = round( min([
                con[con['Extended_Type']=='nonPolyG_PPIH_9298-9387_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='nonPolyC_PPIE_21091-21180_1']['Mean_Value_Green'].values[0],
                ]) / max([
                con[con['Extended_Type']=='nonPolyT_ALDOB_10349-10438_1']['Mean_Value_Green'].values[0]
                ]),2)
            self.nonpolymorphic_red_lowAT_highCG = round( min([
                con[con['Extended_Type']=='nonPolyT_ALDOB_10349-10438_1']['Mean_Value_Red'].values[0],
                ]) / max([
                con[con['Extended_Type']=='nonPolyC_PPIE_21091-21180_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='nonPolyG_PPIH_9298-9387_1']['Mean_Value_Red'].values[0],
                ]),2)
        else:
            # Nonpolymorphic Green (min(CG)/ max(AT)) > 5
            self.nonpolymorphic_green_lowCG_highAT = round( min([
                con[con['Extended_Type']=='NP (C)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='NP (G)']['Mean_Value_Green'].values[0],
                ]) / max([
                con[con['Extended_Type']=='NP (A)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='NP (T)']['Mean_Value_Green'].values[0],
                ]),2)

            # Nonpolymorphic Red (min(AT)/ max(CG)) > 5
            self.nonpolymorphic_red_lowAT_highCG = round( min([
                con[con['Extended_Type']=='NP (A)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='NP (T)']['Mean_Value_Red'].values[0],
                ]) / max([
                con[con['Extended_Type']=='NP (C)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='NP (G)']['Mean_Value_Red'].values[0],
                ]),2)

        # ADDITIONAL tests (not in BeadArray)
        self.negative_control_mean_green = round( np.mean(
        con[con['Control_Type'] == 'NEGATIVE']['Mean_Value_Green'].values,
        ))
        self.negative_control_mean_red = round( np.mean(
        con[con['Control_Type'] == 'NEGATIVE']['Mean_Value_Red'].values,
        ))
        self.norm_A_mean_green = round( np.mean(
        con[con['Control_Type'] == 'NORM_A']['Mean_Value_Green'].values,
        ))
        self.norm_T_mean_green = round( np.mean(
        con[con['Control_Type'] == 'NORM_T']['Mean_Value_Green'].values,
        ))
        self.norm_C_mean_red = round( np.mean(
        con[con['Control_Type'] == 'NORM_A']['Mean_Value_Red'].values,
        ))
        self.norm_G_mean_red = round( np.mean(
        con[con['Control_Type'] == 'NORM_T']['Mean_Value_Red'].values,
        ))

        self.data[sample] = [
            {'col': 'Restoration Green', 'val': self.restoration_green, 'min': -1, 'mid': -0.1, 'max':0, 'formula': "(Restore(Green)/ background) > 0"},
            {'col': 'Staining Green', 'val': self.staining_green, 'max':(5*self.cut), 'formula': f"(Biotin High/ Biotin background) > {5*self.cut}"},
            {'col': 'Staining Red', 'val': self.staining_red, 'max':(5*self.cut), 'formula': f"(DNP High/ DNP background) > {5*self.cut}"},
            {'col': 'Extension Green', 'val':self.extension_green, 'max':(5*self.cut), 'formula': f"min(C or G)/ max(A or T) > {5*self.cut}"},
            {'col': 'Extension Red', 'val':self.extension_red, 'max':(5*self.cut), 'formula': f"max(C or G)/ min(A or T) > {5*self.cut}"},
            {'col': 'Hybridization Green (High/Medium)', 'val':self.hybridization_green_A, 'max':self.cut, 'formula': f"Hyb (High/Med) > {self.cut}"},
            {'col': 'Hybridization Green (Medium/Low)', 'val':self.hybridization_green_B, 'max':self.cut, 'formula': f"Hyb (Med/Low) > {self.cut}"},
            {'col': 'Target Removal Green 1', 'val':self.target_removal_green_1, 'max':self.cut, 'formula': f"(Target ctrl 1 ≤ background) > {self.cut}"},
            {'col': 'Target Removal Green 2', 'val':self.target_removal_green_2, 'max':self.cut, 'formula': f"(Target ctrl 2 ≤ background) > {self.cut}"},
            {'col': 'Bisulfite Conversion I Green C/U', 'val':self.bisulfite_conversion_I_green_CU, 'min':0, 'mid':0.7*self.cut, 'max':self.cut, 'formula': f"BS min(C1,2,_or_3) / BS max(U1, 2, 3) > {self.cut}"},
            {'col': 'Bisulfite Conversion I Green bkg/U', 'val':self.bisulfite_conversion_I_green_bkg_U, 'min':0, 'mid':0.7*self.cut, 'max':self.cut, 'formula': f"BS U ≤ background"},
            {'col': 'Bisulfite Conversion I Red C/U', 'val':self.bisulfite_conversion_I_red_CU, 'min':0, 'mid':0.7*self.cut, 'max':self.cut, 'formula': f"BS min(C1,2,_or_3) / BS max(U1, 2, 3) > {self.cut}"},
            {'col': 'Bisulfite Conversion I Red bkg/U', 'val':self.bisulfite_conversion_I_red_bkg_U, 'min':0, 'mid':0.7*self.cut, 'max':self.cut, 'formula': f"BS U ≤ background"},
            {'col': 'Specificity I Green', 'val':self.specificity_I_green, 'min':0, 'mid':0.7, 'max':self.cut, 'formula': f"GT Mismatch (min(PM)/max(MM)) > {self.cut}"},
            {'col': 'Specificity I Red', 'val':self.specificity_I_red, 'min':0, 'mid':0.7, 'max':self.cut, 'formula': f"GT Mismatch (min(PM)/max(MM)) > {self.cut}"},
            {'col': 'Specificity II', 'val':self.specificity_II, 'min':0, 'mid':0.7, 'max':self.cut, 'formula': f"(S_Red/ S_Green) > {self.cut}"},
            {'col': 'Specificity II Bkg', 'val':self.specificity_II_bkg, 'min':0, 'mid':0.7, 'max':self.cut, 'formula': f"(background/ S_Green) > {self.cut}"},
            {'col': 'Non-polymorphic Green', 'val':self.nonpolymorphic_green_lowCG_highAT, 'min':0, 'mid':0.7*self.cut, 'max':5*self.cut, 'formula': f"(min(CG)/ max(AT)) > {5*self.cut}"},
            {'col': 'Non-polymorphic Red', 'val':self.nonpolymorphic_red_lowAT_highCG, 'min':0, 'mid':0.7*self.cut, 'max':5*self.cut, 'formula': f"(min(AT)/ max(CG)) > {5*self.cut}"},
            {'col': 'Baseline Green', 'val':baseline_G - self.bg_offset, 'formula': "max(Extension (A), Extension (T)) no offset", 'max':400, 'med':200, 'min':100},
            {'col': 'Baseline Red', 'val':baseline_R - self.bg_offset, 'formula': "max(Extension (C), Extension (G)) no offset", 'max':800, 'med':400, 'min':100},
            {'col': 'Negative Baseline G', 'val':self.negative_control_mean_green, 'formula': "mean NEGATIVE Green control probes"},
            {'col': 'Negative Baseline R', 'val':self.negative_control_mean_red, 'formula': "mean NEGATIVE Red control probes"},
            {'col': 'NORM_A', 'val':self.norm_A_mean_green, 'formula': "mean NORM_A control probes Green)", 'max':600, 'med':300, 'min':100},
            {'col': 'NORM_T', 'val':self.norm_T_mean_green, 'formula': "mean NORM_T control probes Green)", 'max':400, 'med':200, 'min':100},
            {'col': 'NORM_C', 'val':self.norm_C_mean_red, 'formula': "mean NORM_C control probes Red)", 'max':2000, 'med':1000, 'min':100},
            {'col': 'NORM_G', 'val':self.norm_G_mean_red, 'formula': "mean NORM_G control probes Red)", 'max':2000, 'med':1000, 'min':100},
        ]

        row = {'Sample': sample}
        row.update({k['col']:k['val'] for k in self.data[sample]})
        self.report.append(row)
        self.formulas.update( {k['col']:k['formula'] for k in list(self.data.values())[0]} )

    def process(self):
        for sample,con in self.control.items():
            self.process_sample(sample, con) # saves everything on top of last sample, for now. testing.

    def save(self):
        writer = pd.ExcelWriter(Path(self.outfilepath,'QC_Report.xlsx'), engine='xlsxwriter')
        report_df= pd.DataFrame(self.report)
        report_df.to_excel(writer, sheet_name='QC_REPORT', startcol=0, startrow=2, header=False, index=False)
        # Get the xlsxwriter objects from the dataframe writer object.
        workbook  = writer.book
        worksheet = writer.sheets['QC_REPORT']
        cell_format = workbook.add_format({'text_wrap':True})
        cell_format.set_align('vjustify')
        cell_format.set_text_wrap(True)
        #cell_format.set_pattern(1)
        #cell_format.set_border(1)
        #cell_format.set_border_color('black')

        def c(col): # converts to an EXCEL COLUMN name, like A or AE
            this_box = f"{chr(col+65)}"
            if 25 < col <= 51: # good for 52 rows wide.
                this_box = f"A{chr((col-26)+65)}"
            if 51 < col <= (77): # good for 78 rows wide.
                this_box = f"B{chr((col-52)+65)}"
            return this_box

        # fix header text wrap
        col_widths = {'Sample': max([len(item['Sample']) for item in self.report])}
        for col,header in enumerate(report_df.columns):
            this_box = f"{c(col)}1"
            worksheet.write(this_box, header, cell_format)
            if header not in self.formulas:
                continue
            this_box = f"{c(col)}2"
            worksheet.write(this_box, self.formulas[header], cell_format)
            col_widths[header] = len(header)

        for row, (sample, meta_row) in enumerate(self.data.items()):
            for col, k in enumerate(meta_row):
                # skipping first row and first column
                this_box = f"{c(col+1)}{row+3}"
                coloring = {'type': '3_color_scale',
                    'min_value':k.get('min',0), 'max_value': k.get('max',1), 'mid_value': k.get('mid', round(k.get('max',1)/2)),
                    'min_color':self.cpass, 'max_color':self.cfail, 'mid_color':self.cmid,
                    "min_type": "num", "mid_type": "num", "max_type": "num"}
                worksheet.conditional_format(this_box, coloring)
                # next: track if any values are longer than the column names, to adjust widths at end
                if len(str(k['val'])) > col_widths[k['col']]:
                    col_widths[k['col']] = len(str(k['val']))
                if len(str(k.get('formula',0))) > col_widths[k['col']]:
                    col_widths[k['col']] = len(str(k.get('formula',0)))

        #worksheet.conditional_format('B2:B8', {'type': '3_color_scale', 'min_value':0, 'max_value':1, 'mid_value':0.5})
        # There is no way to specify “AutoFit” for a column in the Excel file format.
        # This feature is only available at runtime from within Excel.
        # It is possible to simulate “AutoFit” in your application by tracking the maximum width
        # of the data in the column as your write it and then adjusting the column width at the end.
        # setting column widths
        col = 0
        for header,col_width in col_widths.items():
            this_box = f"{c(col+1)}:{c(col+1)}"
            worksheet.set_column(this_box, min(int(0.6*col_width),100))
            print(this_box, min(int(0.6*col_width),100), header)
            col += 1
        worksheet.set_column("A:A", 24) # samples
        writer.save()


def controls_report(**kwargs):
    """Run a clone of Illumina's Bead Array Controls Reporter to generate an excel QC Report.

    This will load the methylprep control_probes.pkl file and perform all the calculations listed in
    https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium_hd_methylation/beadarray-controls-reporter-user-guide-1000000004009-00.pdf
    then produce a tidy color-coded XLSX file with results.

    Note: this function is analogous to methylcheck.plot_controls() except that the output if a color-coded excel sheet instead of image charts.

Required:
    filepath

Optional Arguments:

    outfilepath (None)
        If not specified, QC_Report.xlsx is saved in the same folder with the control probes file.
    colorblind (False)
        Set to True to enabled pass/fail colors that work better for colorblind users.
    bg_offset (3000)
        Illumina's software add 3000 to the baseline/background values as an offset.
        This function does the same, unless you change this value.
    cutoff_adjust (1.0)
        A value of 1.0 will replicate Illumina's definition of pass/fail cutoff values for tests.
        You can increase this number to apply more stringent benchmarks for what constitutes good
        data, typically in the range of 1.0 to 3.0 (as a 1X to 3X cutoff multiplier).
    """
    if 'filepath' not in kwargs:
        raise FileNotFoundError("No control probes file location specified.")
    reporter = BeadArrayControlsReporter(**kwargs)
    reporter.process()
    reporter.save()
