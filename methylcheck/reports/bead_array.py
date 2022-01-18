#io
from pathlib import Path
from io import StringIO
import logging
import fnmatch
# calcs
import warnings
import math
import numpy as np
np.seterr(all='print')
import pandas as pd
from scipy.stats import linregress

# app
import methylcheck

LOGGER = logging.getLogger(__name__)

__all__ = ['ControlsReporter', 'controls_report']


class ControlsReporter():
    """Class used by controls_report() to produce XLSX summary of control probe performance.

    This will load all the methylprep control and raw output data, then perform the calculations recommended by manufacturer
    then produce a tidy color-coded XLSX file with results. This function is analogous to methylcheck.plot_controls() except that the output if a color-coded excel sheet instead of charts.
    Last column "Result" will include OK (green), MARGINAL (yellow), or FAIL (RED) -- as a summary of all other tests.

    If there is a meta data pickle, and there's a Sex or Gender column, it will compare SEX.
        Sex in samplesheet must be reported as "M" or "F" not 0 or 1, or "Male"/"Female" etc.
    Otherwise, it just runs and predicts the sex for non-mouse arrays.

    Note on GCT scores: this uses noob_meth instead of raw, uncorrected meth values to calculate, but the result should be nearly the same.
    """
    input_filenames = {
        'control_probes.pkl': 'control',
        'poobah_values.pkl': 'poobah',
        # FUTURE?? load these separately, and only if there is a reason to run a sex prediction. But sex done by default.
        'noob_meth_values.pkl': 'noob_meth',
        'noob_unmeth_values.pkl': 'noob_unmeth',
    }
    samplesheet_patterns = {
        '*meta_data*.pkl': 'samplesheet', # meta_data is used first, if both available
        '*samplesheet*.csv': 'samplesheet',
        '*sample_sheet*.csv': 'samplesheet',
    }
    # does NOT use m_values or 'beta_values.pkl'

    legacy_columns = {
        # NONE tells code not to include in legacy mode. Sample column is handled in special way.
        #'Sample': 'Sample Name', # leave off; would need to come from sample sheet meta data
        #ADD Sentrix Barcode, Sentrix Position from pkl columns
        'Sample': None,
        'Restoration Green': 'Restoration',
        'Staining Green': 'StainingGreen',
        'Staining Red': 'StainingRed',
        'Extension Green': 'ExtensionGreen',
        'Extension Red': 'ExtensionRed',
        'Hybridization Green (High/Medium)': 'HybridizationHighMedium',
        'Hybridization Green (Medium/Low)': 'HybridizationMediumLow',
        'Target Removal Green 1': 'TargetRemoval1',
        'Target Removal Green 2': 'TargetRemoval2',
        'Bisulfite Conversion I Green C/U': 'BisulfiteConversion1Green',
        'Bisulfite Conversion I Green bkg/U': 'BisulfiteConversion1BackgroundGreen',
        'Bisulfite Conversion I Red C/U': 'BisulfiteConversion1Red',
        'Bisulfite Conversion I Red bkg/U': 'BisulfiteConversion1BackgroundRed',
        'Bisulfite Conversion II Red/Green': 'BisulfiteConversion2',
        'Bisulfite Conversion II bkg/Green': 'BisulfiteConversion2Background',
        'Specificity I Green': 'Specificity1Green',
        'Specificity I Red': 'Specificity1Red',
        'Specificity II': 'Specificity2',
        'Specificity II Bkg': 'Specificity2Background',
        'Non-polymorphic Green': 'NonPolymorphicGreen',
        'Non-polymorphic Red': 'NonPolymorphicRed',
        # additional columns that WON'T appear in legacy report
        'Baseline Green': None,
        'Baseline Red': None,
        'Negative Baseline G': None,
        'Negative Baseline R': None,
        'NORM_A': None,
        'NORM_T': None,
        'NORM_C': None,
        'NORM_G': None,
        'Result': None,
        'Passing Probes': None,
        'Regression NORM_GA': None,
        'Regression NORM_CT': None,
        'Predicted Sex': None,
        'Sex Match': None,
        'GCT score': None,
    }
    untestable_columns = ['Baseline Green',
        'Baseline Red', 'Negative Baseline G',
        'Negative Baseline R', 'NORM_A','NORM_T',
        'NORM_C',    'NORM_G',
        'Passing Probes',
        'Result',
        'Predicted Sex']

    def __init__(self, filepath, outfilepath=None, bg_offset=3000, cutoff_adjust=1.0, colorblind=False,
        roundoff=2, legacy=False, pval=True, pval_sig=0.05, passing=0.7, project_name=None):
        self.filepath = filepath # folder with methyprep processed data
        self.bg_offset = bg_offset
        self.cut = cutoff_adjust # for adjusting minimum passing per test
        self.legacy = legacy # how output XLSX should be formatted
        self.roundoff = 1 if self.legacy else roundoff
        self.pval = pval # whether to include poobah in tests
        self.pval_sig = pval_sig # significance level to define a failed probe
        self.passing = passing # fraction of tests that all need to pass for sample to pass
        self.project_name = project_name # used to name the QC report, if defined.
        # if outfilepath is not provided, saves to the same folder where the pickled dataframes are located.
        if not outfilepath:
            self.outfilepath = filepath
        else:
            self.outfilepath = outfilepath

        for filename in Path(filepath).rglob('*.pkl'):
            if filename.name in self.input_filenames.keys():
                setattr(self, self.input_filenames[filename.name], pd.read_pickle(filename))
        # fuzzy matching samplesheet
        for filename in Path(filepath).rglob('*'):
            if any([fnmatch.fnmatch(filename.name, samplesheet_pattern) for samplesheet_pattern in self.samplesheet_patterns.keys()]):
                #label = next(label for (patt,label) in samplesheet_patterns.items() if fnmatch.fnmatch(filename.name, patt))
                if '.pkl' in filename.suffixes:
                    setattr(self, 'samplesheet', pd.read_pickle(filename))
                elif '.csv' in filename.suffixes:
                    setattr(self, 'samplesheet', pd.read_csv(filename))
                break

        if not hasattr(self,'control'):
            raise FileNotFoundError(f"Could not locate control_probes.pkl file in {filepath}")
        if not hasattr(self,'poobah') and self.pval is True:
            raise FileNotFoundError(f"Could not locate poobah_values.pkl file in {filepath}; re-run and set 'pval=False' to skip calculating probe failures.")
        if hasattr(self,'samplesheet'):
            if isinstance(self.samplesheet, pd.DataFrame):
                # methylprep v1.5.4-6 was creating meta_data files with two Sample_ID columns. Check and fix here:
                if any(self.samplesheet.columns.duplicated()):
                    self.samplesheet = self.samplesheet.loc[:, ~self.samplesheet.columns.duplicated()]
                    LOGGER.info("Removed a duplicate Sample_ID column in samplesheet")
                if 'Sample_ID' in self.samplesheet:
                    self.samplesheet = self.samplesheet.set_index('Sample_ID')
                elif 'Sentrix_ID' in self.samplesheet and 'Sentrix_Position' in self.samplesheet:
                    self.samplesheet['Sample_ID'] = self.samplesheet['Sentrix_ID'].astype(str) + '_' + self.samplesheet['Sentrix_Position']
                    self.samplesheet = self.samplesheet.set_index('Sample_ID')
            else:
                raise TypeError("Meta Data from Samplesheet is not a valid dataframe.")
        if (hasattr(self,'samplesheet') and
            (any(('Gender' in item.title()) for item in self.samplesheet.columns) or
            any(('Sex' in item.title()) for item in self.samplesheet.columns))):
            self.predict_sex = True
            # make sure case is correct
            if ('Gender' in self.samplesheet.columns or 'Sex' in self.samplesheet.columns):
                pass
            else:
                self.samplesheet.columns = [(col.title() if col.lower() in ('sex','gender') else col) for col in self.samplesheet.columns]
        else:
            self.predict_sex = False
        #if hasattr(self,'samplesheet') and self.predict_sex is False:
        #pass # I could add user info that explains why there won't be a sex prediction column.

        self.norm_regressions = {} # sample : all data from calc
        self.sex_extra = {} # sample: all data from get_sex()
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

    def process_sample(self, sample, con):
        """ process() will run this throug all samples, since structure of control data is a dict of DFs
        bg_offset = Background correction offset.
        Default value: 3000
            (applies to all background calculations, indicated with (bkg +x).)

NEGATIVE control probes are used as the baseline for p-val calculations.

see also infinium-hd-methylation-guide-15019519-01.pdf for list of expected intensity per type

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
        self.restoration_green = round( con[con['Extended_Type'].isin(['Restore','neg_ALDOB_3915-4004_1'])]['Mean_Value_Green'].values[0] / baseline_G, self.roundoff)

        if mouse:
            self.staining_green = np.nan
            self.staining_red = np.nan
        else:
            # (Biotin High/Biotin Bkg) > 5
            self.staining_green = round( con[con['Extended_Type'] == 'Biotin (High)']['Mean_Value_Green'].values[0] / con[con['Extended_Type'] == 'Biotin (Bkg)']['Mean_Value_Green'].values[0], self.roundoff)
            # (DNP High > DNP Bkg) > 5
            self.staining_red = round( con[con['Extended_Type'] == 'DNP (High)']['Mean_Value_Red'].values[0] / con[con['Extended_Type'] == 'DNP (Bkg)']['Mean_Value_Red'].values[0], self.roundoff)

        if mouse:
            self.extension_green = round( con[con['Extended_Type'] == 'G_Hairpin2.1_1']['Mean_Value_Green'].values[0] / con[con['Extended_Type'] == 'T_Hairpin2.1_1']['Mean_Value_Green'].values[0], self.roundoff)
            self.extension_red =   round( con[con['Extended_Type'] == 'T_Hairpin2.1_1']['Mean_Value_Red'].values[0] / con[con['Extended_Type'] == 'G_Hairpin2.1_1']['Mean_Value_Red'].values[0], self.roundoff)
        else:
            # GREEN min(C or G)/max(A or T) > 5
            self.extension_green = round( min( con[con['Extended_Type'] == 'Extension (C)']['Mean_Value_Green'].values[0], con[con['Extended_Type'] == 'Extension (G)']['Mean_Value_Green'].values[0]) / max( con[con['Extended_Type'] == 'Extension (A)']['Mean_Value_Green'].values[0], con[con['Extended_Type'] == 'Extension (T)']['Mean_Value_Green'].values[0]), self.roundoff)
            # RED max(C or G)/min(A or T) > 5
            self.extension_red = round( min( con[con['Extended_Type'] == 'Extension (A)']['Mean_Value_Red'].values[0], con[con['Extended_Type'] == 'Extension (T)']['Mean_Value_Red'].values[0]) / max( con[con['Extended_Type'] == 'Extension (C)']['Mean_Value_Red'].values[0], con[con['Extended_Type'] == 'Extension (G)']['Mean_Value_Red'].values[0]), self.roundoff)

        if mouse:
            # Hyb (High/Med)
            self.hybridization_green_A = round( con[con['Extended_Type']=='3_HIGH_MM_50.1_1']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='90_YEAST_3MM_50.1_1']['Mean_Value_Green'].values[0], self.roundoff)
            self.hybridization_green_B = np.nan
        else:
            # Hyb (High/Med) > 1
            self.hybridization_green_A = round( con[con['Extended_Type']=='Hyb (High)']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='Hyb (Medium)']['Mean_Value_Green'].values[0], self.roundoff)
            # Hyb (Med/Low) > 1
            self.hybridization_green_B = round( con[con['Extended_Type']=='Hyb (Medium)']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='Hyb (Low)']['Mean_Value_Green'].values[0], self.roundoff)
            # Hyb (High > Med > Low)
            #self.hybridization_green_C = round( con[con['Extended_Type']=='Hyb (High)']['Mean_Value_Green'].values[0] / con[con['Extended_Type']=='Hyb (Medium)']['Mean_Value_Green'].values[0], self.roundoff)

        if mouse:
            self.target_removal_green_1 = np.nan
            self.target_removal_green_2 = np.nan
        else:
            # Target ctrl 1 ≤ bkg
            self.target_removal_green_1 = round( baseline_G / con[con['Extended_Type'] == 'Target Removal 1']['Mean_Value_Green'].values[0], self.roundoff)
            # Target ctrl 2 ≤ bkg
            self.target_removal_green_2 = round( baseline_G / con[con['Extended_Type'] == 'Target Removal 2']['Mean_Value_Green'].values[0], self.roundoff)

        # con[con['Extended_Type']=='BS_Conversion_I_24_1']['Mean_Value_Green'].values[0],
        if mouse:
            # BS_Conversion_I_54_1, BS_Conversion_I_55_1, BS_Conversion_I_24_1,
            # BS_Conversion_II_5_1, BS_Conversion_II_21_1, BS_Conversion_I_17_1, BS_Conversion_I_72_1
            # higher: 54, 17
            # lower: 24, 55, 72
            self.bisulfite_conversion_I_green_CU = round(
                con[con['Extended_Type']=='BS_Conversion_I_54_1']['Mean_Value_Green'].values[0] /
                con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Green'].values[0], self.roundoff)
            self.bisulfite_conversion_I_green_bkg_U = round( baseline_G / con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Green'].values[0], self.roundoff)
            self.bisulfite_conversion_I_red_CU = round(
                con[con['Extended_Type']=='BS_Conversion_I_54_1']['Mean_Value_Red'].values[0] /
                con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Red'].values[0], self.roundoff)
            self.bisulfite_conversion_I_red_bkg_U = round( baseline_R / con[con['Extended_Type']=='BS_Conversion_I_55_1']['Mean_Value_Red'].values[0], self.roundoff)
            self.bisulfite_conversion_II_red_ratio = round( min([
                con[con['Extended_Type']=='BS_Conversion_II_5_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS_Conversion_II_21_1']['Mean_Value_Red'].values[0]
                ]) / max([
                con[con['Extended_Type']=='BS_Conversion_II_5_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS_Conversion_II_21_1']['Mean_Value_Green'].values[0]
                ]), self.roundoff)
            self.bisulfite_conversion_II_green_bkg = round( baseline_G / max([
                con[con['Extended_Type']=='BS_Conversion_II_5_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS_Conversion_II_21_1']['Mean_Value_Green'].values[0]
                ]), self.roundoff)
        else:
            # BS min(C1, 2, or 3) / BS max(U1, 2, 3) > 1
            # META NOTE: BS Conversion I-C1 is "I C1" in 450k. U1 also differs.
            # META NOTE: I had to ignore I-C3 and I-U3, as these always gave really low intensity outputs. The C1/U2 combo seems to work in practice.
            self.bisulfite_conversion_I_green_CU = round( min([
                con[con['Extended_Type'].isin(['BS Conversion I-C1','BS Conversion I C1'])]['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion I-C2']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='BS Conversion I-C3']['Mean_Value_Green'].values[0]
                ]) / max([
                con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Green'].values[0]
                ]), self.roundoff)
            # Bisulfite Conversion I Green U ≤ bkg | ((bkg + x)/U) > 1
            self.bisulfite_conversion_I_green_bkg_U = round( baseline_G / max([
                con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U4']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U5']['Mean_Value_Green'].values[0]
                ]), self.roundoff)

            # Bisulfite Conversion I Red (C4, 5, 6) / (U4, 5, 6) > 1
            self.bisulfite_conversion_I_red_CU = round( min([
                con[con['Extended_Type'].isin(['BS Conversion I-C4','BS Conversion I C4'])]['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-C5']['Mean_Value_Red'].values[0],
                #con[con['Extended_Type']=='BS Conversion I-C6']['Mean_Value_Red'].values[0]
                ]) / max([
                con[con['Extended_Type'].isin(['BS Conversion I-U4','BS Conversion I U4'])]['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U5']['Mean_Value_Red'].values[0],
                #con[con['Extended_Type']=='BS Conversion I-U6']['Mean_Value_Red'].values[0]
                ]), self.roundoff)
            # Bisulfite Conversion I Red U ≤ bkg | ((bkg + x)/U) > 1
            self.bisulfite_conversion_I_red_bkg_U = round( baseline_R / max([con[con['Extended_Type'].isin(['BS Conversion I-U1','BS Conversion I U1'])]['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U3']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U4']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion I-U5']['Mean_Value_Red'].values[0]
                ]), self.roundoff)

            #### min & max derived by comparing with expected output, because guide manufacturer's reference guide was unclear here.
            # Bisulfite Conversion II min(Red) > C max(Green)
            self.bisulfite_conversion_II_red_ratio = round( min([
                con[con['Extended_Type']=='BS Conversion II-1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion II-2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion II-3']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='BS Conversion II-4']['Mean_Value_Red'].values[0]
                ]) / max([
                con[con['Extended_Type']=='BS Conversion II-1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-3']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-4']['Mean_Value_Green'].values[0]
                ]), self.roundoff)
            # BiSulfite Conversion II C green ≤ bkg | (bkg + x)/ max(Green) > 1
            self.bisulfite_conversion_II_green_bkg = round( baseline_G / max([
                con[con['Extended_Type']=='BS Conversion II-1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-3']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='BS Conversion II-4']['Mean_Value_Green'].values[0]
                ]), self.roundoff)

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
                ), self.roundoff)
            self.specificity_I_red = round(
                con[con['Extended_Type']=='Non_Specific_I_42_1']['Mean_Value_Red'].values[0]
                / max([
                con[con['Extended_Type']=='Non_Specific_I_24_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Non_Specific_I_11_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Non_Specific_I_3_1']['Mean_Value_Red'].values[0],
                ]), self.roundoff)
            self.specificity_II = round( min([
                con[con['Extended_Type']=='Non_Specific_II_17_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Non_Specific_II_1_1']['Mean_Value_Red'].values[0],
                ]) / max([
                con[con['Extended_Type']=='Non_Specific_II_17_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Non_Specific_II_1_1']['Mean_Value_Green'].values[0],
                ]), self.roundoff)
            self.specificity_II_bkg= round( baseline_G / max([
                con[con['Extended_Type']=='Non_Specific_II_17_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Non_Specific_II_1_1']['Mean_Value_Green'].values[0],
                ]), self.roundoff)
        else:
            # ignoring controls 4,5,6 gave me the expected output
            # Specificity I Green (min(PM)/max(MM)) > 1
            self.specificity_I_green = round( min([
                con[con['Extended_Type']=='GT Mismatch 1 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 2 (PM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 3 (PM)']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 4 (PM)']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 5 (PM)']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 6 (PM)']['Mean_Value_Green'].values[0],
                ]) / max([
                con[con['Extended_Type']=='GT Mismatch 1 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 2 (MM)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='GT Mismatch 3 (MM)']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 4 (MM)']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 5 (MM)']['Mean_Value_Green'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 6 (MM)']['Mean_Value_Green'].values[0],
                ]), self.roundoff)

            # ignoring controls 1,2,3 here gave me the expected result
            # Specificity I Red (min(PM)/max(MM)) > 1
            self.specificity_I_red = round( min([
                #con[con['Extended_Type']=='GT Mismatch 1 (PM)']['Mean_Value_Red'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 2 (PM)']['Mean_Value_Red'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 3 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 4 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 5 (PM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 6 (PM)']['Mean_Value_Red'].values[0],
                ]) / max([
                #con[con['Extended_Type']=='GT Mismatch 1 (MM)']['Mean_Value_Red'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 2 (MM)']['Mean_Value_Red'].values[0],
                #con[con['Extended_Type']=='GT Mismatch 3 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 4 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 5 (MM)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='GT Mismatch 6 (MM)']['Mean_Value_Red'].values[0],
                ]), self.roundoff)

            # Specificity 1, Specificity 2, Specificity 3
            # Specificity II (S Red/ S Green) > 1
            self.specificity_II = round( min([
                con[con['Extended_Type']=='Specificity 1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Specificity 2']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='Specificity 3']['Mean_Value_Red'].values[0],
                ]) / max([
                con[con['Extended_Type']=='Specificity 1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 3']['Mean_Value_Green'].values[0],
                ]), self.roundoff)

            # Specificity II (background/ Spec Green) > 1
            self.specificity_II_bkg = round( baseline_G / max([
                con[con['Extended_Type']=='Specificity 1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 2']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='Specificity 3']['Mean_Value_Green'].values[0],
                ]), self.roundoff)

        if mouse:
            self.nonpolymorphic_green_lowCG_highAT = round( min([
                con[con['Extended_Type']=='nonPolyG_PPIH_9298-9387_1']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='nonPolyC_PPIE_21091-21180_1']['Mean_Value_Green'].values[0],
                ]) / max([
                con[con['Extended_Type']=='nonPolyT_ALDOB_10349-10438_1']['Mean_Value_Green'].values[0]
                ]), self.roundoff)
            self.nonpolymorphic_red_lowAT_highCG = round( min([
                con[con['Extended_Type']=='nonPolyT_ALDOB_10349-10438_1']['Mean_Value_Red'].values[0],
                ]) / max([
                con[con['Extended_Type']=='nonPolyC_PPIE_21091-21180_1']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='nonPolyG_PPIH_9298-9387_1']['Mean_Value_Red'].values[0],
                ]), self.roundoff)
        else:
            # Nonpolymorphic Green (min(CG)/ max(AT)) > 5
            self.nonpolymorphic_green_lowCG_highAT = round( min([
                con[con['Extended_Type']=='NP (C)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='NP (G)']['Mean_Value_Green'].values[0],
                ]) / max([
                con[con['Extended_Type']=='NP (A)']['Mean_Value_Green'].values[0],
                con[con['Extended_Type']=='NP (T)']['Mean_Value_Green'].values[0],
                ]), self.roundoff)

            # Nonpolymorphic Red (min(AT)/ max(CG)) > 5
            self.nonpolymorphic_red_lowAT_highCG = round( min([
                con[con['Extended_Type']=='NP (A)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='NP (T)']['Mean_Value_Red'].values[0],
                ]) / max([
                con[con['Extended_Type']=='NP (C)']['Mean_Value_Red'].values[0],
                con[con['Extended_Type']=='NP (G)']['Mean_Value_Red'].values[0],
                ]), self.roundoff)

        # ADDITIONAL tests
        self.negative_control_mean_green = round( np.mean(
        con[con['Control_Type'] == 'NEGATIVE']['Mean_Value_Green'].values,
        ))
        self.negative_control_mean_red = round( np.mean(
        con[con['Control_Type'] == 'NEGATIVE']['Mean_Value_Red'].values,
        ))

        # The Illumina MethylationEPIC BeadChip contains 85 pairs of internal normalization control
        # probes (name with prefix NORM_A, NORM_T, NORM_G or NORM_C), while its predecessor, Illumina
        # HumanMethyl-ation450 BeadChip contains 93 pairs. RELIC first performs a regression on the
        # logarithms of the intensity values of the normalization control probes to derive a quantitative
        # relationship between red and green channels, and then uses the relationship to correct for
        # dye-bias on intensity values for whole array.
        # https://rdrr.io/bioc/ENmix/man/relic.html
        if mouse:
            GA = ['Norm_G38_1', 'Norm_G72_1', 'Norm_G77_1', 'Norm_A38_1', 'Norm_A72_1', 'Norm_A77_1']
            LinregressResult_GA = linregress(
                con[(con['Control_Type'] == 'NORM_G') & (con['Extended_Type'].isin(GA))].sort_values(by='Extended_Type')['Mean_Value_Green'].values,
                con[(con['Control_Type'] == 'NORM_A') & (con['Extended_Type'].isin(GA))].sort_values(by='Extended_Type')['Mean_Value_Red'].values,
            )
            self.regression_NORM_GA = round(LinregressResult_GA.rvalue,2)
            CT = ['Norm_C12_1', 'Norm_C82_1', 'Norm_C84_1', 'Norm_C86_1', 'Norm_C93_1', 'Norm_C99_1',
                  'Norm_T12_1', 'Norm_T82_1', 'Norm_T84_1', 'Norm_T86_1', 'Norm_T93_1', 'Norm_T99_1']
            LinregressResult_CT = linregress(
                con[(con['Control_Type'] == 'NORM_C') & (con['Extended_Type'].isin(CT))].sort_values(by='Extended_Type')['Mean_Value_Green'].values,
                con[(con['Control_Type'] == 'NORM_T') & (con['Extended_Type'].isin(CT))].sort_values(by='Extended_Type')['Mean_Value_Red'].values,
            )
            self.regression_NORM_CT = round(LinregressResult_CT.rvalue,2)
            print(f"{sample} GA r={round(LinregressResult_GA.rvalue,2)} ±{round(LinregressResult_GA.stderr,2)} p<{round(LinregressResult_GA.pvalue,5)} |CT r={round(LinregressResult_CT.rvalue,2)} ±{round(LinregressResult_CT.stderr,2)} p<{round(LinregressResult_CT.pvalue,5)}")
            self.norm_regressions[sample] = {
            'GA': {'rvalue': round(LinregressResult_GA.rvalue,2),
                'pvalue': round(LinregressResult_GA.pvalue,2),
                'stderr': round(LinregressResult_GA.stderr,2),
                'slope': round(LinregressResult_GA.slope,2),
                'intercept': round(LinregressResult_GA.intercept,2),
                },
            'CT': {'rvalue': round(LinregressResult_CT.rvalue,2),
                'pvalue': round(LinregressResult_CT.pvalue,2),
                'stderr': round(LinregressResult_CT.stderr,2),
                'slope': round(LinregressResult_CT.slope,2),
                'intercept': round(LinregressResult_CT.intercept,2),
                },
            }

            """
            this gave terrible results, because probes are apples and oranges:
                drop 2 from T
                drop 1 from A

            CT = [12, 82, 84, 86, 93, 99]
            ['Norm_T11_1', 'Norm_T12_1', 'Norm_T15_1', 'Norm_T18_1',
            'Norm_T1_1', 'Norm_T23_1', 'Norm_T24_1', 'Norm_T26_1', 'Norm_T3_1',
            'Norm_T45_1', 'Norm_T48_1', 'Norm_T4_1', 'Norm_T60_1',
            'Norm_T62_1', 'Norm_T6_1', 'Norm_T73_1', 'Norm_T81_1',
            'Norm_T82_1', 'Norm_T83_1', 'Norm_T84_1', 'Norm_T86_1',
            'Norm_T93_1', 'Norm_T96_1', 'Norm_T99_1']
            ['Norm_C12_1', 'Norm_C13_1', 'Norm_C19_1', 'Norm_C34_1',
            'Norm_C36_1', 'Norm_C43_1', 'Norm_C44_1', 'Norm_C45_1',
            'Norm_C48_1', 'Norm_C49_1', 'Norm_C57_1', 'Norm_C61_1',
            'Norm_C65_1', 'Norm_C73_1', 'Norm_C74_1', 'Norm_C80_1',
            'Norm_C82_1', 'Norm_C84_1', 'Norm_C86_1', 'Norm_C90_1',
            'Norm_C93_1', 'Norm_C99_1']
            GA = [38, 72, 77]
            ['Norm_G28_1', 'Norm_G31_1', 'Norm_G35_1', 'Norm_G38_1',
            'Norm_G50_1', 'Norm_G61_1', 'Norm_G72_1', 'Norm_G77_1',
            'Norm_G91_1']
            ['Norm_A14_1', 'Norm_A15_1', 'Norm_A38_1', 'Norm_A49_1',
            'Norm_A65_1', 'Norm_A72_1', 'Norm_A77_1', 'Norm_A79_1',
            'Norm_A81_1', 'Norm_A95_1']
            """
        else:
            LinregressResult_GA = linregress(
                con[con['Control_Type'] == 'NORM_G'].sort_values(by='Extended_Type')['Mean_Value_Green'].values,
                con[con['Control_Type'] == 'NORM_A'].sort_values(by='Extended_Type')['Mean_Value_Red'].values,
            )
            self.regression_NORM_GA = round(LinregressResult_GA.rvalue,2)

            LinregressResult_CT = linregress(
                con[con['Control_Type'] == 'NORM_C'].sort_values(by='Extended_Type')['Mean_Value_Green'].values,
                con[con['Control_Type'] == 'NORM_T'].sort_values(by='Extended_Type')['Mean_Value_Red'].values,
            )
            self.regression_NORM_CT = round(LinregressResult_CT.rvalue,2)
            print(f"{sample} GA r={round(LinregressResult_GA.rvalue,2)} ±{round(LinregressResult_GA.stderr,2)} p<{round(LinregressResult_GA.pvalue,5)} |CT r={round(LinregressResult_CT.rvalue,2)} ±{round(LinregressResult_CT.stderr,2)} p<{round(LinregressResult_CT.pvalue,5)}")

            # BELOW: including mean values is less useful than providing the regression coefficient for cross-channel linearity.
            #self.norm_A_mean_green = round( np.mean(
            #con[con['Control_Type'] == 'NORM_A']['Mean_Value_Green'].values,
            #))
            #self.norm_T_mean_green = round( np.mean(
            #con[con['Control_Type'] == 'NORM_T']['Mean_Value_Green'].values,
            #))
            #self.norm_C_mean_red = round( np.mean(
            #con[con['Control_Type'] == 'NORM_C']['Mean_Value_Red'].values,
            #))
            #self.norm_G_mean_red = round( np.mean(
            #con[con['Control_Type'] == 'NORM_G']['Mean_Value_Red'].values,
            #))
        if hasattr(self,'noob_meth') and hasattr(self,'gct_scores') and self.gct_scores.get(sample) is not None:
            # sample var should match a column in noob_meth for this to work. And it only runs this function once per batch.
            self.gct_score = self.gct_scores[sample]
        else:
            self.gct_score = None

        # had to flip this to >80% passing, because all tests are ABOVE thresholds
        if hasattr(self,'poobah') and isinstance(self.poobah, pd.DataFrame) and sample in self.poobah.columns:
            self.failed_probes = round( 100*len(self.poobah[sample][ self.poobah[sample] > self.pval_sig ]) / len(self.poobah[sample]), 1)
        else:
            self.failed_probes = 0 # will be removed later

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
            {'col': 'Bisulfite Conversion II Red/Green', 'val':self.bisulfite_conversion_II_red_ratio, 'min':0,  'mid':0.7*self.cut, 'max':self.cut, 'formula': f"BS II min(Red)/max(Grn) > {self.cut}"},
            {'col': 'Bisulfite Conversion II bkg/Green', 'val':self.bisulfite_conversion_II_green_bkg, 'min':0,  'mid':0.7*self.cut, 'max':self.cut, 'formula': f"BS II bkg/Grn > {self.cut}"},
            {'col': 'Specificity I Green', 'val':self.specificity_I_green, 'min':0, 'mid':0.7, 'max':1.0, 'formula': f"GT Mismatch (min(PM)/max(MM)) > 1"}, # guide says DON'T change the threshold
            {'col': 'Specificity I Red', 'val':self.specificity_I_red, 'min':0, 'mid':0.7, 'max':1.0, 'formula': f"GT Mismatch (min(PM)/max(MM)) > 1"}, # guide says DON'T change the threshold
            {'col': 'Specificity II', 'val':self.specificity_II, 'min':0, 'mid':0.7, 'max':self.cut, 'formula': f"(S_Red/ S_Green) > {self.cut}"},
            {'col': 'Specificity II Bkg', 'val':self.specificity_II_bkg, 'min':0, 'mid':0.7, 'max':self.cut, 'formula': f"(background/ S_Green) > {self.cut}"},
            {'col': 'Non-polymorphic Green', 'val':self.nonpolymorphic_green_lowCG_highAT, 'min':0, 'mid':0.7*self.cut, 'max':5*self.cut, 'formula': f"(min(CG)/ max(AT)) > {5*self.cut}"},
            {'col': 'Non-polymorphic Red', 'val':self.nonpolymorphic_red_lowAT_highCG, 'min':0, 'mid':0.7*self.cut, 'max':5*self.cut, 'formula': f"(min(AT)/ max(CG)) > {5*self.cut}"},
            {'col': 'Baseline Green', 'val':baseline_G - self.bg_offset, 'formula': "max(Extension (A), Extension (T)) no offset", 'max':400, 'med':200, 'min':100},
            {'col': 'Baseline Red', 'val':baseline_R - self.bg_offset, 'formula': "max(Extension (C), Extension (G)) no offset", 'max':800, 'med':400, 'min':100},
            {'col': 'Negative Baseline G', 'val':self.negative_control_mean_green, 'formula': "mean NEGATIVE Green control probes"},
            {'col': 'Negative Baseline R', 'val':self.negative_control_mean_red, 'formula': "mean NEGATIVE Red control probes"},
            {'col': 'Regression NORM_GA', 'val':self.regression_NORM_GA, 'formula': "NORM_G (grn) vs NORM_A (red)", 'max':0.8, 'med':0.8, 'min':0.8},
            {'col': 'Regression NORM_CT', 'val':self.regression_NORM_CT, 'formula': "NORM_C (grn) vs NORM_T (red)", 'max':0.8, 'med':0.8, 'min':0.8},
            {'col': 'GCT score', 'val':self.gct_score, 'formula': "mean(oobG extC)/mean(oobG extT)", 'max':1.0, 'med':0.99, 'min':0.93},
            #{'col': 'NORM_A', 'val':self.norm_A_mean_green, 'formula': "mean NORM_A control probes Green)", 'max':600, 'med':300, 'min':100},
            #{'col': 'NORM_T', 'val':self.norm_T_mean_green, 'formula': "mean NORM_T control probes Green)", 'max':400, 'med':200, 'min':100},
            #{'col': 'NORM_C', 'val':self.norm_C_mean_red, 'formula': "mean NORM_C control probes Red)", 'max':1000, 'med':900, 'min':100},
            #{'col': 'NORM_G', 'val':self.norm_G_mean_red, 'formula': "mean NORM_G control probes Red)", 'max':1000, 'med':900, 'min':100},
            {'col': 'Passing Probes', 'val':(100 - self.failed_probes), 'formula': f"(p ≤ {self.pval_sig}) > 80% probes", 'max':80, 'med':80, 'min':80},
        ]

        row = {'Sample': sample}
        if self.pval is not True:
            try:
                self.data[sample].remove({'col': 'Passing Probes', 'val':(100 - self.failed_probes), 'formula': f"(p ≤ {self.pval_sig}) > 80% probes", 'max':80, 'med':80, 'min':80})
            except ValueError:
                pass # poobah file could be missing, in which case it never gets calculated.
        if self.gct_score is None or np.isnan(self.gct_score) is True:
            try:
                self.data[sample].remove({'col': 'GCT score', 'val':self.gct_score, 'formula': "mean(oobG extC)/mean(oobG extT)", 'max':1.0, 'med':0.99, 'min':0.93})
            except ValueError as e:
                print(f'ERROR {e}')
                pass # noob_meth file could be missing, in which case it never gets calculated.
                # and mouse is not supported...

        row.update({k['col']:k['val'] for k in self.data[sample]})
        # DEBUG: everything is rounding OKAY at this point
        #if any([ (len(str(v).split(".")[1]) if '.' in str(v) else 0) > self.roundoff for k,v in row.items()]):
        #    print('ERROR', [len(str(v).split(".")[1]) if '.' in str(v) else 0 for k,v in row.items()] )
        self.report.append(row) # this is converted to excel sheet, but self.data is used to format the data in save()
        self.formulas.update( {k['col']:k['formula'] for k in list(self.data.values())[0]} )
        # process() adds predicted sex and result column

    def process(self):
        if hasattr(self, 'noob_meth'):
            self.gct_scores = methylcheck.bis_conversion_control(self.noob_meth) # a dict of {sentrix_id:score} pairs
            if [v for v in self.gct_scores.values() if v is not None and np.isnan(v) is False] == []:
                self.gct_scores = {}
        else:
            self.gct_scores = {}
        for sample,con in self.control.items():
            self.process_sample(sample, con) # saves everything on top of last sample, for now. testing.

        # predicted_sex, x_median, y_median, x_fail_percent, y_fail_percent,
        if hasattr(self, 'samplesheet') and isinstance(self.samplesheet, pd.DataFrame) and hasattr(self, 'noob_meth') and hasattr(self, 'noob_unmeth'):
            if self.predict_sex:
                LOGGER.info("Predicting Sex and comparing with sample meta data...")
            else:
                LOGGER.info("Predicting Sex...")
            try:
                #print(self.noob_meth.shape, self.noob_unmeth.shape)
                sex_df = methylcheck.get_sex((self.noob_meth, self.noob_unmeth), array_type=None, verbose=False, plot=False, on_lambda=False, median_cutoff= -2, include_probe_failure_percent=False)
            except ValueError as e:
                if str(e).startswith('Unsupported Illumina array type'):
                    # happens with some mouse array versions, but not fatal to the report
                    LOGGER.warning(f"Skipping get prediction: {e}")
                    sex_df = pd.DataFrame()
                else:
                    LOGGER.warning(e)
                    sex_df = pd.DataFrame()
            # merge into processed output (self.report and self.data); merge CORRECTED actual sex values, as M or F
            sex_df = methylcheck.predict.sex._fetch_actual_sex_from_sample_sheet_meta_data(self.filepath, sex_df)

            # row index values are sample names
            for row in sex_df.itertuples():
                try:
                    item_order = [idx for idx,item in enumerate(self.report) if item['Sample'] == row.Index][0]
                except IndexError:
                    # in tests, this happens if samplesheet has data that isn't in process results
                    raise IndexError("651: Report and sex_df not matching. This can happen if the samplesheet contains samples that aren't in processed results.")

                self.data[row.Index].append({'col': 'Predicted Sex', 'val': row.predicted_sex, 'formula': f"X/Y probes"})
                self.report[item_order]['Predicted Sex'] = row.predicted_sex
                if self.predict_sex:
                    self.data[row.Index].append({
                        'col': 'Sex Match',
                        'val': row.sex_matches,
                        'formula': f"Sample sheet M/F",
                        'min':0, 'med':0.5, 'max':1.0})
                    self.report[item_order]['Sex Match'] = row.sex_matches
                    self.formulas['Sex Match'] = "Sample sheet M/F"
                self.formulas['Predicted Sex'] = "X/Y probes"
                # next attrib is not used in XLSX report, but saved to object in case others want to extend this.
                self.sex_extra[row.Index] = {
                    'Predicted Sex': row.predicted_sex,
                    'x_median': row.x_median,
                    'y_median': row.y_median,
                }

        for sample in self.control.keys():
            self.read_sample_result(sample)

    def read_sample_result(self, sample):
        """ save each column's result is as 1 for pass, 0 for fail, then calculate fraction.
        - includes some nuance on how off-the-mark each test is in the Result column (OK, MARGINAL, FAIL)
        """
        passfail = {}
        missed_by_much = {}
        for col in self.data[sample]:
            if col['col'] in self.untestable_columns:
                continue
            if pd.isna(col['val']):
                continue
            if col.get('max') is None:
                print(f"WARNING: no threshold for {col['col']}")
                continue
            passfail[col['col']] = 1 if col['val'] >= col['max'] else 0
            missed_by_amount = round(col['val'] / (col['max'] + 0.00000000001),2)
            if missed_by_amount <= 0: # some tests can go negative, so setting a floor
                # print(f"missed_by_amount: {missed_by_amount} col val {col['val']} col max {col['max']}")
                missed_by_amount = 0.01
            missed_by_much[col['col']] = missed_by_amount


        result = round( sum(passfail.values()) / sum(~np.isnan(list(passfail.values()))), 2) # fraction passing
        # average ratio of value to cutoff per failing test
        failed_tests = [col for col,val in missed_by_much.items() if passfail[col] == 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            how_close = round(np.nanmean(np.clip([val for col,val in missed_by_much.items() if passfail[col] == 0], 0, 100000)),2)

        # percent of columns that passed
        if result >= 1.0:
            result = 'OK'
        elif result >= self.passing and (how_close == 0 or np.isnan(how_close)):
            result = 'MARGINAL (infinite values)'
        elif result >= self.passing and 0.7 <= how_close <= 1.3:
            result = f'OK ({how_close})'
        elif result >= self.passing and 0.5 <= how_close <= 1.5:
            result = f'MARGINAL ({how_close})'
        else: # less than 70 percent of tests passed.
            result = f'FAIL ({how_close})'


        passing_col = [col for col in self.data[sample] if col['col'] == 'Passing Probes']
        if self.pval is True and passing_col != [] and passing_col[0]['val'] < passing_col[0]['max']:
            result = f'FAIL (pval)'
        result_col = {'col': 'Result', 'val':result, 'formula': f"At least {100*self.passing}% tests pass"}
        self.data[sample].append(result_col)
        report_row = min([idx for idx,row in enumerate(self.report) if row['Sample'] == sample])
        self.report[report_row]['Result'] = result_col['val']
        if failed_tests != []:
            self.report[report_row]['Why Failed'] = ', '.join(failed_tests)

    def save(self):
        if self.legacy:
            # change 'col' part of each row in self.data and keys in self.report to new column names
            for sample, row in self.data.items():
                for idx,col in enumerate(row.copy()): # iterate through deep-copy while modifying the original list
                    if self.legacy_columns.get(col['col']) != None:
                        self.data[sample][idx]['col'] = self.legacy_columns[col['col']]
                    else:
                        if col['col'] == 'Sample':
                            # split Sample into 3 columns
                            try:
                                Sentrix_Barcode, Sentrix_Position = col['val'].split('_')
                            except:
                                Sentrix_Barcode = np.nan
                                Sentrix_Position = np.nan
                            self.data[sample].append({'Sentrix Barcode': Sentrix_Barcode})
                            self.data[sample].append({'Sentrix Position': Sentrix_Position})
                            self.data[sample].remove(col)
                            col['col'] = 'Sample Name'
                            self.data[sample].insert(0, col)
                        else:
                            self.data[sample].remove(col)

            # self.report is a list of rows (dicts) -- converted to DF/excel later
            for idx,row_dict in enumerate(self.report):
                for col,val in row_dict.copy().items():
                    if self.legacy_columns.get(col):
                        self.report[idx][self.legacy_columns[col]] = self.report[idx].pop(col)
                    else:
                        if col == 'Sample':
                            try:
                                Sentrix_Barcode, Sentrix_Position = val.split('_')
                            except:
                                Sentrix_Barcode = np.nan
                                Sentrix_Position = np.nan
                            self.report[idx]['Sample Name'] = self.report[idx].pop('Sample')
                            self.report[idx]['Sentrix Barcode'] = Sentrix_Barcode
                            self.report[idx]['Sentrix Position'] = Sentrix_Position
                        else:
                            self.report[idx].pop(col)
            # finally, reorder self.report keys, because this dictates XLSX column order


        # fetch the folder parent name and append to file (usually this is the project name)
        if self.project_name == None:
            self.project_name = Path(self.filepath).resolve().name
        writer = pd.ExcelWriter(Path(self.outfilepath, f"{self.project_name}_QC_Report.xlsx"), engine='xlsxwriter')
        report_df= pd.DataFrame(self.report) #<---- here in pandas 1.3x the decimal places get messed up.
        report_df = report_df.round(self.roundoff)
        report_df.to_excel(writer, sheet_name='QC_REPORT', startcol=0, startrow=(2 if not self.legacy else 1), header=False, index=False)
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
        if self.legacy:
            col_widths = {'Sample Name': max([len(item['Sample Name']) for item in self.report])}
        else:
            col_widths = {'Sample': max([len(item['Sample']) for item in self.report])}
        for col,header in enumerate(report_df.columns):
            this_box = f"{c(col)}1"
            worksheet.write(this_box, header, cell_format)
            col_widths[header] = len(header)
            if header not in self.formulas or self.legacy:
                continue
            this_box = f"{c(col)}2"
            worksheet.write(this_box, self.formulas[header], cell_format)
            if len(self.formulas[header]) > col_widths[header]:
                col_widths[header] = len(self.formulas[header])

        for row, (sample, meta_row) in enumerate(self.data.items()):
            for col, k in enumerate(meta_row):
                # skipping first 2 rows and first column | or 3 cols, 1 row if legacy
                this_box = f"{c(col+1)}{row+3}" if not self.legacy else f"{c(col+3)}{row+2}"
                coloring = {'type': '3_color_scale',
                    'min_value':k.get('min',0), 'max_value': k.get('max',1), 'mid_value': k.get('mid', round(k.get('max',1)/2)),
                    'min_color':self.cpass, 'max_color':self.cfail, 'mid_color':self.cmid,
                    "min_type": "num", "mid_type": "num", "max_type": "num"}
                if k['col'] == 'Result':
                    result_format = workbook.add_format({'bold':True, 'font_color':'red'})
                    coloring = {'type': 'text', 'criteria': 'begins with', 'value': 'F', 'format': result_format}
                worksheet.conditional_format(this_box, coloring)
                # next: track if any values are longer than the column names, to adjust widths at end
                if len(str(k['val'])) > col_widths[k['col']]:
                    col_widths[k['col']] = int( len(str(k['val'])) * 1.5 )

        #worksheet.conditional_format('B2:B8', {'type': '3_color_scale', 'min_value':0, 'max_value':1, 'mid_value':0.5})
        # There is no way to specify “AutoFit” for a column in the Excel file format.
        # This feature is only available at runtime from within Excel.
        # It is possible to simulate “AutoFit” in your application by tracking the maximum width
        # of the data in the column as your write it and then adjusting the column width at the end.
        # setting column widths
        col = 0
        skipcols = ['Sample', 'Sample Name', 'Sentrix Barcode', 'Sentrix Position']
        for header,col_width in col_widths.items():
            if header in skipcols:
                col += 1
                continue # handle these a different way below
            this_box = f"{c(col)}:{c(col)}"
            col_width = min(int(0.6*col_width),60) # makes it word wrap better
            worksheet.set_column(this_box, col_width)
            col += 1
        worksheet.set_column("A:A", 24) # samples
        writer.save()


def controls_report(*args, **kwargs):
    """Generates a color-coded excel/csv QC Report.

CLI: methylcheck controls

    This will load the methylprep control_probes.pkl file and perform all the calculations and produce a tidy color-coded XLSX file with results.

    Note: this function is analogous to methylcheck.plot_controls() except that the output if a color-coded excel sheet instead of image charts.

Required Args:
    filepath (str or Path) to input files

Optional Arguments:

    outfilepath (None)
        If not specified, QC_Report.xlsx is saved in the same folder with the control probes file.
    colorblind (False)
        Set to True to enabled pass/fail colors that work better for colorblind users.
    bg_offset (3000)
        Illumina's software add 3000 to the baseline/background values as an offset.
        This function does the same, unless you change this value. Changing this will
        affect whether some tests pass the cutff_adjust thresholds or not.
    cutoff_adjust (1.0)
        A value of 1.0 will replicate Illumina's definition of pass/fail cutoff values for tests.
        You can increase this number to apply more stringent benchmarks for what constitutes good
        data, typically in the range of 1.0 to 3.0 (as a 1X to 3X cutoff multiplier).
    legacy (False)
        True: changes XLSX column names; columns are TitleCase instead of readable text; drops the formulas row;
        splits sample column into two SentrixID columns. Rounds all numbers to 1 decimal
        instead of 2.
        This will also exclude p-value poobah column and the result (interpretation) column from report.
    roundoff (2, int)
        Option to adjust the level of rounding in report output numbers.
    pval (True)
        True: Loads poobah_values.pkl and adds a column with percent of probes that failed per sample.
    pval_sig (0.05)
        pval significance level to define passing/failing probes
    passing (0.7, float)
        Option to specify the fraction (from 0 to 1.0) of tests that must pass for sample to pass.
        Note that this calculation also weights failures by how badly the miss the threshold value,
        so if any tests is a massive failure, the sample fails. Also, if pval(poobah) is included, and
        over 20% of probes fail, the sample fails regardless of other tests.
    project_name (str, default None)
        If defined, will be used to name the QC report file.
    """
    if len(args) == 1:
        kwargs['filepath'] = args[0]
        LOGGER.warning(f"Assigned argument 0 `{kwargs['filepath']}` to `filepath` kwarg. This function takes 0 arguments.")
    elif len(args) > 1:
        raise AttributeError("This function takes 0 arguments, but {len(args)} arguments found. Specify the filepath using filepath=<path>.")
    if 'filepath' not in kwargs:
        raise FileNotFoundError("No control probes file location specified.")
    reporter = ControlsReporter(**kwargs)
    reporter.process()
    reporter.save()
