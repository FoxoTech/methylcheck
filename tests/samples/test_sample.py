import methylcheck
import pandas as pd
import methylprep # for manifest support
from pathlib import Path
PATH = Path('docs/example_data/mouse')

class TestProcessedSample():
    manifest = methylprep.Manifest(methylprep.ArrayType('mouse'))
    manifest_mouse_probe_types = dict(manifest.mouse_data_frame['Probe_Type'].value_counts())
    manifest_control_probe_types = dict(manifest.control_data_frame['Control_Type'].value_counts())

    def test_mouse_probes(self):
        pd_mu = pd.read_pickle(Path(PATH,'mouse_probes.pkl'))
        mu = methylcheck.load(Path(PATH,'mouse_probes.pkl'), verbose=False, silent=True)
        if not (isinstance(mu, dict) and isinstance(list(mu.values())[0], pd.DataFrame)):
            raise AssertionError()
        if len(mu) != len(pd_mu):
            raise AssertionError(f"Got a {len(mu)} items in mouse_probes.pkl; expected {len(pd_mu)}.")
        print(f"sample count OK")
        df0 = list(mu.values())[0]
        probes_by_type = dict(df0.index.str[:2].value_counts())
        if probes_by_type['cg'] != 5881 and probes_by_type['uk'] != 4186:
            raise AssertionError(f"Mismatch in number of cg and uk(nown) probes in mouse_probes.pkl; expected 5881 cg and 4186 uk.")
        print('mouse cg,uk count OK')

        actual_mouse_probes = dict(df0['Probe_Type'].value_counts())
        probe_counts = {'mu': 6332, 'rp': 4514, 'ch': 2851, 'rs': 291} # based on C20 manifest
        probe_count_errors = {}
        for probe_type, probe_count in probe_counts.items():
            if self.manifest_mouse_probe_types[probe_type] != probe_count:
                probe_count_errors[probe_type] = {'actual': self.manifest_mouse_probe_types[probe_type], 'expected': probe_count}
        if probe_count_errors:
            raise AssertionError(f"mouse probe count errors: {probe_count_errors}")
        print('mouse mu,rp,ch,rs count OK')
        # compare with manifest
        diffs = []
        for probe_type, probe_count in self.manifest_mouse_probe_types.items():
            if probe_count != actual_mouse_probes[probe_type]:
                diffs.append(f"{probe_type}: {actual_mouse_probes[probe_type]} / {probe_count}")
        if diffs:
            print("Probes in manifest NOT in control probes saved:")
            print('\n'.join(diffs))


    def test_control_probes(self):
        con = methylcheck.load(Path(PATH,'control_probes.pkl'), verbose=False, silent=True)
        con0 = list(con.values())[0]
        con_types = dict(con0['Control_Type'].value_counts())
        probe_counts = {'NEGATIVE': 179, 'NORM_T': 24, 'NORM_C': 22, 'NORM_A': 10, 'NORM_G': 9, 'NON-POLYMORPHIC': 3, 'SPECIFICITY I': 3, 'BISULFITE CONVERSION I': 3, 'BISULFITE CONVERSION II': 2, 'SPECIFICITY II': 2, 'HYBRIDIZATION': 2, 'RESTORATION': 1}
        probe_count_errors = {}
        for probe_type, probe_count in probe_counts.items():
            if con_types[probe_type] != probe_count:
                probe_count_errors = {'actual': con_types[probe_type], 'expected': probe_count}
        if probe_count_errors:
            raise AssertionError(f"control probe count differed: {probe_count_errors}")
        print('mouse control_probes count OK')

        diffs = []
        for probe_type, probe_count in self.manifest_control_probe_types.items():
            if not con_types.get(probe_type):
                print(f"ERROR: control output data is missing {probe_type} found in manifest.")
                continue
            if probe_count != con_types[probe_type]:
                diffs.append(f"{probe_type}: {con_types[probe_type]} / {probe_count}")
        if diffs:
            print("Probes in manifest NOT in control probes saved:")
            print('\n'.join(diffs))

    def test_plot_mouse_betas_from_pickle(self):
        mu = methylcheck.load(Path(PATH,'mouse_probes.pkl'), verbose=False, silent=True)
        df0 = list(mu.values())[0]
        df = df0[['beta_value']]
        methylcheck.sample_plot(df, silent=True, save=True)
        methylcheck.beta_density_plot(df, silent=True)
        methylcheck.sample_plot(df, silent=True)
        methylcheck.mean_beta_plot(df, silent=True)
        methylcheck.cumulative_sum_beta_distribution(df, silent=True)
        methylcheck.beta_mds_plot(df, silent=True)
        methylcheck.mean_beta_compare(df,df,silent=True)
        df = df0[['cm_value']]
        methylcheck.beta_density_plot(df, silent=True)
        df = df0[['m_value']]
        methylcheck.beta_density_plot(df, silent=True)
        Path('./beta.png').unlink()
