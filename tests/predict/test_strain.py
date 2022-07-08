from pathlib import Path
import pandas as pd
import numpy as np
import methylcheck


def test_infer_strain():
    """ uses the snp_beta columns of control probes to predict mouse strain """
    filepath = Path('docs/example_data/mouse_test/control_probes.pkl')
    raw = pd.read_pickle(filepath)
    raw = {k: v['snp_beta'] for k,v in raw.items()}
    df = pd.DataFrame(data=raw)
    for sample in df.columns:
        result = methylcheck.predict.infer_strain(df[[sample]])
        result_strain = list(result.values())[0]['best']
        result_pval = list(result.values())[0]['pval']
        if result_strain == 'AKR_J' and round(result_pval,3) > 0.91:
            pass
        else:
            raise ValueError("Mouse strain did not match expected: {mouse_strain} vs AKR_J (conf: {result_pval})")
