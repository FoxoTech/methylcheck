from pathlib import Path
import logging
#app
import methylcheck

class TestQcPlot():

    def test_qc_plot(self):
        filepath = Path('~/methylcheck/docs/example_data/GSE69852')
        methylcheck.run_qc(filepath)
