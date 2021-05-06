import methylcheck
from pathlib import Path
import random
import copy
letters = 'abcdefghjiklmnopqrstuvwxyz'
#PATH = Path('/Volumes/LEGX/GEO/test_pipeline/GSE49618/')
PATH = Path('docs/example_data/GSE69852/')

def test_truncate_custom_tables():

    custom_tables = [
        {
        'title': "8 columns (truncated)",
        'col_names': ["first","second","third", "fourth_really_long_column", "first","second","third", "eight_really_long_column"],
        #'row_names': ["<list of strings, optional>"],
        'data': [[1,2,3,4,5,6,7,8], [4,5,6,7,8,9,10,11],
            [66666,7,8,9,12123,5352,31298014593059340593,453],
            ['a','b','c','d','efjeiefackjaldwaiom aamk','r','g','h'],
            [66666,7,8,9,12123,5352,3129801,453],
            ['a','b','c','d','e','r','g','h'],
            ],
        'order_after': "beta_density_plot",
        'font_size': 'truncate'
        },
        #{"<...second table here...>"}
    ]

    for i in range(random.randrange(20, 40)):
        row = [''.join(random.choice(letters) for j in range(random.randrange(30))) for k in range(8)]
        custom_tables[0]['data'].append(row)

    # drop 1 column
    N = 7
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "7 columns  (truncated)"
    new_table['font_size'] = 'truncate'
    custom_tables.append(new_table)


    # drop 2 columns
    N = 6
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "6 columns  (truncated)"
    new_table['font_size'] = 'truncate'
    custom_tables.append(new_table)

    # drop 3 columns
    N = 5
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "5 columns (truncate)"
    new_table['font_size'] = 'truncate'
    custom_tables.append(new_table)

    # add 1 column
    N = 9
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row + ['blah']
    new_table['col_names'] = new_table['col_names'] + ['blah_col']
    new_table['title'] = "9 columns (truncate)"
    new_table['font_size'] = 'truncate'
    custom_tables.append(new_table)

    report = methylcheck.ReportPDF(
        path=PATH,
        poobah_max_percent=10,
        pval_cutoff=0.01,
        title='QC Report',
        author='FOXO Technologies, inc.',
        subject="QC Report",
        keywords="methylation array",
        outpath=PATH,
        filename='GSE49618_QC_REPORT.PDF',
        poobah=True,
        on_lambda=False,
        custom_tables=custom_tables,
        debug=False,
        order=['beta_density_plot', 'detection_poobah', 'predict_sex', 'mds', 'auto_qc',
        'qc_signal_intensity', 'M_vs_U_compare', 'M_vs_U', 'controls', 'probe_types'],
        runme=True
    )


def test_auto_formatted_custom_tables():

    custom_tables = [
        {
        'title': "8 columns (auto)",
        'col_names': ["first","second","third", "fourth_really_long_column", "first","second","third", "eight_really_long_column"],
        #'row_names': ["<list of strings, optional>"],
        'data': [[1,2,3,4,5,6,7,8], [4,5,6,7,8,9,10,11],
            [66666,7,8,9,12123,5352,31298014593059340593,453],
            ['a','b','c','d','efjeiefackjaldwaiom aamk','r','g','h'],
            [66666,7,8,9,12123,5352,3129801,453],
            ['a','b','c','d','e','r','g','h'],
            ],
        'order_after': "beta_density_plot",
        'font_size': 'auto'
        },
        #{"<...second table here...>"}
    ]

    for i in range(random.randrange(20, 40)):
        row = [''.join(random.choice(letters) for j in range(random.randrange(30))) for k in range(8)]
        custom_tables[0]['data'].append(row)

    # drop 1 column
    N = 7
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "7 columns  (auto)"
    new_table['font_size'] = 'auto'
    custom_tables.append(new_table)


    # drop 2 columns
    N = 6
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "6 columns  (auto)"
    new_table['font_size'] = 'auto'
    custom_tables.append(new_table)

    # drop 3 columns
    N = 5
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "5 columns (auto)"
    new_table['font_size'] = 'auto'
    custom_tables.append(new_table)

    # add 1 column
    N = 9
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row + ['blah']
    new_table['col_names'] = new_table['col_names'] + ['blah_col']
    new_table['title'] = "9 columns (auto)"
    new_table['font_size'] = 'auto'
    custom_tables.append(new_table)

    report = methylcheck.ReportPDF(
        path=PATH,
        poobah_max_percent=10,
        pval_cutoff=0.01,
        title='QC Report',
        author='FOXO Technologies, inc.',
        subject="QC Report",
        keywords="methylation array",
        outpath=PATH,
        filename='GSE49618_QC_REPORT.PDF',
        poobah=True,
        on_lambda=False,
        custom_tables=custom_tables,
        debug=False,
        order=['beta_density_plot', 'detection_poobah', 'predict_sex', 'mds', 'auto_qc',
        'qc_signal_intensity', 'M_vs_U_compare', 'M_vs_U', 'controls', 'probe_types'],
        runme=True
    )


def test_none_formatted_custom_tables():

    custom_tables = [
        {
        'title': "8 columns (none)",
        'col_names': ["first","second","third", "fourth_really_long_column", "first","second","third", "eight_really_long_column"],
        #'row_names': ["<list of strings, optional>"],
        'data': [[1,2,3,4,5,6,7,8], [4,5,6,7,8,9,10,11],
            [66666,7,8,9,12123,5352,31298014593059340593,453],
            ['a','b','c','d','efjeiefackjaldwaiom aamk','r','g','h'],
            [66666,7,8,9,12123,5352,3129801,453],
            ['a','b','c','d','e','r','g','h'],
            ],
        'order_after': "beta_density_plot",
        'font_size': None
        },
        #{"<...second table here...>"}
    ]

    for i in range(random.randrange(20, 40)):
        row = [''.join(random.choice(letters) for j in range(random.randrange(30))) for k in range(8)]
        custom_tables[0]['data'].append(row)

    # drop 1 column
    N = 7
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "7 columns  (none))"
    new_table['font_size'] = None
    custom_tables.append(new_table)


    # drop 2 columns
    N = 6
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "6 columns  (none)"
    new_table['font_size'] = None
    custom_tables.append(new_table)

    # drop 3 columns
    N = 5
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row[:N]
    new_table['col_names'] = new_table['col_names'][:N]
    new_table['title'] = "5 columns (none)"
    new_table['font_size'] = None
    custom_tables.append(new_table)

    # add 1 column
    N = 9
    new_table = copy.deepcopy(custom_tables[0])
    for idx,row in enumerate(new_table['data']):
        new_table['data'][idx] = row + ['blah']
    new_table['col_names'] = new_table['col_names'] + ['blah_col']
    new_table['title'] = "9 columns (none)"
    new_table['font_size'] = None
    custom_tables.append(new_table)

    report = methylcheck.ReportPDF(
        path=PATH,
        poobah_max_percent=10,
        pval_cutoff=0.01,
        title='QC Report',
        author='FOXO Technologies, inc.',
        subject="QC Report",
        keywords="methylation array",
        outpath=PATH,
        filename='GSE49618_QC_REPORT.PDF',
        poobah=True,
        on_lambda=False,
        custom_tables=custom_tables,
        debug=False,
        order=['beta_density_plot', 'detection_poobah', 'predict_sex', 'mds', 'auto_qc',
        'qc_signal_intensity', 'M_vs_U_compare', 'M_vs_U', 'controls', 'probe_types'],
        runme=True
    )
