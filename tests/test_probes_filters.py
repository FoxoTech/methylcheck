# -*- coding: utf-8 -*-
from pathlib import Path
TESTPATH = 'tests'
#app
import methylcheck


def test_list_problem_probes():
    """ confirm the number of probes matches expected counts for diff combinations """
    arrays = ['450k','EPIC']
    pubs450k = [        'Price2013',        'Chen2013',        'Naeem2014',        'DacaRoszak2015']
    pubsEPIC = ['McCartney2016',        'Zhou2016']
    reasons = ['Polymorphism', 'CrossHybridization', 'BaseColorChange', 'RepeatSequenceElements']
    results = [
    ('450k', 'Price2013', 213246),
    ('450k', 'Chen2013', 265410),
    ('450k', 'Naeem2014', 128695),
    ('450k', 'DacaRoszak2015', 89678),
    ('EPIC', 'McCartney2016', 'Polymorphism', 349423),
    ('EPIC', 'McCartney2016', 'CrossHybridization', 370579),
    ('EPIC', 'McCartney2016', 'BaseColorChange', 326274),
    ('EPIC', 'McCartney2016', 'RepeatSequenceElements', 326267),
    ('EPIC', 'Zhou2016', 'Polymorphism', 386308),
    ('EPIC', 'Zhou2016', 'CrossHybridization', 182272),
    ('EPIC', 'Zhou2016', 'BaseColorChange', 178671),
    ('EPIC', 'Zhou2016', 'RepeatSequenceElements', 178671),
    ('EPIC', ['McCartney2016', 'Zhou2016'], ['Polymorphism', 'CrossHybridization', 'BaseColorChange', 'RepeatSequenceElements'], 678407),
    ('EPIC', 'McCartney2016', 'Polymorphism', 548418),
    ('EPIC', 'McCartney2016', 'CrossHybridization', 514120),
    ('EPIC', 'McCartney2016', 'BaseColorChange', 384943),
    ('EPIC', 'McCartney2016', 'RepeatSequenceElements', 384537),
    ('EPIC', 'Zhou2016', 'Polymorphism', 634197),
    ('EPIC', 'Zhou2016', 'CrossHybridization', 338080),
    ('EPIC', 'Zhou2016', 'BaseColorChange', 293870),
    ('EPIC', 'Zhou2016', 'RepeatSequenceElements', 293870),
    ]
    for array in arrays:
        if array == '450k':
            for pub in pubs450k:
                total = len(methylcheck.probes.filters.list_problem_probes(array, [pub]))
                result = (array, pub, total)
                print(result)
                if result not in results:
                    raise AssertionError(f"result {result}")
        if array == 'EPIC':
            for pub in pubsEPIC:
                for reason in reasons:
                    total = len(methylcheck.probes.filters.list_problem_probes(array, [pub, reason]))
                    result = (array, pub, reason, total)
                    print(result)
                    if result not in results:
                        raise AssertionError(f"result {result}")
            total = len(methylcheck.probes.filters.problem_probe_reasons(array, pubsEPIC+reasons))
            result = (array, pubsEPIC, reasons, total)
            print(result)
            if result not in results:
                raise AssertionError(f"result {result}")


def test_problem_probe_reasons():
    """ confirm the number of probes matches expected counts for diff combinations """
    arrays = ['450k','EPIC']
    pubs450k = [        'Price2013',        'Chen2013',        'Naeem2014',        'DacaRoszak2015']
    pubsEPIC = ['McCartney2016',        'Zhou2016']
    reasons = ['Polymorphism', 'CrossHybridization', 'BaseColorChange', 'RepeatSequenceElements']
    for array in arrays:
        if array == '450k':
            for pub in pubs450k:
                df = methylcheck.probes.filters.problem_probe_reasons(array, [pub])
                print(array, pub, df.shape)
        if array == 'EPIC':
            for pub in pubsEPIC:
                for reason in reasons:
                    total = len(methylcheck.probes.filters.problem_probe_reasons(array, [pub, reason]))
                    print(array, pub, reason, total)
            total = len(methylcheck.probes.filters.problem_probe_reasons(array, pubsEPIC+reasons))
            print(array, pubsEPIC, reasons, total)

def other():
    zhou_sketchy_probes = set(methylcheck.list_problem_probes('EPIC', criteria=['Zhou2016']))
    print('zhou_sketchy_probes', len(zhou_sketchy_probes))

    mccartney_sketchy_probes = set(methylcheck.list_problem_probes('EPIC', criteria=['McCartney2016']))
    print('mccartney_sketchy_probes', len(mccartney_sketchy_probes))

    zm_intersection_sketchy_probes = zhou_sketchy_probes.intersection(mccartney_sketchy_probes)
    print('zm_intersection_sketchy_probes', len(zm_intersection_sketchy_probes))

    polymorphism_sketchy_probes = set(methylcheck.list_problem_probes('EPIC', criteria=['Polymorphism']))
    print('polymorphism_sketchy_probes', len(polymorphism_sketchy_probes))

    xhybridizing_sketchy_probes = set(methylcheck.list_problem_probes('EPIC', criteria=['CrossHybridization']))
    print('xhybridizing_sketchy_probes', len(xhybridizing_sketchy_probes))

    basecolorchange_sketchy_probes = set(methylcheck.list_problem_probes('EPIC', criteria=['BaseColorChange']))
    print('basecolorchange_sketchy_probes', len(basecolorchange_sketchy_probes))

    repeat_sketchy_probes = set(methylcheck.list_problem_probes('EPIC', criteria=['RepeatSequenceElements']))
    print('repeat_sketchy_probes', len(repeat_sketchy_probes))

    pxbr_intersection_sketchy_probes = (polymorphism_sketchy_probes.intersection(
        xhybridizing_sketchy_probes).intersection(basecolorchange_sketchy_probes).intersection(
        repeat_sketchy_probes))
    print('pxbr_intersection_sketchy_probes', len(pxbr_intersection_sketchy_probes))


def test_exclude_sex_control_probes():
    test_filepath = 'docs/example_data/GSE69852'
    df = methylcheck.load(test_filepath)
    array = '450k'
    filtered = methylcheck.exclude_sex_control_probes(df, array, no_sex=True, no_control=True, verbose=False)
