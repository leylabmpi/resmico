import os
import pytest
import logging

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_helps(script_runner):
    ret = script_runner.run('resmico', 'filter', '-h')
    assert ret.success, ret.print()

def test_filter(tmpdir, script_runner):
    save_path = tmpdir.mkdir('save_dir')
    pred_file = os.path.join(data_dir, 'UHGG-n9', 'resmico_predictions.csv')
    contig_file = os.path.join(data_dir, 'UHGG-n9', 'GUT_GENOME031704.fna.gz')
    ret = script_runner.run('resmico', 'filter',
                            '--score-cutoff', '0.02',
                            '--outdir', str(save_path),
                            pred_file, contig_file)
    assert ret.success, ret.print()

def test_filter_list(tmpdir, script_runner):
    save_path = tmpdir.mkdir('save_dir')
    pred_file = os.path.join(data_dir, 'UHGG-n9', 'resmico_predictions.csv')
    contig_files = os.path.join(data_dir, 'UHGG-n9', 'contig_files.txt')
    ret = script_runner.run('resmico', 'filter',
                            '--score-cutoff', '0.02',
                            '--outdir', str(save_path),
                            pred_file, contig_files)
    assert ret.success, ret.print()

def test_filter_outfile(tmpdir, script_runner):
    save_path = tmpdir.mkdir('save_dir')
    pred_file = os.path.join(data_dir, 'UHGG-n9', 'resmico_predictions.csv')
    contig_files = os.path.join(data_dir, 'UHGG-n9', 'contig_files.txt')
    ret = script_runner.run('resmico', 'filter',
                            '--score-cutoff', '0.02',
                            '--outdir', str(save_path),
                            '--outfile', 'contigs-filtered.fna',
                            pred_file, contig_files)
    assert ret.success, ret.print()
