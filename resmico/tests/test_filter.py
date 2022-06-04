import os
import pytest
import logging

from resmico.commands import filter_contigs as filter_cmd

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_help():
    args = ['-h']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        filter_cmd.parse_args(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_filter(tmpdir, caplog, capsys):
    caplog.set_level(logging.INFO)
    save_path = tmpdir.mkdir('save_dir')
    pred_file = os.path.join(data_dir, 'cami-gut_pred.csv.gz')
    contig_file = os.path.join(data_dir, 'cami-gut_contigs.fna.gz')
    args = [pred_file, contig_file]
    args = filter_cmd.parse_args(args)
    filter_cmd.main(args)
    captured = capsys.readouterr()


