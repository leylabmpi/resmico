import os
import pytest
import logging

from resmico.commands import train as Train_CMD

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_help():
    args = ['-h']
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Train_CMD.parse_args(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_train(tmpdir, caplog):
    caplog.set_level(logging.INFO)   
    save_path = tmpdir.mkdir('save_dir')
    args = ['--binary-data', '--n-folds', '2', '--n-epochs', '2',
            '--save-path', str(save_path),
            '--feature-files-path',
            os.path.join(data_dir, 'n10_r2/preprocess/')]
    args = Train_CMD.parse_args(args)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Train_CMD.main(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


