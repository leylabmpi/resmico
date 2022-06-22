import os
import pytest
import logging

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_helps(script_runner):
    ret = script_runner.run('resmico', 'train', '-h')
    assert ret.success

def test_train(tmpdir, script_runner):
    save_path = tmpdir.mkdir('save_dir')
    input_path = os.path.join(data_dir, 'n10', 'features')
    ret = script_runner.run('resmico', 'train',
                            '--n-epochs', '2',            
                            '--save-path', str(save_path),
                            '--feature-files-path', input_path)
    assert ret.success, ret.print()

def test_train_filelist(tmpdir, script_runner):
    save_path = tmpdir.mkdir('save_dir')
    input_path = os.path.join(data_dir, 'n10',
                              'features', 'file_list.txt')
    ret = script_runner.run('resmico', 'train',
                            '--n-epochs', '2',            
                            '--save-path', str(save_path),
                            '--feature-files-path', input_path)
    assert ret.success, ret.print()
