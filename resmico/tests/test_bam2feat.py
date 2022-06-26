import os
import pytest
import logging

# test/data dir
test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, 'data')

# tests
def test_helps(script_runner):
    ret = script_runner.run('resmico', 'bam2feat', '-h')
    assert ret.success, ret.print()

