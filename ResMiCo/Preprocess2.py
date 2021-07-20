import argparse
import gzip
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def read_file(fname, features):
    f = gzip.open(fname)

    keys = f.readline().split('\t')
    feature_pos = [keys.index(f) for f in features]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--i', help='Input directory. All .tsv.gz files will be read recursively.', default='./')
    parser.add_argument('--p', help='Number of processes to use for parallelization', default='1')
    parser.add_argument('--features', help='List of features to pre-process', default=['position', 'ref_base'])
    args = parser.parse_args()

    file_list = [str(f) for f in list(Path(args.i).rglob("*.tsv.gz"))]
    print(f'Processing {len(file_list)} *.tsv.gz files found in {args.i} ...');

    pool = Pool(args.p)

if __name__ == '__main__':
    pass
