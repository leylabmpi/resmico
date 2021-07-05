from pathlib import Path
from ratschlab_common.io.bigmatrix import TablesBigMatrixWriter, AxisDescription
import click
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tables

logging.getLogger().setLevel(logging.INFO)


def process_file(in_file, out_file):
    logging.info(f"working on {in_file} {os.path.getsize(in_file)}")

    if out_file.exists():
        out_file.unlink()

    with open(in_file, 'rb') as f:
        loaded_info = pickle.load(f)
        mats = loaded_info[0] 
        sample_dict = loaded_info[-1]
        if len(loaded_info)==3:
            labels = loaded_info[1]
            real_data = False
        elif len(loaded_info)==2:
            real_data = True
        else:
            print('pickle file has only {} dimension'.format(len(loaded_info)))

    logging.info(f"loaded {in_file}")
    id2sample = { v:k for (k,v) in sample_dict.items()}
    samples_str = [id2sample[i] for i in range(0, len(id2sample))]

    mats_size = [m.shape[0] for m in mats]
    offset_ends = np.cumsum(mats_size)

    row_desc_ids = np.concatenate([ np.repeat(i, sz) for (i, sz) in enumerate(mats_size)])

    row_desc_ids = AxisDescription([row_desc_ids], ['id'], [])

    overall_mat = np.concatenate(mats)

    #chunksize_kb = 256
    #chunk_length = int(chunksize_kb * 1024 / (8*overall_mat.shape[1]))

    filter = tables.Filters(complevel=5, complib='blosc:lz4hc')
    out_file.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f"writing {out_file}")
    bm_writer = TablesBigMatrixWriter()
    bm_writer.write(out_file, overall_mat, row_desc_ids,
                    pd.DataFrame(list(range(0, overall_mat.shape[1]))), chunkshape=(1000, overall_mat.shape[1]),
                    compression_filter=filter)
                          #(chunk_length, overall_mat.shape[1]))

    with tables.open_file(out_file, "r+") as h5_file:
        h5_file.create_array(h5_file.root, 'offset_ends', offset_ends)
        h5_file.create_array(h5_file.root, 'samples', samples_str)
        if not real_data:
            h5_file.create_array(h5_file.root, 'labels', np.array(labels))


@click.command()
@click.argument("input_pickle", type=click.Path(readable=True))
@click.argument("output_hdf", type=click.Path())
def main(input_pickle, output_hdf):
    process_file(Path(input_pickle), Path(output_hdf))


if __name__ == "__main__":
    main()

