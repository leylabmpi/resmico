# import
import numpy as np
from DeepMAsED import Utils

def main(args):
    if args.pickle_tsv:
        feature_files_dic = Utils.read_feature_file_table(args.feature_file_table, 
                                                          args.force_overwrite, technology=args.technology)
        Utils.pickle_in_parallel(feature_files_dic, args.n_procs, args.set_target)
    if args.compute_mean_std:
        Utils.compute_sum_sumsq_n(args.feature_file_table, n_feat=21)
    if args.standard_data:
        Utils.standardize_data(args.feature_file_table, args.mean_std_file, args.set_target)
    return
            

if __name__ == '__main__':
    pass
        
