from ResMiCo import Utils


def main(args):
    if args.pickle_tsv:
        if args.real_data:
            feature_files_dic = Utils.read_feature_ft_realdata(args.feature_file_table,
                                                               args.force_overwrite)
            Utils.pickle_in_parallel(feature_files_dic, args.n_procs, set_target=False, real_data=True)
        else:
            feature_files_dic = Utils.read_feature_file_table(args.feature_file_table,
                                                              args.force_overwrite, technology=args.technology)
            Utils.pickle_in_parallel(feature_files_dic, args.n_procs,
                                     args.set_target, real_data=False, v1=args.deepmased_v1)
    if args.compute_mean_std:
        Utils.compute_sum_sumsq_n(args.feature_file_table, n_feat=20)  # todo: features_sel
    if args.standard_data:
        if args.real_data:
            Utils.standardize_data(args.feature_file_table, args.mean_std_file, set_target=False, real_data=True)
        else:
            Utils.standardize_data(args.feature_file_table, args.mean_std_file, args.set_target)
    #     if args.add_pos_feat:
    #         Utils.add_pos_feat(args.feature_file_table, args.rch, args.set_target, args.name_input_folder)
    #     if args.add_feat_h5:
    #         Utils.add_feat_h5(args.input_folder, args.output_folder, args.rch)
    return


if __name__ == '__main__':
    pass
