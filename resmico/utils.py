import _pickle as pickle
import os
import sys
import csv
import gzip
from pathlib import Path
import logging
from toolz import itertoolz
from functools import partial
from collections import defaultdict
import multiprocessing as mp
import tables
import pathos

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, ReLU, BatchNormalization, Add, Cropping1D, AveragePooling1D
from tensorflow.keras.layers import Masking, LayerNormalization, MultiHeadAttention, Dropout

def nested_dict():
    return defaultdict(nested_dict)


def iter_all_values(nested_dictionary):
    for key, value in nested_dictionary.items():
        if type(value) is defaultdict:
            yield from iter_all_values(value)
        else:
            yield value


def get_all_values(nested_dictionary):
    return list(iter_all_values(nested_dictionary))


def compute_sum_sumsq_n(featurefiles_table, n_feat=18):
    """
    This function is applied once to the whole training data to compute
    mean and standard deviation. File is saved in the same directory.
    """
    feat_files = read_feature_file_table(featurefiles_table)
    all_files = get_all_values(feat_files['pkl'])
    feat_sum = np.zeros(n_feat)
    feat_sq_sum = np.zeros(n_feat)
    n_el = 0

    for filename in all_files:
        with open(filename, 'rb') as feat:
            logging.info('openning file: {}'.format(filename))
            x, _, _ = pickle.load(feat)
            for xi in x:
                sum_xi = np.sum(xi, 0)
                sum_sq = np.sum(xi ** 2, 0)
                feat_sum += sum_xi
                feat_sq_sum += sum_sq
                n_el += xi.shape[0]

    path = os.path.split(featurefiles_table)[0]
    np.save(path + '/mean_std', np.array([feat_sum, feat_sq_sum, n_el], dtype=object))
    return


def standardize_file(filename, mean, std, set_target):
    with open(filename, 'rb') as feat:
        if set_target == True:
            x, target_contig, name_to_id = pickle.load(feat)
        else:
            x, name_to_id = pickle.load(feat)
        standard_x = []
        for xi in x:
            standard_x.append((xi - mean) / std)

    with open(filename, 'wb') as f:
        logging.info('Dumping: {}'.format(filename))
        if set_target == True:
            pickle.dump([standard_x, target_contig, name_to_id], f)
        else:
            pickle.dump([standard_x, name_to_id], f)
    return


def standardize_data(feat_file_table, mean_std_file, set_target=True, real_data=False, nprocs=1):
    if real_data:
        feat_files = read_feature_ft_realdata(feat_file_table)
    else:
        feat_files = read_feature_file_table(feat_file_table)
    feat_sum, feat_sq_sum, n_el = np.load(mean_std_file, allow_pickle=True)

    mean = feat_sum / n_el
    std = np.sqrt((feat_sq_sum / n_el - mean ** 2).clip(min=0))
    # do not change refrence, counts, and counts features that are already devided by corresponding coverage (3 features)
    mean[0:8 + 3] = 0
    std[0:8 + 3] = 1
    std[std == 0] = 1.

    print(mean)
    print(std)

    all_files = get_all_values(feat_files['pkl'])

    with pathos.multiprocessing.Pool(nprocs) as pool:
        pool.map(lambda file: standardize_file(file, mean, std, set_target), all_files)
    return


def add_pos_feat(feature_file_table, rch='', set_target=True, name_input_folder='features'):
    name_output_folder = name_input_folder + '_pos'
    Data_dic = read_feature_file_table(feature_file_table)
    all_input_pickles = []
    if len(rch) > 1:
        keys1 = [rch]
    else:
        keys1 = Data_dic['pkl'].keys()

    for key1 in keys1:
        for key2 in Data_dic['pkl'][key1].keys():
            for key3 in Data_dic['pkl'][key1][key2].keys():
                for key4 in Data_dic['pkl'][key1][key2][key3].keys():
                    all_input_pickles.append(Data_dic['pkl'][key1][key2][key3][key4])

    for input_pickle_path in all_input_pickles:
        pickle_with_posf(input_pickle_path, set_target,
                         name_input_folder, name_output_folder)

    return


def pickle_with_posf(input_pickle_path, set_target=True,
                     name_input_folder='features_sel', name_output_folder='features_pos'):
    with open(input_pickle_path, 'rb') as feat:
        if set_target == True:
            xi, yi, n2ii = pickle.load(feat)
        else:
            xi, n2ii = pickle.load(feat)

    new_xi = []
    for cont in xi:
        new_fs = []
        for T in [len(cont) - 1, 1000, 5000]:
            new_fs.append(pos_sin(len(cont), T))
            new_fs.append(pos_cos(len(cont), T))

        new_cont = np.concatenate((cont, np.array(new_fs).T), axis=1)
        new_xi.append(new_cont)

    path_parts = list(Path(input_pickle_path).parts)
    path_parts[path_parts.index(name_input_folder)] = name_output_folder
    features_out = Path(*path_parts)
    Path(features_out.parent).mkdir(parents=True, exist_ok=True)
    with open(features_out, 'wb') as f:
        logging.info('Dumping pickle file {}'.format(features_out))
        if set_target == True:
            pickle.dump([new_xi, yi, n2ii], f)
        else:
            pickle.dump([xi, n2ii], f)


def pos_cos(lencont, T):
    pos_f = np.cos(np.arange(lencont) * 2 * np.pi / T)
    return pos_f


def pos_sin(lencont, T):
    pos_f = np.sin(np.arange(lencont) * 2 * np.pi / T)
    return pos_f


def get_row_val(row, row_num, col_idx, col):
    try:
        x = row[col_idx[col]]
    except (IndexError, KeyError):
        msg = 'ERROR: Cannot find "{}" value in Row{} of feature file table'
        sys.stderr.write(msg.format(col, row_num) + '\n');
        sys.exit(1)
    return str(x)


def find_pkl_file(feat_file, force_overwrite=False):
    if force_overwrite is True:
        #        logging.info('  --force-overwrite=True; creating pkl from tsv file')
        return feat_file, 'tsv'
    else:
        logging.info('  searching for pkl version of tsv file')

    # extract file name is needed
    pkl = os.path.splitext(feat_file)[0]
    if pkl.endswith('.tsv'):
        pkl = os.path.splitext(pkl)[0]

    if os.path.isfile(pkl + '.pkl'):
        logging.info('Found pkl file: {}'.format(pkl))
        #         msg = '  Using the existing pkl file. Use DeepMAsED Preprocess --pickle-tsv'
        #         msg += ' --force-overwrite=True to force-recreate the pkl file from the tsv file'
        #         logging.info(msg)
        return pkl + '.pkl', 'pkl'
    else:
        logging.info('  No pkl found. A pkl file will be created from the tsv file')
        return feat_file, 'tsv'


def read_feature_file_table(feat_file_table, force_overwrite=False, technology='all-asmbl'):
    """ Loads feature file table, which lists all feature tables & associated
    metadata. The table is loaded based on column names.
    Params:
      feat_file_table : str, file path of tsv table
      force_overwrite : bool, force create pkl files?
      technology : str, filter to just specified assembler(s)
    Returns:
      dict{file_type : {richness: {read_depth: {simulation_rep : {assembler : feature_file }}}}}
    """
    if feat_file_table.endswith('.gz'):
        _open = lambda x: gzip.open(x, 'rt')
    else:
        _open = lambda x: open(x, 'r')

    base_dir = os.path.split(feat_file_table)[0]
    D = nested_dict()
    with _open(feat_file_table) as f:
        # load
        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)
        if 'read_length' in col_names:
            longdir = True
        else:
            longdir = False
        # longdir : means resmico simulations with
        # richness/abundance_distribution/simulation_replicate/read_length/sequencing_depth/assembler/
        # folder structure

        # indexing
        colnames = ['richness', 'rep', 'read_depth', 'assembler', 'feature_file']
        if longdir:
            colnames.extend(['abundance_distribution', 'read_length'])
        colnames = {x: col_names.index(x) for x in colnames}

        # formatting rows
        for i, row in enumerate(tsv):
            # i is used only for logging info
            richness = get_row_val(row, i + 2, colnames, 'richness')
            rep = get_row_val(row, i + 2, colnames, 'rep')
            read_depth = get_row_val(row, i + 2, colnames, 'read_depth')
            assembler = get_row_val(row, i + 2, colnames, 'assembler')
            if technology != 'all-asmbl' and assembler != technology:
                msg = 'Feature file table, Row{} => "{}" != --technology; Skipping'
                logging.info(msg.format(i + 2, assembler))
                continue
            if longdir:
                # get_row_val was useful only for printing errors
                abnd_distr = row[colnames['abundance_distribution']]
                read_len = row[colnames['read_length']]
            feature_file = get_row_val(row, i + 2, colnames, 'feature_file')
            if not os.path.isfile(feature_file):
                feature_file = os.path.join(base_dir, feature_file)
            if not os.path.isfile(feature_file):
                msg = 'Feature file table, Row{} => Cannot find file; '
                msg += 'The file provided: {}'
                raise ValueError(msg.format(i + 2, feature_file))
            #            else:
            #                logging.info('Input file exists: {}'.format(feature_file))

            if feature_file.endswith('.pkl'):
                file_type = 'pkl'
            elif feature_file.endswith('.tsv') or feature_file.endswith('.tsv.gz'):
                feature_file, file_type = find_pkl_file(feature_file, force_overwrite)
            else:
                msg = 'Feature file table, Row{} => file extension'
                msg += ' must be ".tsv", ".tsv.gz", or ".pkl"'
                msg += '; The file provided: {}'
                raise ValueError(msg.format(i + 2, feature_file))

            if longdir:
                D[file_type][richness][abnd_distr][rep][read_len][read_depth][assembler] = feature_file
            else:
                D[file_type][richness][read_depth][rep][assembler] = feature_file

    # summary
    sys.stderr.write('#-- Feature file table summary --#\n')
    n_tech = defaultdict(dict)
    if longdir:
        for ft, inf in D.items():
            for rch, info in inf.items():
                for abnd, infoo in info.items():
                    for rep, infooo in infoo.items():
                        for rlen, infoooo in infooo.items():
                            for dp, infooooo in infoooo.items():
                                for tech, filename in infooooo.items():
                                    try:
                                        n_tech[ft][tech] += 1
                                    except KeyError:
                                        n_tech[ft][tech] = 1
    else:
        for ft, inf in D.items():
            for rich, info in inf.items():
                for dep, infoo in info.items():
                    for rep, infooo in infoo.items():
                        for tech, filename in infooo.items():
                            try:
                                n_tech[ft][tech] += 1
                            except KeyError:
                                n_tech[ft][tech] = 1

    msg = 'Assembler = {}; File type = {}; No. of files: {}\n'
    for ft, v in n_tech.items():
        for tech, v in v.items():
            sys.stderr.write(msg.format(tech, ft, v))
    sys.stderr.write('#--------------------------------#\n')
    return D


def read_feature_ft_realdata(feat_file_table, force_overwrite=False):
    """ Loads feature file table, which lists all feature tables & associated
    metadata. The table is loaded based on column names.
    Params:
      feat_file_table : str, file path of tsv table
      force_overwrite : bool, force create pkl files?
    Returns:
      dict{file_type : {sample: {genome : feature_file }}}
    """
    df = pd.read_csv(feat_file_table, sep='\t')
    base_dir = os.path.split(feat_file_table)[0]
    D = nested_dict()

    for raw_id in range(df.shape[0]):
        genome = df.loc[raw_id, 'genome']
        sample = df.loc[raw_id, 'sample']
        feature_file = df.loc[raw_id, 'feature_file']
        feature_file = os.path.join(base_dir, feature_file)
        if not os.path.isfile(feature_file):
            msg = 'Feature file table, Row{} => Cannot find file; '
            msg += 'The file provided: {}'
            raise ValueError(msg.format(raw_id + 2, feature_file))
        if feature_file.endswith('.pkl'):
            file_type = 'pkl'
        elif feature_file.endswith('.tsv') or feature_file.endswith('.tsv.gz'):
            feature_file, file_type = find_pkl_file(feature_file, force_overwrite)
        else:
            msg = 'Feature file table, Row{} => file extension'
            msg += ' must be ".tsv", ".tsv.gz", or ".pkl"'
            msg += '; The file provided: {}'
            raise ValueError(msg.format(raw_id + 2, feature_file))

        D[file_type][sample][genome] = feature_file
    return D


def pickle_in_parallel(feature_files, n_procs, set_target=True, real_data=False, v1=False):
    """
    Pickling feature files using multiproessing.Pool.
    Params:
      feature_files : list of file paths
      n_procs : int, number of parallel processes
      set_target : bool, passed to pickle_data_b()
    """
    if n_procs > 1:
        logging.info('Pickling in parallel with {} threads...'.format(n_procs))
    else:
        logging.info('Pickling...')
    pool = mp.Pool(processes=n_procs)
    # list of lists for input to pool.map
    x = get_all_values(feature_files['tsv'])

    # Pickle in parallel and saving file paths in dict
    if v1:
        func = partial(pickle_data_b_v1, set_target=set_target)
    else:
        func = partial(pickle_data_b, set_target=set_target)
    if n_procs > 1:
        ret = pool.map(func, x)
    else:
        ret = map(func, x)
    for y in ret:
        logging.info(" ")
    return


def pickle_data_b(features_in, set_target=True):
    """
    One time function parsing the tsv file and dumping the 
    values of interest into a pickle file. 
    The input file can be gzip'ed 
    Params:
      x : list, first 2 elements are features_in & features_out
      set_target : bool, set the target (for train) or not (for predict)
    Returns:
      features_out       
    """
    features_out = os.path.join(os.path.split(features_in)[0], 'features.pkl')
    msg = 'Pickling feature data: {} => {}'
    #    logging.info(msg.format(features_in, features_out))

    feat_contig, target_contig = [], []
    name_to_id = {}

    # Dictionary for one-hot encoding
    letter_idx = defaultdict(int)
    # Idx of letter in feature vector
    idx_tmp = [('A', 0), ('C', 1), ('T', 2), ('G', 3)]

    for k, v in idx_tmp:
        letter_idx[k] = v

    idx = 0
    # Read tsv and process features
    if features_in.endswith('.gz'):
        _open = lambda x: gzip.open(x, 'rt')
    else:
        _open = lambda x: open(x, 'r')
    with _open(features_in) as f:
        # load
        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)
        # indexing
        w_contig = col_names.index('contig')
        w_ref = col_names.index('ref_base')
        w_nA = col_names.index('num_query_A')
        w_nC = col_names.index('num_query_C')
        w_nG = col_names.index('num_query_G')
        w_nT = col_names.index('num_query_T')
        w_npropM = col_names.index('num_proper_Match')
        w_orpM = col_names.index('num_orphans_Match')
        w_max_is = col_names.index('max_insert_size_Match')
        w_mean_is = col_names.index('mean_insert_size_Match')
        w_min_is = col_names.index('min_insert_size_Match')
        w_std_is = col_names.index('stdev_insert_size_Match')
        w_mean_mq = col_names.index('mean_mapq_Match')
        w_min_mq = col_names.index('min_mapq_Match')
        w_std_mq = col_names.index('stdev_mapq_Match')
        w_mean_al = col_names.index('mean_al_score_Match')
        w_min_al = col_names.index('min_al_score_Match')
        w_std_al = col_names.index('stdev_al_score_Match')
        w_gc = col_names.index('seq_window_perc_gc')  # try without
        w_npropV = col_names.index('num_proper_SNP')
        w_cov = col_names.index('coverage')

        w_features = [w_max_is, w_min_is, w_mean_is, w_std_is,
                      w_min_mq, w_mean_mq, w_std_mq,
                      w_min_al, w_mean_al, w_std_al,
                      w_gc,
                      w_cov]

        w_num_features = [w_npropM, w_orpM, w_npropV]
        nf = 23  # 4 for reference features, 4 count features, 12 important features, 3 'al_score'

        # formatting rows
        for row in tsv:
            name_contig = row[w_contig]

            # If name not in set, add previous contig and target to dataset
            if name_contig not in name_to_id:
                if idx != 0:
                    # filling missing values with average within contig
                    df_feat = pd.DataFrame(np.array(feat).reshape(-1, nf))
                    # before all missing values were just filled with an average within a contig df_feat.mean()
                    # this results in a very noisy filling,
                    # because missing position are getting different values depending on the contig of origin
                    # maybe, it is better to put -1 (the minimum observed value in the data is 0) everywhere
                    df_feat.fillna(-1, inplace=True)
                    feat_contig.append(df_feat.values)
                    if set_target == True:
                        target_contig.append(float(tgt))
                feat = []

                # Set target (0 or 1; 1=misassembly)
                if set_target == True:
                    w_ext = col_names.index('Extensive_misassembly')
                    tgt = int(row[w_ext])

                # index
                name_to_id[name_contig] = idx
                idx += 1

            # Construct feature vec
            f_countN = [float(row[w_nA]), float(row[w_nC]), float(row[w_nT]), float(row[w_nG])]
            # normalisation, absolute values is coded in coverage
            if np.sum(f_countN) > 1:
                f_countN = f_countN / np.sum(f_countN)  # devision by coverage directly is also possible

            f_num_values = []
            for ind in w_num_features:
                if float(row[w_cov]) > 0:
                    f_num_values.append(float(row[ind]) / float(row[w_cov]))
                else:
                    f_num_values.append(float(row[ind]))

            f_flt_values = []
            for ind in w_features:
                try:
                    f_flt_values.append(float(row[ind]))
                except:
                    if row[ind] == 'NA':
                        f_flt_values.append(None)  # will be filled
                    else:
                        print(ind, row[ind])

            # feat.append(np.concatenate((4 * [0], f_countN, f_flt_values, [int(depth)]))[None, :])
            feat.append(np.concatenate((4 * [0], f_countN, f_num_values, f_flt_values))[None, :])
            feat[-1][0][letter_idx[row[w_ref]]] = 1

    # Append last
    df_feat = pd.DataFrame(np.array(feat).reshape(-1, nf))
    df_feat.fillna(-1, inplace=True)  # df_feat.mean()
    feat_contig.append(df_feat.values)
    if set_target == True:
        target_contig.append(float(tgt))

    # Checking feature object
    assert (len(feat_contig) == len(name_to_id))

    # Save processed data into pickle file
    with open(features_out, 'wb') as f:
        logging.info('Dumping pickle file {}'.format(features_out))
        if set_target == True:
            pickle.dump([feat_contig, target_contig, name_to_id], f)
        else:
            pickle.dump([feat_contig, name_to_id], f)

    return


def pickle_data_b_v1(features_in, set_target=True):
    """
    One time function parsing the csv file and dumping the
    values of interest into a pickle file.
    It generates features exactly in the same format at deepmased v1 and standardize them
    """
    features_out = os.path.join(os.path.split(features_in)[0], 'features.pkl')
    feat_contig, target_contig = [], []
    name_to_id = {}

    mean_tr = [0., 0., 0., 0.,
               1.38582490e+00, 1.22166380e+00, 1.22158993e+00, 1.38745485e+00,
               6.79878240e-03, 5.21650785e+00, 4.22636972e-03]
    std_tr = [1., 1., 1., 1.,
              2.52844001, 2.36866187, 2.36876082, 2.52997747,
              0.11330371, 1.91498052, 0.11517926]

    # Dictionary for one-hot encoding
    letter_idx = defaultdict(int)
    # Idx of letter in feature vector
    idx_tmp = [('A', 0), ('C', 1), ('T', 2), ('G', 3)]

    for k, v in idx_tmp:
        letter_idx[k] = v

    idx = 0
    # Read tsv and process features
    if features_in.endswith('.gz'):
        _open = lambda x: gzip.open(x, 'rt')
    else:
        _open = lambda x: open(x, 'r')
    with _open(features_in) as f:
        # load
        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)
        # indexing
        w_contig = col_names.index('contig')
        w_ext = col_names.index('Extensive_misassembly')
        w_ref = col_names.index('ref_base')
        w_nA = col_names.index('num_query_A')
        w_nC = col_names.index('num_query_C')
        w_nG = col_names.index('num_query_G')
        w_nT = col_names.index('num_query_T')
        w_SNP = col_names.index('num_SNPs')
        w_cov = col_names.index('coverage')
        w_dis = col_names.index('num_discordant')
        w_features = [w_SNP, w_cov, w_dis]
        nf = 11
        # formatting rows
        for row in tsv:
            name_contig = row[w_contig]

            # If name not in set, add previous contig and target to dataset
            if name_contig not in name_to_id:
                if idx != 0:
                    # filling missing values with average within contig
                    df_feat = pd.DataFrame(np.array(feat).reshape(-1, nf))
                    df_feat.fillna(df_feat.mean(), inplace=True)
                    feat_contig.append(df_feat.values)
                    if set_target == True:
                        target_contig.append(float(tgt))
                feat = []

                # Set target (0 or 1; 1=misassembly)
                if set_target == True:
                    tgt = int(row[w_ext])

                # index
                name_to_id[name_contig] = idx
                idx += 1

            # Construct feature vec
            f_countN = [float(row[w_nA]), float(row[w_nC]), float(row[w_nG]), float(row[w_nT])]

            # normalisation, absolute values is coded in coverage

            f_flt_values = []
            for ind in w_features:
                try:
                    f_flt_values.append(float(row[ind]))
                except:
                    if row[ind] == 'NA':
                        f_flt_values.append(None)  # will be filled
                    else:
                        print(ind, row[ind])
            # ref_base	num_query_A	num_query_C	num_query_G	num_query_T	num_SNPs	coverage	num_discordant
            feat_i = np.concatenate((4 * [0], f_countN, f_flt_values))[None, :]
            feat_i = (feat_i - mean_tr) / std_tr
            feat.append(feat_i)
            feat[-1][0][letter_idx[row[w_ref]]] = 1

    # Append last
    df_feat = pd.DataFrame(np.array(feat).reshape(-1, nf))
    df_feat.fillna(df_feat.mean(), inplace=True)
    feat_contig.append(df_feat.values)
    if set_target == True:
        target_contig.append(float(tgt))

    # Save processed data into pickle file
    with open(features_out, 'wb') as f:
        logging.info('Dumping pickle file {}'.format(features_out))
        if set_target == True:
            pickle.dump([feat_contig, target_contig, name_to_id], f)
        else:
            pickle.dump([feat_contig, name_to_id], f)

    return


def load_features_tr(feat_file_table, max_len=10000,
                     technology=None,
                     chunks=True):
    """
    Loads features, pre-process them and returns training. 
    Uses data from both assemblers. 

    Inputs: 
        feat_file_path: path to the table that lists all feature files
        max_len: fixed length of contigs
        technology(=assembler): megahit or metaspades.

    Outputs:
        x, y: lists, where each element comes from one metagenome
    """
    # reading in feature file table
    feat_files = read_feature_file_table(feat_file_table,
                                         technology=technology)

    # for each metagenome simulation rep, combining features from each assembler together
    ## "tech" = assembler
    x, y, ye, yext = [], [], [], []
    for rich, info in feat_files['pkl'].items():
        for dep, infoo in info.items():
            for rep, infooo in infoo.items():
                xtech, ytech = [], []
                for tech, filename in infooo.items():
                    with open(filename, 'rb') as feat:
                        xi, yi, _ = pickle.load(feat)
                        xtech.append(xi)
                        ytech.append(yi)
                x_in_contig, y_in_contig = [], []

                for xi, yi in zip(xtech, ytech):
                    for j in range(len(xi)):
                        len_contig = xi[j].shape[0]
                        if chunks:
                            idx_chunk = 0
                            while idx_chunk * max_len < len_contig:
                                chunked = xi[j][idx_chunk * max_len:
                                                (idx_chunk + 1) * max_len, :]

                                x_in_contig.append(chunked)
                                y_in_contig.append(yi[j])

                                idx_chunk += 1
                        else:
                            x_in_contig.append(xi[j])
                            y_in_contig.append(yi[j])

                # Each element is a metagenome
                x.append(x_in_contig)
                yext.append(np.array(y_in_contig))

    # for binary classification
    y = yext
    return x, y


def load_features(feat_file_table, max_len=10000,
                  technology='megahit',
                  chunks=True,
                  n_procs=1):
    """
    Loads features and returns validation data. 

    Params: 
        data_path: path to directory containing features.pkl
        max_len: fixed length of contigs
        technology: assembler, megahit or metaspades.       
    Returns:
        x, y, i2n: lists, where each element comes from one metagenome, and 
          a dictionary with idx -> (rich, depth, rep, contig_name)
          Dictionary is needed to track number of chunks corresponding to the same contig.
    """

    # Finding feature files
    # reading in feature file table
    feat_files = read_feature_file_table(feat_file_table,
                                         technology=technology)

    # loading pickled feature matrices 
    x, y, ye, yext, n2i = [], [], [], [], []
    shift = 0
    i2n_all = {}
    for rich, info in feat_files['pkl'].items():
        for depth, infoo in info.items():
            for rep, infooo in infoo.items():
                for tech, filename in infooo.items():
                    with open(filename, 'rb') as feat:
                        features = pickle.load(feat)

                    xi, yi, n2ii = features

                    i2ni = reverse_dict(n2ii)

                    x_in_contig, y_in_contig = [], []

                    n2i_keys = set([])
                    for j in range(len(xi)):
                        len_contig = xi[j].shape[0]
                        if chunks:
                            idx_chunk = 0
                            while idx_chunk * max_len < len_contig:
                                chunked = xi[j][idx_chunk * max_len:
                                                (idx_chunk + 1) * max_len, :]

                                x_in_contig.append(chunked)
                                y_in_contig.append(yi[j])

                                i2n_all[len(x_in_contig) - 1 + shift] = (rich, depth, rep, i2ni[j][0])
                                idx_chunk += 1
                                n2i_keys.add(i2ni[j][0])
                        else:
                            x_in_contig.append(xi[j])
                            y_in_contig.append(yi[j])
                            i2n_all[len(x_in_contig) - 1 + shift] = (rich, depth, rep, i2ni[j][0])
                            n2i_keys.add(i2ni[j][0])

                    # Each element is a metagenome
                    x.append(x_in_contig)
                    yext.append(np.array(y_in_contig))

                    # Sanity check
                    assert (len(n2i_keys - set(n2ii.keys())) == 0)
                    assert (len(set(n2ii.keys()) - n2i_keys) == 0)

                    shift = len(i2n_all)

    # for binary classification
    y = yext

    return x, y, i2n_all


# def load_features_nogt(feat_file_table, max_len=10000, 
#                        pickle_only=False,
#                        force_overwrite=False,
#                        chunks=True,
#                        n_procs=1):
#     """
#     Loads features for real datasets. Filters contigs with low coverage.
#     WARNING: `coverage` column assumed to be second-from-last column.

#     Params: 
#       feat_file_table: str, path to feature file table
#       max_len: str, fixed length of contigs
#     Returns:
#       x, y, i2n: lists, where each element comes from one metagenome, and 
#           a dictionary with idx -> (metagenome, contig_name)
#     """
#     # reading in feature file table
#     feat_files = read_feature_file_table(feat_file_table,
#                                          force_overwrite=force_overwrite)

#     # loading pickled feature tables
#     x, y, ye, yext, n2i = [], [], [], [], []
#     shift = 0
#     i2n_all = {}
#     idx_coverage = -2
#     i = 0
#     for rich,info in feat_files['pkl'].items():
#         for depth,infoo in info.items():
#             for rep,infooo in infoo.items():
#                 for tech,filename in infooo.items():
#                     with open(filename, 'rb') as inF:
#                         logging.info('Loading file: {}'.format(filename))
#                         features = pickle.load(inF)

#                     # unpacking
#                     try:
#                         xi, n2ii = features
#                         yi = [-1 for i in range(len(xi))]
#                     except ValueError:
#                         xi, yi, n2ii = features                

#                     # reverse dict
#                     i2ni = reverse_dict(n2ii)

#                     # contigs
#                     n_contigs_filtered = 0
#                     x_in_contig, y_in_contig = [], []
#                     n2i_keys = set([])

#                     for j in range(len(xi)):
#                         len_contig = xi[j].shape[0]

#                         #Filter low coverage
#                         if np.amin(xi[j][:, idx_coverage]) == 0:
#                             n_contigs_filtered += 1
#                             continue
#                         if chunks:
#                             idx_chunk = 0
#                             while idx_chunk * max_len < len_contig:
#                                 chunked = xi[j][idx_chunk * max_len :
#                                                 (idx_chunk + 1) * max_len, :]

#                                 x_in_contig.append(chunked)
#                                 y_in_contig.append(yi[j])

#                                 i2n_all[len(x_in_contig) - 1 + shift] = (i, i2ni[j][0])
#                                 idx_chunk += 1
#                                 n2i_keys.add(i2ni[j][0])
#                         else:
#                             x_in_contig.append(xi[j])
#                             y_in_contig.append(yi[j])
#                             i2n_all[len(x_in_contig) - 1 + shift] = (int(rep), i2ni[j][0])
#                             n2i_keys.add(i2ni[j][0])

#                     # status
#                     msg = 'Contigs filtered due to low coverage: {}'
#                     logging.info(msg.format(n_contigs_filtered))

#                     # Each element is a metagenome
#                     x.append(x_in_contig)
#                     yext.append(np.array(y_in_contig))
#                     # for next loop iteration
#                     shift = len(i2n_all)
#                     i += 1

#     # for binary classification
#     y = yext
#     return x, y, i2n_all


def kfold(x, y, idx_lo, k=5):  # check why not default function
    """Creating folds for k-fold validation
    Params:
      k : number of folds
    Returns:
      4 lists : x_tr, x_val, y_tr, y_val
    """
    # check data
    if len(x) < k:
        msg = 'Number of metagenomes is < n-folds: {} < {}'
        raise IOError(msg.format(len(x), k))

    # Define validation data
    x_tr, y_tr = [], []
    x_val, y_val = [], []

    # setting fold lower & upper
    meta_per_fold = int(len(x) / k)
    lower = idx_lo * meta_per_fold
    upper = (idx_lo + 1) * meta_per_fold

    # creating folds
    for i, xi in enumerate(x):
        if i < lower or i >= upper:  # idx_lo:
            x_tr = x_tr + xi
            y_tr.append(y[i])
        else:
            x_val = x_val + xi
            y_val.append(y[i])

    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)

    return x_tr, x_val, y_tr, y_val


def class_recall_0(y_true, y_pred):
    label = 0
    class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
    y_true = K.cast(y_true, 'int32')
    accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
    class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def class_recall_1(y_true, y_pred):
    label = 1
    class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
    y_true = K.cast(y_true, 'int32')
    accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
    class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


class auc_callback(Callback):
    def __init__(self, val_gen):
        self.val_gen = val_gen

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.val_gen)
        y_true_val = self.val_gen.y
        auc_val = average_precision_score(y_true_val, y_pred_val)
        #         auc_val = average_precision_score(y_true_val[:len(y_pred_val)], y_pred_val)   #last batch was not readed
        print('\rauc_val: %s' % (str(round(auc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def reverse_dict(d):
    """Flip keys and values.
       There can be more than 1 value per key in a new dic.
    """
    r_d = {}
    for k, v in d.items():
        if v not in r_d:
            r_d[v] = [k]
        else:
            r_d[v].append(k)
    return r_d


# def compute_predictions(n2i, generator, model, save_path, save_name):
#     """
#     Computes predictions for a model and generator, aggregating scores for long contigs.

#     Inputs: 
#         n2i: dictionary with contig_name -> list of idx corresponding to that contig.
#         generator: deepmased data generator
#     Output:
#         saves scores for individual contigs
#     """
#     score_val = model.predict_generator(generator)

#     # Compute predictions by aggregating scores for longer contigs
#     score_val = score_val.flatten()
#     scores = {}

#     outfile = os.path.join(save_path, '_'.join([save_name, 'predictions.csv']))
#     write = open(outfile, 'w')
#     csv_writer = csv.writer(write, delimiter='\t')
#     csv_writer.writerow(['MAG', 'Contig', 'Deepmased_score'])

#     for k in n2i:
#         inf = n2i[k][0]
#         sup = n2i[k][-1] + 1
#         if k[0] not in scores:
#             scores[k[0]] = {}

#         # Make sure contig doesnt appear more than once
#         assert(k[1] not in scores[k[0]])

#         # Make sure we have predictions for these indices
#         if sup > len(score_val):
#             continue

#         # Make sure all the labels for the contig coincide
#         #scores[k[0]][k[1]] = {'pred' : score_val[inf : sup]}
#         csv_writer.writerow([k[0], k[1], str(np.max(score_val[inf : sup]))])

#     write.close()
#     logging.info('File written: {}'.format(outfile))


def compute_predictions_y_known(y, n2i, model, dataGen, n_procs, x=False):
    """
    Computes predictions for a model and generator, NOT aggregating scores for long contigs.

    Inputs: 
        n2i: dictionary with (rich, depth, rep, contig_name) -> list of idx corresponding to that contig.
        y and x of the same size, it works only for contigs of the full length!
    Output:
        scores:
            pred: scores for individual contigs
            y: corresponding true labels
    """
    score_val = model.predict_generator(dataGen,
                                        use_multiprocessing=n_procs > 1, workers=n_procs)

    # Compute predictions by aggregating scores for longer contigs
    score_val = score_val.flatten()
    scores = {}
    for k in n2i:
        inf = n2i[k][0]
        sup = n2i[k][-1] + 1
        if k[0] not in scores:
            scores[k[0]] = {}

        # Make sure contig doesnt appear more than once
        assert (k[1] not in scores[k[0]])

        # Make sure we have predictions for these indices
        if sup > len(score_val):
            continue

        # Make sure all the labels for the contig coincide
        assert ((y[inf: sup] == y[inf]).all())

        if x:  # only for chunks=False
            lens_x = [len(i) for i in x]
            assert (len(y) == len(lens_x))
            scores[k[0]][k[1]] = {'y': int(y[inf]), 'pred': score_val[inf: sup], 'len': lens_x[inf]}
        else:
            scores[k[0]][k[1]] = {'y': int(y[inf]), 'pred': score_val[inf: sup]}
    return scores


def _get_sample_index_from_file(f, metadata_func, longdir):
    samples_dict = {}

    with tables.open_file(f, 'r') as h5:
        for s in h5.get_node('/samples'):
            samples_name = '/'.join(metadata_func(f, longdir) + (s.decode('utf-8'),))
            samples_dict[samples_name] = str(f)
    return samples_dict


def _metadata_func(p: Path, longdir=False):
    if longdir:
        return p.parts[-7:-1]
    else:
        return p.parts[-5:-1]


def build_sample_index(base_path: Path, nprocs: int, sdepth=None, rich=None, rep10=False, filter10=False,
                       longdir=False):
    # The typical directory structure looks like:
    # sample/richness/abundance_distribution/simulation_replicate/read_length/sequencing_depth/assembler/
    richness = rich if rich else '*'
    # build the repetition pattern
    rep = '10' if rep10 else '?' if filter10 else '*'
    sequencing_depth = sdepth if sdepth else '*'
    pattern = f'**/{richness}/**/{rep}/**/{sequencing_depth}/**/*.h5'

    part_files = base_path.glob(pattern)

    with pathos.multiprocessing.Pool(nprocs) as pool:
        partial_dicts = pool.map(lambda f: _get_sample_index_from_file(f, _metadata_func, longdir), part_files)

    samples_dict = {}
    for d in partial_dicts:
        samples_dict.update(d)
    return samples_dict


def _read_label_from_file(f, samples):
    sample_ids = [s[0].split('/')[-1] for s in samples]
    with tables.open_file(f, 'r') as h5f:
        sample_lookup = {s.decode('utf-8'): i for i, s in enumerate(h5f.get_node('/samples')[:])}
        # index of s samples within a file
        sample_idx = [sample_lookup[s] for s in sample_ids]
        labels = h5f.get_node('/labels')[sample_idx]
    return labels


def read_all_labels(samples_dict):
    files_dict = itertoolz.groupby(lambda t: t[1], list(map(  # itertoolz.
        lambda s: (s, samples_dict[s]), samples_dict.keys())))
    y = []
    for f, samples in files_dict.items():
        y.extend(_read_label_from_file(f, samples))
    return np.array(y)


def _read_len_from_file(f, samples):
    sample_ids = [s[0].split('/')[-1] for s in samples]
    with tables.open_file(f, 'r') as h5f:
        sample_lookup = {s.decode('utf-8'): i for i, s in enumerate(h5f.get_node('/samples')[:])}
        # index of s samples within a file
        sample_idx = [sample_lookup[s] for s in sample_ids]
        ends = h5f.get_node('/offset_ends')[sample_idx]
        lens = [x - y for x, y in zip(ends, np.concatenate(([0], ends[:-1])))]
    return lens


def read_all_lens(samples_dict):
    # works only when no samples missing from the file
    files_dict = itertoolz.groupby(lambda t: t[1], list(map(  # itertoolz.
        lambda s: (s, samples_dict[s]), samples_dict.keys())))
    y = []
    for f, samples in files_dict.items():
        y.extend(_read_len_from_file(f, samples))
    return np.array(y)


# data reading
# def clip_range(range_orig, seed, max_length):
#     if range_orig[1] - range_orig[0] <= max_length:
#         return range_orig
#
#     np.random.seed(seed)
#     new_start = np.random.randint(range_orig[0], range_orig[1] - max_length)
#
#     return new_start, new_start + max_length
#
#
# def read_data_from_file(f, samples, seed, max_range_length):
#     sample_ids = [s[0].split('/')[-1] for s in samples]
#
#     mats = []
#     # logging.info(f"Reading data from {f}. Reading {len(samples)} samples.")
#     with tables.open_file(f, 'r') as h5f:
#         sample_lookup = {s.decode('utf-8'): i for i, s in enumerate(h5f.get_node('/samples')[:])}
#
#         # index of s samples within a file
#         sample_idx = [sample_lookup[s] for s in sample_ids]
#
#         labels = h5f.get_node('/labels')[sample_idx]
#
#         offsets = h5f.get_node('/offset_ends')[:]
#         ranges = [(offsets[idx - 1] if idx > 0 else 0, offsets[idx]) for idx in sample_idx]
#
#         np.random.seed(seed)
#         range_seeds = np.random.randint(0, 10000000, len(ranges))
#         ranges_to_read = [clip_range(r, s, max_range_length) for (r, s) in zip(ranges, range_seeds)]
#
#         data_h5 = h5f.get_node('/data')
#         for s, e in ranges_to_read:
#             mats.append(data_h5[s:e, :])
#         return mats, labels
#
#
# def file_reading(file_items, max_len):
#     with pathos.multiprocessing.Pool(1) as pool: #doesn't work together with tf.dataset
#         pool_res = pool.map(lambda t:read_data_from_file(t[0], t[1], t[2], max_len), file_items)
#     X, y = [], []
#     for (dl, ll) in pool_res:
#         for (d, l) in zip(dl, ll):
#             X.append(d)
#             y.append(l)
#     return X, y
# data reading

def clip_range(range_orig, seed, max_length):
    if range_orig[1] - range_orig[0] <= max_length:
        return range_orig

    # np.random.seed(seed)
    new_start = np.random.randint(range_orig[0], range_orig[1] - max_length)

    return new_start, new_start + max_length


def read_data_from_file(f, samples, seed, max_range_length):
    sample_ids = [s[0].split('/')[-1] for s in samples]

    mats = []
    # logging.info(f"Reading data from {f}. Reading {len(samples)} samples.")
    with tables.open_file(f, 'r') as h5f:
        sample_lookup = {s.decode('utf-8'): i for i, s in enumerate(h5f.get_node('/samples')[:])}

        # index of s samples within a file
        sample_idx = [sample_lookup[s] for s in sample_ids]

        labels = h5f.get_node('/labels')[sample_idx]

        offsets = h5f.get_node('/offset_ends')[:]
        ranges = [(offsets[idx - 1] if idx > 0 else 0, offsets[idx]) for idx in sample_idx]

        # np.random.seed(seed)
        range_seeds = np.random.randint(0, 1000000, len(ranges))
        ranges_to_read = [clip_range(r, s, max_range_length) for (r, s) in zip(ranges, range_seeds)]

        data_h5 = h5f.get_node('/data')
        for s, e in ranges_to_read:
            mats.append(data_h5[s:e, :])
        return mats, labels


# def file_reading(file_items, max_len):
#     X, y = [], []
#     for t in file_items:
#         dl, ll = read_data_from_file(t[0], t[1], t[2], max_len)
#         for (d, l) in zip(dl, ll):
#             X.append(d)
#             y.append(l)
#     return X, y

def file_reading(file_items, max_len):
    with pathos.multiprocessing.Pool(4) as pool:
        pool_res = pool.map(lambda t: read_data_from_file(t[0], t[1], t[2], max_len), file_items)
    X, y = [], []
    for (dl, ll) in pool_res:
        X.extend(dl)
        y.extend(ll)
    #         for (d, l) in zip(dl, ll):
    #             X.append(d)
    #             y.append(l)
    return X, y


# for resnet
def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def bn_relu(inputs):
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    return relu

'''
Constructs the essential building unit of Resmico: the residual block. 
'''
# ### original with deterministic predictions
def old_residual_block(x, downsample: bool, filters, kernel_size):
    y = Conv1D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding='valid' if downsample else 'same')(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding='valid')(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding='valid')(x)
        # the additional cropping is needed in order to match the size of the y=Conv1D() output, since here we
        # user kernel_size=1
        x = Cropping1D((0,kernel_size//2))(x)
    x = Cropping1D((0, kernel_size-1))(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

'''
Residual block with all valid pading
'''
def residual_block(x, downsample: bool, filters, kernel_size):
    y = Conv1D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding='valid')(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding='valid')(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding='valid')(x)
        # the cropping is needed to make up for the size difference caused by the convolutions on y
        x = Cropping1D((0,kernel_size-1 + kernel_size//2))(x)

    else:
        x = Cropping1D((0, 2*kernel_size-2))(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


def dilated_residual_block(x, dilate: bool, filters, kernel_size):
    ### if dilate==True: apply dilation rate 3

    if filters>64:
        #bottleneck layer
        y = Conv1D(kernel_size=1,
               filters=64,
               padding='valid')(x)
        y = bn_relu(y)
        y = Conv1D(kernel_size=kernel_size,
               dilation_rate=(1 if not dilate else 3),
               filters=filters,
               padding='valid')(y)
        y = bn_relu(y)
        y = Conv1D(kernel_size=1,
               filters=filters,
               padding='valid')(y)
        y = BatchNormalization()(y)
        
        if dilate: 
            x = Conv1D(kernel_size=1,
                       filters=filters,
                       padding='valid')(x)
            x = Cropping1D((0,((kernel_size-1)*3)))(x)
        else:
            x = Cropping1D((0, (kernel_size-1)))(x)
    
    else:
        #residual block with two weighted layers
        y = Conv1D(kernel_size=kernel_size,
                   dilation_rate=(1 if not dilate else 3),
                   filters=filters,
                   padding='valid')(x)
        y = bn_relu(y)
        y = Conv1D(kernel_size=kernel_size,
                   dilation_rate=1,
                   filters=filters,
                   padding='valid')(y)
        y = BatchNormalization()(y)
        if dilate: 
            #also number of filters changed, so conv layer needed
            x = Conv1D(kernel_size=1,
                       filters=filters,
                       padding='valid')(x)
            x = Cropping1D((0,(kernel_size-1)*3 + (kernel_size-1)))(x)
        else:
            x = Cropping1D((0, 2*kernel_size-2))(x)
        
    out = Add()([x, y])
    out = ReLU()(out)
        
    return out

def transformer_encoder(inputs, head_size, num_heads, dropout=0):
    # Normalization and Attention
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization()(res)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
#     x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# for predictions for long contigs
def load_full_contigs(files_dict):
    with pathos.multiprocessing.Pool(2) as pool:
        pool_res = pool.map(lambda t: read_cont_from_file(t[0], t[1]), files_dict.items())
    X = []
    for dl in pool_res:
        for d in dl:
            X.append(d)
    return X


def read_cont_from_file(f, samples):
    sample_ids = [s[0].split('/')[-1] for s in samples]

    mats = []
    with tables.open_file(f, 'r') as h5f:
        sample_lookup = {s.decode('utf-8'): i for i, s in enumerate(h5f.get_node('/samples')[:])}

        # index of s samples within a file
        sample_idx = [sample_lookup[s] for s in sample_ids]

        offsets = h5f.get_node('/offset_ends')[:]
        ranges = [(offsets[idx - 1] if idx > 0 else 0, offsets[idx]) for idx in sample_idx]

        data_h5 = h5f.get_node('/data')
        for s, e in ranges:
            mats.append(data_h5[s:e, :])
        return mats


def n_moves_window(cont_len, window, step):
    if cont_len < window:
        return 0
    else:
        return np.ceil((cont_len - window) / step)


def create_batch_inds(all_lens, inds_sel, memory_limit, fulllen=False):
    batches_inds = []

    if fulllen:
        cur_batch_ind = []
        cur_max_len = 0
        for i, ind_long in enumerate(inds_sel):
            cont_len = all_lens[ind_long]
            if cur_max_len < cont_len:
                cur_max_len = cont_len
            if cur_max_len * (len(cur_batch_ind) + 1) < memory_limit:
                cur_batch_ind.append(ind_long)
            else:
                batches_inds.append(cur_batch_ind)
                cur_batch_ind = []
                cur_max_len = cont_len
                cur_batch_ind.append(ind_long)
        batches_inds.append(cur_batch_ind)

    else:
        cur_batch_ind = []
        cur_sum_lens = 0
        for i, ind_long in enumerate(inds_sel):
            cont_len = all_lens[ind_long]
            if cur_sum_lens + cont_len < memory_limit:
                cur_sum_lens += cont_len
                cur_batch_ind.append(ind_long)
            else:
                batches_inds.append(cur_batch_ind)
                cur_batch_ind = []
                cur_sum_lens = 0
                cur_sum_lens += cont_len
                cur_batch_ind.append(ind_long)
        batches_inds.append(cur_batch_ind)
    return batches_inds


def gen_sliding_mb(X, batch_size, window, step):
    n_feat = X[0].shape[1]
    x_mb = np.zeros((int(batch_size), window, n_feat))
    #     print('x_mb.shape',x_mb.shape)
    mb_pos = 0
    for i, xi in enumerate(X):
        len_contig = xi.shape[0]
        #         print('len_contig',len_contig)
        for idx_chunk in range(int(1 + n_moves_window(len_contig, window, step))):
            start_pos = int(idx_chunk * step)
            end_pos = start_pos + window
            chunked = xi[start_pos: int(min(end_pos, len_contig)), :]
            #             print('mb_pos, 0:chunked.shape[0]',mb_pos,chunked.shape[0])
            x_mb[mb_pos, 0:chunked.shape[0]] = chunked  # padding is happenning here
            mb_pos += 1
    return x_mb


# look at predictions
def add_stats(df, column_name='chunk_scores'):
    chunk_scores = np.array(df[column_name])
    df['min'] = [np.min(list_scores) for list_scores in chunk_scores]
    df['mean'] = [np.mean(list_scores) for list_scores in chunk_scores]
    df['std'] = [np.std(list_scores) for list_scores in chunk_scores]
    df['max'] = [np.max(list_scores) for list_scores in chunk_scores]

    percent_names = ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90']
    pers = np.arange(10, 100, 10)
    result_shape = (df.shape[0], len(pers))
    compute_pers = [np.percentile(list_scores, pers) for list_scores in chunk_scores]
    df[percent_names] = np.array(compute_pers).reshape(result_shape)

    return df


def aggregate_chunks(batches_list, all_lens, all_labels, all_names, all_preds, window, step):
    if all_labels != []:
        dic_predictions = {'cont_name': [], 'length': [], 'label': [], 'chunk_scores': []}
    else:
        dic_predictions = {'cont_index': [], 'length': [], 'chunk_scores': []}

    start_pos = 0

    for cont_inds in batches_list:
        for cont_ind in cont_inds:
            cont_len = all_lens[cont_ind]
            dic_predictions['length'].append(cont_len)

            end_pos = start_pos + int(1 + n_moves_window(all_lens[cont_ind], window, step))
            cont_preds = all_preds[start_pos:end_pos].reshape(-1)

            # contigs of length of length 5k-6k treated the same as 5k
            if (cont_len >= 5000) & (cont_len <= 6000):
                cont_preds = cont_preds[0]
                # the second window is not informative and can be harmful, because has the same weight as the first

            dic_predictions['chunk_scores'].append(cont_preds)
            start_pos = end_pos
            if all_labels != []:
                dic_predictions['cont_name'].append(all_names[cont_ind])
                dic_predictions['label'].append(all_labels[cont_ind])

            else:
                dic_predictions['cont_index'].append(cont_ind)

    return dic_predictions


def update_progress(current: int, total: int, prefix: str, tail: str):
    """
    Displays or updates a console progress bar.
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'. A value at 1 or bigger represents 100%.
    """
    barLength = 100
    status = tail
    progress = current / total
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    current_str = str(current).rjust(len(str(total)))
    text = f'\r{prefix}[{"#" * block + "-" * (barLength - block)}] {current_str}/{total} {status}'
    sys.stdout.write(text)
    sys.stdout.flush()
    
def sma(arr, window_size=2):
    if len(arr)<=window_size:
        return [np.mean(arr)]
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    while i < len(arr) - window_size + 1:
        window_average = np.sum(arr[
          i:i+window_size]) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages