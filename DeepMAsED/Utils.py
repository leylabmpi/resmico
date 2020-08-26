# import
## batteries
import _pickle as pickle
import os
import sys
import csv
import gzip
import glob
import logging
from functools import partial
from collections import defaultdict
import multiprocessing as mp
import tables
import pathos
## 3rd party
from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import IPython
## application


def nested_dict():
    return defaultdict(nested_dict)


def compute_sum_sumsq_n(featurefiles_table, n_feat=21):
    """
    This function is applied once to the whole training data to compute
    mean and standard deviation. File is saved in the same directory.
    """
    feat_files = read_feature_file_table(featurefiles_table)
    
    feat_sum = np.zeros(n_feat)
    feat_sq_sum = np.zeros(n_feat)
    n_el = 0
    
    for rich,info in feat_files['pkl'].items():
        for dep,infoo in info.items():
            for rep,infooo in infoo.items():
                for tech,filename in infooo.items():
                    with open(filename, 'rb') as feat:
                        print('openning file: ',filename)
                        x, _, _ = pickle.load(feat)
                        for xi in x:
                            sum_xi = np.sum(xi, 0)
                            sum_sq = np.sum(xi ** 2, 0)
                            feat_sum += sum_xi
                            feat_sq_sum += sum_sq
                            n_el += xi.shape[0]
                    
    path = os.path.split(featurefiles_table)[0]
    np.save(path+'/mean_std', [feat_sum, feat_sq_sum, n_el])
    return 


def standardize_data(feat_file_table, mean_std_file, set_target=True):
    
    feat_files = read_feature_file_table(feat_file_table)
    feat_sum, feat_sq_sum, n_el = np.load(mean_std_file, allow_pickle=True)

    mean = feat_sum / n_el
    std = np.sqrt((feat_sq_sum / n_el - mean ** 2).clip(min=0))
    # do not change refrence and counts
    mean[0:8] = 0
    std[0:8] = 1
    std[std==0]=1.

    for rich,info in feat_files['pkl'].items():
        for dep,infoo in info.items():
            for rep,infooo in infoo.items():
                for tech,filename in infooo.items():
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


def get_row_val(row, row_num, col_idx, col):
    try:
        x = row[col_idx[col]]
    except (IndexError, KeyError):
        msg = 'ERROR: Cannot find "{}" value in Row{} of feature file table'
        sys.stderr.write(msg.format(col, row_num) + '\n'); sys.exit(1)
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
        
    if os.path.isfile(pkl+'.pkl'):
        logging.info('Found pkl file: {}'.format(pkl))        
        msg = '  Using the existing pkl file. Use DeepMAsED Preprocess --pickle-tsv'
        msg += ' --force-overwrite=True to force-recreate the pkl file from the tsv file'
        logging.info(msg)
        return pkl+'.pkl', 'pkl'
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
        # indexing
        colnames = ['richness', 'rep', 'read_depth', 'assembler', 'feature_file']
        colnames = {x:col_names.index(x) for x in colnames}
        
        # formatting rows
        for i,row in enumerate(tsv):
            # i is used only for logging info
            richness = get_row_val(row, i + 2, colnames, 'richness')
            rep = get_row_val(row, i + 2, colnames, 'rep')
            read_depth = get_row_val(row, i + 2, colnames, 'read_depth')
            assembler = get_row_val(row, i + 2, colnames, 'assembler')
            if technology != 'all-asmbl' and assembler != technology:
                msg = 'Feature file table, Row{} => "{}" != --technology; Skipping'
                logging.info(msg.format(i+2, assembler))
                continue
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
            
            D[file_type][richness][read_depth][rep][assembler] = feature_file

    # summary
    sys.stderr.write('#-- Feature file table summary --#\n')
    n_tech = defaultdict(dict)
    for ft,inf in D.items():
        for rich,info in inf.items():
            for dep,infoo in info.items():
                for rep,infooo in infoo.items():
                    for tech,filename in infooo.items():
                        try:
                            n_tech[ft][tech] += 1
                        except KeyError:
                            n_tech[ft][tech] = 1
    msg = 'Assembler = {}; File type = {}; No. of files: {}\n'
    for ft,v in n_tech.items():
        for tech,v in v.items():
            sys.stderr.write(msg.format(tech, ft, v))
    sys.stderr.write('#--------------------------------#\n')                
    return D


def pickle_in_parallel(feature_files, n_procs, set_target=True):
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
    pool = mp.Pool(processes = n_procs)
    # list of lists for input to pool.map
    x = []
    for rich,info in feature_files['tsv'].items():
        for depth,infoo in info.items():
            for rep,infooo in infoo.items():
                for tech,filename in infooo.items():
                    F = filename
                    pklF = os.path.join(os.path.split(F)[0], 'features.pkl')
                    x.append([F, pklF, tech, depth])
    # Pickle in parallel and saving file paths in dict
    func = partial(pickle_data_b, set_target=set_target)
    if n_procs > 1:
        ret = pool.map(func, x)
    else:
        ret = map(func, x)
    for y in ret:
        logging.info(" ")
    return 

def pickle_data_b(x, set_target=True):
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
    features_in, features_out = x[:2]
    depth = x[3]
    #add tech as a feature x[2]

    msg = 'Pickling feature data: {} => {}'
#    logging.info(msg.format(features_in, features_out))

    feat_contig, target_contig = [], []
    name_to_id = {}

    # Dictionary for one-hot encoding
    letter_idx = defaultdict(int)
    # Idx of letter in feature vector
    idx_tmp = [('A',0) , ('C',1), ('T',2), ('G',3)]

    for k, v in idx_tmp:
        letter_idx[k] = v

    idx = 0
    #Read tsv and process features
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
        w_npropM = col_names.index('num_proper_Match')
        w_orpM = col_names.index('num_orphans_Match')
        w_max_is = col_names.index('max_insert_size_Match')
        w_mean_is = col_names.index('mean_insert_size_Match')
        w_min_is = col_names.index('min_insert_size_Match')
        w_std_is = col_names.index('stdev_insert_size_Match')
        w_mean_mq = col_names.index('mean_mapq_Match')
        w_min_mq = col_names.index('min_mapq_Match')
        w_std_mq = col_names.index('stdev_mapq_Match')
        w_gc = col_names.index('seq_window_perc_gc') #try without
        w_npropV = col_names.index('num_proper_SNP') #try without
        w_cov = col_names.index('coverage')  # WARNING: predict assumes coverage in -2 position
        w_countN = [w_nA, w_nC, w_nT, w_nG]
        w_features = [w_npropM, w_orpM, 
                      w_max_is, w_min_is, w_mean_is, w_std_is,
                      w_min_mq, w_mean_mq, w_std_mq, 
                      w_npropV, w_gc, w_cov]
        nf=21 #4 for refrence feature, 4 count features, 12 important features, 1 seq depth
        
        # formatting rows

        for row in tsv:
            name_contig = row[w_contig]

            # If name not in set, add previous contig and target to dataset
            if name_contig not in name_to_id:
                if idx != 0:
                    #filling missing values with average within contig
                    df_feat = pd.DataFrame(np.array(feat).reshape(-1,nf))
                    df_feat.fillna(df_feat.mean(), inplace=True)
                    feat_contig.append(df_feat.values)
                    if set_target == True:            
                        target_contig.append(float(tgt))
                feat = []
               
                #Set target (0 or 1; 1=misassembly)
                if set_target == True:
                    tgt = int(row[w_ext])
                    
                # index
                name_to_id[name_contig] = idx
                idx += 1

            # Construct feature vec
            f_countN = [float(row[w_nA]), float(row[w_nC]), float(row[w_nT]), float(row[w_nG])]
            # normalisation, absolute values is coded in coverage
            if np.sum(f_countN)>1:
                f_countN = f_countN/np.sum(f_countN) 
            f_flt_values = []
            for ind in w_features:
                try:
                    f_flt_values.append(float(row[ind]))
                except:
                    if row[ind]=='NA':
                        f_flt_values.append(None) # will be filled
                    else: print(ind, row[ind])
    
            feat.append(np.concatenate((4 * [0], f_countN, f_flt_values, [int(depth)]))[None, :])
            feat[-1][0][letter_idx[row[w_ref]]] = 1

    # Append last
    df_feat = pd.DataFrame(np.array(feat).reshape(-1,nf))
    df_feat.fillna(df_feat.mean(), inplace=True)
    feat_contig.append(df_feat.values)
    if set_target == True:
        target_contig.append(float(tgt))
        
    # Checking feature object
    assert(len(feat_contig) == len(name_to_id))

    # Save processed data into pickle file
    with open(features_out, 'wb') as f:
        logging.info('Dumping pickle file {}'.format(features_out))
        if set_target == True:
            pickle.dump([feat_contig, target_contig, name_to_id], f)
        else:
            pickle.dump([feat_contig, name_to_id], f)
            
    return x
        

def load_features_tr(feat_file_table, max_len=10000, 
                     technology = None,
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
    for rich,info in feat_files['pkl'].items():
        for dep,infoo in info.items():
            for rep,infooo in infoo.items():
                xtech, ytech = [], []
                for tech,filename in infooo.items():
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
                                chunked = xi[j][idx_chunk * max_len :
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
                  technology = 'megahit', 
                  chunks=True,
                  n_procs = 1):
    """
    Loads features and returns validation data. 

    Params: 
        data_path: path to directory containing features.pkl
        max_len: fixed length of contigs
        technology: assembler, megahit or metaspades.       
    Returns:
        x, y, i2n: lists, where each element comes from one metagenome, and 
          a dictionary with idx -> (rich, depth, rep, contig_name)
          Dictionary is needed to track number of chuncks corresponding to the same contig.
    """


    # Finding feature files
    # reading in feature file table
    feat_files = read_feature_file_table(feat_file_table,
                                         technology=technology)

    # loading pickled feature matrices 
    x, y, ye, yext, n2i = [], [], [], [], []
    shift = 0
    i2n_all = {}
    for rich,info in feat_files['pkl'].items():
        for depth,infoo in info.items():
            for rep,infooo in infoo.items():
                for tech,filename in infooo.items():
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
                                chunked = xi[j][idx_chunk * max_len :
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

                    #Sanity check
                    assert(len(n2i_keys - set(n2ii.keys())) == 0)
                    assert(len(set(n2ii.keys()) - n2i_keys) == 0)

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



def kfold(x, y, idx_lo, k=5): #check why not default function
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
        if i < lower or i >= upper: # idx_lo:
            x_tr = x_tr + xi
            y_tr.append(y[i])
        else:
            x_val = x_val + xi
            y_val.append(y[i])

    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)

    return x_tr, x_val, y_tr, y_val


def class_recall_0(y_true, y_pred):
    label=0
    class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
    y_true = K.cast(y_true, 'int32')
    accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
    class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def class_recall_1(y_true, y_pred):
    label=1
    class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
    y_true = K.cast(y_true, 'int32')
    accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
    class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


class roc_callback(Callback):
    def __init__(self,val_gen):
        self.val_gen = val_gen   
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}): 
        y_pred_val = self.model.predict_generator(self.val_gen)
        y_true_val = self.val_gen.y
        auc_val = average_precision_score(y_true_val, y_pred_val)
#         auc_val = average_precision_score(y_true_val[:len(y_pred_val)], y_pred_val)   #last batch was not readed    
        print('\rauc_val: %s' % (str(round(auc_val,4))),end=100*' '+'\n')
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
                                        use_multiprocessing=n_procs > 1,workers=n_procs)

    # Compute predictions by aggregating scores for longer contigs
    score_val = score_val.flatten()
    scores = {}
    for k in n2i:
        inf = n2i[k][0]
        sup = n2i[k][-1] + 1
        if k[0] not in scores:
            scores[k[0]] = {}
       
        # Make sure contig doesnt appear more than once
        assert(k[1] not in scores[k[0]])

        # Make sure we have predictions for these indices
        if sup > len(score_val):
            continue
        
        # Make sure all the labels for the contig coincide
        assert((y[inf : sup] == y[inf]).all())
        
        if x: # only for chunks=False
            lens_x = [len(i) for i in x]
            assert(len(y)==len(lens_x))
            scores[k[0]][k[1]] = {'y' : int(y[inf]), 'pred' : score_val[inf : sup], 'len' : lens_x[inf]}
        else:
            scores[k[0]][k[1]] = {'y' : int(y[inf]), 'pred' : score_val[inf : sup]}
    return scores


def _get_sample_index_from_file(f, metadata_func):
    samples_dict = {}

    with tables.open_file(f, 'r' ) as h5:
        for s in h5.get_node('/samples'):
            samples_name = '/'.join(metadata_func(f) + (s.decode('utf-8'), ))
            samples_dict[samples_name] = str(f)
    return samples_dict

def _metadata_func(p: Path):
    return p.parts[-5:-1]

def build_sample_index(base_path: Path, nprocs: int):
    part_files = base_path.glob('**/*.h5')

    with pathos.multiprocessing.Pool(nprocs) as pool:
        partial_dicts = pool.map(lambda f: _get_sample_index_from_file(f, _metadata_func), part_files)

    samples_dict = {}
    for d in partial_dicts:
        samples_dict.update(d)
    return samples_dict