import os
import logging

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression

from resmico import utils


def main(args):
    np.random.seed(args.seed)

    # CPU only instead of GPU
    # if args.cpu_only:
    #     logging.info('Setting env for CPU-only mode...')
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    
    # Load and process data
    # Provide objective to load
    custom_obj = {'class_recall_0':utils.class_recall_0, 'class_recall_1': utils.class_recall_1}
    logging.info('Loading model...')
    ## pkl
    # logging.info('  Loading mstd...')
    # F = os.path.join(args.model_path, args.mstd_name)
    # if not os.path.exists(F):
    #     msg = 'mstd file is not available at data-path: {}'
    #     raise IOError(msg.format(F))
    # with open(F, 'rb') as mstd:
    #     mean_tr, std_tr = pickle.load(mstd)
    ## h5
    # logging.info('  Loading h5...')
    F = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(F):
        msg = 'Model file not available at data-path: {}'
        raise IOError(msg.format(F))    
    model = load_model(F, custom_objects=custom_obj)
    
    # outdir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    feat_files_dic = utils.read_feature_ft_realdata(args.feature_file_table)

    # separate prediction for each genome
    for sample, info in feat_files_dic['pkl'].items():
        for genome, filename in info.items():
            with open(filename, 'rb') as inF:
                logging.info('loading features from {}'.format(filename))
                x, n2i = pickle.load(inF)

            preds = []

            # split into batches
            all_lens = [len(xi) for xi in x]

            inds_sel = np.arange(len(all_lens))[np.array(all_lens) >= args.min_len]
            inds_toolong = np.arange(len(all_lens))[np.array(all_lens) > args.mem_lim]  # change if want 3k
            inds_sel = np.array([el for el in inds_sel if el not in inds_toolong])

            batch_list = utils.create_batch_inds(all_lens, inds_sel, args.mem_lim)
            logging.info("Number of batches: {}".format(len(batch_list)))

            for ind in range(len(batch_list)):
                X = np.array(x)[batch_list[ind]]

                batch_size = 0
                for cont_ind in batch_list[ind]:
                    batch_size += 1 + utils.n_moves_window(all_lens[cont_ind], args.window, args.window / 2)
                x_mb = utils.gen_sliding_mb(X, batch_size, args.window, args.window / 2)
                pred_mb = model.predict(x_mb)
                preds.extend(pred_mb)

            dic_predictions = utils.aggregate_chunks(batch_list, all_lens, all_labels=[],
                                                     all_preds=np.array(preds), window=args.window, step=args.window / 2)
            logging.info('Dictionary created')

            df_preds = pd.DataFrame.from_dict(dic_predictions)
            df_preds = utils.add_stats(df_preds)
            min_len = 1000
            mem_lim = 500000
            df_preds['scale_length'] = (df_preds['length']-min_len)/mem_lim
            
            logging.info('Aggregated chunks with logreg')

            clf = LogisticRegression()
#             w = np.array([[-0.6246, -0.4191, -2.2027, -0.7481, -0.5864, -0.0315, 0.3024, 0.3307,
#                            -1.0160],
#                           [0.4621, 0.6289, 2.0829, 0.8016, 0.6754, -0.2144, -0.3024, -0.2907,
#                            0.9291]])
#             b = np.array([2.6030, -2.4675])
            
            w = np.array([[-0.8070, -0.5010, -0.4407, -1.4237, -2.4168, -0.5307, -0.7208],
                            [ 0.4844,  0.7417,  0.2885,  1.4630,  2.2048,  0.6157,  1.0309]])
            b = np.array([ 2.9220, -2.5793])
            clf.coef_ = w
            clf.intercept_ = b
            clf.classes_ = np.array([-1, 1])

#             features_list = ['min', 'mean', 'max', 'std', 'p10', 'p30', 'p50', 'p70', 'p90']
            features_list = ['scale_length', 'min', 'mean', 'max', 'std', 'p20', 'p80']
            X_logreg = df_preds[features_list]
            y_pred_proba = clf.predict_proba(X_logreg)[:, 1]
            df_preds['y_pred_proba'] = y_pred_proba
            y_pred = np.array(y_pred_proba)>0.5
            genome_quality = str(np.round(1-np.mean(y_pred_proba), 3))

            df_name = args.save_path + '/' + args.save_name + \
                       '_sample_' + sample + '_genome_' + genome + \
                       '_Q' + genome_quality + '.csv'
            df_preds.to_csv(df_name, index=False)
            logging.info("csv table saved: {}".format(df_name))


    # x, y, i2n = Utils.load_features_nogt(args.feature_file_table,
    #                                      force_overwrite=args.force_overwrite,
    #                                      pickle_only=args.pickle_only,
    #                                      n_procs=args.n_procs, chunks=False)
    #
    # logging.info('Loaded {} contigs'.format(len(set(i2n.values()))))
    # n2i = Utils.reverse_dict(i2n)
    # x = [xi for xmeta in x for xi in xmeta]
    # y = np.concatenate(y)
    #
    # logging.info('Running model generator...')
    # dataGen = Models.Generator(x, y, batch_size=args.batch_size, shuffle=False,
    #                            mean_tr=mean_tr, std_tr=std_tr)
    #
    # logging.info('Computing predictions...')
    # scores = Utils.compute_predictions(n2i, dataGen, model,
    #                                    args.save_path, args.save_name)
    #


if __name__ == '__main__':
    pass
