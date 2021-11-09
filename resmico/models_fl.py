import logging
import math
import random
import time
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, \
    MaxPooling1D, Flatten
from tensorflow.keras.layers import Conv1D, Dropout, Dense
from tensorflow.keras.layers import Bidirectional, LSTM
from toolz import itertoolz

from resmico import contig_reader
from resmico.contig_reader import ContigInfo
from resmico import reader
from resmico import utils


class Resmico(object):
    """
    Implements a convolutional network for mis-assembly prediction.
    """

    def __init__(self, config):
        self.max_len = config.max_len
        self.filters = config.filters
        self.n_conv = config.n_conv
        # ref_base uses one-hot encoding, so needs 4 (i.e. an extra 3) input nodes
        self.n_feat = len(config.features) + 3 if 'ref_base' in config.features else 0
        # self.pool_window = config.pool_window
        self.dropout = config.dropout
        self.lr_init = config.lr_init
        self.n_fc = config.n_fc
        self.n_hid = config.n_hid
        self.net_type = config.net_type  # 'lstm', 'cnn_globpool', 'cnn_resnet', 'cnn_lstm'
        self.num_blocks = config.num_blocks
        self.ker_size = config.ker_size
        self.seed = config.seed

        tf.random.set_seed(self.seed)

        if self.net_type == 'fixlen_cnn_resnet':
            inlayer = Input(shape=(self.max_len, self.n_feat), name='input')
        else:
            inlayer = Input(shape=(None, self.n_feat), name='input')

        if self.net_type == 'cnn_globpool':
            x = Conv1D(self.filters, kernel_size=(10),
                       input_shape=(None, self.n_feat),
                       activation='relu', padding='valid', name='1st_conv')(inlayer)
            # x = BatchNormalization(axis=-1)(x)
            #  x = Dropout(rate=self.dropout)(x)

            for i in range(1, self.n_conv - 1):
                x = Conv1D(2 ** i * self.filters, kernel_size=(5),
                           strides=1, dilation_rate=2,
                           activation='relu')(x)
                # x = BatchNormalization(axis=-1)(x)
                x = Dropout(rate=self.dropout)(x)

            x = Conv1D(2 ** self.n_conv * self.filters, kernel_size=(3),
                       strides=1, dilation_rate=2,
                       activation='relu')(x)
            # x = BatchNormalization(axis=-1)(x)
            x = Dropout(rate=self.dropout)(x)

            maxP = GlobalMaxPooling1D()(x)
            avgP = GlobalAveragePooling1D()(x)
            x = concatenate([maxP, avgP])

        elif self.net_type == 'lstm':
            x = Bidirectional(LSTM(20, return_sequences=True), merge_mode="concat")(inlayer)
            x = Bidirectional(LSTM(40, return_sequences=True, dropout=0.0), merge_mode="ave")(x)
            x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.0), merge_mode="ave")(x)
            x = Bidirectional(LSTM(80, return_sequences=False, dropout=0.0), merge_mode="concat")(x)

        elif self.net_type == 'cnn_lstm':
            x = Conv1D(self.filters, kernel_size=(10),
                       input_shape=(None, self.n_feat),
                       activation='relu', padding='valid', name='1st_conv')(inlayer)
            for i in range(1, self.n_conv - 1):
                x = Conv1D(2 ** i * self.filters, kernel_size=(5),
                           strides=1, dilation_rate=2,
                           activation='relu')(x)
                x = Dropout(rate=self.dropout)(x)
            x = Conv1D(2 ** self.n_conv * self.filters, kernel_size=(3),
                       strides=1, dilation_rate=2,
                       activation='relu')(x)
            x = Dropout(rate=self.dropout)(x)
            x = Bidirectional(LSTM(40, return_sequences=False, dropout=0.0), merge_mode="concat")(x)

        elif self.net_type == 'cnn_resnet':
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(None, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            if self.num_blocks == 3:
                num_blocks_list = [2, 5, 2]
            if self.num_blocks == 4:
                num_blocks_list = [2, 5, 5, 2]
            if self.num_blocks == 5:
                num_blocks_list = [2, 3, 5, 5, 2]
            if self.num_blocks == 6:
                num_blocks_list = [2, 3, 5, 5, 3, 2]
            for i in range(len(num_blocks_list)):
                num_blocks = num_blocks_list[i]
                for j in range(num_blocks):
                    x = utils.residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters,
                                             kernel_size=self.ker_size)
                num_filters *= 2

            maxP = GlobalMaxPooling1D()(x)
            avgP = GlobalAveragePooling1D()(x)
            x = concatenate([maxP, avgP])

        elif self.net_type == 'fixlen_cnn_resnet':
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(self.max_len, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            if self.num_blocks == 3:
                num_blocks_list = [2, 5, 2]
            if self.num_blocks == 4:
                num_blocks_list = [2, 5, 5, 2]
            if self.num_blocks == 5:
                num_blocks_list = [2, 3, 5, 5, 2]
            if self.num_blocks == 6:
                num_blocks_list = [2, 3, 5, 5, 3, 2]
            for i in range(len(num_blocks_list)):
                num_blocks = num_blocks_list[i]
                for j in range(num_blocks):
                    x = utils.residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters,
                                             kernel_size=self.ker_size)
                num_filters *= 2

            avgP = AveragePooling1D(pool_size=100, padding='valid')(x)
            maxP = MaxPooling1D(pool_size=100, padding='valid')(x)
            x = concatenate([maxP, avgP])
            x = Flatten()(x)

        for _ in range(self.n_fc):
            x = Dense(self.n_hid, activation='relu')(x)
            x = Dropout(rate=self.dropout)(x)

        x = Dense(1, activation='sigmoid')(x)

        optimizer = tf.keras.optimizers.Adam(lr=self.lr_init)
        self.net = Model(inputs=inlayer, outputs=x)
        self.net.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=[utils.class_recall_0, utils.class_recall_1])

        # self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        #                        monitor='val_loss', factor=0.8,
        #                        patience=5, min_lr = 0.01 * self.lr_init)

    def predict(self, x, **kwargs):
        return self.net.predict(x, **kwargs)

    def print_summary(self):
        self.net.summary(print_fn=logging.info)

    def save(self, path):
        self.net.save(path)


class Generator(tf.keras.utils.Sequence):
    def __init__(self, x, y, max_len, batch_size,
                 shuffle=True):
        self.max_len = max_len
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.n_feat = x[0].shape[1]

        # Shuffle data
        self.indices = np.arange(len(x))
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Reshuffle when epoch ends 
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def generate(self, indices_tmp):
        """
        Generate new mini-batch
        """
        max_contig_len = max(list(map(len, [self.x[ind] for ind in indices_tmp])))
        mb_max_len = min(max_contig_len, self.max_len)

        x_mb = np.zeros((len(indices_tmp), mb_max_len, self.n_feat))

        for i, idx in enumerate(indices_tmp):
            if self.x[idx].shape[0] <= mb_max_len:
                x_mb[i, 0:self.x[idx].shape[0]] = self.x[idx]
            else:
                # cut chunk
                start_pos = np.random.randint(self.x[idx].shape[0] - mb_max_len + 1)
                x_mb[i, :] = self.x[idx][start_pos:start_pos + mb_max_len, :]

        y_mb = [self.y[i] for i in indices_tmp]
        return x_mb, y_mb

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """
        Get new mb
        """
        if self.batch_size * (index + 1) < len(self.indices):
            indices_tmp = \
                self.indices[self.batch_size * index: self.batch_size * (index + 1)]
        else:
            indices_tmp = \
                self.indices[self.batch_size * index:]
        x_mb, y_mb = self.generate(indices_tmp)
        return x_mb, y_mb


class BinaryDataset(tf.keras.utils.Sequence):
    """
    Base class for resmico binary datasets, i.e. datasets reading data from binary files
    (as opposed to the old CSV files)
    """

    def __init__(self, reader: contig_reader, feature_names: list[str]):
        """
       Arguments:
           - reader: ContigReader instance with all the contig metadata
           - feature_names: the names of the features to read and use for training
        """
        self.reader = reader
        self.feature_names = feature_names
        self.expanded_feature_names = feature_names.copy()
        if 'ref_base' in self.expanded_feature_names:
            pos = self.expanded_feature_names.index('ref_base')
            self.expanded_feature_names[pos: pos + 1] = ['ref_base_A', 'ref_base_C', 'ref_base_G', 'ref_base_T']

    def get_bytes_per_base(self):
        return sum(
            [np.dtype(reader.feature_np_types[reader.feature_names.index(f)]).itemsize for f in self.feature_names])


class BinaryDatasetTrain(BinaryDataset):
    def __init__(self, reader: contig_reader, indices: list[int], batch_size: int, feature_names: list[str],
                 max_len: int, num_translations: int, max_translation_bases: int, fraq_neg: float, do_cache: bool,
                 show_progress: bool):

        """
        Arguments:
            - reader: ContigReader instance with all the contig metadata
            - indices: positions of the contigs in #reader that will be used for training
            - batch_size: training batch size
            - feature_names: the names of the features to read and use for training
            - max_len: maximum acceptable length for a contig. Longer contigs are clipped at a random position
            - num_translations: how many variations to select around the breaking point for positive samples
            - max_translation_bases: maximum number of bases to translate left or right
            - fraq_neg: fraction of samples to keep in the overrepresented class (contigs with no misassembly)
            - do_cahe: if True, the generator will cache the features in memory the first time they are
              read from disk
            - show_progress - if true, a progress bar will show the evaluation progress
        """
        BinaryDataset.__init__(self, reader, feature_names)
        logging.info(
            f'Creating training data generator. Batch size: {batch_size}, Max length: {max_len} Frac neg: {fraq_neg}, '
            f'Features: {len(self.expanded_feature_names)}, Contigs: {len(indices)},  Caching: {do_cache}')

        self.batch_size = batch_size
        self.max_len = max_len
        self.num_translations = num_translations
        self.max_translation_bases = max_translation_bases
        self.fraq_neg = fraq_neg
        self.do_cache = do_cache
        self.show_progress = show_progress

        if self.do_cache:
            # the cache maps a batch index to feature_name:feature_data pairs
            self.cache: dict[int, dict[str, np.array]] = {}
        self.negative_idx = [i for i in indices if reader.contigs[i].misassembly == 0]
        self.positive_idx = [i for i in indices if reader.contigs[i].misassembly == 1]

        self.on_epoch_end()  # select negative samples and shuffle indices

        total_length = sum([self.reader.contigs[i].length for i in self.indices])
        mem_gb = total_length * self.get_bytes_per_base() / 1e9
        logging.info(f'Batch count: {int(np.ceil(len(self.indices) / self.batch_size))}')
        logging.info(
            f'Pos samples: {len(self.positive_idx)}. Neg samples: {len(self.negative_idx) * self.fraq_neg:.0f}. '
            f'Total length: {total_length}. Bytes per base: {self.get_bytes_per_base()}. Req memory: {mem_gb:.2f}GB')

        # determine the position of the num_query_A/C/G/T fields, so that we can apply inversion
        self.pos_A = self.pos_C = self.pos_G = self.pos_T = -1
        self.pos_ref = -1
        if 'ref_base_A' in self.expanded_feature_names:
            self.pos_ref = self.expanded_feature_names.index('ref_base_A')
        if 'num_query_A' in self.expanded_feature_names and 'num_query_T' in self.expanded_feature_names:
            self.pos_A = self.expanded_feature_names.index('num_query_A')
            self.pos_T = self.expanded_feature_names.index('num_query_T')
        if 'num_query_G' in self.expanded_feature_names and 'num_query_C' in self.expanded_feature_names:
            self.pos_G = self.expanded_feature_names.index('num_query_G')
            self.pos_C = self.expanded_feature_names.index('num_query_C')

        # used for testing; contains the interval selected from each contig longer than #self.max_len
        self.intervals = []
        # used for testing; tests set this to false in order to make the interval predictable for contigs
        # shorter than max_len
        self.translate_short_contigs = True

    def on_epoch_end(self):
        """
        Re-shuffle the training data on each epoch. When data is being cached in memory, the class will always use
        the same negative samples (to avoid loading new samples from disk). When data is loaded from disk, the negative
        samples will change at the end of each epoch (assuming fraq_neg < 1).
        """
        np.random.shuffle(self.negative_idx)
        negative_count = int(self.fraq_neg * len(self.negative_idx))
        negative_idx = self.negative_idx[:negative_count]
        self.indices = []
        for _ in range(self.num_translations):
            self.indices += self.positive_idx
            self.indices += negative_idx

        np.random.shuffle(self.indices)
        if self.do_cache:
            self.cache_indices = np.arange(len(self))
            np.random.shuffle(self.cache_indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        x, y = self._get_data(index)
        # if self.pos_A > 0 and self.pos_T > 0 and random.randint(0, 1) == 1:
        #     x[:, [self.pos_A, self.pos_T]] = x[:, [self.pos_T, self.pos_A]]
        #     if self.pos_ref >= 0:
        #         x[:, [self.pos_ref, self.pos_ref + 3]] = x[:, [self.pos_ref + 3, self.pos_ref]]
        # if self.pos_G > 0 and self.pos_C > 0 and random.randint(0, 1) == 1:
        #     x[:, [self.pos_C, self.pos_G]] = x[:, [self.pos_G, self.pos_C]]
        #     if self.pos_ref >= 0:
        #         x[:, [self.pos_ref + 1, self.pos_ref + 2]] = x[:, [self.pos_ref + 2, self.pos_ref + 1]]
        return x, y

    def select_intervals(contig_data: list[ContigInfo], max_len: int, translate_short_contigs: bool,
                         max_translation_bases: int):
        """
        Selects intervals from contigs such that the breakpoints are within the interval.
        For contigs shorter than #max_len, the entire contig is selected.
        For contigs with no mis-assemblies, a random interval is selected.
        For contigs with mis-assemblies, an interval is selected such that the first breakpoint is within the interval.
        Returns: a list of intervals, one per contig
        """
        result = []
        for cd in contig_data:
            start_idx = 0
            end_idx = cd.length
            min_padding = 50  # minimum amount of bases to keep around the breakpoint

            if cd.length > max_len:
                # when no breakpoints are present, we can choose any segment within the contig
                # however, if the contig contains a breakpoint, we must choose a segment that includes the breakpoint
                if not cd.breakpoints:
                    start_idx = np.random.randint(cd.length - max_len + 1)
                else:  # select an interval that contains the first breakpoint
                    # TODO: add one item for each breakpoint
                    lo, hi = cd.breakpoints[0]
                    if max_len >= min_padding + (hi - lo):
                        start_lo = max(0, hi - max_len + min_padding)
                        start_hi = min(cd.length - max_len, lo - min_padding)
                        start_idx = np.random.randint(start_lo, start_hi)
                    else:
                        pass  # corner case for tiny tiny max-len, probably never reached
                end_idx = start_idx + max_len
            elif translate_short_contigs:
                if cd.breakpoints:
                    # we have a mis-assembled contig which is shorter than max_len; pick a random starting point
                    # before the breaking point or shift the contig to the right to enforce some translation invariance
                    lo, hi = cd.breakpoints[0]
                    if True:  # np.random.randint(0, 2) == 0:  # flip a coin for left/right shift
                        # in this case, the contig will be left-truncated
                        start_idx = np.random.randint(0, min(max_translation_bases + 1, max(1, lo - min_padding)))
                    else:
                        # end_idx will be larger than cd.length, which signals that the contig needs to be padded with
                        # start_idx zeros to the left
                        start_idx = np.random.randint(0, max(1, min(lo - min_padding, max_len - cd.length)))
                        end_idx = start_idx + cd.length
                # # we need to also shift negative samples, otherwise the network learns that samples starting with zero
                # #  (or ending with zero) are the positive samples and reach perfect training scores and horrible
                # validation scores
                else:
                    start_idx = np.random.randint(0, max_translation_bases + 1)
            result.append((start_idx, end_idx))
        return result

    def _get_data(self, index):
        """
        Return the next mini-batch of size #batch_size
        Parameters:
            - index: the mini-batch to return
        """
        start = timer()
        self.intervals.clear()
        if self.do_cache and index in self.cache:
            if self.show_progress:
                utils.update_progress(index + 1, len(self), 'Training: ', '')
            return self.cache[self.cache_indices[index]]
        batch_indices = self.indices[self.batch_size * index:  self.batch_size * (index + 1)]
        # files to process
        contig_data: list[ContigInfo] = [self.reader.contigs[i] for i in batch_indices]
        y = np.zeros(self.batch_size)
        for i, idx in enumerate(batch_indices):
            y[i] = 0 if self.reader.contigs[idx].misassembly == 0 else 1

        features_data = self.reader.read_contigs(contig_data)
        max_contig_len = max([self.reader.contigs[i].length for i in batch_indices])
        max_len = min(max_contig_len, self.max_len)
        # Create the numpy array storing all the features for all the contigs in #batch_indices
        x = np.zeros((self.batch_size, max_len, len(features_data[0])))

        contig_intervals = BinaryDatasetTrain.select_intervals(contig_data, max_len, self.translate_short_contigs,
                                                               self.max_translation_bases)
        for i, contig_features in enumerate(features_data):
            contig_len = contig_data[i].length
            to_merge = [None] * len(self.expanded_feature_names)
            start_idx, end_idx = contig_intervals[i]
            length = end_idx - start_idx
            for j, feature_name in enumerate(self.expanded_feature_names):
                if end_idx <= contig_len:
                    to_merge[j] = contig_features[feature_name][start_idx:end_idx]
                else:  # contig will be left-padded with zeros
                    assert contig_len == end_idx - start_idx, f'Contig len is {contig_len}, ' \
                                                              f'st-end are {start_idx}-{end_idx}'
                    to_merge[j] = contig_features[feature_name][0:min(max_len - start_idx, contig_len)]
            stacked_features = np.stack(to_merge, axis=-1)  # each feature becomes a column in x[i]
            if end_idx <= contig_len:
                x[i][:length, :] = stacked_features
            else:
                x[i][start_idx:min(max_len, end_idx), :] = stacked_features
        if self.do_cache:
            self.cache[index] = (x, np.array(y))
        if self.show_progress:
            utils.update_progress(index + 1, self.__len__(), 'Training: ', f' {(timer() - start):5.2f}s')
        return x, np.array(y)


class BinaryDatasetEval(BinaryDataset):
    def __init__(self, reader: contig_reader, indices: list[int], feature_names: list[str], window: int, step: int,
                 total_memory_bytes: int, cache_results: bool, show_progress: bool):

        """
        Arguments:
            reader - ContigReader instance used to load data from disk
            indices - the indices of the contigs to be loaded for evaluation
            feature_names - features to load, e.g. ['coverage', 'num_proper_Match']
            window - the maximum contig length; if a contig is longer than #window, it will be chopped
                into smaller pieces at #step intervals
            step - amount by which the sliding window is moved forward through a contig that is longer than #window
            nprocs - number of processes used to load the contig data
            total_memory_bytes - maximum memory desired for contig features in a mini-batch
            cache_results
            show_progress - if true, a progress bar will show the evaluation progress
        """
        logging.info(f'Creating evaluation data generator. Window: {window}, Step: {step}, Caching: {cache_results}')
        BinaryDataset.__init__(self, reader, feature_names)
        indices = sorted(indices, key=lambda x: reader.contigs[x].length)
        self.indices = indices
        self.window = window
        self.step = step
        # creates batches of contigs such that the total memory used by features in each batch is < total_memory_bytes
        # chunk_counts[batch_count][idx] represents the number of chunks for the contig number #idx
        # in the batch #batch_count
        self.batch_list, self.chunk_counts = self._create_batch_list(reader.contigs, indices, total_memory_bytes)

        # flattened ground truth for each eval contig
        self.y = [0 if self.reader.contigs[i].misassembly == 0 else 1 for b in self.batch_list for i in b]
        # the cached results
        self.cache_results = cache_results
        self.show_progress = show_progress
        if cache_results:
            self.data = [None] * len(self.batch_list)

    def _create_batch_list(self, contig_data: list[ContigInfo], indices: list[int], total_memory_bytes: int):
        """ Divide the validation indices into mini-batches of total size < #total_memory_bytes """
        # there seems to be an overhead for each position; 10 is just a guess to avoid running out of memory
        # on the GPU
        bytes_per_base = 10 + 4 * len(self.feature_names)  # assume each feature is a 4-byte float

        current_indices = []
        batch_list = []
        batch_chunk_count = 0  # total number of contig chunks in the current batch
        chunk_counts = []  # number of chunks in each batch
        counts = []
        for idx in indices:
            contig_len = contig_data[idx].length
            # number of chunks for the current contig
            curr_chunk_count = 1 + max(0, math.ceil((contig_len - self.window) / self.step))
            batch_chunk_count += curr_chunk_count
            # check if the new contig still fits in memory; create a new batch if not
            if current_indices and batch_chunk_count * self.window * bytes_per_base > total_memory_bytes:
                logging.debug(f'Added {len(counts)} contigs with {sum(counts)} chunks to evaluation batch')
                batch_list.append(current_indices)
                current_indices = []
                chunk_counts.append(counts)
                counts = []
                batch_chunk_count = curr_chunk_count

            counts.append(curr_chunk_count)
            current_indices.append(idx)
        batch_list.append(current_indices)
        chunk_counts.append(counts)

        return batch_list, chunk_counts

    def group(self, y, method=max):
        """
        Groups results for contigs chunked into multiple windows (because they were too long)
        by selecting the maximum value for each window
        Params:
            - y: the results for the chunked contigs that need to be grouped. If any chunk of a contig was deemed
                 mis-assembled, then the entire contig is marked as mis-assembled
        Returns:
            - the grouped y, containing one entry for each contig in #self.indices
        """
        total_len = 0
        grouped_y_size = 0
        for batch in self.chunk_counts:
            total_len += sum(batch)
            grouped_y_size += len(batch)
        assert len(y) == total_len, f'y has length {len(y)}, chunk_counts total length is {total_len}'
        assert grouped_y_size == len(self.indices), \
            f'Index map has {grouped_y_size} elements, while indices has {len(self.indices)} elements'
        grouped_y = np.zeros(len(self.indices))
        i = 0
        j = 0
        for batch in self.chunk_counts:
            for chunk_count in batch:
                grouped_y[i] = method(y[j:j + chunk_count])
                i += 1
                j += chunk_count
        return grouped_y

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, batch_idx: int):
        """ Return the mini-batch at index #index """
        start = timer()
        # files to process
        indices = self.batch_list[batch_idx]

        stack_time = 0
        if self.cache_results and self.data[batch_idx] is not None:
            all_stacked_features = self.data[batch_idx]
        else:
            contig_data: list[ContigInfo] = [self.reader.contigs[i] for i in indices]
            all_stacked_features = [None] * len(contig_data)
            features_data = self.reader.read_contigs(contig_data)
            assert len(features_data) == len(contig_data)
            to_merge = [None] * len(self.expanded_feature_names)

            start_stack = timer()
            for i, contig_features in enumerate(features_data):
                contig_len = contig_data[i].length
                assert contig_len == len(contig_features[self.expanded_feature_names[0]])
                # each feature in features_data becomes o column in x
                for j, feature_name in enumerate(self.expanded_feature_names):
                    to_merge[j] = contig_features[feature_name]
                stacked_features = np.stack(to_merge, axis=-1)
                all_stacked_features[i] = stacked_features
            stack_time += (timer() - start_stack)
            if self.cache_results:
                self.data[batch_idx] = all_stacked_features

        max_contig_len = max([self.reader.contigs[i].length for i in indices])
        max_len = min(max_contig_len, self.window)

        x = []  # the evaluation data for all contigs in this batch
        # traverse all contig features, break down into multiple contigs if too long, and create a numpy 3D array
        # of shape (contig_count, max_len, num_features) to be used for evaluation
        for stacked_features in all_stacked_features:
            contig_len = stacked_features.shape[0]
            start_idx = 0
            count = 0
            while True:
                end_idx = start_idx + self.window
                if end_idx <= contig_len:
                    x.append(stacked_features[start_idx:end_idx])
                else:
                    np_data = np.zeros((max_len, len(self.expanded_feature_names)))
                    valid_end = min(end_idx, contig_len)
                    np_data[:valid_end - start_idx] = stacked_features[start_idx:valid_end]
                    x.append(np_data)
                start_idx += self.step

                count += 1
                if end_idx >= contig_len:
                    break
        assert (len(x) == sum(self.chunk_counts[batch_idx])), f'{len(x)} vs {sum(self.chunk_counts[batch_idx])}'
        if self.show_progress:
            utils.update_progress(batch_idx + 1, self.__len__(), 'Evaluating: ',
                                  f' {(timer() - start):5.2f}s  {stack_time:5.2f}s')
        return np.array(x)


class GeneratorBigD(tf.keras.utils.Sequence):
    def __init__(self, data_dict, max_len, batch_size,
                 shuffle_data=True, fraq_neg=1, rnd_seed=None, nprocs=4):
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.shuffle_data = shuffle_data
        # self.shuffle = False
        # self.n_feat = 28 #todo: features_sel
        self.rnd_seed = rnd_seed
        self.nprocs = nprocs
        self.time_load = 0

        self.count_epoch = -1

        if self.rnd_seed:
            np.random.seed(self.rnd_seed)
            tf.random.set_seed(self.rnd_seed)

        if self.shuffle_data:
            all_labels = utils.read_all_labels(self.data_dict)
            self.inds_pos = np.arange(len(all_labels))[np.array(all_labels) == 1]
            self.inds_neg = np.arange(len(all_labels))[np.array(all_labels) == 0]
            self.fraq_neg = fraq_neg  # downsampling
            self.num_neg = int(self.fraq_neg * len(self.inds_neg))
            self.size = self.num_neg + len(self.inds_pos)
            self.on_epoch_end()
        else:
            self.indices = np.arange(len(self.data_dict))
            self.size = len(self.indices)

        logging.info('init generator')

    def on_epoch_end(self):
        """
        Reshuffle when epoch ends
        """
        self.count_epoch += 1

        if self.shuffle_data:
            if self.count_epoch % 1 == 0:  # todo: this IF was implemented to update negative class every n epochs
                logging.info('self.count_epoch: {}'.format(self.count_epoch))
                logging.info("downsample over-represented class by  {}".format(self.fraq_neg))
                np.random.shuffle(self.inds_neg)
                self.indices = np.concatenate((self.inds_pos, self.inds_neg[:self.num_neg]))

            logging.info("shuffle")
            np.random.shuffle(self.indices)
            # we do not shuffle and do not downsample for test and validation data

        # logging.info("len(self.indices) {}".format(len(self.indices)))
        # logging.info("self.indices[:10] {}".format(self.indices[:10]))

    def generate(self, indices_tmp):
        """
        Generate new mini-batch
        """

        sample_keys = np.array(list(self.data_dict.keys()))[indices_tmp]
        # files to process
        files_dict = itertoolz.groupby(lambda t: t[1],
                                       list(map(lambda s: (s, self.data_dict[s]), sample_keys)))  # itertoolz.
        # for every file, associate a random number, which can be used to construct random number to sample a range

        file_seeds = np.random.randint(0, 1000000, len(files_dict.items()))

        file_items = [(k, v, s) for (k, v), s in zip(files_dict.items(), file_seeds)]

        # start_load = time.time()
        X, y = utils.file_reading(file_items, self.max_len)

        # duration_load = time.time() - start_load
        # logging.info("time to read one batch {} {}".format(len(indices_tmp), duration_load))

        max_contig_len = max(list(map(len, [x_i for x_i in X])))
        mb_max_len = min(max_contig_len, self.max_len)  # todo: fixed length NN self.max_len
        n_feat = X[0].shape[1]
        x_mb = np.zeros((len(indices_tmp), mb_max_len, n_feat))

        for i, x_i in enumerate(X):
            x_mb[i, 0:x_i.shape[0]] = x_i

        y_mb = y
        return np.array(x_mb), np.array(y_mb)

    def __len__(self):
        return int(np.ceil(self.size / self.batch_size))

    def __getitem__(self, index):
        """
        Get new mb
        """
        start_time = time.time()
        if self.batch_size * (index + 1) < len(self.indices):
            indices_tmp = \
                self.indices[self.batch_size * index: self.batch_size * (index + 1)]
        else:
            indices_tmp = \
                self.indices[self.batch_size * index:]

        # logging.info("generate batch {}".format(index))
        x_mb, y_mb = self.generate(indices_tmp)
        if index % 50 == 0:  # to see some progress
            logging.info("new batch {}".format(index))
            # print("--- {:.1f} seconds ---".format(time.time() - start_time))
        # if index == 111:
        #     logging.info("index {}".format(index))
        #     logging.info("x_mb[:2] {}".format(x_mb[:2]))
        #     logging.info("y_mb[:10] {}".format(y_mb[:10]))
        return x_mb, y_mb


class GeneratorPredLong(tf.keras.utils.Sequence):
    def __init__(self, data_dict, batch_list, window, step,
                 nprocs):  # data_dict contains all data, because indexes are global
        self.data_dict = data_dict
        self.batch_list = batch_list
        self.window = window
        self.step = step
        # self.n_feat = 28 #todo: features_sel
        self.all_lens = utils.read_all_lens(data_dict)
        self.indices = np.arange(len(batch_list))
        self.nprocs = nprocs

    def generate(self, ind):
        sample_keys = np.array(list(self.data_dict.keys()))[self.batch_list[ind]]
        files_dict = itertoolz.groupby(lambda t: t[1], list(
            map(lambda s: (s, self.data_dict[s]), sample_keys)))  # itertoolz.
        # attention: grouping can change order, it is important that indices are sorted
        X = utils.load_full_contigs(files_dict)

        batch_size = 0
        for cont_ind in self.batch_list[ind]:
            batch_size += 1 + utils.n_moves_window(self.all_lens[cont_ind], self.window, self.step)

        x_mb = utils.gen_sliding_mb(X, batch_size, self.window, self.step)
        return x_mb

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, index):
        x_mb = self.generate(index)
        if index % 50 == 0:  # to see some progress
            logging.info("new batch {}".format(index))
        return x_mb


class GeneratorFullLen(tf.keras.utils.Sequence):
    def __init__(self, data_dict, batch_list, nprocs):  # data_dict contains all data, because indexes are global
        self.data_dict = data_dict
        self.batch_list = batch_list
        # self.n_feat = 28 #todo: features_sel
        self.all_lens = utils.read_all_lens(data_dict)
        self.indices = np.arange(len(batch_list))
        self.nprocs = nprocs

    def generate(self, ind):
        sample_keys = np.array(list(self.data_dict.keys()))[self.batch_list[ind]]
        files_dict = itertoolz.groupby(lambda t: t[1], list(
            map(lambda s: (s, self.data_dict[s]), sample_keys)))  # itertoolz.
        X = utils.load_full_contigs(files_dict)

        max_len = max([self.all_lens[cont_ind] for cont_ind in self.batch_list[ind]])
        n_feat = X[0].shape[1]
        x_mb = np.zeros((len(self.batch_list[ind]), max_len, n_feat))

        for i, xi in enumerate(X):
            x_mb[i, 0:xi.shape[0]] = xi  # padding is happenning here
        return x_mb

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, index):
        x_mb = self.generate(index)
        if index % 50 == 0:  # to see some progress
            logging.info("new batch {}".format(index))
        return x_mb


class Generator_v1(tf.keras.utils.Sequence):
    def __init__(self, data_dict, batch_list, window, step,
                 nprocs):  # data_dict contains all data, because indexes are global
        self.data_dict = data_dict
        self.batch_list = batch_list
        self.window = window
        self.step = step
        # self.n_feat = 28 #todo: features_sel
        self.all_lens = utils.read_all_lens(data_dict)
        self.indices = np.arange(len(batch_list))
        self.nprocs = nprocs

    def generate(self, ind):
        sample_keys = np.array(list(self.data_dict.keys()))[self.batch_list[ind]]
        files_dict = itertoolz.groupby(lambda t: t[1], list(
            map(lambda s: (s, self.data_dict[s]), sample_keys)))  # itertoolz.
        X = utils.load_full_contigs(files_dict)

        batch_size = 0
        for cont_ind in self.batch_list[ind]:
            batch_size += 1 + utils.n_moves_window(self.all_lens[cont_ind], self.window, self.step)

        x_mb = utils.gen_sliding_mb(X, batch_size, self.window, self.step)
        x_mb = np.expand_dims(x_mb, -1)
        return x_mb

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, index):
        x_mb = self.generate(index)
        if index % 50 == 0:  # to see some progress
            logging.info("new batch {}".format(index))
        return x_mb