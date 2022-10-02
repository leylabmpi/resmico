import logging
import math
import time
import random
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, \
    MaxPooling1D, Flatten, Layer
from tensorflow.keras.layers import Conv1D, Dropout, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from tensorflow.python.ops import array_ops

from toolz import itertoolz
from typing import Dict, List

from resmico.contig_reader import ContigReader
from resmico.contig_reader import ContigInfo
from resmico import reader
from resmico import utils


@tf.keras.utils.register_keras_serializable()
class GlobalMaskedMaxPooling1D(GlobalMaxPooling1D):
    """
    Global max pooling operation for 1D temporal data with optional masking.
    """

    def __init__(self, data_format='channels_last', **kwargs):
        super().__init__(data_format=data_format, **kwargs)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            mask = array_ops.expand_dims(
                mask, 2 if self.data_format == 'channels_last' else 1)
            inputs = tf.minimum(inputs, (2 * mask - 1) * np.inf)
            inputs = super().call(inputs)
        return inputs


@tf.keras.utils.register_keras_serializable()
class ArgMaxSumPooling(Layer):
    """
    Reduces an input tensor of shape (batch_size, steps, channels) to a tensor of shape (batch_size, steps),
    by summing the 'maximum features' at each step. A 'maximum feature' is a feature that has the largest value
    across all steps. Note that it is possible that some steps have no maximum features at all, while other steps
    may have multiple maximum features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.data_format = 'channels_last'

    def call(self, inputs, mask=None):
        channel_idx = 2 if self.data_format == 'channels_last' else 1
        time_idx = 1 if self.data_format == 'channels_last' else 2
        max_idx = K.argmax(inputs, axis=time_idx)  # shape (batch_size, channels), values in [0..steps)
        max_idx_mask = K.one_hot(max_idx, inputs.shape[time_idx])  # shape (batch_size, channels, steps)
        max_idx_mask = K.permute_dimensions(max_idx_mask, [0, 2, 1])  # shape (batch_size, steps, channels)

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)  # shape (batch_size, steps)
            mask = array_ops.expand_dims(mask, channel_idx)
            max_idx_mask *= mask  # shape (batch_size, steps, channels)
        masked_input = inputs * max_idx_mask
        input = K.sum(masked_input, axis=channel_idx, keepdims=True)
        return input  # shape (batch_size, steps)

    def compute_mask(self, inputs, mask=None):
        return None


def construct_convolution_lambda(model: Model):
    """
    Builds and returns a lambda that computes the size of the padded/unpadded convolution layer output 
    for the given model. The function traverses the model layer by layer, and updates the lambda function 
    whenever it sees a 1D convolution.
    The returned lambda has two parameters:
      - the contig length (int). The lambda will return the convolved contig length
      - pad (bool): If true, positions that needed padding in order to be convolved are not masked out.
        If false, any position that needed padding in order to be computed will be masked out.
    """
    result = lambda contig_len, pad: contig_len
    for layer in model.layers:
        if isinstance(layer, Conv1D):
            if layer.kernel_size[0] == 1:  # the strided convolution in the residual layer, doesn't affect output size
                continue
            # padding of a layer can be 'valid' or 'same'; if the padding is 'valid' (as in the first convolution
            # layer of ResMiCo), then the output size is shrunk by (kernel_size-1)
            kernel_size = layer.kernel_size[0]
            stride = layer.strides[0]
            dilation_rate = layer.dilation_rate[0]
            output_reduction = (kernel_size - 1) * dilation_rate if layer.padding == 'valid' else 0
            result_n = \
                lambda x, pad, f=result, stride=stride, kernel_size=kernel_size, \
                        dilation_rate=dilation_rate, output_reduction=output_reduction: \
                    math.ceil((f(x, pad) - output_reduction) / stride) if pad \
                        else 1 + (f(x, pad) - kernel_size * dilation_rate) // stride
            result = result_n
    return result


class Resmico(object):
    """
    Implements a convolutional network for mis-assembly prediction.
    """

    def __init__(self, config):
        self.max_len = config.max_len
        self.filters = config.filters
        self.n_conv = config.n_conv
        # ref_base uses one-hot encoding, so needs 4 (i.e. an extra 3) input nodes
        self.n_feat = len(config.features) + (3 if 'ref_base' in config.features else 0)
        # self.pool_window = config.pool_window
        self.dropout = config.dropout
        self.lr_init = config.lr_init
        self.n_fc = config.n_fc
        self.n_hid = config.n_hid
        self.net_type = config.net_type  # 'lstm', 'cnn_globpool', 'cnn_resnet', 'cnn_lstm'
        self.num_blocks = config.num_blocks
        self.ker_size = config.ker_size
        self.seed = config.seed
        mask = None

        tf.random.set_seed(self.seed)

        # function that returns the size of the convoluted output (that goes into the global pooling layers)
        # for an input of given size; only defined for 'cnn_resnet'
        self.convoluted_size = None

        self.fixed_length = self.is_fixed_length(self.net_type)
        inlayer = Input(shape=(self.max_len if self.fixed_length else None, self.n_feat), name='input',
                        dtype='float32')

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
            mask = Input(shape=(None,), name='mask', dtype='bool')
            x = Bidirectional(LSTM(8, return_sequences=True), merge_mode="concat")(inlayer, mask=mask)
            x = Bidirectional(LSTM(8, return_sequences=True, dropout=self.dropout), merge_mode="ave")(x)
            x = Bidirectional(LSTM(16, return_sequences=False, dropout=self.dropout), merge_mode="concat")(x)
            self.convoluted_size = lambda x, pad: x
            
        elif self.net_type == 'gru':
            mask = Input(shape=(None,), name='mask', dtype='bool')
            x = Bidirectional(GRU(8, return_sequences=True), merge_mode="ave")(inlayer, mask=mask)
            x = Bidirectional(GRU(16, return_sequences=False, dropout=self.dropout), merge_mode="concat")(x)
            
            self.convoluted_size = lambda x, pad: x
            
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
            
        elif self.net_type == 'dilate_cnn_resnet_avg':
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(None, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            # for each block group, create the requested number of residual blocks
            for i, num_blocks in enumerate(self._get_blocks(self.num_blocks)):
                for j in range(num_blocks):
                    x = utils.dilated_residual_block(x, dilate=(j == 0 and i != 0), filters=num_filters,
                                             kernel_size=self.ker_size)
                num_filters *= 2

            # lambda function that computes the data size after applying all the convolutional
            # layers with and without padding; if the 'pad' parameter is True, the output is computed
            # for a convolutional layer with padding='same', otherwise for padding='valid'
            # if we don't mask the zero-padded values, the convoluted size can be anything
            tmp_model = Model(inputs=inlayer, outputs=x)  # dummy model used only in next line
            self.convoluted_size = construct_convolution_lambda(tmp_model)
            mask_size = self.convoluted_size(self.max_len, True) if self.fixed_length else None
            mask = Input(shape=(mask_size,), name='mask', dtype='bool')

            x = GlobalAveragePooling1D()(x, mask=mask)
        
        elif self.net_type == 'transformer':
        
            x = utils.transformer_encoder(inlayer, head_size=8, num_heads=1, dropout=0.1)
            x = utils.transformer_encoder(x, head_size=16, num_heads=1, dropout=0.1)
            x = utils.transformer_encoder(x, head_size=32, num_heads=1, dropout=0.1)
            
            tmp_model = Model(inputs=inlayer, outputs=x)  # dummy model used only in next line
            self.convoluted_size = construct_convolution_lambda(tmp_model)
            
            mask_size = self.convoluted_size(self.max_len, True) if self.fixed_length else None
            mask = Input(shape=(mask_size,), name='mask', dtype='bool')
            x = GlobalAveragePooling1D()(x, mask=mask)
            
        elif self.net_type in ['cnn_resnet', 'cnn_resnet_argmax', 'cnn_resnet_avg', 'cnn_resnet_brnn']:
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(None, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            # for each block group, create the requested number of residual blocks
            for i, num_blocks in enumerate(self._get_blocks(self.num_blocks)):
                for j in range(num_blocks):
                    x = utils.residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters,
                                             kernel_size=self.ker_size)
                num_filters *= 2

            # lambda function that computes the data size after applying all the convolutional
            # layers with and without padding; if the 'pad' parameter is True, the output is computed
            # for a convolutional layer with padding='same', otherwise for padding='valid'
            # if we don't mask the zero-padded values, the convoluted size can be anything
            tmp_model = Model(inputs=inlayer, outputs=x)  # dummy model used only in next line
            self.convoluted_size = construct_convolution_lambda(tmp_model)
            
            mask_size = self.convoluted_size(self.max_len, True) if self.fixed_length else None
            mask = Input(shape=(mask_size,), name='mask', dtype='bool')

            if self.net_type == 'cnn_resnet':
                avgP = GlobalAveragePooling1D()(x, mask=mask)
                maxP = GlobalMaskedMaxPooling1D()(x, mask=mask)
                x = concatenate([maxP, avgP])
            elif self.net_type == 'cnn_resnet_argmax':
                x = ArgMaxSumPooling()(x, mask=(mask))  # shape (batch_size, steps)
                x = utils.residual_block(x, downsample=False, filters=16, kernel_size=self.ker_size)
                x = GlobalMaxPooling1D()(x)
            elif self.net_type == 'cnn_resnet_brnn':
                x = Bidirectional(LSTM(16, return_sequences=False), 
                                  merge_mode="concat")(x, mask=mask)
            else:  # cnn_resnet_avg
                x = GlobalAveragePooling1D()(x, mask=mask)

        elif self.net_type == 'fixlen_cnn_resnet':
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(self.max_len, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            for i, num_blocks in enumerate(self._get_blocks(self.num_blocks)):
                for j in range(num_blocks):
                    x = utils.residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters,
                                             kernel_size=self.ker_size)
                num_filters *= 2

            avgP = AveragePooling1D(pool_size=100, padding='valid')(x)
            maxP = MaxPooling1D(pool_size=100, padding='valid')(x)
            x = concatenate([maxP, avgP])
            x = Flatten()(x)

        elif self.net_type == 'fixlen_resnet_maxglob': #used for shap feature importance v2
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(self.max_len, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            for i, num_blocks in enumerate(self._get_blocks(self.num_blocks)):
                for j in range(num_blocks):
                    x = utils.old_residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters,
                                                 kernel_size=self.ker_size)
                num_filters *= 2
                # this is needed only to avoid errors, mask is not used later
            tmp_model = Model(inputs=inlayer, outputs=x)  # dummy model used only in next line
            self.convoluted_size = lambda x, pad: 1
            
            mask_size = self.convoluted_size(self.max_len, True) if self.fixed_length else None
            mask = Input(shape=(mask_size,), name='mask', dtype='bool')
            ###
            x = MaxPooling1D(pool_size=597, padding='valid')(x)
            x = Flatten()(x)
            
        elif self.net_type == 'fixlen_resnet_avgglob': #used for shap feature importance v2
            x = BatchNormalization()(inlayer)
            x = Conv1D(self.filters, kernel_size=10,
                       input_shape=(self.max_len, self.n_feat),
                       padding='valid', name='1st_conv')(x)
            x = utils.relu_bn(x)
            num_filters = self.filters
            for i, num_blocks in enumerate(self._get_blocks(self.num_blocks)):
                for j in range(num_blocks):
                    x = utils.residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters,
                                                 kernel_size=self.ker_size)
                num_filters *= 2
                # this is needed only to avoid errors, mask is not used later
            tmp_model = Model(inputs=inlayer, outputs=x)  # dummy model used only in next line
            self.convoluted_size = lambda x, pad: 1
            mask_size = self.convoluted_size(self.max_len, True) if self.fixed_length else None
            mask = Input(shape=(mask_size,), name='mask', dtype='bool')
            ###
            x = AveragePooling1D(pool_size=1205, padding='valid')(x)
            x = Flatten()(x)

        for _ in range(self.n_fc):
            x = Dense(self.n_hid, activation='relu')(x)
            x = Dropout(rate=self.dropout)(x)

        x = Dense(1, activation='sigmoid')(x)

        # need to clip the gradient for cnn_resnet_avg, otherwise we get NaNs in the weights
        optimizer = tf.keras.optimizers.Adam(lr=self.lr_init, clipnorm=1.0, clipvalue=0.5)
        inputs = [inlayer, mask]
        self.net = Model(inputs=inputs, outputs=x)
        self.net.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=[utils.class_recall_0, utils.class_recall_1])

    @staticmethod
    def _get_blocks(num_blocks: int):
        if num_blocks == 1: #to experiments
            return [1, 2, 2, 2, 1]
        if num_blocks == 3:
            return [2, 5, 2]
        if num_blocks == 4:
            return [2, 5, 5, 2]
        if num_blocks == 5:
            return [2, 3, 5, 5, 2]
        if num_blocks == 6:
            return [2, 3, 5, 5, 3, 2]
        if num_blocks == 7:
            return [2, 4, 4, 2]

    @staticmethod
    def is_fixed_length(network_type: str):
        return network_type in ['fixlen_cnn_resnet', 'cnn_resnet_argmax', 'fixlen_resnet_maxglob', 'fixlen_resnet_avgglob']

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

    def __init__(self, reader: ContigReader, feature_names: List[str], convoluted_size, pad_to_max_len: bool):
        """
       Arguments:
           - reader: ContigReader instance with all the contig metadata
           - feature_names: the names of the features to read and use for training
        """
        self.reader = reader
        self.feature_names = feature_names
        self.expanded_feature_names = feature_names.copy()
        self.convoluted_size = convoluted_size
        self.pad_to_max_len = pad_to_max_len
        if 'ref_base' in self.expanded_feature_names:
            pos = self.expanded_feature_names.index('ref_base')
            self.expanded_feature_names[pos: pos + 1] = ['ref_base_A', 'ref_base_C', 'ref_base_G', 'ref_base_T']

    def get_bytes_per_base(self):
        return sum(
            [np.dtype(reader.feature_np_types[reader.feature_names.index(f)]).itemsize for f in self.feature_names])


class BinaryDatasetTrain(BinaryDataset):
    def __init__(self, reader: ContigReader, indices: List[int], batch_size: int, feature_names: List[str],
                 max_len: int, num_translations: int, max_translation_bases: int, fraq_neg: float, do_cache: bool,
                 show_progress: bool, convoluted_size, pad_to_max_len: bool, weight_factor: int):

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
            - convoluted_size - function that computes the size of the convoluted output for an input of size n
            - pad_to_max_len - if true, all batches will be padded to max-len, even if the longest contig in the batch
              is shorter (this guarantees fixed-length input)
            - weight_factor - if different than 0, contigs are weighed by min(1, contig_len/factor) during training
        """
        BinaryDataset.__init__(self, reader, feature_names, convoluted_size, pad_to_max_len)
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
        self.weight_factor = weight_factor

        if self.do_cache:
            # the cache maps a batch index to feature_name:feature_data pairs
            self.cache: Dict[int, Dict[str, np.array]] = {}
        self.negative_idx = [i for i in indices if reader.contigs[i].misassembly == 0]
        self.positive_idx = [i for i in indices if reader.contigs[i].misassembly != 0]

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

        # the last requested mask and index: used by MaskedDatasetTrain to return the mask
        self.last_mask = None
        self.last_idx = None

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

    def select_intervals(contig_data: List[ContigInfo], max_len: int, translate_short_contigs: bool,
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
            min_size = 2000 #allow chunks shorter than window size
            min_size = min(min_size, max_len) #to pass tests with window < 2k

            if cd.length > max_len:
                # when no breakpoints are present, we can choose any segment within the contig
                # however, if the contig contains a breakpoint, we must choose a segment that includes the breakpoint
                if not cd.breakpoints:
                    start_idx = np.random.randint(cd.length - min_size + 1)
                else:  # select an interval that contains the random breakpoint
                    # TODO: add one item for each breakpoint
                    lo, hi = random.choice(cd.breakpoints)
                    if max_len >= min_padding + (hi - lo):
                        start_lo = max(0, hi - max_len + min_padding)
                        start_hi = min(cd.length - min_size, lo - min_padding)
                        start_idx = np.random.randint(start_lo, start_hi)
                    else:
                        pass  # corner case for tiny tiny max-len, probably never reached
                end_idx = min(start_idx + max_len, end_idx)
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

    def __getitem__(self, index):
        """
        Return the next mini-batch of size #batch_size. The shape of the output is (batch_size, length, feature_count).

        Parameters:
            - index: the index of the mini-batch to return
        """
        start = timer()
        self.intervals.clear()
        if self.do_cache and index in self.cache:
            if self.show_progress:
                utils.update_progress(index + 1, len(self), 'Training: ', '')
            return self.cache[self.cache_indices[index]]
        batch_indices = self.indices[self.batch_size * index:  self.batch_size * (index + 1)]
        # files to process
        contig_data: List[ContigInfo] = [self.reader.contigs[i] for i in batch_indices]
        y = np.zeros(self.batch_size)
        weights = np.ones(self.batch_size, dtype=np.float32)
        for i in range(len(batch_indices)):
            y[i] = 0 if contig_data[i].misassembly == 0 else 1
        if self.weight_factor > 0:
            for i in range(len(batch_indices)):
                weights[i] = min(1, (contig_data[i].length/self.weight_factor)**2)
#                 weights[i] = min(100, (contig_data[i].length/self.weight_factor)**4)

        features_data = self.reader.read_contigs(contig_data)
        max_contig_len = max([self.reader.contigs[i].length for i in batch_indices])
        max_len = self.max_len if self.pad_to_max_len else min(max_contig_len, self.max_len)
#         #TODO
#         max_len += 50
        
        # Create the numpy array storing all the features for all the contigs in #batch_indices
        x = np.zeros((self.batch_size, max_len, len(features_data[0])), dtype=np.float32)
        # it's important to initialize the mask to all ones and then set to zero the padded values rather than the
        # other way around, otherwise we create a mask of all zeros for incomplete batches -> NaN in averaging
        mask = np.ones((self.batch_size, self.convoluted_size(max_len, True)), dtype=np.bool)

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
            x[i][:length, :] = stacked_features
            mask[i][self.convoluted_size(length, False):] = 0
        self.last_mask = mask
        self.last_idx = index
        if self.do_cache:
            self.cache[index] = (x, mask), y, weights
        if self.show_progress:
            utils.update_progress(index + 1, self.__len__(), 'Training: ', f' {(timer() - start):5.2f}s')
        return (x, mask), y, weights


class BinaryDatasetEval(BinaryDataset):
    def __init__(self, reader: ContigReader, indices: List[int], feature_names: List[str], window: int, step: int,
                 total_memory_bytes: int, cache_results: bool, show_progress: bool, convoluted_size,
                 pad_to_max_len: bool, batch_size=300):

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
            cache_results - if true, data is cached in memory after the first evaluation step
            show_progress - if true, a progress bar will show the evaluation progress
            convoluted_size - lambda that computes the size of the convoluted output for an input of size n
            pad_to_max_len - if true, all batches will be padded to max-len, even if the longest contig in the batch
              is shorter (this guarantees fixed-length input)
        """
        logging.info(f'Creating evaluation data generator. Window: {window}, Step: {step}, Caching: {cache_results}')
        BinaryDataset.__init__(self, reader, feature_names, convoluted_size, pad_to_max_len)
        self.indices = indices  # sorted(indices, key=lambda x: reader.contigs[x].length)
        self.window = window
        self.step = step
        # creates batches of contigs such that the total memory used by features in each batch is < total_memory_bytes
        # chunk_counts[batch_count][idx] represents the number of chunks for the contig number #idx
        # in the batch #batch_count
        self.batch_list, self.chunk_counts = self._create_batch_list(reader.contigs, self.indices, 
                                                                     total_memory_bytes, batch_size)

        # flattened ground truth for each eval contig
        self.y = [0 if self.reader.contigs[i].misassembly == 0 else 1 for b in self.batch_list for i in b]
        # the cached results
        self.cache_results = cache_results
        self.show_progress = show_progress
        if cache_results:
            self.data = [None] * len(self.batch_list)

    def _create_batch_list(self, contig_data: List[ContigInfo], indices: List[int], 
                           total_memory_bytes: int, batch_size: int):
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
            if len(current_indices) > batch_size or current_indices and batch_chunk_count * self.window * bytes_per_base > total_memory_bytes:
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

    def group(self, y, method=max, model=None):
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
        if method=='arr':
            grouped_y = []
        else:
            grouped_y = np.zeros(len(self.indices))
        i = 0
        j = 0
        for batch in self.chunk_counts:
            for chunk_count in batch:
                scores = y[j:j + chunk_count]
                if method == 'arr':
                    grouped_y.append(scores.reshape(-1))
                elif method == 'sma':
                    scores = utils.sma(scores, 2)
                    grouped_y[i] = max(scores)
                elif method == 'prob':
                    grouped_y[i] = utils.prob_score_aggr(scores)
                elif method == 'calib_prob':
                    grouped_y[i] = utils.calib_prob_score_aggr(scores, model)
                else:
                    grouped_y[i] = method(scores)
                i += 1
                j += chunk_count
        return grouped_y

    def group_emb(self, y, method=np.mean):
        """
        Groups results for contigs chunked into multiple windows (because they were too long)
        by applying mean across all windows
        Params:
            - y: the results for the chunked contigs that need to be grouped. 
        Returns:
            - the grouped y, containing one vector for each contig in #self.indices
        """
        total_len = 0
        grouped_y_size = 0
        for batch in self.chunk_counts:
            total_len += sum(batch)
            grouped_y_size += len(batch)
        assert len(y) == total_len, f'y has length {len(y)}, chunk_counts total length is {total_len}'
        assert grouped_y_size == len(self.indices), \
            f'Index map has {grouped_y_size} elements, while indices has {len(self.indices)} elements'
        grouped_y = np.empty(len(self.indices), dtype=object)
        i = 0
        j = 0
        for batch in self.chunk_counts:
            for chunk_count in batch:
                if chunk_count > 1:
                    arr = method(y[j:j + chunk_count], axis=0)
                else:
                    arr = y[j]
                grouped_y[i] = '|'.join(str(elem) for elem in arr)
                i += 1
                j += chunk_count
        return grouped_y

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, batch_idx: int):
        """
        Return the mini-batch at index #index
        The function returns a tuple ((x,mask),y), where x has the shape (batch_size, max-contig-len, feature_count) and
        mask has shape (batch_size, max-contig-len). The mask indicates which positions in the output of the last
        convolutional layer are not affected by padding (and should thus be considered by the maxpool and average pool
        layers). The y value, representing the labels) is in this case unused and is a zero array of size batch_size.
        """
        start = timer()
        # files to process
        indices = self.batch_list[batch_idx]

        stack_time = 0
        if self.cache_results and self.data[batch_idx] is not None:
            all_stacked_features = self.data[batch_idx]
        else:
            contig_data: List[ContigInfo] = [self.reader.contigs[i] for i in indices]
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
                stacked_features = np.column_stack(to_merge)
                all_stacked_features[i] = stacked_features.astype(np.float32)  # otherwise it's float64, too much RAM
            stack_time += (timer() - start_stack)
            if self.cache_results:
                self.data[batch_idx] = all_stacked_features

        max_contig_len = max([self.reader.contigs[i].length for i in indices])
        max_len = self.window if self.pad_to_max_len else min(max_contig_len, self.window)
#         #TODO
#         max_len += 50
        
        # the evaluation data for all contigs in this batch
        batch_size = sum(self.chunk_counts[batch_idx])
        x = np.zeros((batch_size, max_len, len(self.expanded_feature_names)), dtype=np.float32)
        # the size of the convoluted output (the output that goes into the max/avg global pooling layer)
        # for the longest contig (including positions that needed partial padding)
        mask = np.zeros((batch_size, self.convoluted_size(max_len, pad=True)), dtype=np.bool)
        # traverse all contig features, break down into multiple contigs if too long, and create a numpy 3D array
        # of shape (batch_size, max_len, num_features) to be used for evaluation
        idx = 0
        for stacked_features in all_stacked_features:
            contig_len = stacked_features.shape[0]
            start_idx = 0
            while True:
                end_idx = start_idx + self.window
                if end_idx < contig_len:
                    x[idx] = stacked_features[start_idx:end_idx] #[:end_idx - start_idx]
                    assert max_len == self.window #+50
                    # keep only positions that didn't need padding in order to be computed (pad=False)
                    mask[idx][:self.convoluted_size(max_len, pad=False)] = 1
                    idx += 1
                else:
#                     ###force at least 5000 bases in the last chunk, as the network hasn't seen contigs shorter than 1K
                    if self.window > 5000 and contig_len > self.window and contig_len - start_idx < 5000:
                        start_idx = contig_len - 5000
#                     if self.window > 1000 and contig_len > self.window and contig_len - start_idx < 1000:
#                         start_idx = contig_len - 1000
                    x[idx][:contig_len - start_idx] = stacked_features[start_idx:contig_len]
                    mask[idx][:self.convoluted_size(contig_len - start_idx, pad=False)] = 1
                    idx += 1
                    break

                start_idx += self.step

        if self.show_progress:
            utils.update_progress(batch_idx + 1, self.__len__(), 'Evaluating: ',
                                  f' {(timer() - start):5.2f}s  {stack_time:5.2f}s')
        return (x, mask), np.zeros(batch_size, dtype=np.bool)
    