# import
## batteries
import logging
from toolz import itertoolz
import pathos
import tables
import time
## 3rd party
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, Dropout, Dense
## application
from DeepMAsED import Utils

class deepmased(object):
    """
    Implements a convolutional network for misassembly prediction. 
    """
    def __init__(self, config):
        self.max_len = config.max_len
        self.filters = config.filters
        self.n_conv = config.n_conv
        self.n_features = config.n_features
        # self.pool_window = config.pool_window
        self.dropout = config.dropout
        self.lr_init = config.lr_init
        self.n_fc = config.n_fc
        self.n_hid = config.n_hid

        #self.net = Sequential()
        inlayer = Input(shape=(None, self.n_features), name='input')

        x = Conv1D(self.filters, kernel_size=(10),
                            input_shape=(None, self.n_features),
                            activation='relu', padding='valid', name='1st_conv')(inlayer)
       # x = BatchNormalization(axis=-1)(x) todo: bn
       #  x = Dropout(rate=self.dropout)(x)

        for i in range(1, self.n_conv-1):
            x = Conv1D(2 ** i * self.filters, kernel_size=(5),
                                strides=1, dilation_rate=2,
                                activation='relu')(x)
            # x = BatchNormalization(axis=-1)(x) todo: bn
            x = Dropout(rate=self.dropout)(x)
            
        x = Conv1D(2 ** self.n_conv * self.filters, kernel_size=(3),
                    strides=1, dilation_rate=2,
                    activation='relu')(x)
        # x = BatchNormalization(axis=-1)(x) todo: bn
        x = Dropout(rate=self.dropout)(x)
        
        maxP = GlobalMaxPooling1D()(x)
        avgP = GlobalAveragePooling1D()(x)
        x = concatenate([maxP, avgP])

        optimizer = tf.keras.optimizers.Adam(lr=self.lr_init)

        for _ in range(self.n_fc):
            x = Dense(self.n_hid, activation='relu')(x)
            x = Dropout(rate=self.dropout)(x)

        x = Dense(1, activation='sigmoid')(x)

        #MirroredStrategy is used instead
        # if self.n_gpu>1:
        #     self.net = Utils.make_parallel(Model(inputs=inlayer,outputs=x), self.n_gpu)
        # else:
        #     self.net = Model(inputs=inlayer,outputs=x)
        self.net = Model(inputs=inlayer, outputs=x)
        self.net.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=[Utils.class_recall_0, Utils.class_recall_1])


        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                               monitor='val_loss', factor=0.8,
                               patience=5, min_lr = 0.01 * self.lr_init)

    def predict(self, x):
        return self.net.predict(x)

    def predict_generator(self, x):
        return self.net.predict_generator(x)

    def print_summary(self):
        print(self.net.summary())

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
        max_contig_len = max(list(map(len,[self.x[ind] for ind in indices_tmp])))
        mb_max_len = min(max_contig_len, self.max_len)
      
        x_mb = np.zeros((len(indices_tmp), mb_max_len, self.n_feat))

        for i, idx in enumerate(indices_tmp):
            if self.x[idx].shape[0]<=mb_max_len:
                x_mb[i, 0:self.x[idx].shape[0]] = self.x[idx]
            else:
                #cut chunk
                start_pos = np.random.randint(self.x[idx].shape[0]-mb_max_len+1)
                x_mb[i, :] = self.x[idx][start_pos:start_pos+mb_max_len,:] 

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
              self.indices[self.batch_size * index : self.batch_size * (index + 1)]
        else:
            indices_tmp = \
              self.indices[self.batch_size * index : ]           
        x_mb, y_mb = self.generate(indices_tmp)
        return x_mb, y_mb



class GeneratorBigD(tf.keras.utils.Sequence):
    def __init__(self, data_dict, max_len, batch_size,
                 shuffle=True, fraq_neg=1, rnd_seed=None, nprocs=4):
        self.max_len = max_len
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.shuffle = shuffle
        self.n_feat = 21
        self.rnd_seed = rnd_seed
        self.nprocs = nprocs
        self.time_load = 0

        all_labels = Utils.read_all_labels(self.data_dict)
        self.inds_pos = np.arange(len(all_labels))[np.array(all_labels) == 1]
        self.inds_neg = np.arange(len(all_labels))[np.array(all_labels) == 0]
        self.fraq_neg = fraq_neg  #downsampling
        self.num_neg = int(self.fraq_neg*len(self.inds_neg))
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Reshuffle when epoch ends
        """
        if self.shuffle:
            logging.info("shuffle and downsample over-represented class by  {}".format(self.fraq_neg))
            np.random.shuffle(self.inds_neg)
            self.indices = np.concatenate((self.inds_pos, self.inds_neg[:self.num_neg]))
            np.random.shuffle(self.indices)
        else:
            #we do not shuffle and do not downsample for test and validation data
            self.indices = np.arange(len(self.data_dict))
        logging.info("len(self.indices) {}".format(len(self.indices)))
        
            
    def generate(self, indices_tmp):
        """
        Generate new mini-batch
        """

        sample_keys = np.array(list(self.data_dict.keys()))[indices_tmp]
        # files to process
        files_dict = itertoolz.groupby(lambda t: t[1], list(itertoolz.map(lambda s: (s, self.data_dict[s]), sample_keys)))
        # for every file, associate a random number, which can be used to construct random number to sample a range
        if self.rnd_seed:
            np.random.seed(self.rnd_seed)
        file_seeds = np.random.randint(0, 1000000, len(files_dict.items()))
        
        file_items = [(k, v, s) for (k ,v), s in zip(files_dict.items(), file_seeds)]

        start_load = time.time()
        X, y = Utils.file_reading(file_items, self.max_len)

        # duration_load = time.time() - start_load
        # logging.info("time to read one batch {} {}".format(len(indices_tmp), duration_load))

        max_contig_len = max(list(map(len,[x_i for x_i in X])))
        mb_max_len = min(max_contig_len, self.max_len)
      
        x_mb = np.zeros((len(indices_tmp), mb_max_len, self.n_feat))

        for i, x_i in enumerate(X):
            x_mb[i, 0:x_i.shape[0]] = x_i

        y_mb = y
        return np.array(x_mb), np.array(y_mb)

            
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """
        Get new mb
        """
        if self.batch_size * (index + 1) < len(self.indices):
            indices_tmp = \
              self.indices[self.batch_size * index : self.batch_size * (index + 1)]
        else:
            indices_tmp = \
              self.indices[self.batch_size * index : ]

        # logging.info("generate batch {}".format(index))
        x_mb, y_mb = self.generate(indices_tmp)
        if index%20==0: #to see some progress
            logging.info("new batch {}".format(index))
        return x_mb, y_mb
