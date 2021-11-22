import numpy as np
import pytest
import unittest
from unittest.mock import patch, MagicMock

from resmico import models_fl
from resmico import contig_reader
from resmico import reader

from resmico.contig_reader import ContigInfo


class TestBinaryDatasetTrain(unittest.TestCase):
    def test_select_intervals(self):
        contig_data = [ContigInfo('Contig1', '/tmp/c1', 1000, 0, 0, 0, []),
                       ContigInfo('Contig2', '/tmp/c2', 1000, 0, 0, 0, [(100, 100)]),
                       ContigInfo('Contig3', '/tmp/c3', 1000, 0, 0, 0, [(800, 900)])]
        max_len = 500
        for i in range(50):
            intervals = models_fl.BinaryDatasetTrain.select_intervals(contig_data, max_len,
                                                                      translate_short_contigs=True,
                                                                      max_translation_bases=30)
            self.assertTrue(0 <= intervals[0][0] <= 500)
            self.assertTrue(500 <= intervals[0][1] <= 1000)
            self.assertTrue(0 <= intervals[1][0] <= 50)
            self.assertTrue(500 <= intervals[1][1] <= 550)
            self.assertTrue(450 <= intervals[2][0] <= 500, f'Start is {intervals[2][0]}')
            self.assertTrue(500 <= intervals[2][1] <= 1000)

    def test_select_intervals_translate_short(self):
        contig_data = [
            ContigInfo('Contig1', '/tmp/c1', 300, 0, 0, 0, [(200, 210)]),
        ]
        max_len = 350
        for i in range(50):
            intervals = models_fl.BinaryDatasetTrain.select_intervals(contig_data, max_len, True, 30)
            if intervals[0][1] - intervals[0][0] < 300:  # contig was truncated to left
                self.assertTrue(0 <= intervals[0][0] <= 150)
                self.assertEqual(300, intervals[0][1])
            else:  # contig will be shifted to left
                self.assertTrue(0 <= intervals[0][0] <= 50)
                self.assertEqual(300 + intervals[0][0], intervals[0][1],
                                 f'Intervals are: {intervals[0][0]}  {intervals[0][1]}')

    @patch('resmico.models_fl.BinaryDatasetTrain.select_intervals')
    def test_contig_selection(self, mock_intervals):
        """
        Make sure that the returned contig features (when using translations) are correct.

        The test mocks reader.read_contigs and BinaryDatasetTrain.select_intervals, and checks if the returned
        (x,y) tuple of BinaryDatasetTrain.__get_item__() is correct.
        """
        for use_cython in [False, True]:
            for cached in [False, True]:
                features = ['num_query_A', 'coverage', 'num_SNPs']
                ctg_reader = contig_reader.ContigReader('data/preprocess/', features, 1, use_cython)
                c1 = ContigInfo('Contig1', '/tmp/c1', 500, 0, 0, 0, [])
                c2 = ContigInfo('Contig2', '/tmp/c2', 300, 0, 0, 1, [(100, 100)])
                c3 = ContigInfo('Contig3', '/tmp/c3', 1000, 0, 0, 1, [(800, 900)])
                ctg_reader.contigs = [c1, c2, c3]

                contigs_data = []
                st = 0
                for c in ctg_reader.contigs:
                    contig_data = {}
                    for f in features:
                        feature_data = np.arange(start=st, stop=st + c.length, dtype=float)
                        contig_data[f] = feature_data
                        st += c.length
                    contigs_data.append(contig_data)
                ctg_reader.read_contigs = MagicMock(
                    return_value=[contigs_data[0], contigs_data[0], contigs_data[0], contigs_data[1], contigs_data[1],
                                  contigs_data[1], contigs_data[2], contigs_data[2], contigs_data[2]])

                indices = np.arange(len(ctg_reader))
                batch_size = 10
                max_len = 500
                data_gen = models_fl.BinaryDatasetTrain(ctg_reader, indices, batch_size, features, max_len,
                                                        num_translations=3, max_translation_bases=10, fraq_neg=1.0,
                                                        do_cache=cached, show_progress=False,
                                                        convoluted_size=(lambda x, pad: x))
                data_gen.indices.sort()  # indices will now be 0,0,0,1,1,1,2,2,2
                self.assertEqual(9, len(data_gen.indices))  # we have 3 contigs, each translated 3 times
                self.assertEqual([0, 0, 0, 1, 1, 1, 2, 2, 2], data_gen.indices)
                mock_intervals.return_value = [
                    # 1st contig: full contig, cropped to left 100, cropped to left 10
                    (0, 500), (100, 500), (10, 500),
                    (0, 300), (50, 300), (40, 300),  # 2nd contig
                    (500, 1000), (450, 950), (440, 900)  # 3rd contig
                ]

                (x, mask), y = data_gen.__getitem__(0)
                self.assertEqual((batch_size, max_len, len(features)), x.shape)
                # the last zero is just padding
                self.assertIsNone(np.testing.assert_array_equal([0, 0, 0, 1, 1, 1, 1, 1, 1, 0], y))

                # first contig, 1st translation (no translation, full contig)
                for i in range(500):
                    self.assertEqual(i, x[0][i][0])
                    self.assertEqual(500 + i, x[0][i][1])
                    self.assertEqual(1000 + i, x[0][i][2])
                # first contig, 2nd translation (shift 100 to left)
                for i in range(400):
                    self.assertEqual(100 + i, x[1][i][0])
                    self.assertEqual(600 + i, x[1][i][1])
                    self.assertEqual(1100 + i, x[1][i][2])
                for i in range(400, 500):
                    for j in range(3):
                        self.assertEqual(0, x[1][i][j])

                # first contig, 2nd translation (shift 10 to left)
                for i in range(490):
                    self.assertEqual(10 + i, x[2][i][0])
                    self.assertEqual(510 + i, x[2][i][1])
                    self.assertEqual(1010 + i, x[2][i][2])
                for i in range(490, 500):
                    for j in range(3):
                        self.assertEqual(0, x[2][i][j])

                # 2nd contig 1st translation
                for i in range(300):
                    self.assertEqual(1500 + i, x[3][i][0])
                    self.assertEqual(1800 + i, x[3][i][1])
                    self.assertEqual(2100 + i, x[3][i][2])
                for i in range(300, 500):
                    for j in range(3):
                        self.assertEqual(0, x[3][i][j])

                # 2nd contig 2nd translation (truncate 50 bases from the left)
                for i in range(250):
                    self.assertEqual(1550 + i, x[4][i][0])
                    self.assertEqual(1850 + i, x[4][i][1])
                    self.assertEqual(2150 + i, x[4][i][2])
                for i in range(250, 500):
                    for j in range(3):
                        self.assertEqual(0, x[4][i][j])

                # 2nd contig 3rd translation (truncate 40 bases from the left)
                for i in range(260):
                    self.assertEqual(1540 + i, x[5][i][0])
                    self.assertEqual(1840 + i, x[5][i][1])
                    self.assertEqual(2140 + i, x[5][i][2])
                for i in range(260, 500):
                    for j in range(3):
                        self.assertEqual(0, x[5][i][j])

                # 3rd contig 1st translation (last 500 bases of contig 3)
                for i in range(500):
                    self.assertEqual(2900 + i, x[6][i][0])
                    self.assertEqual(3900 + i, x[6][i][1])
                    self.assertEqual(4900 + i, x[6][i][2])

                # 3rd contig 2nd translation (bases 450 to 950 of contig 3)
                for i in range(500):
                    self.assertEqual(2850 + i, x[7][i][0])
                    self.assertEqual(3850 + i, x[7][i][1])
                    self.assertEqual(4850 + i, x[7][i][2])

                # 3rd contig 3rd translation (bases 440 to 950 of contig 3)
                for i in range(460):
                    self.assertEqual(2840 + i, x[8][i][0])
                    self.assertEqual(3840 + i, x[8][i][1])
                    self.assertEqual(4840 + i, x[8][i][2])

                # padding
                for i in range(500):
                    for j in range(3):
                        self.assertEqual(0, x[9][i][j])

    def test_gen_train_data(self):
        for cached in [False, True]:
            ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
            indices = np.arange(len(ctg_reader))
            batch_size = 10
            num_translations = 1
            data_gen = models_fl.BinaryDatasetTrain(ctg_reader, indices, batch_size, reader.feature_names, 500,
                                                    num_translations=1, max_translation_bases=0, fraq_neg=1.0,
                                                    do_cache=cached, show_progress=False,
                                                    convoluted_size=(lambda x, pad: x))
            data_gen.translate_short_contigs = False  # so that we know which interval is selected
            # set these to -1 in order to enforce NOT swapping A/T and G/C (for data enhancement)
            data_gen.pos_A = data_gen.pos_ref = data_gen.pos_C = -1
            # unshuffle the indices, so that we can make assertions about the returned data
            data_gen.indices = [0, 1]
            self.assertEqual(1, len(data_gen))
            (train_data, mask), y = data_gen[0]
            # even if we only have 2 samples, the remaining are filled with zero to reach the desired batch size
            self.assertEqual(batch_size, len(train_data))

            expected_y = np.zeros(batch_size)
            expected_y[0] = 1
            self.assertIsNone(np.testing.assert_array_equal(y, expected_y))

            # # train_data[0][0] - first position in first contig, train_data[0][5] 5th position in 1st contig
            self.assertIsNone(
                np.testing.assert_array_equal(train_data[0][0][0:6], np.array([1, 0, 0, 0, 2, 1])))
            self.assertIsNone(
                np.testing.assert_array_equal(train_data[0][5][0:6], np.array([1, 0, 0, 0, 0, 0])))

            # # train_data[1][0] - first position in 2nd contig, train_data[1][5] 5th position in 2nd contig
            self.assertIsNone(
                np.testing.assert_array_equal(train_data[1][0][0:6], np.array([1, 0, 0, 0, 1, 1])))
            self.assertIsNone(
                np.testing.assert_array_equal(train_data[1][5][0:6], np.array([1, 0, 0, 0, 0, 0])))


class TestBinaryDatasetEval(unittest.TestCase):
    bytes_per_base = 10 + sum(  # 10 is the overhead also added in Models_Fl.BinaryDataEval
        [np.dtype(ft).itemsize for ft in reader.feature_np_types])

    @pytest.mark.skip(reason="Not sorting by length at the moment.")
    def test_sort_by_contig_len(self):
        """ Make sure that contigs are sorted in increasing order by length """
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        ctg_reader.contigs = [ContigInfo('Contig1', '/tmp/c1', 300, 0, 0, 0, []),
                              ContigInfo('Contig2', '/tmp/c2', 200, 0, 0, 0, []),
                              ContigInfo('Contig3', '/tmp/c3', 100, 0, 0, 0, [])]
        indices = np.arange(len(ctg_reader))

        gpu_memory_bytes = 1010 * self.bytes_per_base
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 500, 250, gpu_memory_bytes,
                                                False, False, convoluted_size=(lambda x, pad: x))
        self.assertEqual(eval_data.indices, [2, 1, 0])

    def test_batching_one_per_batch(self):
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        ctg_reader.contigs = [ContigInfo('Contig1', '/tmp/c1', 1000, 0, 0, 0, []),
                              ContigInfo('Contig2', '/tmp/c2', 1000, 0, 0, 0, []),
                              ContigInfo('Contig3', '/tmp/c3', 1000, 0, 0, 0, [])]
        indices = np.arange(len(ctg_reader))

        gpu_memory_bytes = 1010 * self.bytes_per_base
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 500, 250, gpu_memory_bytes,
                                                False, False, convoluted_size=(lambda x, pad: x))
        self.assertEqual(3, len(eval_data.chunk_counts))
        for i in range(len(eval_data.chunk_counts)):
            self.assertEqual(1, len(eval_data.chunk_counts[i]))
            self.assertEqual(3, eval_data.chunk_counts[i][0])

    def test_batching_multiple_per_batch(self):
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        ctg_reader.contigs = [ContigInfo('Contig1', 'data/preprocess/features_binary', 500, 0, 246, 0, []),
                              ContigInfo('Contig2', 'data/preprocess/features_binary', 500, 246, 183, 0, []),
                              ContigInfo('Contig3', 'data/preprocess/features_binary', 500, 0, 246, 0, [])]
        indices = np.arange(len(ctg_reader))
        gpu_memory_bytes = 1600 * self.bytes_per_base
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 250, 200, gpu_memory_bytes,
                                                False, False, convoluted_size=(lambda x, pad: x))
        # check that Contig1 and Contig2 are in the first batch (with 3 chunks each) and Contig3 is in the second batch
        # (also with 3 chunks)
        # 1st batch, 2 contigs, 3 chunks each
        self.assertEqual(2, len(eval_data.chunk_counts))
        self.assertEqual(2, len(eval_data.chunk_counts[0]))
        self.assertEqual(3, eval_data.chunk_counts[0][0])
        self.assertEqual(3, eval_data.chunk_counts[0][1])
        # 2nd batch, 1 contig, 3 chunks
        self.assertEqual(1, len(eval_data.chunk_counts[1]))
        self.assertEqual(3, eval_data.chunk_counts[1][0])

        (x,_),_ = eval_data[0]
        self.assertEqual(6, len(x))
        (x, _), _ = eval_data[1]
        self.assertEqual(3, len(x))

    def test_gen_eval_data(self):
        for cached in [False, True]:
            ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
            indices = np.arange(len(ctg_reader))
            eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 500, 250, 1e6, cached,
                                                    False, convoluted_size=(lambda x, pad: x))
            self.assertEqual(1, len(eval_data))
            self.assertEqual(2, len(eval_data.batch_list[0]))
            (x,_), _ = eval_data[0]
            self.assertIsNone(
                np.testing.assert_array_equal(x[0][0][0:6], np.array([1, 0, 0, 0, 2, 1])))
            self.assertIsNone(
                np.testing.assert_array_equal(x[0][5][0:6], np.array([1, 0, 0, 0, 0, 0])))

            self.assertTrue(all(a == b for a, b in zip(x[1][0][0:6], [1, 0, 0, 0, 1, 1])))
            self.assertTrue(all(a == b for a, b in zip(x[1][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_gen_eval_data_short_window(self):
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        indices = np.arange(len(ctg_reader))
        window = 50
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, window, 30, 1e6, False,
                                                False, convoluted_size=(lambda x, pad: x))
        feature_count = len(eval_data.expanded_feature_names)

        self.assertEqual(1, len(eval_data))  # one batch total
        self.assertEqual(2, len(eval_data.batch_list[0]))  # the one batch has 2 contigs
        self.assertEqual([16, 16], eval_data.chunk_counts[0])  # each of the contigs of length 500 has 16 chunks

        total_chunks = 16 + 16  # 16 for the first contig of length 500, 16 for the 2nd contig of length 500
        (x, _), _ = eval_data[0]
        self.assertEqual((total_chunks, window, feature_count), x.shape)

        # check the first 6 features in the 0th and 5th positions of the first chunk in first contig (reference_A/C/G/T,
        # coverage, num_query_A)
        (x, _), _ = eval_data[0]
        self.assertTrue(all(a == b for a, b in zip(x[0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(x[0][5][0:6], [1, 0, 0, 0, 0, 0])))

        # check the first 6 features in the 0th and 5th positions of first chunk in 2nd contig
        self.assertTrue(all(a == b for a, b in zip(x[16][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(x[16][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_group(self):
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        indices = np.arange(len(ctg_reader))
        total_memory_bytes = 1e6
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 50, 30, total_memory_bytes,
                                                False, False, convoluted_size=(lambda x, pad: x))
        (x, _), _ = eval_data[0]
        self.assertEqual(32, len(x))

        y = np.zeros(32)
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([0, 0]), eval_data.group(y)))

        y = np.zeros(32)
        y[5] = 1
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1, 0]), eval_data.group(y)))
        y[0:15] = 1
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1, 0]), eval_data.group(y)))

        y[16] = 1
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1, 1]), eval_data.group(y)))

    def test_group_two_batches(self):
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        indices = np.arange(len(ctg_reader))
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 50, 30, 500, False, False,
                                                convoluted_size=(lambda x, pad: x))

        self.assertEqual(2, len(eval_data))
        (x, _), _ = eval_data[0]
        self.assertEqual(16, len(x))
        (x, _), _ = eval_data[1]
        self.assertEqual(16, len(x))

        y = np.zeros(32)
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([0, 0]), eval_data.group(y)))

        y = np.zeros(32)
        y[5] = 1
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1, 0]), eval_data.group(y)))
        y[0:15] = 1
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1, 0]), eval_data.group(y)))

        y[16] = 1
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1, 1]), eval_data.group(y)))

    def test_gen_eval_data_cached(self):
        ctg_reader = contig_reader.ContigReader('data/preprocess/', reader.feature_names, 1, False)
        indices = np.arange(len(ctg_reader))
        eval_data = models_fl.BinaryDatasetEval(ctg_reader, indices, reader.feature_names, 500, 250, 1e6, True, False,
                                                convoluted_size=(lambda x, pad: x))
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        (x, _), _ = eval_data[0]
        self.assertTrue(all(a == b for a, b in zip(x[0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(x[0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(x[1][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(x[1][5][0:6], [1, 0, 0, 0, 0, 0])))


class TestResmico(unittest.TestCase):
    def setUp(self):
        args = MagicMock()
        args.n_hid = 50
        args.net_type = 'cnn_resnet'
        args.filters = 16
        args.features = ['ref_base']
        args.num_blocks = 4
        args.ker_size = 5
        args.lr_init = 1e-3
        self.args = args

    def test_convolved_output_size_no_masking(self):
        self.args.mask_padding = False
        model = models_fl.Resmico(self.args)
        for is_padding in [False, True]:
            for i in range(100, 10):
                self.assertEqual(i, model.convoluted_size(i, is_padding))

    def test_convolved_output_size(self):
        self.args.mask_padding = True
        model = models_fl.Resmico(self.args)
        self.assertIsNotNone(model.convoluted_size)
        self.assertEqual(14, model.convoluted_size(512, False))
        self.assertEqual(78, model.convoluted_size(1024, False))
        self.assertEqual(133, model.convoluted_size(1071, True))
