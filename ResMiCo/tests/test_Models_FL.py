import numpy as np
import unittest

from .. import Models_FL
from .. import ContigReader
from .. import Reader

from ..ContigReader import ContigInfo


class TestBinaryData(unittest.TestCase):
    def test_gen_eval_data(self):
        for cached in [False, True]:
            reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
            indices = np.arange(len(reader))
            batch_size = 10
            data_gen = Models_FL.BinaryData(reader, indices, batch_size, Reader.feature_names, 500, 1.0, cached, False)
            # unshuffle the indices, so that we can make assertions about the returned data
            data_gen.indices = [0, 1]
            self.assertEqual(1, len(data_gen))
            train_data, y = data_gen[0]
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


class TestBinaryDataEval(unittest.TestCase):
    bytes_per_base = 10 + sum(  # 10 is the overhead also added in Models_Fl.BinaryDataEval
        [np.dtype(ft).itemsize for ft in Reader.feature_np_types])

    def test_batching_one_per_batch(self):
        reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
        reader.contigs = [ContigInfo('Contig1', '/tmp/c1', 1000, 0, 0, 0),
                          ContigInfo('Contig2', '/tmp/c2', 1000, 0, 0, 0),
                          ContigInfo('Contig3', '/tmp/c3', 1000, 0, 0, 0)]
        indices = np.arange(len(reader))

        gpu_memory_bytes = 1010 * self.bytes_per_base
        eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 500, 250, gpu_memory_bytes, False,
                                             False)
        self.assertEqual(3, len(eval_data.chunk_counts))
        for i in range(len(eval_data.chunk_counts)):
            self.assertEqual(1, len(eval_data.chunk_counts[i]))
            self.assertEqual(3, eval_data.chunk_counts[i][0])

    def test_batching_multiple_per_batch(self):
        reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
        reader.contigs = [ContigInfo('Contig1', 'data/preprocess/features_binary', 500, 0, 246, 0),
                          ContigInfo('Contig2', 'data/preprocess/features_binary', 500, 246, 183, 0),
                          ContigInfo('Contig3', 'data/preprocess/features_binary', 500, 0, 246, 0)]
        indices = np.arange(len(reader))
        gpu_memory_bytes = 1600 * self.bytes_per_base
        eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 250, 200, gpu_memory_bytes, False,
                                             False)
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

        self.assertEqual(6, len(eval_data[0]))
        self.assertEqual(3, len(eval_data[1]))

    def test_gen_eval_data(self):
        for cached in [False, True]:
            reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
            indices = np.arange(len(reader))
            eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 500, 250, 1e6, cached, False)
            self.assertEqual(1, len(eval_data))
            self.assertEqual(2, len(eval_data.batch_list[0]))
            self.assertIsNone(
                np.testing.assert_array_equal(eval_data[0][0][0][0:6], np.array([1, 0, 0, 0, 2, 1])))
            self.assertIsNone(
                np.testing.assert_array_equal(eval_data[0][0][5][0:6], np.array([1, 0, 0, 0, 0, 0])))

            self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][0][0:6], [1, 0, 0, 0, 1, 1])))
            self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_gen_eval_data_short_window(self):
        reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 50, 30, 1e6, False, False)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        # 16 for the first contig of length 500, 16 for the 2nd contig of length 500
        self.assertEqual(32, len(eval_data[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_group(self):
        reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
        indices = np.arange(len(reader))
        total_memory_bytes = 1e6
        eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 50, 30, total_memory_bytes, False,
                                             False)
        self.assertEqual(32, len(eval_data[0]))

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
        reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 50, 30, 500, False, False)

        self.assertEqual(2, len(eval_data))
        self.assertEqual(16, len(eval_data[0]))
        self.assertEqual(16, len(eval_data[1]))

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
        reader = ContigReader.ContigReader('data/preprocess/', Reader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, Reader.feature_names, 500, 250, 1e6, True, False)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][5][0:6], [1, 0, 0, 0, 0, 0])))
