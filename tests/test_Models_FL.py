import numpy as np
import unittest

from ResMiCo import Models_FL
from ResMiCo import ContigReader
from ResMiCo.ContigReader import ContigInfo


class TestBinaryData(unittest.TestCase):
    def test_gen_eval_data(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        indices = np.arange(len(reader))
        batch_size = 10
        data_gen = Models_FL.BinaryData(reader, indices, batch_size, ContigReader.feature_names, 500, 1.0)
        # unshuffle the indices, so that we can make assertions about the returned data
        data_gen.indices = [0,1]
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
    def test_batching_one_per_batch(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        reader.contigs = [ContigInfo('Contig1', '/tmp/c1', 1000, 0, 0, 0),
                          ContigInfo('Contig2', '/tmp/c2', 1000, 0, 0, 0),
                          ContigInfo('Contig3', '/tmp/c3', 1000, 0, 0, 0)]
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 500, 250, 1010, False)
        self.assertEqual(3, len(eval_data.chunk_counts))
        for i in range(len(eval_data.chunk_counts)):
            self.assertEqual(1, len(eval_data.chunk_counts[i]))
            self.assertEqual(3, eval_data.chunk_counts[i][0])

    def test_batching_multiple_per_batch(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False, False)
        reader.contigs = [ContigInfo('Contig1', './data/preprocess/features_binary', 500, 0, 246, 0),
                          ContigInfo('Contig2', './data/preprocess/features_binary', 500, 246, 183, 0),
                          ContigInfo('Contig3', './data/preprocess/features_binary', 500, 0, 246, 0)]
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 250, 200, 1000, False)
        self.assertEqual(2, len(eval_data.chunk_counts))
        self.assertEqual(2, len(eval_data.chunk_counts[0]))
        self.assertEqual(3, eval_data.chunk_counts[0][0])
        self.assertEqual(3, eval_data.chunk_counts[0][1])
        self.assertEqual(1, len(eval_data.chunk_counts[1]))
        self.assertEqual(3, eval_data.chunk_counts[1][0])

        self.assertEqual(6, len(eval_data[0]))
        self.assertEqual(3, len(eval_data[1]))

    def test_gen_eval_data(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 500, 250, 1e5, False)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_gen_eval_data_short_window(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 1e5, False)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        # 16 for the first contig of length 500, 16 for the 2nd contig of length 500
        self.assertEqual(32, len(eval_data[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_group(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 1e5, False)
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
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 500, False)

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
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1, False)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 500, 250, 1e5, True)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][5][0:6], [1, 0, 0, 0, 0, 0])))
