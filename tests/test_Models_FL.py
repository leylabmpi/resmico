import numpy as np
import unittest

from ResMiCo import Models_FL
from ResMiCo import ContigReader


class TestBinaryData(unittest.TestCase):
    def test_gen_eval_data(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        data_gen = Models_FL.BinaryData(reader, indices, 10, ContigReader.feature_names, 500)
        self.assertEqual(1, len(data_gen))
        eval_data, y = data_gen[0]
        self.assertEqual(2, len(eval_data))

        self.assertIsNone(np.testing.assert_array_equal(y, np.array([1, 0])))

        # # eval_data[0][0] - first position in first contig, eval_data[0][5] 5th position in 1st contig
        self.assertIsNone(
            np.testing.assert_array_equal(eval_data[0][0][0:6], np.array([1, 0, 0, 0, 2, 1])))
        self.assertIsNone(
            np.testing.assert_array_equal(eval_data[0][5][0:6], np.array([1, 0, 0, 0, 0, 0])))

        # # eval_data[1][0] - first position in 2nd contig, eval_data[1][5] 5th position in 2nd contig
        self.assertIsNone(
            np.testing.assert_array_equal(eval_data[1][0][0:6], np.array([1, 0, 0, 0, 1, 1])))
        self.assertIsNone(
            np.testing.assert_array_equal(eval_data[1][5][0:6], np.array([1, 0, 0, 0, 0, 0])))

    def test_gen_eval_data_short_window(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 1e5)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        # 16 for the first contig of length 500, 16 for the 2nd contig of length 500
        self.assertEqual(32, len(eval_data[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_group(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 1e5)
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
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 500)

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


class TestBinaryDataEval(unittest.TestCase):
    def test_gen_eval_data(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 500, 250, 1e5)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][1][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_gen_eval_data_short_window(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 1e5)
        self.assertEqual(1, len(eval_data))
        self.assertEqual(2, len(eval_data.batch_list[0]))
        # 16 for the first contig of length 500, 16 for the 2nd contig of length 500
        self.assertEqual(32, len(eval_data[0]))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][0][0:6], [1, 0, 0, 0, 2, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][0][5][0:6], [1, 0, 0, 0, 0, 0])))

        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][0][0:6], [1, 0, 0, 0, 1, 1])))
        self.assertTrue(all(a == b for a, b in zip(eval_data[0][16][5][0:6], [1, 0, 0, 0, 0, 0])))

    def test_group(self):
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 1e5)
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
        reader = ContigReader.ContigReader('./data/preprocess/', ContigReader.feature_names, 1)
        indices = np.arange(len(reader))
        eval_data = Models_FL.BinaryDataEval(reader, indices, ContigReader.feature_names, 50, 30, 500)

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
