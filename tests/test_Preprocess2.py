import numpy as np
import unittest

from ResMiCo import Preprocess2


class TestPreprocess2(unittest.TestCase):
    def test_read_file_static_fields(self):
        feature_info, coverage_idx = Preprocess2.get_feature_info('./data/preprocess/features1.tsv.gz',
                                                                  ['assembler', 'contig'],
                                                                  ['s', 's'])
        result = Preprocess2.read_file('./data/preprocess/features1.tsv.gz', feature_info, -1)
        self.assertEqual('megahit', result[0])
        self.assertEqual('k141_15249', result[1])

    # reading the same fields, but in reverse order; makes sure out of order fields are correctly handled
    def test_read_file_static_fields_inverted(self):
        feature_info, coverage_idx = Preprocess2.get_feature_info('./data/preprocess/features1.tsv.gz',
                                                                  ['contig', 'assembler'],
                                                                  ['s', 's'])
        result = Preprocess2.read_file('./data/preprocess/features1.tsv.gz', feature_info, -1)
        self.assertEqual('k141_15249', result[0])
        self.assertEqual('megahit', result[1])

    # reads an entire file, make sure fields are correct
    def test_read_file(self):
        feature_info, coverage_idx = Preprocess2.get_feature_info('./data/preprocess/features1.tsv.gz',
                                                                  ['mean_mapq_Match', 'min_mapq_Match', 'contig',
                                                                   'assembler', 'seq_window_entropy',
                                                                   'seq_window_perc_gc'],
                                                                  ['f', 'f', 's', 's', 'f', 'f'])
        result = Preprocess2.read_file('./data/preprocess/features1.tsv.gz', feature_info, -1)
        self.assertEqual('k141_15249', result[2])
        self.assertEqual('megahit', result[3])

        # check mean_mapq_Match
        self.assertIsNone(
            np.testing.assert_array_equal(result[0][0], [np.nan, np.nan, np.nan, 24, np.nan, 24, 24, 24, 24]))
        self.assertEqual(120, result[0][1])  # 120 = 5 * 24

        # check min_mapq_Match
        self.assertIsNone(
            np.testing.assert_array_equal(result[1][0], [np.nan, np.nan, np.nan, 24, np.nan, 24, 24, 24, 24]))
        self.assertEqual(120, result[1][1])  # 120 = 5 * 24

        # check seq_window_entropy
        self.assertIsNone(
            np.testing.assert_allclose(result[4][0],
                                       np.array([1.252, 1.459, 1.459, 1.918, 1.792, 1.459, 1.459, 0.918, 0.65],
                                                dtype=float)))
        self.assertAlmostEqual(12.366, result[4][1], places=3)

        # check seq_window_perc_gc
        self.assertIsNone(
            np.testing.assert_allclose(result[5][0],
                                       np.array([0.333, 0.5, 0.5, 0.5, 0.333, 0.167, 0.167, 0, 0],
                                                dtype=float)))
        self.assertAlmostEqual(2.5, result[5][1], places=3)

    def test_preprocess(self):
        all_data = Preprocess2.preprocess(1, './data/preprocess/',
                                          ['assembler', 'contig', 'mean_mapq_Match', 'min_mapq_Match',
                                           'seq_window_entropy', 'seq_window_perc_gc', 'num_query_A', 'coverage'],
                                          ['s', 's', 'f', 'f', 'f', 'f', 'i', 'i'])

        self.assertTrue(('metaspades', 'NODE_6433_length_1652_cov_1.829054') in all_data)
        self.assertTrue(('megahit', 'k141_15249') in all_data)

        result1 = all_data[('megahit', 'k141_15249')]
        result2 = all_data[('metaspades', 'NODE_6433_length_1652_cov_1.829054')]

        # check that all float and int fields have 0 average and 1 standard deviation
        for idx in range(2, 6):
            self.assertAlmostEqual(0, np.nanmean(np.concatenate((result1[idx], result2[idx]))), places=5)
            self.assertAlmostEqual(1, np.nanstd(np.concatenate((result1[idx], result2[idx]))), places=5)

        # check the coverage
        self.assertIsNone(
            np.testing.assert_array_equal(result1[7], [4, 1, 1, 1, 1, 1, 1, 1, 1]))
        # check num_query_A
        self.assertIsNone(
            np.testing.assert_array_equal(result1[6], [0.5, 0, 0, 0, 0, 1, 0, 1, 0]))
        # check mean_mapq_Match
        self.assertIsNone(
            np.testing.assert_allclose(result1[2],
                                       np.array([np.nan, np.nan, np.nan, -1.3416, np.nan, -1.3416, -1.3416, -1.3416,
                                                 -1.3416], dtype=float), rtol=1e-3))
        self.assertIsNone(
            np.testing.assert_allclose(result2[2],
                                       np.array([0.745356, 0.745356, 0.745356, 0.745356, 0.745356, 0.745356, 0.745356,
                                                 0.745356, 0.745356], dtype=float), rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
