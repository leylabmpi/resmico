import numpy as np
import unittest

from ResMiCo import Preprocess2


class TestReadContig(unittest.TestCase):
    def test_read_from_file(self):
        result = Preprocess2.read_contig_data('./data/preprocess/Contig2.gz', Preprocess2.feature_names)
        self.assertEqual(500, len(result['contig']))
        self.assertEqual(498 * 'A' + 'CC', result['contig'])

        coverage = result['coverage']
        for pos in range(0, len(result['contig'])):
            self.assertEqual(2 if 420 <= pos < 425 or pos < 5 else 0, coverage[pos])
            self.assertEqual(0, result['num_discordant'][pos])
            self.assertEqual(1 if pos == 0 else 0.5 if pos < 5 else 0, result['num_query_A'][pos])
            self.assertEqual(0.5 if 420 <= pos < 425 else 0, result['num_query_C'][pos])
            self.assertEqual(0.5 if 0 < pos < 5 else 0, result['num_query_G'][pos])
            self.assertEqual(0.5 if 420 <= pos < 425 else 0, result['num_query_T'][pos])

            self.assertIsNone(np.testing.assert_equal(-28 if pos == 0 else 0 if pos < 5 else np.nan,
                                                      result['min_al_score_Match'][pos]));
            self.assertIsNone(np.testing.assert_equal(0 if pos < 5 else np.nan, result['max_al_score_Match'][pos]));

            self.assertIsNone(
                np.testing.assert_array_equal(425 if pos < 5 else np.nan, result['min_insert_size_Match'][pos]))
            self.assertTrue(result['mean_insert_size_Match'][pos] == 425 and pos < 5 or pos >= 5 and np.isnan(
                result['mean_insert_size_Match'][pos]))
            self.assertTrue(result['stdev_insert_size_Match'][pos] == 0 and pos == 0 or pos > 0 and np.isnan(
                result['stdev_insert_size_Match'][pos]))
            self.assertIsNone(
                np.testing.assert_array_equal(425 if pos < 5 else np.nan, result['max_insert_size_Match'][pos]))

            self.assertIsNone(np.testing.assert_array_equal(6 if pos < 5 else np.nan, result['min_mapq_Match'][pos]))
            self.assertIsNone(np.testing.assert_array_equal(7 if pos == 0 else 6 if pos < 5 else np.nan,
                                                            result['max_mapq_Match'][pos]))
            self.assertIsNone(
                np.testing.assert_array_equal(np.nan if pos >= 5 else 6.5 if pos == 0 else 6, result['mean_mapq_Match'][
                    pos]))
            self.assertEqual(0.5 if 0 < pos < 5 else 1 if 420 <= pos < 425 else 0, result['num_proper_SNP'][pos])
            self.assertEqual(0 if pos < 498 else 25 if pos == 498 else 50, result['seq_window_perc_gc'][pos])
            self.assertEqual(1 if 9 <= pos < 30 else 0, result['Extensive_misassembly_by_pos'][pos])

    def test_normalize_zero_mean_one_stdev(self):
        result = Preprocess2.read_contig_data('./data/preprocess/Contig2.gz', Preprocess2.feature_names)

        mean_std_dev = {}
        for fname in Preprocess2.float_feature_names:
            mean_std_dev[fname] = (0, 1)

        old_result = {fname: np.copy(result[fname]) for fname in result.keys()}

        Preprocess2.normalize_contig_data(result, mean_std_dev)
        for fname in Preprocess2.float_feature_names:
            self.assertIsNone(np.testing.assert_array_equal(old_result[fname], result[fname]))

    def test_normalize_zero_mean_two_stdev(self):
        result = Preprocess2.read_contig_data('./data/preprocess/Contig2.gz', Preprocess2.feature_names)

        mean_std_dev = {}
        for fname in Preprocess2.float_feature_names:
            mean_std_dev[fname] = (0, 2)

        old_result = {fname: np.copy(result[fname]) for fname in result.keys()}

        Preprocess2.normalize_contig_data(result, mean_std_dev)
        for fname in Preprocess2.float_feature_names:
            self.assertIsNone(np.testing.assert_array_equal(old_result[fname] / 2, result[fname]))

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
                                           np.array(
                                               [0.745356, 0.745356, 0.745356, 0.745356, 0.745356, 0.745356, 0.745356,
                                                0.745356, 0.745356], dtype=float), rtol=1e-3))

    if __name__ == '__main__':
        unittest.main()
