import numpy as np
import unittest

from ResMiCo import ContigReader
from ResMiCo import Reader


class TestReadContig(unittest.TestCase):
    def test_read_from_file(self):
        input_file = open('./data/preprocess/features_binary', 'rb')
        result = ContigReader._read_contig_data(input_file, Reader.feature_names)
        self.assertEqual(500, len(result['ref_base_A']))
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1] * 498 + [0, 0]), result['ref_base_A']))
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([0] * 498 + [1, 1]), result['ref_base_C']))
        self.assertTrue(not np.any(result['ref_base_G']))
        self.assertTrue(not np.any(result['ref_base_T']))

        coverage = result['coverage']
        for pos in range(0, len(result['ref_base_A'])):
            self.assertEqual(2 if 420 <= pos < 425 or pos < 5 else 0, coverage[pos])
            self.assertEqual(0, result['num_discordant'][pos])
            self.assertEqual(1 if pos == 0 else 0.5 if pos < 5 else 0, result['num_query_A'][pos])
            self.assertEqual(0.5 if 420 <= pos < 425 else 0, result['num_query_C'][pos])
            self.assertEqual(0.5 if 0 < pos < 5 else 0, result['num_query_G'][pos])
            self.assertEqual(0.5 if 420 <= pos < 425 else 0, result['num_query_T'][pos])

            self.assertIsNone(np.testing.assert_equal(-28 if pos == 0 else 0 if pos < 5 else np.nan,
                                                      result['min_al_score_Match'][pos]))
            self.assertIsNone(np.testing.assert_equal(0 if pos < 5 else np.nan, result['max_al_score_Match'][pos]))

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
            self.assertIsNone(np.testing.assert_array_equal(np.nan if pos >= 5 else 6.5 if pos == 0 else 6,
                                                            result['mean_mapq_Match'][pos]))
            self.assertEqual(0.5 if 0 < pos < 5 else 1 if 420 <= pos < 425 else 0, result['num_proper_SNP'][pos])
            self.assertEqual(0 if pos < 498 else 25 if pos == 498 else 50, result['seq_window_perc_gc'][pos])
            self.assertEqual(1 if pos < 20 else 0, result['Extensive_misassembly_by_pos'][pos])

    def test_normalize_zero_mean_one_stdev(self):
        input_file = open('./data/preprocess/features_binary', 'rb')
        old_result = ContigReader._read_contig_data(input_file, Reader.feature_names)

        reader = ContigReader.ContigReader('./data/preprocess/', Reader.feature_names, 1, False)
        for fname in Reader.float_feature_names:
            reader.means[fname] = 0
            reader.stdevs[fname] = 1

        result = reader._read_and_normalize(reader.contigs[0])

        for fname in Reader.float_feature_names:
            # replace NANs with 0, as that's what the normalization in ContigReader does
            nan_pos = np.isnan(old_result[fname])
            old_result[fname][nan_pos] = 0
            self.assertIsNone(np.testing.assert_array_equal(old_result[fname], result[fname]))

    def test_normalize_zero_mean_two_stdev(self):
        input_file = open('./data/preprocess/features_binary', 'rb')
        old_result = ContigReader._read_contig_data(input_file, Reader.feature_names)

        reader = ContigReader.ContigReader('./data/preprocess/', Reader.feature_names, 1, False)
        for fname in Reader.float_feature_names:
            reader.means[fname] = 0
            reader.stdevs[fname] = 2

        result = reader._read_and_normalize(reader.contigs[0])

        for fname in Reader.float_feature_names:
            # replace NANs with 0, as that's what the normalization in ContigReader does
            nan_pos = np.isnan(old_result[fname])
            old_result[fname][nan_pos] = 0
            self.assertIsNone(np.testing.assert_array_equal(old_result[fname] / 2, result[fname]))

    def test_read_three_features(self):
        input_file = open('./data/preprocess/features_binary', 'rb')
        reader = ContigReader.ContigReader('./data/preprocess/', [Reader.feature_names[0], Reader.feature_names[1],
                                                                  Reader.feature_names[3]], 1, False)
        result = reader.read_contigs(reader.contigs)

        # we read 2 contigs in total
        self.assertEqual(2, len(result))
        # each contig has 6 features (because 'ref_base' is one-hot encoded into 4 features)
        self.assertEqual(6, len(result[0]))
        self.assertEqual(6, len(result[1]))

        self.assertEqual(500, len(result[0]['ref_base_A']))
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([1] * 498 + [0, 0]), result[0]['ref_base_A']))
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([0] * 498 + [1, 1]), result[0]['ref_base_C']))
        self.assertTrue(not np.any(result[0]['ref_base_G']))
        self.assertTrue(not np.any(result[0]['ref_base_T']))
        coverage = result[0]['coverage']
        for pos in range(0, len(result[0]['ref_base_A'])):
            self.assertEqual(2 if 420 <= pos < 425 or pos < 5 else 0, coverage[pos])
            self.assertEqual(0.5 if 420 <= pos < 425 else 0, result[0]['num_query_C'][pos])

    if __name__ == '__main__':
        unittest.main()
