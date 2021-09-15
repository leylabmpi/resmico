import numpy as np
import unittest

from ResMiCo import ContigReader


class TestReadContig(unittest.TestCase):
    def test_read_from_file(self):
        result = ContigReader.read_contig_data('./data/preprocess/Contig2.gz', ContigReader.feature_names)
        self.assertEqual(500, len(result['ref_base']))
        self.assertEqual(498 * 'A' + 'CC', result['ref_base'])

        coverage = result['coverage']
        for pos in range(0, len(result['ref_base'])):
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
            self.assertEqual(1 if pos < 20 else 0, result['Extensive_misassembly_by_pos'][pos])

    def test_normalize_zero_mean_one_stdev(self):
        result = ContigReader.read_contig_data('./data/preprocess/Contig2.gz', ContigReader.feature_names)

        mean_std_dev = {}
        for fname in ContigReader.float_feature_names:
            mean_std_dev[fname] = (0, 1)

        old_result = {fname: np.copy(result[fname]) for fname in result.keys()}

        ContigReader.normalize_contig_data(result, mean_std_dev)
        for fname in ContigReader.float_feature_names:
            self.assertIsNone(np.testing.assert_array_equal(old_result[fname], result[fname]))

    def test_normalize_zero_mean_two_stdev(self):
        result = ContigReader.read_contig_data('./data/preprocess/Contig2.gz', ContigReader.feature_names)

        mean_std_dev = {}
        for fname in ContigReader.float_feature_names:
            mean_std_dev[fname] = (0, 2)

        old_result = {fname: np.copy(result[fname]) for fname in result.keys()}

        ContigReader.normalize_contig_data(result, mean_std_dev)
        for fname in ContigReader.float_feature_names:
            self.assertIsNone(np.testing.assert_array_equal(old_result[fname] / 2, result[fname]))

    if __name__ == '__main__':
        unittest.main()
