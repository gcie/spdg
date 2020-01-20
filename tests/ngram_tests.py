import unittest

import numpy
import spdg


class NgramTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_subgrams(self):
        ngram = spdg.Ngram(3)
        ngram[(0, 1, 0)] = 3
        ngram[(0, 1, 1)] = 2
        ngram[(1, 1, 0)] = 5
        ngram.norm()

        assert ngram[(0, 1, 0)] == 0.3
        assert ngram.subgram((0, 1))[0] == 0.6

    def test_ngram_from_data(self):
        data = numpy.array([[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2]])
        ngram = spdg.ngram.from_data(data, 3)
        self.assertEqual(ngram[(1, 1, 1)], 1 / 2)
        self.assertEqual(ngram[(1, 1, 2)], 1 / 3)
        self.assertEqual(ngram[(1, 2, 2)], 1 / 6)

    def test_random_ngram(self):
        ngram = spdg.ngram.random(4, 12, 3)

        self.assertEqual(len(ngram), 12)
        self.assertEqual(len(ngram.sample()), 4)

        with self.assertRaises(spdg.errors.ParameterError):
            spdg.ngram.random(3, 30, 2)

        with self.assertRaises(spdg.errors.ParameterError):
            spdg.ngram.random(0, 3, 2)

        with self.assertRaises(spdg.errors.ParameterError):
            spdg.ngram.random(3, 3, 0)


if __name__ == '__main__':
    unittest.main()
