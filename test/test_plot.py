from unittest import TestCase

import numpy as np
import pandas as pd

from mlnext import plot


class TestHelper(TestCase):

    def test_get_segments(self):

        x = pd.DataFrame(np.ones((10, 1)))
        y = pd.DataFrame([0, 1, 1, 1, 0, 0, 1, 1, 0, 0])

        expected = [0, 1, 4, 6, 8, 9]

        result = plot._get_segments(x, y)

        self.assertListEqual(expected, result)

    def test_truncate_length(self):

        x1 = pd.DataFrame(np.ones((20, 1)))
        x2 = pd.DataFrame(np.ones((14, 1)))

        result = plot._truncate_length(x1, x2)

        for r in result:
            pd.testing.assert_frame_equal(r, x2)

    def test_truncate_length_None(self):

        x1 = pd.DataFrame(np.ones((20, 1)))
        x2 = pd.DataFrame(np.ones((14, 1)))

        result = plot._truncate_length(x1, x2, None)

        pd.testing.assert_frame_equal(result[0], x2)
        pd.testing.assert_frame_equal(result[1], x2)
        self.assertEqual(result[2], None)
