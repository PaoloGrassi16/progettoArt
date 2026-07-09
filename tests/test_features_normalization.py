import unittest

import numpy as np

from musicalai.audio.features import denormalize_log_mel, normalize_log_mel


class FeatureNormalizationTest(unittest.TestCase):
    def test_normalization_round_trip(self):
        values = np.array([[-80.0, -40.0, 0.0]], dtype=np.float32)
        normalized = normalize_log_mel(values, top_db=80.0)
        restored = denormalize_log_mel(normalized, top_db=80.0)
        np.testing.assert_allclose(values, restored, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

