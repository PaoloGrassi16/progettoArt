import unittest

import numpy as np

from musicalai.audio.specaugment import specaugment
from musicalai.config import SpecAugmentConfig


class SpecAugmentTest(unittest.TestCase):
    def test_specaugment_preserves_shape_and_dtype(self):
        spec = np.ones((128, 96), dtype=np.float32)
        config = SpecAugmentConfig(
            freq_mask_param=16,
            time_mask_param=20,
            num_freq_masks=2,
            num_time_masks=2,
            replace_with_mean=False,
        )
        augmented = specaugment(spec, config, seed=123)
        self.assertEqual(augmented.shape, spec.shape)
        self.assertEqual(augmented.dtype, np.float32)
        self.assertLessEqual(float(augmented.min()), 1.0)


if __name__ == "__main__":
    unittest.main()

