import unittest

from musicalai.scenarios.gaming_vr import build_plan as gaming_plan
from musicalai.scenarios.pro_audio import build_plan as pro_audio_plan
from musicalai.scenarios.therapy_masking import build_plan as therapy_plan


class ScenarioPlanTest(unittest.TestCase):
    def test_gaming_maps_electronic_to_disco(self):
        plan = gaming_plan(intensity=0.8, target="electronic")
        self.assertEqual(plan.target_genre, "disco")
        self.assertGreaterEqual(plan.confidence_threshold, 0.85)

    def test_pro_audio_maps_lofi_to_hiphop(self):
        plan = pro_audio_plan(style="lofi_glitch")
        self.assertEqual(plan.target_genre, "hiphop")
        self.assertTrue(plan.controls["export_stems"])

    def test_therapy_maps_ambient_to_classical(self):
        plan = therapy_plan(noise_level=0.7, target="ambient")
        self.assertEqual(plan.target_genre, "classical")
        self.assertTrue(plan.controls["avoid_transients"])


if __name__ == "__main__":
    unittest.main()

