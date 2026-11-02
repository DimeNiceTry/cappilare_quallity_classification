"""Numerical checks with known answers, independent of any trained model."""
import unittest
import numpy as np
from scipy.special import softmax
from evaluate_calibration import (calibration_bins, check_disjoint, fit_temperature,
                                  nll_from_logits, top_label_ece)


class CalibrationChecks(unittest.TestCase):
    def test_hand_calculated_weighted_error(self):
        # First bin: 2/4 * |.65-.5|; second: 2/4 * |.85-1| = .15.
        p = np.array([[.6, .4], [.7, .3], [.8, .2], [.9, .1]])
        self.assertAlmostEqual(top_label_ece(np.array([0, 1, 0, 0]), p, 4), .15)

    def test_zero_and_one_are_counted_and_empty_bins_are_safe(self):
        error, bins = calibration_bins([0, .1, 1], [0, 1, 1], 10)
        self.assertAlmostEqual(error, .3)
        self.assertEqual(sum(b["count"] for b in bins), 3)
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[-1]["count"], 1)

    def test_completely_wrong_confident_predictions(self):
        self.assertEqual(top_label_ece(np.array([1, 0]), np.eye(2)), 1)

    def test_uninformative_but_calibrated_constant_predictor(self):
        p = np.tile([.5, .3, .2], (10, 1))
        self.assertAlmostEqual(top_label_ece(np.array([0] * 5 + [1] * 3 + [2] * 2), p), 0)

    def test_invalid_input_is_rejected(self):
        for y, p in [(np.array([0]), [[.2, .2]]), (np.array([2]), [[.5, .5]]),
                     (np.array([0]), [[np.nan, .5]]), (np.array([]), np.empty((0, 2))),
                     (np.array([.5]), [[.5, .5]])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                top_label_ece(y, p)

    def test_temperature_preserves_argmax_and_reduces_calibration_nll(self):
        z = np.tile([4., 0., 0.], (10, 1))
        y = np.array([0] * 6 + [1] * 2 + [2] * 2)
        t = fit_temperature(y, z)
        self.assertGreater(t, 1)
        self.assertLess(nll_from_logits(y, z, t), nll_from_logits(y, z))
        np.testing.assert_array_equal(softmax(z, axis=1).argmax(1), softmax(z / t, axis=1).argmax(1))
        self.assertAlmostEqual(softmax(z / t, axis=1)[0, 0], .6, places=5)

    def test_patient_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "patient overlap"):
            check_disjoint([{"file": "a"}], ["p1"], [{"file": "b"}], ["p1"])


if __name__ == "__main__":
    unittest.main()
