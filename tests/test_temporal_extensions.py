import unittest

import numpy as np
import torch

from experiments.temporal_extensions import (
    acceleration_consistency_loss,
    savgol_pose_smooth,
)


class TemporalExtensionsTest(unittest.TestCase):
    def test_acc_identical(self):
        y = torch.randn(2, 81, 17, 3)
        self.assertEqual(float(acceleration_consistency_loss(y, y)), 0.0)

    def test_acc_gradient(self):
        y = torch.randn(2, 81, 17, 3)
        x = (y + torch.randn_like(y) * 0.1).requires_grad_()
        loss = acceleration_consistency_loss(x, y)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_acc_short_clip(self):
        y = torch.randn(1, 2, 17, 3)
        x = y.clone().requires_grad_()
        acceleration_consistency_loss(x, y).backward()
        self.assertEqual(float(x.grad.abs().sum()), 0.0)

    def test_sg_preserves_polynomial(self):
        t = np.arange(81, dtype=np.float64)
        curve = 0.01 * t * t + 0.3 * t + 2
        seq = np.broadcast_to(curve[None, :, None, None], (2, 81, 17, 3)).copy()
        filtered = savgol_pose_smooth(seq, window_length=9, polyorder=3)
        np.testing.assert_allclose(filtered, seq, atol=1e-10)

    def test_sg_clip_independence(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 81, 17, 3))
        y = savgol_pose_smooth(x, window_length=9, polyorder=3)
        altered = x.copy()
        altered[1] += 10000
        z = savgol_pose_smooth(altered, window_length=9, polyorder=3)
        np.testing.assert_array_equal(y[0], z[0])

    def test_sg_disabled(self):
        x = np.ones((1, 81, 17, 3), dtype=np.float32)
        y = savgol_pose_smooth(x, window_length=0)
        np.testing.assert_array_equal(y, x)
        self.assertIsNot(y, x)

    def test_sg_invalid_window(self):
        x = np.ones((1, 81, 17, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            savgol_pose_smooth(x, window_length=8)
        with self.assertRaises(ValueError):
            savgol_pose_smooth(x, window_length=83)
        with self.assertRaises(ValueError):
            savgol_pose_smooth(x, window_length=9, polyorder=9)


if __name__ == '__main__':
    unittest.main()
