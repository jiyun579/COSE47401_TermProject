"""Smoke test the source-to-source variants without a model or GPU."""
import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parent.parent / "scripts/install_temporal_variants.py"
SPEC = importlib.util.spec_from_file_location("installer", PATH)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


SAMPLE_SOURCE = (
    "import torch\n"
    "def parse_args():\n    opts = parser.parse_args()\n"
    "def train_one_epoch():\n"
    "        loss_3d_velocity = loss_velocity(pred, y)\n"
    "        loss_total = loss_3d_pos + \\\n"
    "                    args.lambda_3d_velocity * loss_3d_velocity + \\\n"
    "                    args.lambda_lv * loss_lv + \\\n"
    "        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)\n"
    "def evaluate():\n    results_all = np.concatenate(results_all)\n"
    "def train():\n"
    "        loss_names = ['3d_velocity', 'angle', 'angle_velocity', 'total']\n"
    "        train_one_epoch(args, model, train_loader, optimizer, device, losses)\n"
    "        wandb.log({\n"
    "                'train/loss_3d_velocity': losses['3d_velocity'].avg,\n"
    "        })\n"
    "def main():\n    args = get_config(opts.config)\n"
)


class TestInstaller(unittest.TestCase):
    def test_acc_patch(self):
        result = INSTALLER.make_accel(SAMPLE_SOURCE)
        self.assertIn("args.lambda_acc * loss_acc", result)
        self.assertIn("losses['acceleration'].update", result)
        self.assertIn("train/loss_acceleration", result)
        compile(result, '<test train_accel>', 'exec')

    def test_sg_patch(self):
        result = INSTALLER.make_sg(SAMPLE_SOURCE)
        self.assertIn("--sg-window", result)
        self.assertIn("results_all = savgol_pose_smooth(", result)
        compile(result, '<test eval_sg>', 'exec')

    def test_source_structure_check(self):
        with self.assertRaises(ValueError):
            INSTALLER.make_accel("import torch\n")


if __name__ == '__main__':
    unittest.main()
