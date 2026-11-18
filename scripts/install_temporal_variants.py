"""Generate train_accel.py and eval_sg.py from this machine's patched train.py.

Deliberately NEVER modifies train.py or the baseline training config/checkpoints.
This preserves local AP3D evaluation fixes that may not have been committed.

Run from the repository root:
    python scripts/install_temporal_variants.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(
            f"Unable to locate unique insertion point for {label}: "
            f"found {count} matches. The local train.py differs from the expected version; "
            f"no changes have been written."
        )
    return text.replace(old, new, 1)


def make_accel(src: str) -> str:
    s = replace_once(
        src,
        "import torch\n",
        "import torch\nfrom experiments.temporal_extensions import acceleration_consistency_loss\n",
        "acceleration import",
    )
    s = replace_once(
        s,
        "        loss_3d_velocity = loss_velocity(pred, y)\n",
        "        loss_3d_velocity = loss_velocity(pred, y)\n"
        "        loss_acc = acceleration_consistency_loss(pred, y)\n",
        "differentiable acceleration loss",
    )
    s = replace_once(
        s,
        "                    args.lambda_3d_velocity * loss_3d_velocity + \\\n",
        "                    args.lambda_3d_velocity * loss_3d_velocity + \\\n"
        "                    args.lambda_acc * loss_acc + \\\n",
        "total loss",
    )
    s = replace_once(
        s,
        "        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)\n",
        "        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)\n"
        "        losses['acceleration'].update(loss_acc.item(), batch_size)\n",
        "loss meter",
    )
    s = replace_once(
        s,
        "'3d_velocity', 'angle', 'angle_velocity', 'total'",
        "'3d_velocity', 'acceleration', 'angle', 'angle_velocity', 'total'",
        "loss meter registration",
    )
    s = replace_once(
        s,
        "                'train/loss_3d_velocity': losses['3d_velocity'].avg,\n",
        "                'train/loss_3d_velocity': losses['3d_velocity'].avg,\n"
        "                'train/loss_acceleration': losses['acceleration'].avg,\n",
        "wandb logging",
    )
    s = replace_once(
        s,
        "        train_one_epoch(args, model, train_loader, optimizer, device, losses)\n",
        "        train_one_epoch(args, model, train_loader, optimizer, device, losses)\n"
        "        print(f\"[ACC] epoch {epoch} acceleration loss: \"\n"
        "              f\"{losses['acceleration'].avg:.6f}, lambda_acc={args.lambda_acc}\", flush=True)\n",
        "terminal logging",
    )
    return s


def make_sg(src: str) -> str:
    s = replace_once(
        src,
        "import torch\n",
        "import torch\nfrom experiments.temporal_extensions import savgol_pose_smooth\n",
        "savgol import",
    )
    s = replace_once(
        s,
        "    opts = parser.parse_args()\n",
        "    parser.add_argument('--sg-window', type=int, default=9, "
        "help='Odd SG window; 0 disables SG for baseline evaluation')\n"
        "    parser.add_argument('--sg-polyorder', type=int, default=3)\n"
        "    opts = parser.parse_args()\n",
        "SG arguments",
    )
    s = replace_once(
        s,
        "    results_all = np.concatenate(results_all)\n",
        "    results_all = np.concatenate(results_all)\n"
        "    if args.sg_window:\n"
        "        print(f'[SG] smoothing predictions: window={args.sg_window}, '\n"
        "              f'polyorder={args.sg_polyorder}', flush=True)\n"
        "        results_all = savgol_pose_smooth(\n"
        "            results_all, window_length=args.sg_window,\n"
        "            polyorder=args.sg_polyorder\n"
        "        )\n"
        "    else:\n"
        "        print('[SG] disabled: original baseline predictions', flush=True)\n",
        "SG prediction postprocessing",
    )
    s = replace_once(
        s,
        "    args = get_config(opts.config)\n",
        "    args = get_config(opts.config)\n"
        "    args.sg_window = opts.sg_window\n"
        "    args.sg_polyorder = opts.sg_polyorder\n",
        "SG options in config",
    )
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly replace previously generated experiment files')
    opt = parser.parse_args()

    base_train = ROOT / 'train.py'
    base_config = ROOT / 'configs/ap3d/MotionAGFormer-base-81.yaml'
    if not base_train.is_file():
        raise SystemExit(f"Missing original source: {base_train}")
    if not base_config.is_file():
        raise SystemExit(f"Missing AP3D config: {base_config}")

    src = base_train.read_text(encoding='utf-8')
    if 'DataReaderAP3D' not in src or 'AP3D-81' not in src:
        raise SystemExit(
            'Your local train.py does not contain the AP3D DataReader integration. '
            'Do not generate variants from the unmodified upstream train.py.'
        )
    if 'lambda_acc' in src or 'sg_window' in src:
        raise SystemExit('train.py already contains temporal experiment changes; refusing to duplicate them.')

    base_yaml = base_config.read_text(encoding='utf-8')
    if 'lambda_acc:' in base_yaml:
        raise SystemExit('Baseline YAML already defines lambda_acc; expected an unmodified baseline config.')
    if 'n_frames: 81' not in base_yaml:
        raise SystemExit('Expected n_frames: 81 in the AP3D baseline YAML.')

    output = {
        ROOT / 'train_accel.py': make_accel(src),
        ROOT / 'eval_sg.py': make_sg(src),
        ROOT / 'configs/ap3d/MotionAGFormer-base-81-acc.yaml': (
            base_yaml.rstrip() + '\n\n'
            '# Additional supervised acceleration-consistency loss.\n'
            '# Units: normalized coordinates per frame squared.\n'
            'lambda_acc: 5.0\n'
        ),
    }

    for path in output:
        if path.exists() and not opt.overwrite:
            raise SystemExit(
                f"Output already exists: {path}. Remove it or use --overwrite "
                "after inspecting any local changes. No files written."
            )

    for path, content in output.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        print(f'[CREATED] {path.relative_to(ROOT)}')

    print('\nSuccess. Baseline train.py and its config remain untouched.')
    print('Before running GPU workloads, check python -m py_compile train_accel.py eval_sg.py')


if __name__ == '__main__':
    main()
