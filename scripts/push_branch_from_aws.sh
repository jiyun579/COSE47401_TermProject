#!/usr/bin/env bash
# Safe, opt-in upload of temporal variants to a separate GitHub branch.
# Never edits the running project or the remote main branch.
set -euo pipefail

REPO_URL='https://github.com/jiyun579/COSE47401_TermProject.git'
BRANCH='feat/temporal-savgol-acceleration-loss'
SOURCE="$HOME/COSE47401_TermProject"
DEST="$HOME/COSE47401_TermProject_upload"
ZIP="$HOME/motionagformer_branch_upload.zip"

for file in "$SOURCE/train.py" \
            "$SOURCE/AthletePose3D/pose_3d/datareader_ap3d.py" \
            "$SOURCE/configs/ap3d/MotionAGFormer-base-81.yaml" \
            "$ZIP"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing required file: $file" >&2
        exit 1
    fi
done
if [[ -e "$DEST" ]]; then
    echo "ERROR: $DEST already exists. Refusing to overwrite it." >&2
    exit 1
fi

# A separate clone avoids modifying in-flight training and uncommitted work.
git clone "$REPO_URL" "$DEST"
cd "$DEST"
git switch -c "$BRANCH"
unzip -n "$ZIP"

cp "$SOURCE/train.py" ./train.py
cp "$SOURCE/AthletePose3D/pose_3d/datareader_ap3d.py" ./AthletePose3D/pose_3d/datareader_ap3d.py
mkdir -p configs/ap3d
cp "$SOURCE/configs/ap3d/MotionAGFormer-base-81.yaml" ./configs/ap3d/MotionAGFormer-base-81.yaml

# Run the local AP3D-adapted code generation (not the unpatched GitHub main train.py).
PYTHON="$SOURCE/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
"$PYTHON" scripts/install_temporal_variants.py
"$PYTHON" -m py_compile train_accel.py eval_sg.py
"$PYTHON" -m unittest discover -s tests -p 'test_installer.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_temporal_extensions.py' -v

git add \
    train.py train_accel.py eval_sg.py \
    AthletePose3D/pose_3d/datareader_ap3d.py \
    configs/ap3d/MotionAGFormer-base-81.yaml \
    configs/ap3d/MotionAGFormer-base-81-acc.yaml \
    experiments/__init__.py experiments/temporal_extensions.py \
    scripts/install_temporal_variants.py scripts/push_branch_from_aws.sh \
    tests/test_installer.py tests/test_temporal_extensions.py \
    README_KO.md BRANCH_UPLOAD_KO.md

if git diff --cached --name-only | grep -E '(^|/)(checkpoint|data)/|\.pkl$|\.pth(\.tr)?$'; then
    echo 'ERROR: Dataset/checkpoint detected in staged changes. Refusing to commit.' >&2
    exit 1
fi

echo '=== Staged source files ==='
git diff --cached --stat

git commit -m 'feat: add SG smoothing and acceleration consistency loss for AP3D'
git push -u origin "$BRANCH"
echo "Uploaded branch: https://github.com/jiyun579/COSE47401_TermProject/tree/$BRANCH"
