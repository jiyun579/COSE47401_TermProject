# GitHub 실험 브랜치 업로드 안내

원격 저장소: https://github.com/jiyun579/COSE47401_TermProject
예정 브랜치: `feat/temporal-savgol-acceleration-loss`

## 포함된 코드

- `experiments/temporal_extensions.py`: Savitzky–Golay 필터와 미분 가능한 가속도 오차 손실
- `scripts/install_temporal_variants.py`: **기존에 AP3D를 연결해 둔 로컬 `train.py`**에서 `eval_sg.py`와 `train_accel.py`를 생성하는 설치 스크립트
- `tests/test_temporal_extensions.py`, `tests/test_installer.py`: 단위 테스트
- `README_KO.md`: 학습·평가 방법, 정합성 문제, 비교 실험 설명

**주의**: GitHub의 현재 `main`에는 AWS에서 수정한 AP3D 통합 `train.py`와 `configs/ap3d/MotionAGFormer-base-81.yaml`이 없으므로, 이 패키지로부터 생성한 두 실행 파일은 AWS의 패치된 코드 기반입니다. 실험 브랜치를 완전히 재현하려면 그 파일들 및 `AthletePose3D/pose_3d/datareader_ap3d.py`의 AWS 수정본도 함께 커밋해야 합니다. 원본 데이터 `.pkl`과 체크포인트는 절대 커밋하지 마세요.

## 추천: AWS에서 사용 중인 프로젝트를 그대로 두고 별도 클론으로 푸시

1. 본 ZIP을 AWS 홈(`~/`)에 전송하세요. 아래에서 ZIP 이름은 `motionagformer_branch_upload.zip`이라고 가정합니다.
2. AWS 터미널에서 아래 명령을 실행하세요.

```bash
cd ~
git clone https://github.com/jiyun579/COSE47401_TermProject.git COSE47401_TermProject_upload
cd COSE47401_TermProject_upload
git switch -c feat/temporal-savgol-acceleration-loss
unzip -n ~/motionagformer_branch_upload.zip

# 현재 가동 중인 EC2 프로젝트의 AP3D 수정 사항을 안전하게 복사 (원본은 변경 안 함)
cp ~/COSE47401_TermProject/train.py ./train.py
cp ~/COSE47401_TermProject/AthletePose3D/pose_3d/datareader_ap3d.py ./AthletePose3D/pose_3d/datareader_ap3d.py
mkdir -p configs/ap3d
cp ~/COSE47401_TermProject/configs/ap3d/MotionAGFormer-base-81.yaml ./configs/ap3d/

# AP3D 학습/평가 실행 파일 생성
python scripts/install_temporal_variants.py
python -m py_compile train_accel.py eval_sg.py
python -m unittest discover -s tests -p 'test_installer.py' -v
python -m unittest discover -s tests -p 'test_temporal_extensions.py' -v

# 새 브랜치에 코드만 커밋: 데이터와 체크포인트는 제외
git add train.py train_accel.py eval_sg.py \
  AthletePose3D/pose_3d/datareader_ap3d.py \
  configs/ap3d/MotionAGFormer-base-81.yaml \
  configs/ap3d/MotionAGFormer-base-81-acc.yaml \
  experiments/__init__.py experiments/temporal_extensions.py \
  scripts/install_temporal_variants.py \
  tests/test_installer.py tests/test_temporal_extensions.py \
  README_KO.md BRANCH_UPLOAD_KO.md

git diff --cached --stat
git commit -m "feat: add SG smoothing and acceleration loss experiments for AP3D"
git push -u origin feat/temporal-savgol-acceleration-loss
```

GitHub 인증이 필요하면 해당 저장소의 쓰기 권한이 있는 GitHub 계정으로 로그인하세요. 실행 전에 `git diff --cached --stat`으로 데이터·체크포인트가 올라가지 않는 것을 확인하세요. `git push`는 `main`을 변경하지 않으며 새 브랜치를 업로드합니다.

## 왜 두 실행 파일을 그대로 ZIP에 넣지 않았는가?

두 실행 파일은 이미 AWS에서 수정해 둔 `train.py`의 AP3D 데이터 리더 선택, 검증용 평가 수정을 그대로 보존해야 합니다. 이 세부 변경은 현재 원격 저장소의 `main`에서 확인되지 않습니다. 따라서 확인할 수 없는 수정본을 추측해서 복제하지 않고, **AWS 원본을 복사한 별도 클론에서 안전하게 생성하는 방식**을 사용합니다.
