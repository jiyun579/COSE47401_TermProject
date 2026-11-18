# MotionAGFormer Base-81: Temporal Variant Add-on

이 파일은 기존 `train.py`, AP3D 데이터셋, 학습 체크포인트를 **수정하지 않습니다.**

`install_temporal_variants.py`는 **AWS EC2에 존재하는 사용자 버전** `train.py`(AP3D DataReader가 추가된 버전)를 읽어 동일한 데이터 로딩·평가 로직을 재사용하는 두 개의 파일을 생성합니다.

## 1. 설치

1. ZIP을 EC2의 `~/COSE47401_TermProject/`로 전송합니다.
2. 가상환경을 활성화하고 압축을 풉니다.

```bash
cd ~/COSE47401_TermProject
source .venv/bin/activate
unzip -n motionagformer_temporal_variants.zip
python scripts/install_temporal_variants.py
python -m py_compile train_accel.py eval_sg.py
python -m unittest discover -s tests -p 'test_*extensions.py' -v
```

프로젝트의 기존 `tests`에 다양한 테스트가 존재하므로 위 명령은 새 모듈 테스트만 실행합니다. `test_installer.py` 검사도 하려면:

```bash
python -m unittest discover -s tests -p 'test_installer.py' -v
```

Python에서 `scipy`가 없다면 기존 `.venv`에서 `python -m pip install scipy`만 실행합니다. `torch`는 재설치하지 않습니다.

### 설치 후 생성되는 파일

- `train_accel.py`: 기존 학습 손실 + 가속도 일관성 손실
- `eval_sg.py`: 사전학습 체크포인트 추론 결과의 SG 평활화 평가 (`--sg-window 0`이면 미적용)
- `configs/ap3d/MotionAGFormer-base-81-acc.yaml`: 기존 AP3D 설정을 복사한 가속도 손실 실험 설정 (`lambda_acc: 5.0`)

재실행할 때 이미 생성한 파일이 있으면 중단하도록 했습니다. 기존 파일 덮어쓰기에는 명시적으로 `--overwrite`가 필요합니다.

## 2. SG 필터 실험 — 추가 학습 불필요

Baseline 모델의 동일한 체크포인트를 **각각 SG 미적용 / 적용**하여 평가합니다. 81프레임 클립 각각의 시간축에 대해 관절별 XYZ에 `scipy.signal.savgol_filter(mode='interp')`를 수행합니다.

> Baseline GPU 학습이 진행 중이라면 두 평가 실행을 동시에 수행하지 마세요. 학습이 완료됐거나 GPU가 비었을 때 아래를 순서대로 실행하세요.

```bash
mkdir -p logs
python -u eval_sg.py \
  --config configs/ap3d/MotionAGFormer-base-81.yaml \
  --checkpoint checkpoint/ap3d_base81_preliminary \
  --checkpoint-file best_epoch.pth.tr \
  --eval-only --num-cpus 4 --wandb-run-id sg-baseline \
  --sg-window 0 \
  > logs/eval_baseline_no_sg.log 2>&1

python -u eval_sg.py \
  --config configs/ap3d/MotionAGFormer-base-81.yaml \
  --checkpoint checkpoint/ap3d_base81_preliminary \
  --checkpoint-file best_epoch.pth.tr \
  --eval-only --num-cpus 4 --wandb-run-id sg-window9 \
  --sg-window 9 --sg-polyorder 3 \
  > logs/eval_baseline_sg9.log 2>&1

for f in logs/eval_baseline_no_sg.log logs/eval_baseline_sg9.log; do
  echo "==== $f ===="
  grep -E '\[SG\]|Protocol #1 Error|Acceleration error|Protocol #2 Error' "$f" | tail -8
done
```

SG가 고속 동작의 급격한 변화를 지울 수 있으니 `--sg-window 5 --sg-polyorder 2` 및 `9,3`을 비교하세요. 필터는 *예측 좌표에만* 적용하며 정답과 학습 가중치는 건드리지 않습니다.

## 3. 가속도 손실 실험 — 별도 모델 학습

기본 손실함수에 다음 항만 추가합니다.

```
L_total = L_baseline + lambda_acc * L_acc
L_acc = mean(||(P_pred[t+2]-2P_pred[t+1]+P_pred[t])
             - (P_gt[t+2]-2P_gt[t+1]+P_gt[t])||_2)
```

이는 무조건 예측 가속도를 0으로 만드는 평활화가 아니라, **실제 정답의 가속도와 예측의 차이**를 낮추는 손실입니다. 기존 `lambda_3d_velocity` 설정을 그대로 유지합니다.

현재 학습과 겹치지 않도록 Baseline 완료 후 별도의 디렉터리로 학습합니다.

```bash
nohup python -u train_accel.py \
  --config configs/ap3d/MotionAGFormer-base-81-acc.yaml \
  --new-checkpoint checkpoint/ap3d_base81_acc \
  --num-cpus 4 --seed 0 --wandb-run-id ap3d-acc-001 \
  > logs/ap3d_base81_acc.log 2>&1 < /dev/null &

echo "PID: $!"
tail -f logs/ap3d_base81_acc.log
```

비교를 위한 공정성: Baseline과 Acceleration 모델은 동일한 데이터 분할, 에포크 수, 학습률 및 seed를 사용합니다. 본 스크립트는 가속도 모델을 Baseline 체크포인트에서 미세조정하지 않고 처음부터 학습합니다.

가속도 손실은 **정규화 좌표/frame² 기준**입니다. 별도의 프레임레이트 변환 없이 물리 단위 `mm/s²`라고 표기하지 마세요.

## 4. 알려진 데이터 문제 (중요)

기존 AP3D 사전 분할 클립에 이상이 확인됐습니다: train 550/19,086, test 74/3,067 (`rm_camera_2` 구간 등). 특히 74개 검증 클립이 `valid.pkl` 재구성 결과와 일치하지 않습니다.

따라서 기존 `train.py`의 `evaluate()`가 출력하는 MPJPE와 Acceleration Error는 **잠정값**이며 논문/최종 보고서에서 확정 성능으로 인용하면 안 됩니다. 데이터 이상을 해결한 동일한 검증 세트에서 Baseline, SG, Acc 변형을 모두 다시 평가해야 합니다.

## 5. 어떤 부분이 달라졌는가?

### SG
- `eval_sg.py`는 기존 `train.py`와 동일한 AP3D 데이터 로딩/평가/모델 생성 로직을 포함합니다.
- `results_all = np.concatenate(results_all)` 다음에만 **옵션에 따라** SG 함수를 호출합니다.
- 독립 클립 처리이므로 서로 다른 비디오 클립을 이어 붙여 평활화하지 않습니다.

### Acc
- `train_accel.py`는 원본을 복제해 `acceleration_consistency_loss`를 training loop의 손실 합에 추가한 것입니다.
- 매 epoch의 가속도 손실을 콘솔에 출력하고 WandB 로깅(선택)에도 별도 항목을 더합니다.
- 가속도 손실의 가중치는 새 YAML의 `lambda_acc`로 조절합니다.

## 6. 유의사항

- Baseline 학습 중에 이 설치를 하더라도 원래의 `train.py`, `configs/ap3d/MotionAGFormer-base-81.yaml`을 수정하지 않으므로 이미 실행된 Python 학습에는 영향을 주지 않습니다.
- 새로 생성한 파일을 Git에 올릴 경우 AP3D `.pkl` 파일 등 라이선스 데이터가 포함되지 않도록 유의하세요.
- `train_accel.py`와 `eval_sg.py`는 현재 프로젝트의 패치된 `train.py`를 기반으로 생성되므로, 원본 upstream `train.py`에 AP3D가 연결되어 있지 않으면 설치 프로그램이 거부합니다.
