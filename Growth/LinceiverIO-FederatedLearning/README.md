# README

## Requirements
- Python 3.8.20
- PyTorch 1.10.2
- einops 0.8.0
- tensorboardX 2.1

설치 명령어:
```bash
pip install torch==1.10.2 torchvision==0.11.3 einops==0.8.0 tensorboardX==2.1
```

## Project Overview

본 프로젝트는 Federated Learning 환경에서 **Perceiver IO 스타일** 모델을 사용하여 FashionMNIST 데이터셋을 학습하는 코드입니다.

- **`federated_main_per.py`**: 메인 실행 파일. Federated 학습 로직이 구현되어 있습니다.
- **`federated_batch.sh`**: 여러 hyperparameter 조합에 대해 batch로 `federated_main_per.py`를 실행하는 스크립트입니다.
- **`options.py`**: 커맨드라인 옵션 파서를 정의합니다.
- **`sampling.py`**: 데이터 샘플링 방식을 정의합니다 (IID/Non-IID 분할).
- **`test_with_sharing_lat.py`**: Shared Latent Perceiver 모델 및 Head 구조 정의.
- **`utils.py`**: 데이터셋 로드 및 기타 유틸리티 함수들.

## How to Run

### 1. Single Experiment

```bash
python federated_main_per.py --epochs 50 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 10 --lr 0.01 --k 128 --test_no 0 --dataset fmnist --iid 1
```

### 2. Batch Run

```bash
bash federated_batch.sh
```

- 다양한 실험 설정(`k`, `test_no` 등)을 자동 반복 실행합니다.

### Options 설명
| 옵션명 | 설명 | 기본값 |
|:------|:-----|:------|
| `--epochs` | 전체 학습 라운드 수 | 10 |
| `--num_users` | 전체 클라이언트 수 | 100 |
| `--frac` | 매 라운드에 참여할 클라이언트 비율 | 0.1 |
| `--local_ep` | 각 클라이언트 로컬 학습 epoch 수 | 10 |
| `--local_bs` | 로컬 배치 사이즈 | 10 |
| `--lr` | 학습률 | 0.01 |
| `--momentum` | SGD momentum (미사용) | 0.5 |
| `--test_no` | 실험 번호 (결과 구분용) | 0 |
| `--model` | 모델 이름 | perceiver-io-linstyle |
| `--dataset` | 사용할 데이터셋 (mnist, fmnist, cifar) | fmnist |
| `--iid` | 1이면 IID, 0이면 Non-IID 분포 | 1 |
| `--unequal` | Non-IID 데이터 불균등 분할 여부 | 0 |
| `--k` | Linformer projection dimension | 128 |

### 결과 저장
- 모든 실험 결과는 `./save/fmnist/` 폴더 아래에 **자동 저장**됩니다.
- 파일명 예시: `fed_linceiverio_report_{local_ep}_{k}_{test_no}.txt`
- 내부에는 각 라운드별 Loss, 정확도, 최종 실행시간 등이 기록됩니다.
