# README

## Requirements
- `Python` 3.8.20
- `PyTorch` 1.10.2
- `einops` 0.8.0

(추가로 torchvision이 자동 설치되어야 합니다.)

```bash
pip install torch==1.10.2 torchvision==0.11.3 einops==0.8.0
```

## How to Run

### Step 1: 파일 설명
- `main.py`: 단일 실험 실행 스크립트 (모델 학습 및 평가)
- `perceiver_io_linstyle.py`: Perceiver IO 모델 (Linformer 스타일) 정의 파일
- `run_batch.sh`: 여러 실험(batch run) 자동 실행 스크립트

### Step 2: run_batch.sh 실행 방법

```bash
bash run_batch.sh
```

- 스크립트 내부에서 여러 다른 `k` 값과 `test_no`를 조합하여 `main.py`를 반복 실행합니다.
- 각 실행마다 지정된 `k`, `test_no`에 해당하는 결과 폴더가 자동으로 생성됩니다.

### Step 3: main.py 주요 실행 옵션

`run_batch.sh`는 내부적으로 아래와 같은 명령어를 실행합니다.

```bash
python main.py --model perceiver-io-linstyle --epochs=10 --batch_size=64 --lr=5e-4 --weight_decay=1e-4 --k=128 --test_no=1
```

#### Arguments 설명
| 인자 | 설명 | 기본값 |
|:----|:----|:----|
| `--model` | 사용할 모델 종류. `perceiver-io` 또는 `perceiver-io-linstyle` 선택 | perceiver-io |
| `--epochs` | 학습할 에폭 수 | 10 |
| `--batch_size` | 배치 크기 | 64 |
| `--lr` | 학습률 | 5e-4 |
| `--weight_decay` | Weight decay 값 | 1e-4 |
| `--k` | Linformer 압축 차원 | 128 |
| `--test_no` | 실험 번호 (결과 폴더 구분용) | 없음 |

### Step 4: 결과 저장

- 결과 폴더(`./results/{k}/{test_no}/` 또는 `./results-pio/{k}/{test_no}/`)가 **자동 생성**됩니다.
- 각 폴더에는 다음과 같은 파일들이 저장됩니다:
  - 모델 파라미터 (`.pth`)
  - 학습 및 테스트 loss/accuracy 기록 (`.txt`)
  - Loss 및 Accuracy 플롯 이미지 (`.png`)

## 참고
- `device = cuda` 가용 여부에 따라 자동으로 GPU를 사용합니다.
- 데이터셋은 Fashion-MNIST를 자동 다운로드합니다.


