# README

## Requirements

- Python 3.8+
- PyTorch 1.10.2
- torchvision 0.11.3
- einops 0.8.0
- matplotlib

설치 명령어:

```bash
pip install torch==1.10.2 torchvision==0.11.3 einops==0.8.0 matplotlib
```

## Project Overview

본 프로젝트는 Perceiver IO에 Linformer 기반 Adaptive Attention을 통합한 **Linceiver IO** 모델을 사용하여, 입력 복잡도에 따라 압축 차원 k가 **동적으로 조절**되는지를 실험하는 코드입니다.

- `lin_mnist.py`: MNIST (흑백 이미지) 데이터셋을 대상으로 실험을 수행하는 스크립트
- `lin_cifar.py`: CIFAR-10 (컬러 이미지) 데이터셋을 대상으로 실험을 수행하는 스크립트
- `outputs/`: 학습 결과 그래프(`accuracy`, `adaptive k`, `loss`)가 저장되는 폴더

두 실험 모두 동일한 Linceiver IO 구조를 사용하며, 입력 복잡도 차이에 따른 `adaptive k`의 변화 양상을 시각화합니다.

## How to Run

### 1. MNIST 실험 실행

```bash
python lin_mnist.py
```

### 2. CIFAR-10 실험 실행

```bash
python lin_cifar.py
```


실행 시 `./` 디렉토리에 다음과 같은 결과 이미지가 생성됩니다:

- `mnist_accuracy.png`, `mnist_k.png`
- `cifar_acc.png`, `cifar_k.png`

## 실험 목적

- 단순한 입력(MNIST)과 복잡한 입력(CIFAR-10)에서 Linceiver IO의 `adaptive k`가 어떻게 다르게 조절되는지 관찰합니다.
- `effective k`는 학습 중 `sigmoid(gate_k) * alpha_k`로 계산되며, 낮은 차원으로 수렴할수록 연산 효율이 높습니다.

## 결과 해석 예시

- MNIST에서는 `k`가 빠르게 감소하고 낮은 값에서 수렴하여 고효율 학습 구조를 형성합니다.
- CIFAR-10에서는 `k`가 점진적으로 감소하고 일정 수준 이하로는 유지되어 정보 손실을 방지합니다.

## 구조 설명

| 파일명 | 설명 |
| --- | --- |
| `lin_mnist.py` | MNIST 기반 Adaptive k 실험 코드 |
| `lin_cifar.py` | CIFAR-10 기반 Adaptive k 실험 코드 |
| `PerceiverIO` | 모델 본체 및 attention 구조 정의 포함 |
| `LinformerAttention` | Adaptive k를 구현하는 핵심 attention 모듈 |
| `outputs/` | 시각화된 결과 이미지 저장 폴더 |
