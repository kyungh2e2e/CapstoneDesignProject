# Adaptive K Test

## Source Code

| 파일명 | 설명 |
|--------|------|
| `lin_mnist.py` | MNIST(흑백 이미지) 데이터셋을 대상으로 실험을 수행하는 스크립트 |
| `lin_cifar.py` | CIFAR-10(컬러 이미지) 데이터셋을 대상으로 실험을 수행하는 스크립트 |
| `Perceiver/` | Perceiver IO 모델 본체 및 Attention 구조 정의 포함 |
| `LinformerAttention/` | Adaptive k를 구현하는 핵심 Attention 모듈 |
| `outputs/` | 시각화된 실험 결과(Accuracy, Adaptive k, Loss 등) 이미지 저장 폴더 |

---

## ⚙️ How to Build

본 프로젝트는 별도의 빌드가 필요하지 않습니다. Python 스크립트 실행만으로 모든 기능이 동작합니다.

---

## 📦 How to Install

다음 명령어로 필수 라이브러리를 설치하세요:

```bash
pip install torch==1.10.2 torchvision==0.11.3 einops==0.8.0 matplotlib
```

**필수 환경:**
- Python 3.8 이상
- PyTorch 1.10.2
- torchvision 0.11.3
- einops 0.8.0
- matplotlib

실행에 오랜 시간이 소요될 수 있으므로 GPU를 사용 가능한 환경에서 실험을 진행하시는 걸 추천드립니다.

---

## 🧪 How to Test

### 1. MNIST 실험 실행

```bash
python lin_mnist.py
```

### 2. CIFAR-10 실험 실행

```bash
python lin_cifar.py
```

### 결과 확인

각 스크립트를 실행하면 `./outputs/` 디렉토리에 다음과 같은 이미지가 생성됩니다:

- `mnist_accuracy.png`, `mnist_k.png`
- `cifar_accuracy.png`, `cifar_k.png`

---

## 📁 Sample Data

- MNIST 및 CIFAR-10 데이터셋은 `torchvision.datasets`를 통해 **자동 다운로드**됩니다.
- 별도의 수동 다운로드나 외부 데이터 준비는 필요하지 않습니다.

---

## 📚 Used Open Source Libraries

본 프로젝트는 다음의 오픈소스 라이브러리를 활용합니다:

- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크 (License: BSD)
- [Torchvision](https://github.com/pytorch/vision) - 데이터셋 및 변환 도구 (License: BSD)
- [einops](https://github.com/arogozhnikov/einops) - 텐서 재구성 유틸리티 (License: MIT)
- [matplotlib](https://matplotlib.org/) - 시각화 도구 (License: PSF)

---

## 🧠 Project Overview

이 프로젝트는 Perceiver IO에 Linformer 기반 Adaptive Attention을 적용한 **Linceiver IO** 모델을 구현합니다. 입력 복잡도에 따라 압축 차원 `k`를 **동적으로 조절**하는 구조로, 효율성과 정보 보존을 동시에 추구합니다.

### 특징:
- 단순한 입력(MNIST)과 복잡한 입력(CIFAR-10)에서 adaptive k 조절 실험
- `effective k`는 학습 중 `sigmoid(gate_k) * alpha_k`로 계산되어 효율적인 차원 사용 달성

---

## 📈 실험 결과물 및 해석 예시

- **MNIST**: `k` 값이 빠르게 감소하여 저차원에서도 안정적으로 학습 가능
- **CIFAR-10**: 고차원 입력에도 점진적인 차원 축소가 적용되어 정보 손실 없이 효율 유지

- 아래는 MNIST와 CIFAR-10에 대한 Linceiver IO의 학습 결과를 시각화한 그래프입니다:
<img src = "https://github.com/kyungh2e2e/CapstoneDesignProject/blob/79daad0108ffe71408325609568af5d1e586a4c3/AdaptiveK%EA%B2%B0%EA%B3%BC.png" width="600"/>

**MNIST 결과 요약**

| 항목 | 값 |
|------|----|
| 초기 Accuracy | 약 0.82 |
| 최종 Accuracy | 약 0.965+ |
| k 감소 범위 | 60 → 6 |
| 특징 | 단순 이미지 → 과감한 차원 축소 |

- Accuracy는 1 epoch부터 빠르게 상승해 10 epoch 안에 95%에 도달하며, 이후 미세한 진동을 보이며 수렴하였습니다.
- Adaptive k는 epoch가 진행됨에 따라 빠르게 감소하여, 30 epoch 기준 평균 6 수준으로 매우 낮아졌습니다.
- 이는 MNIST가 단순한 입력 구조를 가지므로, 모델이 많은 attention 차원을 제거해도 정확도를 유지할 수 있음을 보여줍니다.
- Adaptive Attention은 계산량을 줄이면서도 높은 정확도를 유지하는 데 효과적임을 확인할 수 있습니다.

---

**CIFAR-10 결과 요약**

| 항목 | 값 |
|------|----|
| 초기 Accuracy | 약 0.32 |
| 최종 Accuracy | 약 0.56+ |
| k 감소 범위 | 79 → 61 |
| 특징 | 복잡 이미지 → 차원 유지 |

- Accuracy는 초반 32%에서 시작하여 완만하게 상승, 35 epoch 기준 약 55~56% 수준에 도달하였습니다.
- Adaptive k는 초기 값 79에서 시작해 35 epoch 기준 61까지 천천히 감소하였으며, MNIST에 비해 훨씬 많은 차원을 유지하였습니다.
- 이는 CIFAR-10처럼 시각적으로 복잡한 데이터셋에서는 더 많은 정보 보존이 필요함을 보여줍니다.
- Adaptive Attention은 정보 손실을 피하면서도 연산 효율을 개선하려는 전략을 잘 수행하고 있음을 알 수 있습니다.

---

**결론:**  
MNIST와 CIFAR-10 모두에서 Adaptive k는 입력 복잡도에 따라 다르게 조절되며, 정보 보존과 연산 최적화 간의 균형을 자동으로 학습하는 경향을 보여줍니다.

---

## ▶️ 실행 스크립트 예시

다음과 같은 스크립트를 통해 전체 실험을 한 번에 실행할 수 있습니다:

```bash
#!/bin/bash
echo "Running MNIST..."
python lin_mnist.py

echo "Running CIFAR-10..."
python lin_cifar.py

echo "Done. Results saved in outputs/"
```
---

## 🚀 Google Colab에서 바로 실행

Python 스크립트 실행 대신 Google Colabatory 파일로도 실행 가능합니다.

Linceiver IO의 Adaptive K 실험을 직접 실행해보고 싶다면 아래 Colab 버튼을 눌러보세요:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Capstone-IT-in/CapstoneDesignProject/blob/main/Growth/AdaptiveKTest/AdaptiveKTest.ipynb)

해당 ipynb파일을 사용해 설명한 튜토리얼 블로그 글도 있으니 참고해주세요.

✍️ 관련 글 보기: [Adaptive K 실험 정리 블로그 포스트](https://kyungcotry.tistory.com/9)

