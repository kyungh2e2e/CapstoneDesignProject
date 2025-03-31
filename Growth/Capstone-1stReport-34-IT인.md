<!-- Template for PROJECT REPORT of CapstoneDesign 2025-2H, initially written by khyoo -->
<!-- 본 파일은 2025년도 컴공 졸업프로젝트의 <1차보고서> 작성을 위한 기본 양식입니다. -->
<!-- 아래에 "*"..."*" 표시는 italic체로 출력하기 위해서 사용한 것입니다. -->
<!-- "내용"에 해당하는 부분을 지우고, 여러분 과제의 내용을 작성해 주세요. -->

# Team-Info
| (1) 과제명 | *Linceiver IO: Reducing Computational Cost in Federated Learning with Adaptive Low-Rank Perceiver IO*
|:---  |---  |
| (2) 팀 번호 / 팀 이름 | *34-IT인* |
| (3) 팀 구성원 | 심영은 (2194058): 리더, *실험용 서버구축 및 운영, 연합학습 환경 설계 및 구현, 통계분석 및 시각화* <br> 김경희 (2276034): 팀원, *모델 경량화 및 효율성 최적화, seqeunce length에 적응적으로 대응하는 모델 설계 및 구현* <br> 이채원 (2171039) : 팀원, *팀 레포지토리 및 산출물 관리, 모델 최적화 및 성능 개선, 실험 확장을 위한 다중 데이터셋 통합 구축 및 데이터 전처리*			 |
| (4) 팀 지도교수 | *이형준 교수님* |
| (5) 과제 분류 | *연구 과제* |
| (6) 과제 키워드 | *Perceiver IO, Linformer, Federated Learning*  |
| (7) 과제 내용 요약 | 현대의 분산 데이터 환경에서는 클라이언트들이 서로 다른 데이터 유형을 보유하고 있으며, 이는 연합학습의 성능 저하로 이어진다. 본 과제는 멀티모달 연합학습에서 이질성 문제(Heterogeneity Gap)를 해결하면서도, 자원이 부족한 디바이스에서도 사용 가능한 경량 모델을 개발하고자 한다. 이 프로젝트에서는 범용성이 뛰어난 Perceiver IO 모델을 컴퓨팅 자원이 부족한 연합학습 클라이언트에 적용하기 위한 경량화 모델인 Linceiver IO를 제안한다. Linceiver IO의 경우 Perceiver IO의 attention 연산에 Linformer 구조를 적용하여 계산 복잡도를 줄일 수 있다. |


<br>

# Project-Summary
| 항목 | 내용 |
|:---  |---  |
| (1) 문제 정의 |  |
| (2) 기존연구와의 비교 | *유사한 과제/연구/서비스/시스템의 예를 들고, 각각의 장단점을 기술할 것. 특히, 본 과제가 유사과제에 대하여 갖는 장점을 부각할 것* |
| (3) 제안 내용 | 1. **Multi-modal 연합학습의 유연성 향상** <br> 기존 Multi-modal 연합학습은 학습 가능한 모달리티가 정해져 있다. 즉, 이미지-텍스트를 학습하는 모델이라면 그 외의 영상이나 음성과 같은 다른 input에 대한 일반화 성능이 떨어지는 한계가 존재한다. 하지만, 본 연구는 다양한 모달리티의 데이터를 통합적으로 처리 가능한 멀티모달 모델인 Perceiver IO를 활용하여 client가 사용하는 모달리티에 제약을 받지 않는 보다 유연한 연합학습 환경을 제안한다.  <br><br> 2. **Perceiver IO의 경량화 및 연합학습 통합**  <br> Perceiver IO는 앞서 말한 것처럼 다양한 모달리티의 데이터를 모두 학습할 수 있는 범용성 있는 multi-modal 학습을 가능하게 하는 강력한 모델이다. 하지만, 이 모델은 attention 연산을 반복하여 수행하기 때문에 계산 복잡도가 높고, 따라서 자원이 한정되어 있는 연합학습 환경에서의 적용이 어렵다는 한계를 가진다. 또한, Perceiver IO라는 모델이 기존 중앙 학습 환경에서의 적용을 염두에 두고 만들어진 모델이기 때문에, 이 모델 자체를 연합 학습 환경에서 사용하기 위해서는 모델 구조의 수정이 필요하다. 이 문제를 해결하기 위해, 본 연구는 Perceiver IO의 attention 연산을 경량화하였고, 연합 학습 환경에 맞추어 구조를 수정하였다. 마지막으로, attention 연산 경량화에 있어 데이터를 압축하는 파라미터를 동적으로 선택하도록 하여, 최적화가 더욱 효율적으로 수행될 수 있도록 하였다. |
| (4) 기대효과 및 의의 | 1. **다양한 모달리티 조합을 수용하는 범용 연합학습 프레임워크 실현** <br> 본 연구는 Perceiver IO의 다양한 모달리티를 통합할 수 있는 능력을 활용해, 다양한 조합의 모달리티를 받아들일 수 있는 연합학습 환경을 제안한다. 이는 기존의 제한된 모달리티 조합만을 처리할 수 있는 연합학습 연구들과 차별화되며, 현실적인 데이터 다양성을 반영한 시스템 설계라는 점에서 높은 의의가 있다. <br> 2. **연합학습 환경에 적합한 고성능 멀티모달 모델의 경량화**  <br> 계산량이 많고 메모리 사용이 큰 Perceiver IO 모델을 경량화하고 구조를 수정함으로써, 자원이 제한된 디바이스(예: 모바일, 센서 장치)에서도 학습 참여가 가능하도록 하였다. 이는 연합학습의 실제 적용 가능성을 크게 높이며, 다양한 실세계 응용(의료, 스마트홈, 사용자 행동 분석 등)에 바로 활용될 수 있다. <br> 3. **학술적 기여와 후속 연구 기반 마련** <br> 본 프로젝트는 Perceiver IO 기반의 경량화 및 구조 수정이라는 새로운 시도를 통해, 범용 멀티모달 모델의 연합학습 적용 가능성을 실험하였다. 이는 이와 같은 범용 멀티모달 모델의 최적화와 분산 학습 환경에서의 적용과 관련된 연구에 있어 기초적인 참고 사례로 사용될 수 있는 의의를 가진다. |
| (5) 주요 기능 리스트 | 1. **Linformer를 통한 Perceiver IO 경량화** <br> Perceiver IO는 반복적인 attention 연산을 수행해 연합학습 환경에서의 적용은 어려운 계산 복잡도가 큰 무거운 모델이라는 한계점을 가진다. 이 점을 해결하기 위해 Transformer의 attention 연산을 선형 시간에 수행할 수 있도록 경량화한 모델인 Linformer를 사용하였고, 이를 적용한 Perceiver IO를 Linformer IO로 명명했다. 이를 통해 계산 복잡도를 O(N^2)에서 O(kN)으로 줄이면서, 모델의 크기와 추론 시간을 모두 감소시킬 수 있게 된다. <br> 2. **연합학습을 위한 Perceiver IO 구조 설계** <br> Perceiver IO라는 모델이 중앙 학습을 위한 모델이고, 기존의 다른 모델들과는 다르게 weight를 합산하여 나누고 다시 분배할 수 있는 구조를 가진 모델이 아니기 때문에, 이를 연합학습 환경에 통합하기 위한 독자적인 구조를 설계하는 것이 필요했다. 이를 위해 본 연구에서는 Perceiver IO를 크게 Shared Latent Perceiver와 Perceiver Head로 나누어 Perceiver IO 모델로도 연합 학습이 가능하도록 하였다. <br> 3. **동적으로 데이터 압축 정도 선택 구현** <br> Linformer에서의 근사에서는 projection 차원 k가 큰 영향을 미친다. 이 축소 과정을 본 연구에서는 특정 값으로 설정하는 것이 아니라, 데이터에 따라 변동 가능하도록 하여 데이터 압축과 효율성 사이의 적절한 사이를 찾아 효과적으로 모델의 경량화가 가능하도록 하였다.|

<br>
 

# Project-Design & Implementation

| 항목 | 내용 |
|:---  |---  |
| (1) 요구사항 정의 |   **1. 경량화** <br> - Linceiver IO는 Perceiver IO에 비해 빠른 학습 속도를 보이고, 더 적은 용량의 모델 크기를 갖춰야 한다. <br> - Attention 연산의 복잡도를 기존 $O(N^2)$에서 $O(kN)$으로 줄이기 위한 Low-Rank Projection 기반의 구조(Linformer 방식)를 적용해야 한다. <br> - 모델 파라미터 수는 가능한 한 최소화하여 연합학습 클라이언트에서 동작할 수 있도록 경량화를 달성해야 한다. <br><br> **2. 범용성** <br> - Linceiver IO는 다양한 크기의 데이터, 더 나아가 다양한 모달리티 데이터에서 안정적인 학습 효과를 보여야 한다. <br> - Linceiver IO의 projection 차원 k는 학습 과정에서 동적으로 변경 가능하며, 정확도와 학습 속도 및 용량에서 최적의 Trade-off를 만족하는 수치로 설정되어야 한다. <br><br> **3. 연합학습 적용성** <br> - 기존 Perceiver IO가 연합학습에 적용된 사례가 없는 만큼, 본 프로젝트는 Linceiver IO의 구조를 연합학습 시스템에 최적화하여 새롭게 설계해야 한다. <br> - 연합학습 환경에서 Shared Latent Perceiver와 개별 클라이언트의 Perceiver Head로 구성된 하이브리드 구조를 적용해야 한다. <br> - 클라이언트 단은 독립적인 데이터셋을 기반으로 로컬 학습을 수행하며, 중앙 서버는 각 클라이언트로부터 데이터를 수합하지 않고, Shared Latent Perceiver의 가중치를 받아 평균화하는 방식으로 학습을 통합해야 한다. <br> - 학습된 모델은 중앙집중형 학습 모델 대비 유사하거나 더 나은 성능을 달성해야 한다. |
| (2) 전체 시스템 구성 | <img src="https://github.com/yeongeunshim/CapstoneDesignProject/blob/main/%EC%8B%9C%EC%8A%A4%ED%85%9C%EA%B5%AC%EC%A1%B0.jpg?raw=true" alt="시스템 구성도" width="480"/> <br><br> **1. User (연구팀원)** <br> - 개인 PC 또는 노트북에서 VS Code를 실행하고 원격 서버에 접속함 <br> - 서버 내부의 코드 작성, 실행, 디버깅을 수행함 <br><br> **2. VS Code Remote - SSH** <br> - 사용자의 로컬 VS Code에서 서버에 안전하게 접속하게 해주는 도구 <br> - 코드 수정, 실험 실행 등을 로컬에서 바로 하듯이 원격으로 수행 가능 <br><br> **3. ML/DL 개발환경** <br> - Python과 PyTorch 기반의 소프트웨어 환경이 구성되어 있음 <br> - 아나콘다 가상환경으로 통일된 패키지 및 라이브러리 버전 유지 <br><br> **4. GPU 서버 환경** <br> - 실제 연산이 수행되는 장소로, 고성능 GPU가 장착된 서버임 <br> - Ubuntu 리눅스 OS를 기반으로 실행되며, PyTorch에서 GPU를 활용해 모델 학습과 추론을 처리함 |
| (3) 주요엔진 및 기능 설계 | **1. VS Code Remote SSH** <br> - 팀원과 서버 간의 연결을 중계하는 원격 개발 도구로, 로컬에서 VS Code를 실행하고 Remote SSH 확장을 통해 서버에 접속함 <br> - 서버 내부 파일을 직접 수정하거나 실행 가능하며, Microsoft에서 개발한 오픈소스 확장 기능임 <br><br> **2. Python 3.8.20** <br> - 프로젝트의 주 개발 언어로, 다양한 머신러닝 라이브러리와의 높은 호환성으로 채택됨 <br><br> **3. PyTorch 1.10.2** <br> - Perceiver IO 및 Linceiver IO 모델 구현, 학습 및 추론에 사용되는 Python 기반 딥러닝 프레임워크 <br><br> **4. Ubuntu 20.04.6** <br> - 서버 운영체제로 사용되는 오픈소스 기반 리눅스 배포판 <br><br> **5. NVIDIA GeForce RTX 3070** <br> - CUDA 기반 GPU 연산 가속이 가능한 고성능 그래픽카드로, PyTorch와 호환되어 빠른 연산을 지원함 <br><br> **6. Anaconda 가상환경** <br> - 패키지 의존성 충돌 방지를 위해 모든 개발자가 동일한 환경을 사용함 <br> - 환경 복제를 통해 실험의 재현성 확보 가능 |
| (4) 주요 기능의 구현 | **1. Perceiver IO의 Attention 연산을 경량화** <br> 본 논문에서는 범용성이 뛰어난 Perceiver IO 모델을 컴퓨팅 자원이 부족한 연합학습 클라이언트에 적용하기 위한 경량화 모델인 Linceiver IO를 제안한다. Linceiver IO의 경우 Perceiver IO의 attention 연산에 low-rank projection을 활용하는 Linformer 구조를 적용하여 계산 복잡도를 $O(N^2)$에서 $O(kN)$로 줄일 수 있다. 설계한 Linceiver IO의 구조는 아래와 같다. <br><br> <img src="https://github.com/yeongeunshim/CapstoneDesignProject/blob/main/linceiver%20io%20%EA%B5%AC%EC%A1%B0.jpg?raw=true" alt="Linceiver IO 구조도" width="480"/> <br><br> **2. 다양한 사이즈 및 모달리티 데이터에 범용적으로 사용 가능한 모델 설계** <br> Low-Rank Projection에서 사용하는 k값은 학습가능한 파라미터를 추가하여 학습과정에서 다른 파라미터들과 같이 업데이트 된다. 입력 데이터에 따라 각 Low-rank Projection 차원의 상대적 중요도를 동적으로 결정하여, 필요한 정보만 효과적으로 선택하고, 불필요한 차원의 정보를 억제함으로써 효율적인 학습을 수행할 수 있다. <br><br> **3. 설계한 모델에 맞는 새로운 연합학습 구조 제안** <br> Perceiver IO 구조를 연합학습에 적용한 사례가 없기 때문에, 설계한 모델이 연합학습 환경에서 잘 동작할 수 있게끔 연합학습 구조와 시스템, 정책 등을 독자적으로 설계하였다. 본 연구에서는 Linceiver IO를 Shared Latent Perceiver, Perceiver Head로 분리하여 연합학습 시스템을 설계했다. Shared Latent Perceiver는 연합학습 환경에서 모든 클라이언트가 공유하는 백본 네트워크로 동작하며, 클라이언트들이 학습한 가중치를 수집하고 평균화하여 지속적으로 업데이트하도록 했다. 한편, 클라이언트는 자체적인 Embedding Layer, Perceiver Head와 함께 학습 가능한 Query Vector를 보유하고 있으며, 독립적인 로컬 데이터셋을 활용하여  학습한다. 전체 연합학습 시스템의 구조도는 아래와 같다. <br><br> <img src="https://github.com/yeongeunshim/CapstoneDesignProject/blob/main/%EC%97%B0%ED%95%A9%ED%95%99%EC%8A%B5%EA%B5%AC%EC%A1%B0.jpg?raw=true" alt="연합학습 구조도" width="480"/> |
| (5) 기타 | *기타 사항을 기술* |

<br>
