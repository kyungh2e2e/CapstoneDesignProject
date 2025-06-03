# 24-2 캡스톤디자인과창업프로젝트 34팀 - IT인

## 👩‍💻 멤버

|[심영은](https://github.com/yeongeunshim)|[김경희](https://github.com/kyungh2e2e)|[이채원](https://github.com/chae-jpg)|
|:--:|:--:|:--:|
|팀장|팀 레포지토리 관리|팀 자원 및 산출물 관리|


## 📚 연구 주제

**Linceiver IO: Reducing Computational Cost in
Federated Learning with Adaptive Low-Rank
Perceiver IO** (Linceiver IO: 적응형 저차원 Perciever IO를 활용한 연합 학습의 연산 비용 절감)

## 📁 폴더 구조

<pre style="background-color: #1F3737; padding: 10px; border-radius: 5px; color: #ffffff;">
<code>
Start
├── FL_Test
│   └── Federated_Learning_Flower_Pytorch.ipynb  # Flower를 이용한 연합학습 구현 파일입니다.
└── Linformer_Test
    └── Lin_VS_Trans.ipynb  # Linformer 와 Transformer 모델의 성능 비교 테스트 파일입니다.
Growth
├── AdaptiveKTest
│   └── lin_cifar.py # MNIST (흑백 이미지) 데이터셋을 대상으로 Adaptive K 알고리즘 검증 실험을 수행하는 스크립트 입니다.
│   └── lin_mnist.py # CIFAR-10 (컬러 이미지) 데이터셋을 대상으로 Adaptive K 알고리즘 검증 실험을 수행하는 스크립트 입니다.
├── LinceiverIO-StandAlone
│   └── main.py # 단일 실험 실행 스크립트입니다.
│   └── perceiver_io_linstyle.py # Linceiver IO 모델이 정의된 파일입니다.
│   └── run_batch.sh # 다양한 k에 대한 Linceiver IO 실험을 batch로 실행하는 스크립트입니다. 
│   └── report_batch.py # 실험 수행 후 생성된 결과를 평균내어 report.txt로 변환해주는 스크립트입니다.
└── LinceiverIO-FederatedLearning
    └── federated_main_per.py # 단일 연합학습 실험 실행 스크립트입니다.
    └── federated_batch.sh # 여러 hyperparameter 조합에 대해 batch로 federated_main_per.py를 실행하는 스크립트입니다.
    └── options.py # 커맨드라인 옵션 파서를 정의합니다.
    └── sampling.py # 데이터 샘플링 방식을 정의합니다.
    └── utils.py # 데이터셋 로드 및 기타 유틸리티 함수를 정의합니다.
    └── test_with_sharing_lat.py # Linceiver IO 모델을 연합학습을 위해 Shared Backbone과 Perceiver Head로 분리하여 정의합니다.
</code>
</pre>

## 🔗 관련 링크

[그라운드 룰](https://github.com/Capstone-IT-in/CapstoneDesignProject/blob/main/Ground_Rule.md)] <br>
[그로쓰 실험 영상](https://youtu.be/PSTCc0QR7L0?feature=shared)

