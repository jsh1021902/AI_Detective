# 2023 경영경제대학 학술제

## 개요
> **2023 경영경제대학 학술제** <br/> 
> **프로젝트 기간: 2023.05.02 ~ 2023.05.31** <br/>

<br>

## 프로젝트 소개
- 합성데이터의 성능 평가를 위한 통계적 데이터 분석 방법을 고찰해보는 프로젝트틀 진행했습니다.

- `Data Loading and Initial Exploration`
  - 이미지를 2D형식(32x32 픽셀)으로 재구성하고 데이터 세트 전체의 픽셀값에 대한 통계적 요약(평균, 표준편차 등)을 제공합니다.
  - [mnist_test.ipynb](https://github.com/jsh1021902/AI_Detective/blob/main/code/mnist_test.ipynb)
  - [mnist_분포확인.ipynb](https://github.com/jsh1021902/AI_Detective/blob/main/code/mnist_%EB%B6%84%ED%8F%AC%ED%99%95%EC%9D%B8.ipynb)
- `t-SNE Visualization & Correlation Analysis`
  - 차원 축소 기법인 t-SNE(t-distributed Stochastic Neighbor Embedding)을 적용하여 고차원 데이터를 2차원 공간에서 시각화합니다.
  - 가짜 MNIST 이미지의 픽셀별 상관 행렬을 계산하고 시각화하여 데이터 세트 전체에서 픽셀이 얼마나 유사한지 또는 다른지 식별합니다.
  - [mnist_분포확인_test.ipynb](https://github.com/jsh1021902/AI_Detective/blob/main/code/mnist_%EB%B6%84%ED%8F%AC%ED%99%95%EC%9D%B8_test.ipynb)
  - [mnist_분포확인_ver2.ipynb](https://github.com/jsh1021902/AI_Detective/blob/main/code/mnist_%EB%B6%84%ED%8F%AC%ED%99%95%EC%9D%B8_ver2.ipynb)



### 연구 관련 자료 조사
- 연구 관련 자료 조사는 다음과 같습니다.
  ```text
  Reference : https://m.dt.co.kr/contents.html?article_no=2023050102109931081005
  ```
<br>

### 코드 구현을 위한 참고자료
- 코드 구현을 위한 참고자료는 다음과 같습니다.
  ```text
  Reference : Fourier Features Let Networks LearnHigh Frequency Functions in Low Dimensional Domains (https://bmild.github.io/fourfeat/)
  https://pseudo-lab.github.io/Tutorial-Book/chapters/GAN/Ch2-EDA.html
  https://github.com/fengliu90/DK-for-TST
  ```
<br>

### 참고 문헌
- 연구 관련 참고 문헌은 다음과 같습니다.
  ```text
  (DCGAN) Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In ICLR, 2016.
  Sam Bond-Taylor, Adam Leach, Yang Long, Chris G. Willcocks “Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models”
  Constructing and Visualizing High-Quality Classifier Decision Boundary Maps dagger (https://pure.rug.nl/ws/portalfiles/portal/118190644/information_10_00280.pdf)
  Yang Song & Stefano Ermon. [“Generative modeling by estimating gradients of the data distribution.”](https://arxiv.org/abs/1907.05600) NeurIPS 2019.
  Prafula Dhariwal & Alex Nichol. [“Diffusion Models Beat GANs on Image Synthesis."](https://arxiv.org/abs/2105.05233) arxiv Preprint arxiv:2105.05233 (2021). [[code](https://github.com/openai/guided-diffusion)]
  ```
<br>

----

## 사용 기술

### Environment
![Pycharm](https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)

### Development
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![scikit](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
<br>
----
## 합성데이터의 성능 평가를 위한 통계적 데이터 분석 방법 요약
- 자세한 코드는 [mnist_분포확인_test.ipynb](https://github.com/jsh1021902/AI_Detective/blob/main/code/mnist_%EB%B6%84%ED%8F%AC%ED%99%95%EC%9D%B8_test.ipynb) 참고

### 연구의 필요성
    - 양질의 데이터 부족
        - 내부 데이터 보안, 저작권, 개인정보 보호, 특수 도메인 라벨러 확보, 데이터 편향성 등의 문제로 양질의 데이터 확보 애로
    - 롱테일 데이터 부재
        - 발생하지 않았거나 발생 빈도가 낮은 다수 사건(Long Tail Data) 등 현실에서 구하기 어려운 데이터 존재
        - 수 많은 데이터들이 존재하지만 실제로 쓸 만한 데이터들이 부족한 풍요속의 빈곤을 해결하기 위해 나온 것이 합성 데이터이다.
    - 합성 데이터 생성 글로벌 시장 규모 증가
        - 합성 데이터 생성 글로벌 시장 규모는 2021년 기준 1억 2,330만 달러를 기록했으며 2022년부터 2030년까지 연평균 34.8%로 성장 예상
<img src="https://github.com/user-attachments/assets/55268b13-d89f-4b6c-aeeb-5b3ba839b87c" width="500" />
<img src="https://github.com/user-attachments/assets/76195089-4c39-4580-9c6d-1e735b4e8531" width="500" />


### 주제 선정 이유
    - GPT는 디코더만으로 구성된 트랜스포머 모델이다
    - Encoder-Decoder Attention 제거 :
        - GPT는 디코더만으로 구성되기 때문에 encoder와 비교하는 encoder-decoder Attention Layer가 필요가 없다.
        - 기존 코드에서 두 번째 서브 레이어를 제거한다
        ---------------------------------------------------------------------------------------
                attention2 = MultiHeadAttention(
                    d_model, num_heads, name="attention_2")(inputs={
                          'query': attention1,
                          'key': enc_outputs,
                          'value': enc_outputs,
                          'mask': padding_mask
                      })

                # 마스크드 멀티 헤드 어텐션의 결과는
                # Dropout과 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
                attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
                attention2 = tf.keras.layers.LayerNormalization(
                  epsilon=1e-6)(attention2 + attention1)
       -----------------------------------------------------------------------------------------
    - 인코더 출력 제거 : encoder_outputs 제거
    - 마스킹 방식 : padding mask 제거, look_ahead_mask 만 사용하면됨

<br>

----
회고 및 결론
---
### 회고
<details>
  <summary><b>Transformer 구현 회고</b></summary>
  <div markdown="1">
    <li> 배운 점 </li>
      <ul>
        <li>t-SNE 알고리즘을 이용하여 2차원으로 축소할 수 있다는 것을 알 수 있었습니다.</li>
        <li>custom 모델 저장하는 방법을 배웠다 </li>
        <li>숫자를 제거하는 전처리 제거만으로도 대답이 확연히 달라지는 것을 볼 수 있었다 </li>
        <li>underfitting 상황을 생각해서 epoch을 높였더니 성능이 향상되었다</li>
      </ul>
    <li> 아쉬운 점 </li>
      <ul>
        <li>검정 방식에서 현재는 Generator가 생성한 합성 데이터를 커널 밀도 추정과 스미르노프 검정, 맨-휘트니 검정으로만 실제 데이터와 비교하였다.</li>
        <li>추후 연구에서는 two-samples test와 같은 방식과 다른 다양한 검정 방식으로도 비교할 예정이다.</li>
        <li>위의 연구를 좀 더 해서 검정할 때 좀 더 대표성 있는 데이터로 처리를 한다면 더 정확한 예측이 가능할 것으로 보인다.</li>
      </ul>
    <li> 느낀 점 </li>
      <ul>
        <li>어려운 개념이라도 노력하면 이해할 수 있다는 것을 느꼈다</li>
        <li>챗봇도 결국 어떤 데이터를 학습하냐에 따라 대답이 달라진다</li>
      </ul>
    <li> 어려웠던 점 </li>
      <ul>
        <li>transformer의 구조를 이해하는데 어려웠다</li>
        <li>custom 모델 저장하는 데 config 설정하는 것이 어려웠다</li>
      </ul>
  </div>
</details>

### 결론
- 커널 밀도 추정 과정에서는 합성데이터의 경계가 모호한데, 이는 해당 데이터를 학습함으로써 추후 복잡한 데이터를 학습하거나 새로운 데이터를 예측할 때 성능을 높일 가능성이 있습니다.
- 이는 현재 딥러닝 모델의 Overconfidence 문제를 해결하여 값을 구분하는 경계를 부드럽게 하는 역할을 할 수 있습니다​.
<br>

---
## 디렉토리 구조
```bash
├── README.md
├── code : 관련 코드
├── img : 코드 실행 결과 이미지들
└── materials : 관련 보고서
```
