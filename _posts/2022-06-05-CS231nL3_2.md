---
layout: single
title:  "CS231n_Lecture 3 (2/2)"
categories: CS231n
# tag: [data science, vision, cs231n]
use_math: true
toc: true
toc_sticky: true
author_profile: false
published: true

---

# CS231n_Lecture 3 | Loss Functions and Optimization (2/2)

# Optimization

- Loss가 최소가 되는 W값을 찾는 과정을 Optimization(최적화) 라고 함.
- 강의 에서는 Optimization 방법으로 #1 Random search와 #2 Follow the slope 두 가지를 설명함.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled.png)

- Random Search의 경우 W값을 Random으로 바꿔 가면서 최적의 Loss 값을 찾는 방법임.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%201.png)

- 현재 최고 수준의 결과 (State of the art)로 95%의 정확도가 나오는 반면 Random search의 경우 정확도가 15.5%로 낮음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%202.png)

- Follow the slop의 경우는 쉽게 말해 산을 내려갈 때 가장 경사진 반대 방향으로 내려가는 것과 같음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%203.png)

- Follow the slope는 Gradient descent 라고 부르며, 그래프의 Gradient (기울기)를 계산하고 가장 경사가 가파른 반대 방향으로 진행함.
- Graph가 1차원 식이라면, 미분식으로 계산 할 수 있고 수식을 하나씩 세워서 계산하는 Gradient를 Numerical gradient라고 함.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%204.png)

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%205.png)

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%206.png)

- Numerical gradient를 계산 예시
- h를 미세하게 변경해 가면서 Gradient를 계산하며, 많은 계산이 필요하여 계산이 느림.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%207.png)

- Loss(dW)를 다 계산하여 값을 구하는 것은 시간과 노력이 많이 듦.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%208.png)

- 뉴턴과 라이프니치가 (서로 각자) 개발한 미분공식 (Chain Rule)을 활용하면 미분값을 더 쉽게 구할 수 있음.
- 이를 Analytic gradient 라고함.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%209.png)

- Analytic gradient를 계산한 예시.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2010.png)

- Numerical gradient는 정확하지 않고, 느리지만 사용하기 쉬움
- Analytic gradient는 정확하고 빠르지만 사용하기 어려움.
- 일반적으로 Analytic gradient를 사용하고 Debugging 시 Numerical gradient를 사용함. (Gradient check)

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2011.png)

- Gradient descent (경사하강법)는 최저 Loss를 찾아가는 과정임.
- Learning rate (step size)는 가장 중요한 hyper-parameter로 모델 크기나, Regularization strength등을 고려하여 산정함.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2012.png)

- 붉은 부분은 Loss가 낮은 곳이고, 파란 부분은 Loss가 높은 곳임.
- Negative gradient로 이동할 시 Minima에 도달 할 수 있음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2013.png)

- 3차원에서 Local Minima와 Global Minima
- Local Minima에 빠지는 경우 Learning Rate를 조절하여 Global Minima로 갈 수 있음.

> ![https://blog.kakaocdn.net/dn/cSKIGH/btqT90rmzbL/pkmZUvubjhB01yTrVFlrQK/img.gif](https://blog.kakaocdn.net/dn/cSKIGH/btqT90rmzbL/pkmZUvubjhB01yTrVFlrQK/img.gif)

>  Optimizer에 따른 Minima를 찾는 과정  


> **출처**
> [Gradient Descent Algorithms](https://yjjo.tistory.com/6)

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2014.png)

- 전체 데이터 셋에 대해서 Loss를 계산하는 것은 시간도 많이 걸리고 자원도 많이 듦.
- Stochastic Gradient Descent (확률적 경사하강법)을 사용하여 Mini-batch(샘플링된 데이터)를 사용하여 전체 Loss 합을 추정함.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2015.png)

[Multiclass SVM optimization demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)

- 해당 웹사이트에서 Weight와 Bias값이 바뀜에 따라 Linear Classifier가 어떻게 동작 하는지 직관적으로 알아 볼 수 있음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2016.png)

- 지금까지의 강의에서 Image Classification에 Linear Classifier를 사용했는데, Pixel 값에 Linear Classifier를 적용하는 것은 좋은 방법이 아님.
- Deep Learning 이전에는 Image Feature를 Linear Classifier에 넣는 방식으로 Image Classification을 했었음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2017.png)

- Pixel 에서는 Linear Classifier로 분류 할 수 없는 이미지도 Feature transform을 통해서 Linear classifier로 분류가 가능함

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2018.png)

- 간단한 Feature의 예는 Color Histogram을 들 수 있음.
- Hue color spectrum을 이용하여 모든 픽셀의 색들을 Bucket에 나누어 담고 어떤 색이 가장 많은지 알아 볼 수 있음. (개구리 사진의 경우 녹색이 가장 많음)

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2019.png)

- 앞선 강의에서 Hubel과 Wiesel이 인간의 시각 체계에서 Edge를 인식하는 것이 중요하다는 걸 발견 했다는 얘기를 했었음.
- Histogram of Oriented Gradients는 이미지를 8x8의 픽셀 영역으로 나누고 각 영역의 Edge 방향을 측정하여 가장 지배적인 Edge 방향을 계산함.
- 그 후 Edge 방향을 각각의 Edge Bucket에 담으면 어떤 영역에 어떤 종류의 Edge가 있는 지를 알 수 있음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2020.png)

- Image Classification의 또 다른 방법은 자연어 처리로부터 영감을 받은 Bag of words가 있음.
- 대량의 이미지를 받아 랜덤하게 이미지를 작은 부분으로 자르고 K-means 와 같은 알고리즘으로 Clustering 한 후 각 Cluster에 단어 Class를 붙여줌.
- 이를 통해 이미지의 색과 Edge를 기반으로 한 분류를 할 수 있음.

![Untitled](../../images/2022-06-05-CS231n_Lecture3_2/Untitled%2021.png)

- Deep Learning 이전에는 Image Classification을 할 때 이미지로 부터 특징을 추출해 비교하는 방식을 사용했었음.
- ConvNet(합성곱 신경망)은 Model이 스스로 이미지로 부터 직접 Feature를 뽑아내는 방식임.

- **출처**

[**Stanford University Youtube CS231n_Lecture3**](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3)