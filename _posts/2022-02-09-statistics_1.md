---
layout: single
title:  "확률과 통계 기초_1. 통계용어정리"
categories: Statistics
# tag: [data science, statistics]
use_math: true
toc: true
toc_sticky: true
author_profile: false
published: true
---

### 1) 최빈값(Mode)

- 가장 빈번하게 나타나는 데이터 값
- 도수분포표에서 가장 긴 막대
- 최빈값이 두개 이상인 경우가 많음
    - 이봉분포(bimodal distribution)
    - 다봉분포(multimodal distribution)

### 2) 평균(Mean)

- 모든 값의 총합을 값의 개수로 나눈 값. (outlier에 민감/서로 다른 표본들에서 안정적인 경향)

$$
\overline{X} = \frac{1}{n}\sum_{i=1}^{n}{x_i}
$$

- 절사 평균(truncated mean) : 값들을 크기 순으로 정렬한 후, 양 끝에서 일정 개수의 값들을 삭제 한 뒤 남은 값들을 가지고 구한 평균
- 가중 평균(weighted mean) : 각 데이터 값에 사용자가 지정한 가중치를 곱한 값들의 총합을 다시 가중치의 총합으로 나눈 값
- Average : 산술평균(arithmetic mean)

### 3) 중간값(median)

- 데이터를 일렬로 정렬했을 때, 한가운데 위치하는 값 (outlier에 robust 함)
    - Robust : 극단값들에 민감하지 않다는 의미
- 가중 중간값(weighted median) : 데이터에 가중치를 곱하고 구한 중간값

### 4) 편차(deviation)

- 이탈도 라고도 부름
- 관측값과 위치 추정값 사이의 차이
- 편차의 총 합은 0

### 5) 오차제곱합(sum of squared errors);SS

- 편차의 총합이 0이므로 제곱하여 더함
    
    $$
    \sum_{i=1}^{n}({X_i-\overline{X}})^2
    $$
    

### 6) 분산(variance)

- 평균과의 편차를 제곱한 값들의 합을 n-1로 나눈 값 (n은 데이터 개수)
    - n이 아닌 n-1로 나누는 이유: 자유도(degree of freedom)을 주어 편향(biased)을 피하기 위함

$$
\frac{1}{n-1}\sum_{n=1}^{n}(X_i-\overline{X})^2
$$

### 7) 표준편차(standard deviation);SD

- 분산의 제곱근
- 표준편차가 작다 - 데이터들이 평균에 가깝다
- 표준편차가 크다 - 데이터들이 평균과 멀다
    
    $$
    \sqrt{\frac{1}{n-1}\sum_{n=1}^{n}(X_i-\overline{X})^2}
    $$
    

### 8) 표준화(standardization)

- 중심화를 한 이후에 척도화를 하는 과정
    - 중심화 : 모든 데이터에 평균 값을 빼줌
    - 척도화 : 모든데이터를 표준편차로 나눔
- 표준화를 하여도 데이터의 분포는 변하지 않음 → **데이터의 성질은 같다**

$$
z = \frac{X-\overline{X}}{s}
$$