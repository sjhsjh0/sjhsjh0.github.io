---
layout: single
title:  "확률과통계기초_2.확률용어정리"
categories: Statistics
# tag: [data science, vision, cs231n]
use_math: true
toc: true
toc_sticky: true
author_profile: false
published: true

---


### 1. Probability (확률)
- 정확한 확률은 측정이 불가능 함.
- 반복시행을 통해 상대도수(relative frequency)를 측정하여 확률을 추정함.
(relative frequency 를 무한히 반복 하여 확률에 근사)
    - relative frequency ($\hat{p_i}$ )를 무한히 반복하여 추정한 확률 ($p_i$)

$$
\hat{p}_i = N_n(O_i)/n
$$

$$
p_i = \lim_{n \to \infty}N_n(O_i)/n
$$

### 2. Random Variable (확률 변수)
- 확률 분포 또는 무작위 프로세스에 의해 결정되는 값을 가지는 변수
    - ex) 동전을 던졌을 때 앞이 나오면 0, 뒤가 나오면 1 이라고 했을 때, 0과1은 동전이 앞면이 나올 확률(0.5)과 뒷면이 나올 확률(0.5)로 결정되므로, 0,1은 확률 변수 임.
- 이산(discrete)확률변수와 연속(continuous)확률 변수가 있음.
    - 이산확률변수 : 동전, 주사위등 어떤 시행의 결과가 유한하고 세세하게 정의되어 있는 경우.
        - 이산확률변수의 분포는 probability mass function (pmf)로 나타냄.
            
![주사위의 이산확률 분포](../../images/2023-01-24-확률과통계기초_2.확률용어정리/Untitled.png){: .align-center}
            
                  
            
$0 \le p(x) \le1$ (확률의 범위)
$\sum_{x}p(x) = 1$  (모든 확률의 합은 1)
$P(X \in B) = \sum_{x\in B}{p(x)}$ 
( X = random variable (주사위를 던지기 전), x = instance(주사위를 던진 후))

    
- 연속확률변수 : 시간, 속도, 온도 등 확률분포에 따라 값이 연속적으로 변하는 변수.
    - 연속확률변수의 분포는 probability density function (pdf)로 나타냄.
            
![pdf](../../images/2023-01-24-확률과통계기초_2.확률용어정리/Untitled%201.png){: .align-center}
            
$$
\lim_{n \to \infty} {P(x<X\le x+ \Delta x) \over \Delta x}
$$

$f(x) \ge 0 ; f(x) >1$  ( 확률 변수는 0보다 크며, 1보다도 클수도 있음.)
$\int_{-\infty}^{\infty} f(x)dx = 1$ (pdf의 총합은 1) 
            
### 3. Joint Probability (결합 확률)
- 확률 변수가 여러개 일때 이들을 함계 고려하는 확률.
    - ex) 동전의 앞면과 주사위의 3이 함께 나올 확률
- x,y가 독립일때, $p(x,y) = p(x)p(y)$

### 4. Conditional Probability (조건부 확률)
- 주어진 사건이 일어났을 때 다른 사건이 일어날 확률.
    - ex) 동전의 앞면이 나왔을 때, 주사위 3이 나올 확률
$P(A｜B) = {P(A\cap B)\over P(B)}$ (B가 일어났을 때 A가 일어날 확률)

### 5. Bayes Rule
- 두 확률 변수의 사전 확률(prior probability)과 사후 확률(posterior probability) 사이의 관계를 나타내는 정리.
    - 사전확률, 사후확률 정의


    > [[링크] 베이즈 정리의 의미](https://angeloyeo.github.io/2020/01/09/Bayes_rule.html)



- $p(x｜y)$ 
=$P(A｜B)$=${｜A\cap B｜\over｜B｜}={｜A\cap B｜/｜U｜ \over ｜B｜ / ｜U｜}={P(A\cap B)\over P(B)}={p(x,y)\over p(y)}$


- $P(A\cap B)$ = $P(A｜B)P(B)$ = $P(B｜A)P(A)$
- $P(A｜B) = {P(B｜A)*P(A) \over P(B)}$
    
- $P(A｜B)$  : 사후확률(posterior), $P(A)$ : 사전확률(prior), $P(B)$ : 증거(evidence) , $P(B｜A)$ : 우도/가능도(likelihood)
    
- likelihood 설명
 
    > [[링크] 확률(Probability) vs 가능도(Likelihood)](https://jinseob2kim.github.io/probability_likelihood.html)

