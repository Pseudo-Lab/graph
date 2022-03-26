---
layout: post
title: 'The PageRank Citation Ranking Bringing Order to the Web'
author: Sumin.Han
date: 2022-03-25 13:03
tags: [pagerank]
use_math: true
---

<!--more-->

# 목차

1. PageRank의 배경
2. PageRank 정의 & 개념
    1. 웹 그래프
    2. PageRank란?
    3. PageRank에 대한 직관
3. Simple PageRank
    1. 투표 (vote) 관점에서의 Simple PageRank
    2. random walk 관점에서의 Simple PageRank
    3. Simple PageRank 관점 정리
    4. Simple PageRank의 계산
    5. Simple PageRank의 문제점
4. ~~Simple~~ PageRank
    1. Spider trap과 Dead end의 해결 방법
    2. ~~Simple~~ PageRank 계산 예시
5. PageRank의 결과
    1. Convergence Properties
6. + PageRank를 이해하는 데 필요한 수학
    1. Stochastic matrix
    2. Perron-Frobenius theorem
7. + 질문

---

![]()

![](/files/posts/The-PageRank-Citation-Ranking-Bringing-Order-to-the-Web/fig_01.png)

---

# 3 Simple PageRank

## 3.3 Simple PageRank 관점 정리

- 투표 관점에서 정의한 rank
    - == random walk 관점에서의 stationary distribution
    - (== 논문 수식)

---

- **투표 관점에서 정의한 rank**
    
$$
r_j = \sum_{i \rightarrow j}^{} \frac{r_i}{d_i}
$$
    
- $d_i$ :
    - out-degree
    - node $i$의 forward link (out-edge, out-link) 수
