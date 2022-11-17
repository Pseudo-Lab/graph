---
layout: post
use_math: true

title: 'How Powerful are Graph Neural Networks'
author: Jaeshik.Shin
tags: [GIN]
date: 2022-07-25 04:30
---

ㅤ안녕하세요, 가짜연구소 Groovy Graph 팀의 신재식입니다. 이 글은, '[How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)' 논문을 읽고, 정리한 글 입니다.

ㅤ혹시 내용중 잘못된 점이나 보완할 점 있다면 댓글로 알려주시면 감사하겠습니다. 그럼 시작하겠습니다.

<!--more-->

---

### 목차

1. Introduction
2. Preliminaries
3. Theoretical Framework : Overview
4. Building Powerful Graph Neural Networks
5. Less Powerful but Still Interesting GNNs
6. Other Related work
7. Experiments
8. Conclustion
9. Reference

---

# 1 Introduction

- GNN은 노드 분류, link 예측, graph 분류 작업에서 좋은 성과를 보임
    - Moleculses,social,biological,financial networks 분야
- 신규 도메인의 GNN은 경험에 기반한 직관, 실험적 시행착오를 기반하고 있음
- GNN의 속성과 한계에 대한 이론적 이해는 부족하며, GNN 표현 방식에 대한 분석도 제한적
- 방법론
    - 주어진 노드의 이웃에 대한 벡터세트를 반복 가능한 요소가 있는 multi-set으로 나타냄
    - GNN의 인접 집계(aggregation)는 다중 집합에 대한 집계 기능으로 생각할 수 있음
    - GNN이 서로 다른 multi-set을 서로 다른 표현으로 aggregation할 수 있어야 함

## 1.1 본 논문에서 제안하고자 하는 내용
- GNN의  표현을 분석하기 위한 이론적 프레임워크 제시
- GNN 변형이 얼마나 표현력이 있는지 분석
- WL Test(Weisfeilier-Lehman,웨이스페일러-레흐만 테스트)에서 아이디어를 얻음
    - WL 테스트는 GNN과 유사한데, 이웃의 특징 벡터를 집계하여 주어진 노드의 벡터를 반복적으로 업데이트
    - 다른 노드을 다른 기능 벡터에 맵핑하여 업데이트 하는 방식이 장점
- GNN이 WL 처럼 집계하는 시스템의 표현력이 좋고, 다른 노드에  매핑할 수 있는 기능을 모델링 할 경우 GNN이 WL처럼 분류하는데 큰 판별 성능을 가질 수 있다고 판단

## 1.2 Main Results
1. GNN이 그래프 구조를 구별할 때  WL 테스트만큼 판별 성능이 뛰어난 것을 증명
2. GNN 결과가 WL 테스트만큼 neighborhood aggregation 및 그래프 판독 기능을 가질 수 있는 파라미터 설정
3. GCN 및 GraphSAGE와 같은 GNN 변형으로 구분할 수 없는 그래프 구조를 식별하고 GNN 기반 모델과 같은 그래프 구조의 종류를 정확하게 분별 가능
4. 간단한 신경 아키텍처인 **GIN(Graph Isomorphism Network)을 개발**하고 Discriminative/Representational value과 WL test와 동일함을 증명

---

# 2. Preliminaries

- Node Classification
    - 각 노드 v ∈ V에는 연관된 레이블 y_v가 있고 목표는 v의 레이블이 y_v = f(h_v)로 예측될 수 있도록 v의 표현 벡터 h_v를 학습하는 것
- Graph Classification
    - {G_1 , ..., G_N } ⊆ G 와 레이블 {y_1 , ..., y_N } ⊆ Y 가 주어지면 예측에 도움이 되는 표현 벡터 h_G를 학습하는 것을 목표

## 2.1 Graph Neural Networks(GNN)

- GNN은 그래프 구조와 노드  X_v를 사용하여 노드 h_v / 전체 그래프 h_G의 표현 벡터를 학습
- GNN은 neighborhood aggregation strategy 전략을 따르며, neighbor의 representations을 집계하여 반복적으로 업데이트 함
- k 반복 집계 후 노드의 표현은 k-hop 네트워크 이웃 내의 구조적 정보를 caputre함
- GNN의 k번째 레이어의 수식
    - Aggregate함수는 multiset에 정의된 함수이며 주로 summation사용
    - Combine함수에서는 전 단계에서 수집한 정보 a_v와, 현재의 feautre vector h_v를 사용해서 Node의 새로운 feature vector에 Update
    <img src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F7db7cb8c-f7dd-4a84-83b3-f1246399356d%2FUntitled.png?table=block&id=4b87148c-3295-4e1e-ad53-442f748c0437&spaceId=333f96cf-396d-45ff-8331-232d41bd4d55&width=1920&userId=b188a853-05ac-4bc5-92c3-a256967ab022&cache=v2"width="400px">
    - h_v(k) : feature vector of node v at the k-th iteration/layer

## 2.2 Weisfeilier-Lehman test(WL test)
- Graph Isomorphism problem : 두 그래프가 위상적으로 동일한지 여부를 확인하는 부분이지만, 이 부분은 challenging problem
- 그래프 동형의 Weisfeiler-Lehman(WL) 테스트는 광범위한 그래프 클래스를 구별하는 효과적이고 계산적으로 효율적인 테스트
- 1차원 형태인 "navie vertex refinement"는 GNN의 이웃 집합과 유사
- WL 테스트는
    - (1) 노드와 그 이웃의 레이블을 반복적으로 집계하고
    - (2) 집계된 레이블을 고유한 새 레이블로 해시(다양한 길이를 가진 데이터를 고정된 거리를 가진 데이터로 매핑한 값)
- 알고리즘은 일부 반복에서 두 그래프 사이의 노드 레이블이 다른 경우 두 그래프가 non-isomorphic이라고 결정
- WL 테스트를 기반으로 그래프 간의 유사도를 측정하는 WL 하위 트리 커널을 제안
    - 커널 : WL 테스트는 여러번 반복하여 노드 레이블 수를 그래프의 특징 벡터로 사용
- 직관적으로 WL 테스트의 k번째 반복에서 노드의 레이블 :
    - 노드에 뿌리를 둔 k의 하위 트리 구조를 나타냄
    - 중간 그림: WL 테스트가 다른 그래프를 구별하기 위해 사용하는 루트 하위 트리 구조(파란색 노드).
    - 오른쪽 그림: GNN의 aggregating 기능이 노드 이웃의 multi set을 캡처하는 경우 GNN은 재귀 방식으로 루트 하위 트리를 캡처 → WL 테스트만큼의 성능이 나옴
- 동형 그래프의 예시(Graph1 = Graph2)
- 동형 그래프 계산 과정
    - 생긴 형태는 다르게 보이지만, 노드와 연결되어 있는 링크(엣지)가 동일하므로 동형 그래프

⇒WL 알고리즘의 목표 : 

- 두 그래프 간의 동형을 판정하는 것
- k번의 반복 후에도 두 그래프가 같은 컬러일 경우, 두 그래프는 동형일 가능성이 있음을 알려줌

---

# 3 Theoretical Framework : Overview

- 위 그림 설명 :
    - GNN은 각 노드의 기능 벡터를 재귀적으로 업데이트하여 네트워크 구조와 주변의 다른 노드의 기능이 정의됨
- 인접 노드 집합의 특징 벡터가 multi-set을 형성함
    - 다른 노드가 동일한 특징 벡터를 가질 수 있기 때문에 동일한 요소가 여러 번 나타날 수 있음
- Multi-set(다중 집합)
    - 다중 집합은 해당 요소에 대해 여러 instance를 허용하는 집합의 일반화된 개념
    - 더 형식적인 다중 집합은 2-tuple로 구성된 집합을 X = (S, m)로 표현한다면,
        - S:  고유한 요소로 구성된 X의 기본 집합
        - m : S → N≥1은 요소의 다중성을 나타냄
- GNN의 표현력을 연구하기 위해 GNN이 임베딩 공간의 동일한 위치에 두 개의 노드를 매핑할 때를 분석
- 직관적으로,  GNN은 해당 노드에서 ‘**동일한 기능을 가진’ ‘동일한 하위 트리 구조’**가 있는 경우에만, 두 노드를 동일한 위치에 매핑함
- 하위 트리 구조는 노드 이웃을 통해 재귀적으로 정의되기 때문에(그림참고),
    - GNN이 두 개의 이웃(즉, 두 개의 다중 집합)을 동일한 임베딩 또는 매핑하는지 여부에 대한 관점으로 분석접근이 가능
- GNN의 집계 체계를 신경망이 나타낼 수 있는 다중 집합에 대한 함수 클래스로 추상화하고 multi-set function을 나타낼 수 있는지 연구

---

# 4. Building Powerful Graph Neural Networks
- WL test와 GNN의 representational power의 관계에 대해 알아볼 예정
- GNN은 임베딩 공간의 다른 표현에 매핑하여 다른 그래프 구조를 구별할 수 있음
    - But, 두 개의 서로 다른 그래프를 서로 다른 임베딩에 매핑하는 기능은 어려운 그래프 동형 현상 문제
- 즉, ‘**동형 그래프가 동일한 표현에 매핑**’되고 ‘**비동형 그래프가 다른 표현에 매핑**’ 형태가 좋은 형태

- **Lemma 2**. G1과 G2를 두 개의 비동형 그래프라고 가정
        
    - 그래프 신경망 A : G → R^d가 G1과 G2를 **다른 임베딩에 매핑하면**
    - Weisfeiler-Lehman 그래프 동형 테스트도 G1과 G2가 동형이 아닌 것으로 결정
    
    ⇒ Lemma 2를 통해 WL Test로 구별해내지 못하는 그래프들에 대해서 GNN도 역시 구별해내지 못한다는 것을 강조
    
    ⇒ Lemma 2의 증명은 WL Test는 feature vector를 update하는 과정이 injective하다는 점을 설명
    
    - GNN의 neighborhood aggregation이 injective하다면 WL Test만큼의 성능이 가능한가?
        - Theorem 3에서 답변해줌

- **Theorem 3.** A : G → R^d를 GNN이라고 가정
        
    - 충분한 수의 GNN 레이어를 사용하여 A는 동형에 대한 WL 테스트가 비동형으로 결정한 모든 그래프 G1 및 G2를 다음 조건이 충족되는 경우 다른 임베딩에 매핑.
        
        a) A는 반복적으로 노드 기능을 집계 및 업데이트, 함수 f는 multiset에서 작동하고 공집합은 주입식임
                
        (b) 노드 기능의 다중 집합에서 작동하는 A의 그래프 수준을 판독, h_v(k)는 주입식임
        
    
    ⇒ Aggregate, combine, readout 함수가 multiset에 대해서 injective 일 때, **GNN은 WL Test와 같은 분별성능을 가질 수 있다**라는 결론
    
    *injective function(단사 함수) : 정의역 원소에 대해 다른 치역 원소로 맵핑되는 함수 / 입력값의 정보를 온전히 출력값으로 전달하는 함수
    
- GNN의 이점
    - WL 테스트의 노드 기능 벡터는 본질적으로 원-핫 인코딩이므로 하위 트리 간의 **유사성을 확인할 수 없음**
    - 대조적으로 Theorem 3의 기준을 만족하는 GNN은 하위 트리를 저차원 공간에 injective하는 방법을 학습하여 WL 테스트를 일반화하여,
        - **이를 통해 GNN은 다른 구조를 구별할 수 있을 뿐만 아니라 유사한 그래프 구조를 유사한 임베딩에 매핑하고 그래프 구조 간의 종속성을 확인하는 방법 학습 가능**
        
        ⇒ WL test는 그래프가 다르다는 것은 알아도, 얼마나 다른지에 대해서 알 수 없음!

## 4.1 Graph Isomorphism Network(GIN)

- Theorem 3의 조건을 증명할 수 있는 GIN(Graph Isomorphism Network)이라는 간단한 아키텍처를 개발
    - 이 모델은 WL 테스트를 일반화하므로 GNN 간의 최대 판별력의 성능 표현이 가능
- Neighbor aggregation에 대한 injective multiset functions modeling하기 위해 "Deep Multiset" 이론 개발
    - Deep Multiset : 신경망을 사용하여 범용성 다중 집합 함수를 parameterizing하는 이론
- Lemma5과 Corollary 6을 통해서 Aggregate와 Combine 함수가 multiset에 대해 injective한 함수가 존재하는지 확인해야 함

## 4.2 Graph-Level Readout of GIN

- GIN을 학습한 노드 임베딩은 `노드 분류` 및 `링크 예측`과 같은 작업에 직접 사용 가능
    - 그래프 분류 작업을 위해 우리는 개별 노드의 임베딩이 주어지면 전체 그래프의 임베딩을 생성하는 다음 "판독" 기능 가능
- 그래프 수준 판독에서 중요한 부분은 하위 트리 구조에 해당하는 Node representaion이 반복 횟수가 증가함에 따라 범위가 global해짐
    - layer가 많으면 Global한 특성만 남고, Layer가 적으면 local한 특성만 남는 것 ⇒ 적당한 Layer의 수가 필요
    - GNN의 고질적인 문제인 over-smoothing의 문제라고 해석할 수 있음
        - over-smoothing : 그래프 신경망의 layer 수가 증가하면서 정점의 임베딩이 서로 유사해지는 현상
- 적절한 Layer를 위해 layer의 graph representation을 concatenation을 모두 합쳐줌

- Theorem 3와 Corollary 6에 따라 GIN이 위 식의 READOUT을 대체하는 경우,
    - 동일한 반복의 모든 노드 기능을 합산하면, WL 테스트와 WL 하위 트리 커널 일반화 가능
    - Readout(=graph representation)

---

# 5 Less Powerful but Still Interesting GNNs
- GCN 및 GraphSAGE를 포함하여 Theorem 3의 조건을 충족하지 않는 GNN을 연구
    - Theorem 3의 핵심 아이디어 : Aggregate, combine, readout 함수가 multiset에 대해서 injective 일 때, **GNN은 WL Test와 같은 분별성능을 가질 수 있다**라는 결론
- 위 같은 GNN의 변형 모델이 WL 테스트보다 설명력이 좋지 않지만, 그럼에도 불구하고 GCN과 같은 평균 집계(Mean Aggregator)가 있는 모델은 노드 분류 작업에 대해 잘 수행
- 이를 더 잘 이해하기 위해 다양한 GNN 변형이 그래프에 대해
    - 캡처할 수 있는 것과
    - 캡처할 수 없는 것
        - 위 2개를 정확하게 특성화하고 그래프를 통한 학습의 의미를 파악
- Graph를 학습하는 데에 있어 다른 GNN의 변형 버전들이 어떻게 Graph에 대해 학습하고 정보를 습득하는지에 대한 설명

---

# 6 Other Related work

- GNN의 속성에 대해 수학적으로 연구하는 작업은 상대적으로 적음
- 가장 초기의 GNN 모델(Scarselli et al., 2009b)이 확률로 측정 가능한 함수를 근사할 수 있음을 보여줌
- Lei et al. (2017)은 제안된 아키텍처가 그래프 커널의 RKHS에 있음을 보여주지만 어떤 그래프를 구별할 수 있는지 명시적으로 연구하지는 않음
- sum 집계 및 MLP 인코딩(Battaglia et al., 2016; Scarselli et al., 2009b; Duvenaud et al., 2015)을 포함하여 대부분의 GNN 기반 아키텍처가 제안
- 다른 GNN 아키텍처와 달리 GIN(Graph Isomorphism Network)은 단순하지만 Powerful함

---

# 7 Experiments

- GIN과 GNN 변형의 훈련 및 테스트 성능을 평가 및 비교
    - 본 실험의 목적은 모델이 단지 Input Node Feature에 의존하지 않고, Network 구조를 학습하도록 하는 것임
        - 따라서 생물 정보 그래프에서 노드는 범주형 입력 기능을 갖지만 소셜 네트워크에서는 기능이 없음
        - 소셜 네트워크의 경우 같이 노드 기능을 생성
            - REDDIT 데이터 세트의 경우 모든 노드 기능 벡터를 동일하게 설정
            - 다른 소셜 그래프의 경우 노드 정도의 원-핫 인코딩을 사용
- 데이터 세트
    - 9개의 그래프 분류 벤치마크 사용
        - 4개의 생물정보학 데이터세트 : MUTAG, PTC, NCI1, PROTEINS
        - 5개의 소셜 네트워크 데이터세트 :COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY 및 REDDIT-MULTI5K

### 모델 및 구성

- GIN과 GNN 변형을 평가
- GIN 프레임워크에서 두 가지 변형을 고려
    - (1) Eq4.1에서 경사하강법으로 학습시킨 ε을 GIN-ε라고 함
    - (2) Eq4.1에서 ε을 0인 경우를  GIN-0이라고 함 
      
- **높은 Representational Power를 갖고 있는 모델일수록 Training set에서의 정확도도 높을 것**

- 그림 4: GIN,  GNN 변형 및 WL 하위 트리 커널의 훈련 세트 성능.

- 하이퍼 매개변수 조정 리스트
    - (1) 은닉 유닛의 수 ∈ {16, 32} 생물정보학 그래프, 64 소셜 그래프
    - (2) 배치 크기 ∈ {32, 128};
    - (3) 조밀한 층 이후의 드롭아웃 비율 ∈ {0, 0.5}
    - (4) 에포크의 수, 즉 10겹에 걸쳐 평균화된 최고의 교차 검증 정확도를 가진 단일 에포크가 선택
- 데이터 세트 크기가 작기 때문에 하이퍼파라미터 선택이 유효성 검사 세트를 사용하여 수행되는 대체 설정은 매우 불안정
    - (예: MUTAG의 경우 유효성 검사 세트에는 18개의 데이터 포인트만 포함됨).
- 추가로 5개의 GNN 레이어(입력 레이어 포함), 64 크기의 은닉 유닛, 128 크기의 미니배치 및 0.5 드롭아웃 비율과 같은 데이터 세트에서 모든 하이퍼 매개변수가 고정된 다양한 GNN의 훈련 정확도를 확인함
    - 비교를 위해 WL 하위 트리 커널의 훈련 정확도가 확인되며, 여기서 반복 횟수를 4로 설정했으며 이는 5개의 GNN 레이어와 비슷함

### BASELINE

- 아래 리스트와 비교
    - (1) C-SVM이 사용된 WL 하위 트리 커널 분류기
        - 조정하는 하이퍼파라미터는 SVM의 C와 WL 반복 횟수 ∈ {1, 2, . . . , 6}
    - (2) Defusion-convolution 신경망(DCNN) 및 Deep Graph CNN(DGCNN)
    - (3) Anonymous Walk Embedding(AWL) 딥 러닝 방법과 AWL의 경우 원본 논문에 기재된 정확도

## 7.1 Results

- 표 1: Test-Set 분류 정확도(%).
    - 볼드체 : GIN의 정확도가 GNN 변형 중에서가장 높지 않은 데이터 세트에서 유의 수준 10%에서 paired t-test사용하여 GIN의 성능 평가

### Training-set Performance

- 훈련 정확도를 비교하여 GNN의 표현력에 대한 이론적 분석을 검증
    - 더 높은 표현력을 가진 모델은 더 높은 훈련 세트 정확도가 필수
- 그림 4는 동일한 하이퍼 매개변수 설정을 사용하여 GIN 및 GNN 변형의 train curve보여줌
- 첫째, 이론적으로 가장 표현력이 좋은 GNN, 즉 GIN-ε 및 GIN-0은 모든 훈련 세트에 거의 완벽하게 적합가능
    - 이에 비해 mean/max 풀링 또는 1-layer 퍼셉트론을 사용하는 GNN 변형은 많은 데이터 세트에서 심하게 과소적합(underfit)됨
- Training-set 경향
    - 특히 훈련 정확도 패턴은 모델의 표현력에 따른 순위와 일치함
    - MLP가 있는 GNN 변형은 1-layer 퍼셉트론이 있는 GNN보다 더 높은 훈련 정확도를 갖는 경향이 있음
    - Sum aggregator의 GNN은 mean/max 풀링 집계자가 있는 GNN보다 훈련 세트에 더 잘 맞는 경향이 있음
- 본 데이터 세트에서 GNN의 훈련 정확도는 WL 하위 트리 커널의 정확도를 결코 초과하지 않음
    - 이는 GNN이 일반적으로 WL 테스트보다 판별력이 낮기 때문이라고 예상됨
        - 예를 들어, IMDBBINARY에서 어떤 모델도 훈련 세트에 완벽하게 맞을 수 없으며 GNN은 WL 커널과 기껏해야 동일한 훈련 정확도를 달성함
    - 이 패턴은 WL 테스트가 집계 기반 GNN의 표현 용량에 대한 upper bound을 제공한다는 본 논문의 결과와 일치함
        - 그러나 WL 커널은 노드 기능을 결합하는 방법을 배울 수 없음

### Test-set Performance

- 높은 표현력을 가진 GNN이 그래프 구조를 정확하게 포착하여 잘 일반화할 수 있다고 가정
- 표 1은 GIN(Sum–MLP) 및 기타 GNN 변형 및 최신 base line의 테스트 정확도를 비교
- 첫째, GIN, 특히 GIN-0은 9개 데이터 세트 모두에서 GNN 변형보다 성능이 우수(또는 유사한 성능을 달성)하여 SOTA달성
- GIN은 상대적으로 많은 수의 훈련 그래프가 포함된 소셜 네트워크 데이터 세트에서 강점이 있음
    - Reddit 데이터 세트의 경우 모든 노드는 노드 기능과 동일한 스칼라를 공유
    - 여기서 GIN과 sum-aggregation GNN은 그래프 구조를 정확하게 포착하고 다른 모델보다 높은 성능을 보여줌
- GIN(GIN-0 및 GIN-ε)을 비교하면 GIN-0이 약간 그러나 일관되게 GIN-ε을 능가함

---

# 8 Conclusion

- GNN의 표현력에 대한 추론을 위한 이론적 토대를 개발
- GNN 변형의 표현 능력에 대한 boundary를 증명
- Neighbor set framework에서 증명할 수 있는  GNN 설계
- Future work :
    - 그래프로 학습하기 위한 훨씬 더 강력한 아키텍처를 위해 negibor aggregation(또는 메시지 전달)보다 성능을 향상시키는 것
    - GNN의 속성을 이해하고 개선하고 최적화 환경을 더 잘 이해하는 것

⇒ MLP(Multi-Layer Perceptron)이 injective function일 때 GIN은 WL만큼 강한 성능을 가질 수 있음

 *Training process에서 MLP가 injective할지는 보장할 수 없음
 
 ---
 
 # 9 Reference
 
1. GIN : [https://harryjo97.github.io/paper review/How-Powerful-are-Graph-Neural-Networks/](https://harryjo97.github.io/paper%20review/How-Powerful-are-Graph-Neural-Networks/)
2. GIN : [https://greeksharifa.github.io/](https://greeksharifa.github.io/)
3. WL Test : [https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/)
4. Injective Function : [https://velog.io/@stapers/Lecture-9-Theory-of-Graph-Neural-Networks](https://velog.io/@stapers/Lecture-9-Theory-of-Graph-Neural-Networks)
5. GIN : [https://junklee.tistory.com/125](https://junklee.tistory.com/125)


