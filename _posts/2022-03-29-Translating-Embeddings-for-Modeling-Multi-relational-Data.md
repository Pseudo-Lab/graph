---
layout: post
use_math: true

title: 'Translating Embeddings for Modeling Multi-relational Data'
author: Daehwan.Oh
date: 2022-03-29
tags: [TransE]
---

### 목차
1. Objective
2. Dataset
3. Evaluation Metric
4. Method
5. Result
6. Limit

# Objective
Entity, relation을 embedding 할 수 있어야한다.

기존 방법에 비해 학습하기 쉬워야하며 파라미터 갯수가 적어야 한다.

큰 데이터셋에 대해 scalable 해야합니다.

<img width="1058" alt="스크린샷 2022-05-05 오후 1 30 27" src="https://user-images.githubusercontent.com/25244851/166863816-00e5b020-705c-4294-9b3b-d380aba83d39.png">

# Dataset
Freebase, WordNet 데이터셋에서 사용되는 Relation

<img width="293" alt="스크린샷 2022-05-05 오후 1 32 26" src="https://user-images.githubusercontent.com/25244851/166863956-895240b6-3d26-4c52-884d-bb801fc7e520.png">

실제 Freebase 데이터셋의 Entity와 Relation 예시

<img width="703" alt="스크린샷 2022-0![Uploading 스크린샷 2022-05-05 오후 1.35.08.png…]()
5-05 오후 1 33 09" src="https://user-images.githubusercontent.com/25244851/166863991-c5de4511-a3a2-439f-a3c5-6b6365cbe477.png">

# Evaluation Metric
벡터의 합과 벡터간의 거리(distance / dissimilarity)가 사용됩니다.

<img width="745" alt="스크린샷 2022-05-05 오후 1 35 45" src="https://user-images.githubusercontent.com/25244851/166864170-2c6050e7-c63a-4f3d-aabb-aacb1e8547a2.png">

## rank

거리값을 기준을 벡터들을 정렬할 수 있고 그 중 테스트셋의 정답 벡터의 순위로 'rank'를 구할 수 있습니다.

'rank'의 평균값을 통해 테스트셋에 대한 'mean-rank'를 구할수 있습니다.

또한 상위 10 순위에 포함되는지 유무를 통해 'hits@10(%)' 비율을 구할 수 있습니다.

<img width="340" alt="스크린샷 2022-05-05 오후 1 43 40" src="https://user-images.githubusercontent.com/25244851/166864704-27d53b55-49c1-4dcc-a4c1-be4109158a70.png">

# Method

<img width="555" alt="스크린샷 2022-05-05 오후 1 44 08" src="https://user-images.githubusercontent.com/25244851/166864730-835f140f-af21-4d84-8c08-38c43797fadd.png">

## Algorithm

<img width="562" alt="스크린샷 2022-05-05 오후 1 44 23" src="https://user-images.githubusercontent.com/25244851/166864739-7e025561-34f4-4d8e-96cb-331c2a7a9fe9.png">

# Result

Link prediction Task에서 이전의 방법에 비해 좋은 성능을 보였습니다.

큰 데이터셋(FB1M)에 대해서도 동작하였고 좋은 성능을 보입니다.

모델의 파라미터의 갯수도 적은 편(0.81M)입니다. 기존 방법들과 비교했을 때 Unstructured를 제외하면 가장 적습니다.

<img width="664" alt="스크린샷 2022-05-05 오후 1 51 18" src="https://user-images.githubusercontent.com/25244851/166865288-bc85c481-2d22-4ef2-85fa-b1795667b23d.png">
