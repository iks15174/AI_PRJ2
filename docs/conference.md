# 회의록

### 2022-05-22

1. data agumentation을 통해 train data의 양을 늘려보자
2. 기본 CNN 모델의 depth를 깊게 만들어자
3. 모델이 깊어질 경우 gradient vanishing을 막을 수 있는 방법을 적용해 보자
   - L2 regulation
   - Batch Normalization
4. 적절한 filter size 선택하기

### 2022-05-25

1. class 별 정확도 계산 기능 추가. trash의 정확도가 낮은 것을 확인할 수 있다. 추측되는 이유는 1. data의 갯수가 적어서 2. cardboard와 이미지가 비슷하다. 두가지 이다.

2. result

````
Epoch: 0
Train Loss: 2.5267837047576904, accuracy: 41.87405776977539
Valid loss: 1.2351734638214111, accuracy: 55.79268264770508
Accuracy for class: cardboard is 80.4 %
Accuracy for class: glass is 18.5 %
Accuracy for class: metal is 39.3 %
Accuracy for class: paper is 75.9 %
Accuracy for class: plastic is 72.1 %
Accuracy for class: trash is 29.4 %
Epoch: 1
Train Loss: 1.4043346643447876, accuracy: 52.7714958190918
Valid loss: 1.4444680213928223, accuracy: 52.439022064208984
Accuracy for class: cardboard is 41.3 %
Accuracy for class: glass is 26.2 %
Accuracy for class: metal is 32.1 %
Accuracy for class: paper is 89.2 %
Accuracy for class: plastic is 54.1 %
Accuracy for class: trash is 64.7 %
Epoch: 2
Train Loss: 1.1308189630508423, accuracy: 59.74736022949219
Valid loss: 1.3639541864395142, accuracy: 50.91463088989258
Accuracy for class: cardboard is 76.1 %
Accuracy for class: glass is 47.7 %
Accuracy for class: metal is 28.6 %
Accuracy for class: paper is 54.2 %
Accuracy for class: plastic is 49.2 %
Accuracy for class: trash is 58.8 %
Epoch: 3
Train Loss: 0.9858716130256653, accuracy: 64.34766387939453
Valid loss: 1.148551344871521, accuracy: 61.585365295410156
Accuracy for class: cardboard is 80.4 %
Accuracy for class: glass is 35.4 %
Accuracy for class: metal is 37.5 %
Accuracy for class: paper is 85.5 %
Accuracy for class: plastic is 75.4 %
Accuracy for class: trash is 23.5 %
Epoch: 4
Train Loss: 0.846425473690033, accuracy: 68.51432800292969
Valid loss: 1.0680702924728394, accuracy: 64.63414764404297
Accuracy for class: cardboard is 60.9 %
Accuracy for class: glass is 66.2 %
Accuracy for class: metal is 39.3 %
Accuracy for class: paper is 88.0 %
Accuracy for class: plastic is 70.5 %
Accuracy for class: trash is 17.6 %
Epoch: 5
Train Loss: 0.7433792948722839, accuracy: 72.30392456054688
Valid loss: 1.1101417541503906, accuracy: 64.93901824951172
Accuracy for class: cardboard is 67.4 %
Accuracy for class: glass is 50.8 %
Accuracy for class: metal is 51.8 %
Accuracy for class: paper is 84.3 %
Accuracy for class: plastic is 72.1 %
Accuracy for class: trash is 35.3 %
Epoch: 6
Train Loss: 0.6809184551239014, accuracy: 74.54751586914062
Valid loss: 1.0390526056289673, accuracy: 66.7682876586914
Accuracy for class: cardboard is 71.7 %
Accuracy for class: glass is 44.6 %
Accuracy for class: metal is 58.9 %
Accuracy for class: paper is 84.3 %
Accuracy for class: plastic is 77.0 %
Accuracy for class: trash is 41.2 %
Epoch: 7
Train Loss: 0.6568846702575684, accuracy: 75.9992446899414
Valid loss: 1.0308512449264526, accuracy: 67.37804412841797
Accuracy for class: cardboard is 80.4 %
Accuracy for class: glass is 60.0 %
Accuracy for class: metal is 50.0 %
Accuracy for class: paper is 89.2 %
Accuracy for class: plastic is 59.0 %
Accuracy for class: trash is 41.2 %
Epoch: 8
Train Loss: 0.585312008857727, accuracy: 78.31825256347656
Valid loss: 1.0885998010635376, accuracy: 63.71950912475586
Accuracy for class: cardboard is 80.4 %
Accuracy for class: glass is 61.5 %
Accuracy for class: metal is 39.3 %
Accuracy for class: paper is 88.0 %
Accuracy for class: plastic is 52.5 %
Accuracy for class: trash is 29.4 %
Epoch: 9
Train Loss: 0.5761536359786987, accuracy: 79.20437622070312
Valid loss: 1.0214850902557373, accuracy: 65.54877471923828
Accuracy for class: cardboard is 76.1 %
Accuracy for class: glass is 63.1 %
Accuracy for class: metal is 46.4 %
Accuracy for class: paper is 80.7 %
Accuracy for class: plastic is 65.6 %
Accuracy for class: trash is 35.3 %
Epoch: 10
Train Loss: 0.5134789347648621, accuracy: 81.18401336669922
Valid loss: 1.5563862323760986, accuracy: 61.585365295410156
Accuracy for class: cardboard is 73.9 %
Accuracy for class: glass is 49.2 %
Accuracy for class: metal is 66.1 %
Accuracy for class: paper is 55.4 %
Accuracy for class: plastic is 80.3 %
Accuracy for class: trash is 23.5 %
Epoch: 11
Train Loss: 0.4982297420501709, accuracy: 81.76847839355469
Valid loss: 1.3180619478225708, accuracy: 63.41463088989258
Accuracy for class: cardboard is 82.6 %
Accuracy for class: glass is 38.5 %
Accuracy for class: metal is 51.8 %
Accuracy for class: paper is 85.5 %
Accuracy for class: plastic is 67.2 %
Accuracy for class: trash is 23.5 %
Epoch: 12
Train Loss: 0.41502153873443604, accuracy: 85.2752685546875
Valid loss: 1.5019099712371826, accuracy: 62.499996185302734
Accuracy for class: cardboard is 89.1 %
Accuracy for class: glass is 30.8 %
Accuracy for class: metal is 69.6 %
Accuracy for class: paper is 74.7 %
Accuracy for class: plastic is 62.3 %
Accuracy for class: trash is 29.4 %
Epoch: 13
Train Loss: 0.4120248854160309, accuracy: 85.40724182128906
Valid loss: 1.4144072532653809, accuracy: 63.1097526550293
Accuracy for class: cardboard is 80.4 %
Accuracy for class: glass is 33.8 %
Accuracy for class: metal is 58.9 %
Accuracy for class: paper is 78.3 %
Accuracy for class: plastic is 72.1 %
Accuracy for class: trash is 35.3 %
Epoch: 14
Train Loss: 0.3489047884941101, accuracy: 86.91553497314453
Valid loss: 1.4636691808700562, accuracy: 63.71950912475586
Accuracy for class: cardboard is 76.1 %
Accuracy for class: glass is 75.4 %
Accuracy for class: metal is 32.1 %
Accuracy for class: paper is 78.3 %
Accuracy for class: plastic is 59.0 %
Accuracy for class: trash is 35.3 %  ```
````

### 2022-05-27

validation 결과가 수렴하지 않고 너무 튀는게 제일 문제. 단적으로 말해 모델 성능과 무관하게 학습 그래프가 전혀 예쁘지가 않다.. 결과의 설득력 문제

다만 validation 성능이 가장 좋았던 모델을 선택해 test에 대해 돌려보면 최대 정확도 80% 근처까지 나옴

더 시도해보아야 할 것:

- overfit 막기 위해 앙상블 적용

  - 제일 유의미할 것 같은 시도는 trash 클래스 이진분류 모델 만들고 먼저 적용하기

- pretrain 모델 적용
  - 이미지 데이터의 일반적인 특징조차도 학습을 제대로 못 해서 너무 튀는 것일 가능성도 큼
