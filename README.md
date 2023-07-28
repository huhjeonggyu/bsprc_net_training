# bsprc_net_training (version 0.0.1)

Monte-Carlo simulation을 이용하여, 임의의 model parameter, option variable에 대해, option의 time value(=price-payoff)가 산출된 대량의 data를 생성하고, 
이후 neural network로 가상의 data를 최대한 잘 학습하여, network가 사실상 옵션의 closed pricing formula로 간주되는 것을 목표로 합니다. 
이러한 접근 방법은 원칙적으로 모든 option pricing model에 대해 feasibility를 제공하는 것을 가능하게 합니다.
현재 구현은 data 생성은 CPU, GPU를 모두 지원하며, network 학습은 오로지 GPU로 이루어집니다. 
또한, CPU의 병렬 처리 능력을 보완하기 위해 10개의 multi-process가 돌아갑니다. 병렬 처리 연산이 많이 필요하면 GPU가 빠르나 그렇지 않으면 CPU가 빠를 수도 있습니다.

중요한 노트북(.ipynb) 파일은 다음과 같습니다.
1. sample.ipynb: data 생성
2. train.ipynb: network 학습
3. test.ipynb: network 성능 테스트

이외의 파이썬(.py) 파일은 다음과 같습니다.
1. model.py: network 정의
2. utils.py: 다양한 utility 함수 정의

또한, 두 개의 디렉토리가 있으며 역할은 다음과 같습니다.
1. data: 생성된 data가 저장되며 data/train에는 training 파일이 data/test에는 test용 파일이 각각 저장됩니다.
2. net: 학습된 network가 저장됩니다. network의 파일 이름에는 훈련에 사용한 data의 개수가 포함되어 있습니다.

이외에도 다음과 같은 사실이 중요합니다.
1. 각 data 파일은 10,000개의 값을 가지고 있습니다.
2. model parameter와 option variable의 범위는, 변동성 sigma는 0.01에서 1까지, 만기 T는 0.01에서 1까지, 행사가 K는 log(K)/sqrt(T)가 -2에서 2까지 포괄합니다.
3. network는 평범한 MLP로 node 1000개의 hidden layer 2개를 가지고, sigma, T, K를 받아들여, option의 time value tv를 내놓습니다.
4. network는 ADAM optimizer를 이용해 학습되며, traning data 중 30%는 learning rate decay 조건 검사를 위한 용도로만 사용됩니다.
5. validation dataset의 loss가 더이상 떨어지지 않는다면 early stopping rule이 적용됩니다. (언제 학습을 종료할지 정하는 것은 매우 중요한 문제입니다..)
6. 실험이 잘 되었다면, data 개수를 10배씩 늘릴 때, test.ipynb의 R2는 대략 10배씩 증가하고, MSE는 대략 10배씩 감소될 것이 기대됩니다.

## 사용법
1. sample.ipynb : data 생성
한개당 10,000개의 data가 저장된다는 사실을 기억하세요.
* train_data_num : training data 파일의 개수
* test_data_num : test data 파일의 개수
2. train.ipynb : network 학습
* data_num : 몇 개의 data를 사용하여 network를 훈련할지 정하세요. (10,000의 배수)
3. test.ipynb : network 성능 테스트
* train_data_num : 얼만큼 data를 사용해 학습된 network을 테스트 할지 정하세요. (10,000의 배수)
* test_data_num : test data의 개수를 정하세요. 가급적 생성한 test data를 모두 사용하세요. (10,000의 배수)
