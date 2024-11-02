import numpy as np
import torch


def binary_encoder(input_size):
    def wrapper(num):  # 입력된 숫자를 input_size 길이의 이진 리스트로 변환한다.
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (input_size - len(ret)) + ret

    return wrapper


def training_test_gen(x, y):
    assert len(x) == len(y)
    indices = np.random.permutation(range(len(x)))
    split_size = int(0.9 * len(indices))
    trX = x[indices[:split_size]]
    trY = y[indices[:split_size]]
    teX = x[indices[split_size:]]
    teY = y[indices[split_size:]]
    return trX, trY, teX, teY


def get_numpy_data(intpu_size=10, limit=1000):
    x = []
    y = []
    encoder = binary_encoder(intpu_size)
    for i in range(limit):  # 0부터 limit까지의 수를 이진수로 변환하여 x에 저장하고 대응되는 결과를 y에 저장.
        x.append(encoder(i))
        if i % 15 == 0:  # 출력값은 그대로 출력, Fizz, Buzz, FizzBuzz로 총 4개이므로 크기가 4인 벡터로 반환.
            y.append([1, 0, 0, 0])
        elif i % 5 == 0:
            y.append([0, 1, 0, 0])
        elif i % 3 == 0:
            y.append([0, 0, 1, 0])
        else:
            y.append([0, 0, 0, 1])
    return training_test_gen(np.array(x), np.array(y))


# 초매개변수
epochs = 500
batches = 64
lr = 0.01
input_size = 10
output_size = 4
hidden_size = 100

trX, trY, teX, teY = get_numpy_data(input_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

x = torch.from_numpy(trX).to(device=device, dtype=dtype)
y = torch.from_numpy(trY).to(device=device, dtype=dtype)
w1 = torch.randn(input_size, hidden_size, requires_grad=True, device=device, dtype=dtype)
w2 = torch.randn(hidden_size, output_size, requires_grad=True, device=device, dtype=dtype)
b1 = torch.randn(1, hidden_size, requires_grad=True, device=device, dtype=dtype)
b2 = torch.randn(1, output_size, requires_grad=True, device=device, dtype=dtype)


def network_execution_over_whole_dataset():
    pass


for i in range(epochs):
    network_execution_over_whole_dataset()
