import torch

tensor_uninitialized = torch.Tensor(3, 2)  # 초기화 되지 않은 3 * 2 텐서를 생성, 쓰레기 값이 들어있다.
tensor_rand_initialized = torch.rand(3, 2)  # [0, 1) 구간의 임의의 값으로 초기화된 3 * 2 텐서를 생성.
tensor_with_ones = torch.ones(3, 2)  # 1로 초기화된 3 * 2 텐서를 생성.
tensor_with_zeros = torch.zeros(3, 2)  # 0으로 초기화된 3 * 2 텐서를 생성.

py_lst = [[1, 2],
          [3, 4],
          [5, 6]]

tensor_float = torch.FloatTensor(py_lst)  # 파이선 리스트를 텐서로 변환. 원소의 자료형은 float로 변환된다.
tensor_int = torch.IntTensor(py_lst)  # 파이선 리스트를 텐서로 변환. 원소의 자료형은 int로 변환된다.

# 이외에도 HarfTensor, ByteTensor, ShortTensor 등 다양한 자료형의 텐서로 변환할 수 있다.
# 물론 torch.BoolTensor(3, 2)와 같은 형태로도 초기화되지 않은 텐서를 선언할 수 있다.

# size와 shape는 동일한 정보를 제공한다. 대신 size는 메소드이고 shape는 속성이다.
# shape는 파이선의 tuple 자료형의 하위 자료형이므로 tuple의 모든 연산을 사용 가능하다. ex) shape[0]
size = tensor_rand_initialized.size()
shape = tensor_rand_initialized.shape
print(f'size == shape ? : {size == shape}')

# 텐서 합 예시
a = torch.ones(3, 2)
b = torch.ones(3, 2)
c = torch.ones(3, 3)
print(f'a + b = {a + b}')
print(f'a + 1 = {a + 3}')
# print(f'a + c = {a + c}') # 텐서의 형태가 동일해야 한다.

# 텐서 곱 예시
print(f'a * 4 = {a * 4}')
print(f'a * tensor_int = {a * tensor_int}')  # 대응되는 원소끼리 곱한다.

# 텝서 행렬 곱 예시
d = torch.Tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(f'a @ d = {a @ d}')  # 좌항의 열과 우항의 행이 동일해야한다.

# out-place 연산자
print(f'a, before out-placed calc: {a}')
a.add(b)
print(f'a, after out-placed calc: {a}')

# in-place 연산자: 피연산자의 값이 변경된다.
print(f'a, before in-placed calc: {a}')
a.add_(b)  # in-place 연산자는 트레일링 언더스코어 규칙을 따른다.
print(f'a, after in-placed calc: {a}')

print(f'a[0, 1]: {a[0, 1]}')  # 텐서 인덱싱
print(f'a[0:2]: {a[0:2]}')  # 텐서 로우 슬라이싱
print(f'a[:, 0:1]: {a[:, 0:1]}')  # 텐서 칼럼 슬라이싱

# 텐서 결합 (Concatenation): 연결하는 차원을 제외한 나머지 차원의 크기가 동일해야 한다.
print(f'torch.cat((a, b)): {torch.cat((a, b), dim=0)}')
print(f'torch.cat((a, b)): {torch.cat((a, b), dim=1)}')

# 텐서 스택 (Stack): 새로운 차원을 생성하여 연결하며, 모든 텐서의 크기가 동일해야 한다.
print(f'torch.stack((a, b)): {torch.stack((a, b))}')  # shape = (2, 3, 2)

# 텐서 분할
print(f'torch.split(a, 1, dim=1) : {torch.split(a, 1, dim=1)}')  # 나뉘어진 텐서의 개별 크기를 지정.
print(f'torch.split(a, [1, 2]): {torch.split(a, [1, 2])}')  # 텐서의 개별 크기를 리스트로 지정할 수 있다.
print(f'torch.chunk(a, 2): {torch.chunk(a, 2)}')  # 지정된 개수만큼 텐서를 균등 분할.

# 텐서 압축 / 확장
tensor_randn = torch.randn(1, 2, 1, 3, 1)
print(f'tensor_randn.squeeze() : {tensor_randn.squeeze().shape}')  # 매개변수를 통해 특정 차원만 압축할 수 있다.
print(f'tensor_randn.unsqueeze() : {tensor_randn.unsqueeze(0).shape}')  # 0번째에 새로운 차원 추가
