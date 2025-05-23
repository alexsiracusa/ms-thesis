import torch
from Sequential2D import IterativeSequential2D, FlatIterativeSequential2D

sizes = [2500, 2500, 2500]

I = torch.nn.Identity()
f1 = torch.nn.Linear(2500, 2500)
f2 = torch.nn.Linear(2500, 2500)
f3 = torch.nn.Linear(2500, 2500)
f4 = torch.nn.Linear(2500, 2500)
f5 = torch.nn.Linear(2500, 2500)
f6 = torch.nn.Linear(2500, 2500)

blocks = [
    [I,     None,   None],
    [f1,    f2,     f3  ],
    [f4,    f5,     f6  ],
]

device = torch.device('cuda')
num_iterations = 10

# Sequential2D
model = IterativeSequential2D(blocks, num_iterations=num_iterations)
data = torch.zeros((1, 2500))

model = model.to(device)
data = data.to(device)

for _ in range(1):
    output = model.forward(data)
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


# Flat Sequential2D
model = FlatIterativeSequential2D(blocks, sizes, num_iterations=num_iterations)
data = torch.zeros((1, 7500))

model = model.to(device)
data = data.to(device)

for _ in range(1):
    output = model.forward(data)
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


# Linear
model = torch.nn.Linear(7500, 7500)
data = torch.zeros((1, 7500))

model = model.to(device)
data = data.to(device)

for _ in range(num_iterations):
    data = model.forward(data)
print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")




