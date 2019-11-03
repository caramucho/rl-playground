import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


samples = 100
x_train = [2 + i/samples * 5.5 for i in range(samples)]
y_train = [100*math.sin(x) for x in x_train]

plt.plot(x_train, y_train)


class regression(nn.Module):
    def __init__(self):
        super(regression, self).__init__()
        self.my_para = nn.Parameter(torch.randn(7))
        # self.my_c = nn.Parameter(torch.randn(1))
        # self.my_b = nn.Parameter(torch.randn(1))
        # self.my_a = nn.Parameter(torch.randn(1))
        # self.my_c = nn.Parameter(torch.Tensor([10]))
        # self.my_b = nn.Parameter(torch.Tensor([-8]))
        # self.my_a = nn.Parameter(torch.Tensor([1]))

        # self.not_param = Variable(torch.randn(1), requires_grad=True)

    def forward(self, x):
        res = [self.my_para[i] * x**i for i in range(7)]
        return torch.sum(res)


learning_rate = 1
model = regression()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10000
for epoch in range(num_epochs):
    inputs = torch.Tensor(x_train)
    targets = torch.Tensor(y_train)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


samples = 100
x_test = [1.5 + i/samples * 6.5 for i in range(samples)]
y_test = [model.my_d*x*x*x + model.my_a * x *
          x + model.my_b*x+model.my_c for x in x_test]

plt.plot(x_test, y_test)
plt.plot(x_train, y_train)
