import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 3.1], dtype=np.float32)
y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 1.3], dtype=np.float32)
x_train = x_train.reshape(-1, 1) #行列を15*1に
y_train = y_train.reshape(-1, 1)
input_size = 1
output_size = 1
num_epochs = 1000000  #学習を繰り返す回数（エポック数）
learning_rate = 0.0002 #学習率１度にどれだけモデルのパラメータを変化させるか
class LinearRegression(nn.Module):#線形回帰用mport torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 3.1], dtype=np.float32)
y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 1.3], dtype=np.float32)
x_train = x_train.reshape(-1, 1) #行列を15*1に
y_train = y_train.reshape(-1, 1)
input_size = 1
output_size = 1
num_epochs = 1000000  #学習を繰り返す回数（エポック数）
learning_rate = 0.0002 #学習率１度にどれだけモデルのパラメータを変化させるか
class LinearRegression(nn.Module):#線形回帰用にnn.Moduleを継承したクラスを作製
  def __init__(self, input_size, output_size):
    super(LinearRegression, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    out = self.linear(x)
    return out
model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss() #平均二乗誤差で評価
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    inputs = torch.tensor(x_train)
    targets = torch.tensor(y_train)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
torch.save(model.state_dict(), 'model.pkl')
predicted = outputs.detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
