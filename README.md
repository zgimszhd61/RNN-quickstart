# RNN-quickstart

循环神经网络（RNN）是一种专门用于处理序列数据的神经网络算法。它的核心特点是网络中存在循环结构，这使得网络能够保持对之前信息的记忆，从而对序列中的数据进行有效处理。RNN广泛应用于自然语言处理、语音识别、时间序列预测等领域。

下面我将给出一个使用Python和PyTorch框架的具体例子，我们将通过一个简单的RNN来实现对正弦波数据的预测。这个例子将展示如何构建和训练一个循环神经网络，用于预测给定历史数据点后的下一个值。

首先，请确保你已经安装了PyTorch。如果还未安装，可以通过运行 `pip install torch` 来安装。

接下来是完整的代码示例：

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(0)
np.random.seed(0)

# 生成模拟数据：正弦波
t = np.linspace(0, 10, 1000)
data = np.sin(t)

# 将数据转换为RNN的输入格式
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

seq_length = 20
sequences = create_inout_sequences(data, seq_length)

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(RNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)
        output, hidden = self.rnn(input_seq, h0)
        predictions = self.linear(output[:, -1])
        return predictions

# 实例化模型，定义损失函数和优化器
model = RNN()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    for seq, labels in sequences:
        optimizer.zero_grad()
        seq = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
        labels = torch.FloatTensor(labels).unsqueeze(0)
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'epoch: {epoch} loss: {single_loss.item():10.8f}')

# 测试模型
test_inputs = data[-seq_length:].tolist()
model.eval()
for i in range(20):
    seq = torch.FloatTensor(test_inputs[-seq_length:])
    with torch.no_grad():
        test_inputs.append(model(seq.view(1, seq_length, 1)).item())

# 绘制预测结果
plt.figure(figsize=(12,6))
plt.title('RNN Sin Wave Prediction')
plt.xlabel('t')
plt.ylabel('Sin(t)')
plt.plot(t, data, label='Original Data')
plt.plot(t[-(seq_length+20):], test_inputs[-(seq_length+20):], label='Predicted Data')
plt.legend()
plt.show()
```

这段代码首先生成一个正弦波数据，然后构建一个RNN模型来预测给定历史数据后的下一个点。我们通过训练这个网络，使其能够预测接下来的正弦波值。最后，我们用一组测试数据来评估模型的预测效果，并将结果绘制出来。

