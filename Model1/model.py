import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Model1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        # self.hidden = (torch.zeros(self.num_layers, 1, self.hidden_size), torch.zeros(self.num_layers, 1, self.hidden_size))    

    def forward(self, x):
        output, _ = self.lstm(x.view(8, 1, -1))
        output = self.linear(output.view(8, -1))
        return output

def train(model, dataloader, loss_fn, optimizer):
    train_loss = 0
    model.train()
    train_acc = 0
    for (X, y) in dataloader:
        output = model(X)
        loss = loss_fn(output.squeeze()[-1], y.squeeze()[-1])
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += abs(output.squeeze()[-1]-y.squeeze()[-1])/y.squeeze()[-1]
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(train_loss, train_acc)

def test(model, X, y, loss_fn):
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for x in range(len(X)):
            output, hn = model(X[x], hn)
            test_loss += loss_fn(output, y[x])
        test_loss /= len(X)
    print(test_loss)


def get_data():
    df = pd.read_csv('weekly_data.csv')
    pointer = 0
    df_values = df.to_numpy()
    n = len(df_values)
    input_data = []
    output_data = []
    while pointer < n - 9:
        in_sequence = []
        out_sequence = []
        val = []
        for i in range(0, 8):
            try:
                for st in df_values[pointer+i][2].split(" "):
                    val.append(float(st))
            except:
                val.append(5.0)
            in_sequence.append([df_values[pointer+i][1], np.mean(val)])
            out_sequence.append(df_values[pointer+i+1][1])
        input_data.append(in_sequence)
        output_data.append(out_sequence)
        pointer += 1
    return torch.FloatTensor(input_data), torch.FloatTensor(output_data)

def main():
    torch.autograd.set_detect_anomaly(True)
    X, y = get_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, shuffle=True)
    input_size, hidden_size, num_layers = 2, 20, 4
    model = Model1(input_size, hidden_size, num_layers)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    for _ in range(5):
        train(model, dataloader, loss_fn, optimizer)

main()