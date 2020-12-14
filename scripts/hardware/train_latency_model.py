import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import optim
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from aw_nas.hardware.utils import Prim


def reg_data_to_feature(dataset, table=None):
    block_sums = dataset[:, 0].reshape(-1, 1)
    prims = dataset[:, 2]
    for data in prims:
        for p in data:
            p.pop("performances", None)
    avg_block_sums = block_sums if table is None else np.array(
        [sum([table[Prim(**p)] for p in data])
         for data in prims]).reshape(-1, 1)
    overall = dataset[:, 1]
    num_prims = np.array([len(x) for x in dataset[:, 2]]).reshape(-1, 1)
    return np.concatenate([num_prims, avg_block_sums], 1), overall
    # return avg_block_sums, overall


def rnn_data_to_feature(dataset, table=None):
    prims = dataset[:, 2]
    block_latency = [[p.pop("performances")["latency"] for p in data]
                     for data in prims]
    if table is not None:
        block_latency = [[table[Prim(**p)] for p in data] for data in prims]
    overall = dataset[:, 1]
    return block_latency, overall


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, device="cpu"):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True).to(self.device)
        self.linear = nn.Linear(hidden_layer_size, output_size).to(self.device)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
        

    def forward(self, seq, bs=1):
        self.hidden_cell = (torch.zeros(1, bs, self.hidden_layer_size).to(self.device),
                                    torch.zeros(1, bs, self.hidden_layer_size).to(self.device))
        seq = rnn_utils.pack_sequence([torch.tensor(s).reshape(-1, 1).to(self.device) for s in seq], enforce_sorted=False)
        lstm_out, self.hidden_cell = self.lstm(seq, self.hidden_cell)
        lstm_out, index = rnn_utils.pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.permute([1, 0, 2])
        select = torch.zeros(lstm_out.shape[:2]).scatter_(1, index.reshape(-1, 1) - 1, 1).to(torch.bool).to(self.device)
        lstm_out = lstm_out[select]
        predictions = self.linear(lstm_out)
        return predictions[:, -1]

    def predict(self, input_seqs):
        return self.forward(input_seqs, bs=len(input_seqs)).cpu().detach().numpy()

    def fit(self, train_X, train_y, epochs=200, bs=1024):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for i in range(epochs):
            for j in range(len(train_X) // bs + 1):
                seq = train_X[j * bs: (j + 1) * bs]
                label = train_y[j * bs: (j + 1) * bs]
                optimizer.zero_grad()
                pred = self.forward(seq, bs=bs)
                loss = loss_function(pred, torch.tensor(label.astype(np.float32)).to(pred.device))
                loss.backward()
                optimizer.step()

            if i % 5 == 0:
                print("epoch: {:3}, loss: {:10.5f}".format(
                    i, loss.item()))
        return self


def train_and_test(model, data_to_feature_fn, **kwargs):
    with open("samples.pkl", "rb") as fr:
        samples = pickle.load(fr)

    with open("test_model.pkl", "rb") as fr:
        table = pickle.load(fr)["table"]
        table = {Prim(**k): v for k, v in table}

    train_feature, train_y = data_to_feature_fn(samples["train"], None)
    model.fit(train_feature, train_y, **kwargs)
    test_feature, test_y = data_to_feature_fn(samples["test"], table)
    pred_y = model.predict(test_feature)
    err = (pred_y - test_y) / test_y
    print("%e, %e, %e" % (err.max(), err.mean(), err.std()))


def reg_main():
    model = linear_model.LinearRegression()
    train_and_test(model, reg_data_to_feature)


def mlp_main():
    clf = MLPRegressor(solver='lbfgs',
                       alpha=1e-4,
                       hidden_layer_sizes=(10, 10, 10),
                       random_state=1)
    train_and_test(clf, reg_data_to_feature)


def lstm_main():
    model = LSTM(1, 20, 1, device="cuda")
    train_and_test(model, rnn_data_to_feature)


if __name__ == "__main__":
    lstm_main()
