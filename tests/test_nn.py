"""
Modified version of https://github.com/jettify/pytorch-optimizer/blob/master/tests/test_optimizer_with_nn.py
"""
import pytest
import numpy as np
import torch
from torch import nn

import optorch


@pytest.fixture
def dataset():
    seed = 42
    rng = np.random.RandomState(seed)
    N = 100
    D = 2

    X = rng.randn(N, D) * 2

    # center the first N/2 points at (-2,-2)
    mid = N // 2
    X[:mid, :] = X[:mid, :] - 2 * np.ones((mid, D))

    # center the last N/2 points at (2, 2)
    X[mid:, :] = X[mid:, :] + 2 * np.ones((mid, D))

    # labels: first N/2 are 0, last N/2 are 1
    Y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    return x, y


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred


def ids(v):
    return '{} {}'.format(v[0].__name__, v[1:])


def build_lookahead(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)


optimizers = [
    (optorch.AdaDelta, {'weight_decay': 0.001, 'momentum': 0.5}, 200),
    (optorch.AdaGrad, {'lr': 1.0}, 200),
    (optorch.Adam, {'lr': 1.0}, 200),
    (optorch.AdaMax, {'lr': 0.1}, 200),
    (optorch.AdamW, {'lr': 1.0, 'weight_decay': 0.001}, 200),
    (optorch.AMSGrad, {'lr': 0.0001}, 200),
    (optorch.Nadam, {}, 200),
    (optorch.Nesterov, {}, 200),
    (optorch.QHAdam, {'lr': 1.0}, 200),
    (optorch.QHM, {}, 200),
    (optorch.RMSProp, {'lr': 1.0}, 200),
    (optorch.SGD, {'lr': 1.0}, 200),
    (optorch.SGDM, {'lr': 1.0}, 200),
    (optorch.SGDW, {'lr': 1.0, 'weight_decay': 0.001}, 200),


]


@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_basic_nn_modeloptimizer_config(dataset, optimizer_config):
    torch.manual_seed(42)
    x_data, y_data = dataset
    model = LogisticRegression()

    loss_fn = nn.BCELoss()
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)
    init_loss = None
    for _ in range(iterations):
        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)
        if init_loss is None:
            init_loss = loss
        optimizer.zero_grad()
        loss.backward(create_graph=True)
        optimizer.step()

    loss = 2.0 * loss.item()
    assert init_loss.item() > loss, (optimizer_class, loss)


