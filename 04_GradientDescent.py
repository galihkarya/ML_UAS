# Gradient numpy

import numpy as np

# f = w * x
# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0
def forward(x):
    return w * x
def loss(y, y_predicted):
    return((y_predicted-y)**2).mean()
# gradient
# MSE = 1/N * (w * x - y) ** 2
# dJ/dw = 1/N 2x (w * x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
print(f'Prediction before training: f(5) = {forward(5):.3f}')
# training
iteration = 20
learning_rate = 0.01
for epoch in range(iteration):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient
    dw = gradient(X, X, y_pred)
    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'prediction after training: f(5) = {forward(5):.3f}')


# Gradient torch

import torch

# f = w * x
# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forward(x):
    return w * x
def loss(y, y_predicted):
    return((y_predicted-y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')
# training
iteration = 100
learning_rate = 0.01
for epoch in range(iteration):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient = backward pass
    l.backward() # dl/dw
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'prediction after training: f(5) = {forward(5):.3f}')