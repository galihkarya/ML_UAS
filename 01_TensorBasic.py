import torch
import numpy as np

# x = torch.empty(3)
# print(x)

# x = torch.empty(2, 3)
# print(x)
print('empty function')
x = torch.empty(2, 2, 2, 3)
print(x)

print('\nrandom fucntion')
x = torch.rand(2, 2)
print(x)

print('\nzeros function')
x = torch.zeros(2, 2)
print(x)

print('\nones function')
# x = torch.ones(2, 2)
x = torch.ones(2, 2, dtype=torch.double)
# print(x)
# print(x.dtype)
print(x.size())

x = torch.rand(2, 2)
y = torch.rand(2, 2)
# print(x)
# print(y)

# z = x + y
# z = torch.add(x, y)

# z = x - y
# z = torch.sub(x, y)

# z = x * y
# z = torch.mul(x, y)

# z = x / y
# z = torch.div(x, y)

# print(z)
y.add_(x)
print(y)


x = torch.rand(5, 3)
print(x)
# print(x[:, 0]) 
# print(x[1, :])
print(x[1, 1].item())

x = torch.rand(4, 4)
y = x.view(-1, 8)
print(y)

a = torch.ones(5)
print(a)
b = a.numpy()
# print(type(b))
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
a += 1
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # z.numpy()
    z = z.to('cpu')

x = torch.ones(5, requires_grad=True)
print(x)