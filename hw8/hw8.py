import torch

step = 0.01

x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

for _ in range(1000):

    loss = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8
    loss.backward()
    
    with torch.no_grad():
        x -= step * x.grad
        y -= step * y.grad
        z -= step * z.grad
        
        x.grad.zero_()
        y.grad.zero_()
        z.grad.zero_()

print(f'x={x.item():.4f} y={y.item():.4f} z={z.item():.4f}')