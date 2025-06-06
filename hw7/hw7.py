from micrograd.engine import Value

step = 0.01

x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

def loss_function(x, y, z):
    return (x - 1) ** 2 + (y - 2) ** 2 + (z - 3) ** 2  

for _ in range(10000):
    x.grad = 0
    y.grad = 0
    z.grad = 0

    loss = loss_function(x, y, z)
    loss.backward()  

    x.data -= step * x.grad
    y.data -= step * y.grad
    z.data -= step * z.grad

print(f'x={x.data:.4f} y={y.data:.4f} z={z.data:.4f}')