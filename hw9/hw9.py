import torch
import matplotlib.pyplot as plt

x=torch.empty(100).uniform_(0,10)
noise=torch.empty(100).uniform_(-2.5,2.5)
y=-x+noise

w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)

optimizer=torch.optim.SGD([w,b],lr=0.01)

for _ in range(1000):
    optimizer.zero_grad()
    pred=w*x+b
    loss=torch.nn.functional.mse_loss(pred,y)
    loss.backward()
    optimizer.step()
    if (not _) or (not((_+1)%100)):
        print(f"{_+1}/1000, Loss:{loss.item()}")

y_pred=w.detach()*x+b.detach()
plt.plot(x,y,'ro',label='Original data')
plt.plot(x,y_pred,label='Fitted line')
plt.legend()
plt.show()