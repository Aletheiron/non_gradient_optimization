import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#Exogenous parameteres
device = torch.device("cpu")

EPOCHS=400000


hidden_units=10

lr=3e-03

#Data Generation

data=torch.arange(0,120,0.01).unsqueeze(dim=1)
y=(data**0.5-134*torch.log(data+2)+8*torch.randn(size=data.size())+0.9*data**0.9)*10*torch.sin(data)+5*data-0.03*data**2+0.05*data**3-0.0007*data**4


#plt.plot(data.cpu().numpy(),y_pred.cpu().numpy(), label='Model')
plt.plot(data.cpu().numpy(),y.cpu().numpy(), label='Real Data')
plt.legend()
plt.show()