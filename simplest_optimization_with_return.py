import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import copy
#Exogenous parameteres
device = torch.device("cpu")

EPOCHS=1000000


hidden_units=30

lr=3e-04

#magn=1

#Data Generation

data=torch.arange(0,120,0.01).unsqueeze(dim=1)
y=(data**0.5-134*torch.log(data+2)+8*torch.randn(size=data.size())+0.9*data**0.9)*10*torch.sin(data)+5*data-0.03*data**2+0.05*data**3-0.0007*data**4

#Create training/test sets
train_split=int(0.8*len(data))
X_train,y_train=data[:train_split],y[:train_split]
X_test,y_test=data[train_split:],y[train_split:]


X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)

#General Model

class Simple_Model(nn.Module):
    
    def __init__(self, hidden_units:int):
        super().__init__()
        
        self.linear1=nn.Linear(in_features=1, out_features=hidden_units)
        self.linear2=nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear3=nn.Linear(in_features=hidden_units, out_features=1)
    
    def forward(self, x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        x=F.relu(x)
        x=self.linear3(x)
        
        return x

#Loss function
loss=nn.MSELoss()


#Instances of a model 

model=Simple_Model(hidden_units=hidden_units).to(device)
optimizer=torch.optim.Adam(params=model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.001, total_iters=1000)

#Train loop and tracking

loss_all=[]
test_loss_all=[]

loss_all_for_checking=[]

epoch_n=[]

start=timer()


for epoch in range(EPOCHS):
    
    with torch.no_grad():
        logits=model(X_train)
        loss_normal=loss(logits, y_train)
        
        # Around here we should implement new systeme of change. We need remember the direction of the last change of the parameters. 
        # Or generate tensor with required shapes and remember it
        # Then keep change at the same rate. Mean adding this pregenerated tensor to params. 
        # First change is random. change=lr*torch.randn(param.shape)?
        # 
        # if loss function value is lower or equal of the previous value: loss_all[-1]>=loss_all[-2] for the second epoch.
        # if not we randomly change weights again.
        
        if epoch==0:
            
            loss_all.append(loss_normal.log10().item())
            
            random_params_vector_change = []
            
            #Should we remember old params for return to them?
            old_param_list=[]
            
            for param in model.parameters():
                
                old_param=param.data
                old_param_list.append(old_param)
            
            for param in model.parameters():
                # Generate a random tensor with the same shape as the parameter
                # using a standard normal distribution (mean=0, std=1)
                random_tensor = copy.deepcopy(torch.randn(param.shape))
                #print(f'changing vector for params: {random_tensor}')
                random_params_vector_change.append(random_tensor)
                #print(random_params_vector_change)
                
            for i, param in enumerate(model.parameters()):
                
                param.data = param.data + lr* random_params_vector_change[i]
            
        
        if epoch!=0:
            
            loss_all.append(loss_normal.log10().item())
            
            if loss_all[-1]<=loss_all[-2]:
                
                old_param_list=[]
            
                for param in model.parameters():
                    
                    old_param=param.data
                    old_param_list.append(old_param)
                
                
                for i, param in enumerate(model.parameters()):
                
                    param.data = param.data + lr* random_params_vector_change[i]
            
            else:
                
                    #And we should eliminate the last unconvinient loss

                    #loss_all.pop()
                    
                    
                    for i, param in enumerate(model.parameters()):
                
                        param.data = old_param_list[i]

                    random_params_vector_change = []
            
                    for param in model.parameters():
                        # Generate a random tensor with the same shape as the parameter
                        # using a standard normal distribution (mean=0, std=1)
                        random_tensor = copy.deepcopy(torch.randn(param.shape))
                        random_params_vector_change.append(random_tensor)
                        
                    for i, param in enumerate(model.parameters()):
                
                        param.data = param.data + lr* random_params_vector_change[i]
            
        
        # optimizer.zero_grad(set_to_none=True)
        # loss_normal.backward()
        # optimizer.step()
        #scheduler.step()
        
        if epoch%1000==0:
            print (f"normal  losses: {loss_normal.log10()}")
            loss_all_for_checking.append(loss_normal.log10().item())
            epoch_n.append(epoch)
            with torch.no_grad():
                test_logits=model(X_test)
                test_loss=loss(test_logits,y_test)
                print(f'normal test loss: {test_loss.log10()}')
                test_loss_all.append(test_loss.log10().item())

end=timer()
print(end-start)


#print(len(test_loss_all)), print(len(epoch_n))

with torch.no_grad():
    y_pred=model(data.to(device))

plt.plot(data.cpu().numpy(),y_pred.cpu().numpy(), label='Model')
plt.plot(data.cpu().numpy(),y.cpu().numpy(), label='Real Data')
plt.legend()
plt.show()

#Plot results
plt.plot(epoch_n, loss_all_for_checking, label='Train loss')
plt.plot(epoch_n,test_loss_all, label='Test loss')
plt.legend()
plt.show()