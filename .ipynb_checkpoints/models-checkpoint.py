import numpy as np 

import torch
import torch.nn.functional as F

def get_loader(X, y): 
    training_set = [[X[i], y[i]] for i in range(len(X))]
    return torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True)


class PNet(torch.nn.Module): 
    def __init__(self, problem, DEVICE): 
        super(PNet, self).__init__() 
        
        self.problem = problem
        self.DEVICE=DEVICE
        
        self.n_locations = problem.n_locations
        self.linear = torch.nn.Linear(self.n_locations, self.n_locations, device=DEVICE) 
    
    def forward(self, w, z): 
        n = w.shape[0]
        zw = z[torch.arange(w.shape[0], device=self.DEVICE), w]
        inp = torch.zeros(n, self.n_locations, device=self.DEVICE)
        inp[torch.arange(n, device=self.DEVICE), w] = zw
        return self.linear(inp)
    
class FNet(torch.nn.Module): 
    def __init__(self, problem, DEVICE): 
        super(FNet, self).__init__() 
    
        self.problem = problem
        self.DEVICE=DEVICE
        
        self.n_locations = problem.n_locations
        self.linear = torch.nn.Linear(self.n_locations, self.n_locations, device=DEVICE) 

    def forward(self, w): 
        try:
            inp = F.one_hot(w, num_classes=self.n_locations).float()
            return self.linear(inp)
        except Exception as e: 
            print(w.shape, inp.shape)
            print(e)
        

def loss_fn(p, f, W, Z, problem): 
    f_pred = f(W)
    p_pred = p(W, f_pred)
    nv_pred = problem.get_objective(p_pred, f_pred)
    nv_true = problem.get_objective(p(W, Z), Z)
    
    return torch.mean((nv_pred - nv_true) ** 2)
    
def train_e2e(f_model, p_model, W_train, Z_train, problem,  lr=1e-4, EPOCHS=100, DEVICE='cpu'):
    print("training e2e")
    n = W_train.shape[0] 
        
    data_loader = get_loader(W_train, Z_train)
    optimizer = torch.optim.Adam(f_model.parameters(), lr=lr)
    
    losses = [] 
    for epoch in range(EPOCHS):
        for data in data_loader:
            optimizer.zero_grad() 
            
            w, z = data 
            
            loss = loss_fn(p_model, f_model, w, z, problem).mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.cpu().item())
            
        if epoch % (EPOCHS//10) == 0: 
            print("epoch:", epoch, " ", np.mean(losses[-100:]))
    return f_model


def train_p(p_model, W_train, Z_train, problem, lr=1e-4, EPOCHS=100, DEVICE='cpu'): 
    print("training 2nd stage (end-to-end)")
    n = W_train.shape[0] 
        
    data_loader = get_loader(W_train, Z_train)
    optimizer = torch.optim.Adam(p_model.parameters(), lr=lr)
    
    losses = [] 
    for epoch in range(EPOCHS):
        for data in data_loader:
            optimizer.zero_grad() 
            
            w, z = data 
            
            pred = p_model(w, z) 
            loss = problem.get_objective(pred, z).mean()
        
            loss.backward()
            optimizer.step()
            
            losses.append(loss.cpu().item())
            
        if epoch % (EPOCHS//10) == 0: 
            print("epoch:", epoch, " ", np.mean(losses[-100:]))
            
    return p_model

def train_2stage(p_model, W_train, Z_train, problem, lr=1e-4, EPOCHS=100): 
    print("training 2nd stage (mse)")
    n = W_train.shape[0] 
        
    data_loader = get_loader(W_train, Z_train)
    optimizer = torch.optim.Adam(p_model.parameters(), lr=lr)
    
    losses = [] 
    for epoch in range(EPOCHS):
        for data in data_loader:
            optimizer.zero_grad() 
            
            w, z = data 
            
            pred = p_model(w, z) 
            loss = F.mse_loss(pred, z)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.cpu().item())
            
        if epoch % (EPOCHS//10) == 0: 
            print("epoch:", epoch, " ", np.mean(losses[-100:]))
            
    return p_model
    

def decision_e2e(p, f, problem): 
    f_pred = f(problem.actions)
    p_pred = p(problem.actions, f_pred)
    obj_pred = problem.get_objective(p_pred, f_pred) 
    return torch.argmin(obj_pred).cpu().item()