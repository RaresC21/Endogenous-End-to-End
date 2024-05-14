#/usr/bin/env python3

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

import model_classes
from constants import *
from data import get_decision_mask, get_decision_mask_

def get_loader(X, y): 
    training_set = [[X[i], y[i]] for i in range(len(X))]
    return torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True)




def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0)).mean(0)

def task_loss_no_mean(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
        params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0))

def rmse_loss(mu_pred, Y_actual):
    return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()

def rmse_loss_weighted(mu_pred, Y_actual, weights):
    return ((weights * (mu_pred - Y_actual)**2).mean(dim=0).sqrt()).sum()


def run_rmse_net(model, variables, X_train, Y_train, EPOCHS=1000):
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(EPOCHS):
        opt.zero_grad()
        model.train()
        train_loss = nn.MSELoss()(
            model(variables['X_train_']), variables['Y_train_'])
        train_loss.backward()
        opt.step()


    model.eval()
    model.set_sig(variables['X_train_'], variables['Y_train_'])

    return model


def run_weighted_rmse_net(X_train, Y_train, X_test, Y_test, params):
    weights = torch.ones(Y_train.shape, device=DEVICE)
    for i in range(10):
        model, weights2 = run_weighted_rmse_net_helper(X_train, Y_train, X_test, Y_test, params, weights, i)
        weights = weights2.detach()
    return model

def run_weighted_rmse_net_helper(X_train, Y_train, X_test, Y_test, params, weights, i):
    X_train_ = torch.tensor(X_train[:,:-1], dtype=torch.float, device=DEVICE)
    Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_test_ = torch.tensor(X_test[:,:-1], dtype=torch.float, device=DEVICE)
    Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)

    model = model_classes.Net(X_train[:,:-1], Y_train, [200, 200])
    if USE_GPU:
        model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    solver = model_classes.SolveScheduling(params)
    for j in range(100):

        model.train()
        batch_train_weightrmse(100, i*100 + j, X_train_.data, Y_train_.data, model, opt, weights.data)

    # Rebalance weights
    model.eval()
    mu_pred_train, sig_pred_train = model(X_train_)
    Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
    weights2 = task_loss_no_mean(
        Y_sched_train.float(), Y_train_, params)
    if USE_GPU:
        weights2 = weights2.cuda()
    model.set_sig(X_train_, Y_train_)

    return model, weights2

def train_end_to_end(X_train, Y_train, variables, params, EPOCHS, DEVICE): 
    model = model_classes.Net(X_train, Y_train, [200, 200]).to(DEVICE)
    
    data_loader = get_loader(variables["X_train_"], variables["Y_train_"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    losses = [] 
    for epoch in range(EPOCHS):
        for data in data_loader:
            optimizer.zero_grad() 
            
            x, y = data 
            v = model(x)[0]
            loss = task_loss(v, y, params).mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.cpu().item())
            
        if epoch % (EPOCHS//10) == 0: 
            print("epoch:", epoch, " ", np.mean(losses[-10:]))
    return model

def train_pnet(e_net, X_train, Y_train, params, EPOCHS, DEVICE): 
    model = model_classes.PNet(e_net, DEVICE).to(DEVICE)
    mask_data = get_decision_mask(Y_train, DEVICE)
    
    data_loader = get_loader(Y_train, mask_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    batch = 50
    n_data = X_train.shape[0]
    
    losses = [] 
    for epoch in range(EPOCHS):
        for i in range(0, n_data, batch):
            optimizer.zero_grad() 
            
            x = X_train[i:i+batch,:]
            y = Y_train[i:i+batch,:]
            w = np.random.randint(0,24)
            m = get_decision_mask_(y, DEVICE, w)
            
            v = model(x, y, m)
            loss = task_loss(v, y, params).mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.cpu().item())
            
        if epoch % (EPOCHS//10) == 0: 
            print("epoch:", epoch, " ", np.mean(losses[-100:]))
    return model


def train_fnet(model, p_net, xx, yy, variables, params, EPOCHS, DEVICE): 
    # model = model_classes.FNet(xx, yy, [200, 200]).to(DEVICE)
    X_train = variables["X_train_"]
    Y_train = variables["Y_train_"]

    batch = 50
    n_data = X_train.shape[0]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)

    losses = [] 
    best_state = None
    best_loss = 1e5
    
    for epoch in range(EPOCHS):
        for i in range(0, n_data, batch):
            optimizer.zero_grad() 
            
            x = X_train[i:i+batch,:]
            y = Y_train[i:i+batch,:]

            w = np.random.randint(0,24)
            m = get_decision_mask_(y, DEVICE, w)
            
            f_pred = model(x, w)
            p_pred = p_net(x, f_pred, m) 
            p_pred_true = p_net(x, y, m)
            
            v1 = f_pred[:,:w]
            v1_true = p_net.e_net(x)[:,:w]
            v2 = p_pred[:,w:]
            v_2true = p_pred_true[:,w:]
            
            v = torch.cat((v1_true, v2), 1)
            v_true = torch.cat((v1_true, v_2true), 1)
                        
            loss_mine = task_loss(v, f_pred, params)
            loss_true = task_loss(v_true, y, params)
            loss = torch.mean(((loss_mine - loss_true) ** 2))
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.cpu().item())
            
        if np.mean(losses[-20:]) < best_loss: 
            best_loss = np.mean(losses[-20:])
            best_state = model.state_dict().copy()
            
            
        if epoch % (EPOCHS//100) == 0: 
            print("epoch:", epoch, " ", np.mean(losses[-100:]))
    model = model_classes.FNet(xx, yy, [200, 200])
    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model

def single_loss(p_net, x, y, w, params): 
    m = get_decision_mask_(y, DEVICE, w)

    p_pred = p_net(x, y, m)

    v1 = p_net(x, y, torch.zeros_like(y))[:,:w]
    v2 = p_pred[:,w:]
    v = torch.cat((v1, v2), 1)

    loss = task_loss(v, y, params).mean()
    return loss.item()
        

def train_reward_learner(p_net, xx, yy, variables, params, EPOCHS, DEVICE):
    model = model_classes.RNet(xx, yy, [200, 200]).to(DEVICE)
    X_train = variables["X_train_"]
    Y_train = variables["Y_train_"]
        
    batch = 50
    n_data = X_train.shape[0]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = [] 
    
    for epoch in range(EPOCHS):
        for i in range(0, n_data, batch):
            optimizer.zero_grad() 
            
            w = np.random.randint(0,24)
            x = X_train[i:i+batch,:]
            y = Y_train[i:i+batch,:]
            l = single_loss(p_net, x, y, w, params)
            
            pred = model(x, w) 
            # print(pred.shape)
            loss = torch.mean((pred - l) ** 2)
            loss.backward() 
            optimizer.step()
            
            losses.append(loss.item())
        if epoch % (EPOCHS // 100) == 0: 
            print(epoch, np.mean(losses[-100:]))
    return model 

def batch_train_weightrmse(batch_sz, epoch, X_train_t, Y_train_t, model, opt, weights_t):

    batch_data_ = torch.empty(batch_sz, X_train_t.size(1), device=DEVICE)
    batch_targets_ = torch.empty(batch_sz, Y_train_t.size(1), device=DEVICE)
    batch_weights_ = torch.empty(batch_sz, weights_t.size(1), device=DEVICE)

    size = batch_sz

    for i in range(0, X_train_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_ = torch.empty(size, X_train_t.size(1), device=DEVICE)
            batch_targets_ = torch.empty(size, Y_train_t.size(1), device=DEVICE)
            batch_weights_ = torch.empty(size, weights_t.size(1), device=DEVICE)

        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]
        batch_weights_.data[:] = weights_t[i:i+size]

        opt.zero_grad()
        preds = model(batch_data_)[0]

        ((batch_weights_ * (preds - batch_targets_)**2).mean(dim=0).sqrt()).sum().backward()

        opt.step()

        print ('Epoch: {}, {}/{}'.format(epoch, i+batch_sz, X_train_t.size(0)))
       

def run_task_net(model, variables, params, X_train, Y_train, args):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    solver = model_classes.SolveScheduling(params)

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(1000):
        opt.zero_grad()
        model.train()
        mu_pred_train, sig_pred_train = model(variables['X_train_'])
        Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
        train_loss = task_loss(
            Y_sched_train.float(),variables['Y_train_'], params)
        train_loss.sum().backward()

        model.eval()
        mu_pred_test, sig_pred_test = model(variables['X_test_'])
        Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
        test_loss = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)

        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])
        Y_sched_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
        hold_loss = task_loss(
            Y_sched_hold.float(), variables['Y_hold_'], params)

        opt.step()

        print(i, train_loss.sum().item(), test_loss.sum().item(), 
            hold_loss.sum().item())


        # Early stopping
        hold_costs.append(hold_loss.sum().item())
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()
                best_model = model_classes.Net(
                    X_train[:,:-1], Y_train, [200, 200])
                best_model.load_state_dict(model_states[idx])
                if USE_GPU:
                    best_model = best_model.cuda()
                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model


def eval_net(which, model, variables, params):
    solver = model_classes.SolveScheduling(params)

    model.eval()
    mu_pred_train, sig_pred_train = model(variables['X_train_'])
    mu_pred_test, sig_pred_test = model(variables['X_test_'])

    if (which == "task_net"):
        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])

    # Eval model on rmse
    train_rmse = rmse_loss(mu_pred_train, variables['Y_train_'])
    test_rmse = rmse_loss(mu_pred_test, variables['Y_test_'])

    if (which == "task_net"):
        hold_rmse = rmse_loss(mu_pred_hold, variables['Y_hold_'])

    # with open(
    #     os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
    #     np.save(f, train_rmse)

    # with open(
    #     os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
    #     np.save(f, test_rmse)

    # if (which == "task_net"):
    #     with open(
    #         os.path.join(save_folder, '{}_hold_rmse'.format(which)), 'wb') as f:
    #         np.save(f, hold_rmse)

    # Eval model on task loss
    # Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
    # train_loss_task = task_loss_no_mean(
    #     Y_sched_train.float(), variables['Y_train_'], params)

    Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
    test_loss_task = task_loss_no_mean(
        Y_sched_test.float(), variables['Y_test_'], params)
    print(test_loss_task.detach().cpu().numpy())

    # if (which == "task_net"):
    #     Y_sched_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
    #     hold_loss_task = task_loss_no_mean(
    #         Y_sched_hold.float(), variables['Y_hold_'], params)

    # np.save(os.path.join(save_folder, '{}_train_task'.format(which)), train_loss_task.detach().cpu().numpy())
    # np.save(os.path.join(save_folder, '{}_test_task'.format(which)), test_loss_task.detach().cpu().numpy())

    # if (which == "task_net"):
    #     np.save(os.path.join(save_folder, '{}_hold_task'.format(which)), hold_loss_task.detach().cpu().numpy())
    return test_loss_task.detach().numpy()