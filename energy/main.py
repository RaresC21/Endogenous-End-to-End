import pandas as pd
import numpy as np
from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
try: import setGPU
except ImportError: pass
import warnings
import torch
import model_classes, nets
from constants import *
from model_classes import SolvePointQP
from data import get_data, get_decision_mask, get_decision_mask_
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")


def task_loss_no_mean(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
        params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0))

def pred_obj(x, w): 
    f_pred = f_net(x, w)
    m = get_decision_mask_(f_pred, DEVICE, w)

    p_pred = p_net(x, f_pred, m)

    v2 = p_pred[:,w:]
    v1 = p_net(x, torch.zeros_like(f_pred), torch.zeros_like(f_pred))[:,:w]

    v = torch.cat((v1, v2), 1)

    loss = nets.task_loss(v, f_pred, params).mean()
    return loss.item()


def eval_decision(x, y, w):
    f_pred = f_net(x, w)
    m = get_decision_mask_(y, DEVICE, w)

    p_pred = p_net(x, y, m)
    
    v1 = p_net(x, y, torch.zeros_like(y))[:,:w]
    v2 = p_pred[:,w:]
            
    v = torch.cat((v1, v2), 1)

    loss = nets.task_loss(v, y, params).mean()
    return loss.item()


def get_decisions(X_test_pt): 
    vanilla = [] 
    ours = [] 
    random = []
    optimal = []
    two_stage = []
    all_costs = []
    end_to_end_net.eval()
    f_net.eval()
    p_net.eval()
    reward_losses = []
    my_decisions = []
    
    # get decisions for the cost-learner/reward-learner
    all_rewards = []
    for w in range(24):
        r = reward_learner(X_test_pt, w).cpu().detach()
        all_rewards.append(list(r))
    all_rewards = np.array(all_rewards)
    reward_actions = np.argmin(all_rewards, axis=0)

    for indx in range(len(X_test_pt)):
        X_t = X_test_pt[indx:indx+1,:]
        Y_t = Y_test_pt[indx:indx+1,:]

        e2e_pred = p_net(X_t, torch.zeros_like(Y_t), torch.zeros_like(Y_t))
        e2e = nets.task_loss(e2e_pred, Y_t, params).mean().item() 
        two_stage_cost = nets.task_loss(model_rmse(X_t), Y_t, params).mean().item()

        best_cost = 1e5 
        best_w = -1
        cur_costs = [] 
        for w in range(24): # get best_decision w using predicted cost
            c = pred_obj(X_t, w)
            if c < best_cost: 
                best_cost = c
                best_w = w

        my_decisions.append(best_w)
        our = eval_decision(X_t, Y_t, best_w)

        r = np.random.randint(24)
        rand = eval_decision(X_t, Y_t, r)

        best_cost = 1e5 
        best_dec = -1
        uniform_random = 0
        for w in range(24): # get cost of each decision
            cur = eval_decision(X_t, Y_t, w)
            if cur < best_cost: 
                best_cost = cur 
                best_dec = w
            uniform_random += cur
            cur_costs.append(cur)
        all_costs.append(cur_costs)

        reward_loss = eval_decision(X_t, Y_t, reward_actions[indx]) 
        reward_losses.append(reward_loss)

        optimal.append(best_cost)
        vanilla.append(e2e)    
        ours.append(our)
        random.append(uniform_random / 24)
        two_stage.append(two_stage_cost)
        
    return optimal, vanilla, ours, random, two_stage, reward_losses, np.array(all_costs)


if __name__ == '__main__':    

    X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data(DEVICE) 
    variables = {'X_train_': X_train_pt, 'Y_train_': Y_train_pt, 
            'X_test_': X_test_pt, 'Y_test_': Y_test_pt}

    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}

    mask = get_decision_mask(Y_train_pt, DEVICE)

    width = 200
    EPOCHS_rmse = 4000
    EPOCHS = 4000

    print("TRAINING 2-STAGE")
    model_rmse = model_classes.Net(X_train[:,:-1], Y_train, [width, width]).to(DEVICE)
    model_rmse = nets.run_rmse_net(model_rmse, variables, X_train, Y_train, EPOCHS_rmse)
    model_rmse.eval()

    print("TRAINING END-to-END")
    end_to_end_net = nets.train_end_to_end(X_train[:,:-1], Y_train, variables, params, EPOCHS, DEVICE)
    end_to_end_net.eval()


    print("TRAIN P-MODEL")
    p_net = nets.train_pnet(end_to_end_net, X_train_pt, Y_train_pt, params, EPOCHS, DEVICE)
    p_net.eval();

    print("TRAIN F-MODEL")
    f_net = model_classes.FNet(X_train[:,:-1], Y_train, [width, width]).to(DEVICE)
    f_net = nets.train_fnet(f_net, p_net, X_train[:,:-1], Y_train, variables, params, EPOCHS, DEVICE)
    f_net.eval()

    print("TRAIN cost learner")
    reward_learner = nets.train_reward_learner(p_net, X_train[:,:-1], Y_train, variables, params, EPOCHS, DEVICE)

    print("\n")
    print("Evaluating")
    optimal, vanilla, ours, random, two_stage, reward_losses, all_costs = get_decisions(X_test_pt)
    single_action = 1 # baseline: always choose action of observing hour 1
    
    print("-------------- MEAN COSTS --------------")
    print("VANILLA e2e", np.mean(vanilla))
    print("Our app    ", np.mean(ours))
    print("Reward     ", np.mean(reward_losses))
    print("random     ", np.mean(random))
    print("optimal    ", np.mean(optimal))
    print("2-s        ", np.mean(two_stage))
    print("single ac  ", np.mean(all_costs[:,single_action]))


    print("-------------- MEDIAN COSTS --------------")
    QUANTILE = 0.5
    print("VANILLA e2e  ", np.quantile(vanilla, QUANTILE))
    print("Our app      ", np.quantile(ours, QUANTILE))
    print("random       ", np.quantile(random, QUANTILE))
    print("reward       ", np.quantile(reward_losses, QUANTILE))
    print("optimal      ", np.quantile(optimal, QUANTILE))
    print("single ac    ", np.quantile(all_costs[:,single_action], QUANTILE))
    print("2-stage      ", np.quantile(two_stage, QUANTILE))

    bins = 20

    cm = 1/2.54  # centimeters in inches
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(40*cm, 40*cm))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(80*cm, 20*cm))
    ax1.hist([ours, optimal], label=['endogenous E2E', 'optimal'], bins=bins, histtype='step', range=[0, 0.6])
    ax1.legend(fontsize="18")
    ax2.hist([all_costs[:,single_action], optimal], label=['single action', 'optimal'], bins=bins, histtype='step', range=[0,0.6])
    ax2.legend(fontsize="18")
    ax3.hist([random, optimal], label=['random', 'optimal'], bins=bins, histtype='step', range=[0,0.6])
    ax3.legend(fontsize="18")
    ax4.hist([reward_losses, optimal], label=['cost learner', 'optimal'], bins=bins, histtype='step', range=[0,0.6])
    ax4.legend(fontsize="18")
    plt.savefig('plots/cost_distribution_long_.png')

