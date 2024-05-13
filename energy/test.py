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

# import matplotlib.pyplot as plt
# import tkinter
# import matplotlib
# import matplotlib.pyplot as plt

from model_classes import SolvePointQP

from data import get_data, get_decision_mask, get_decision_mask_


warnings.simplefilter("ignore")

def pred_obj(x, w): 
    f_pred = f_net(x)[:, w*24 : w*24 + 24]
    m = get_decision_mask_(f_pred, DEVICE, w)

    p_pred = p_net(x, f_pred, m)

    v1 = f_pred[:,:w]
    v2 = p_pred[:,w:]
            
    v = torch.cat((v1, v2), 1)

    loss = nets.task_loss(v, f_pred, params).mean()
    return loss.item()


def eval_decision(x, y, w):
    f_pred = f_net(x)[:, w*24 : w*24 + 24]
    m = get_decision_mask_(f_pred, DEVICE, w)

    p_pred = p_net(x, y, m)
    
    v1 = f_pred[:,:w]
    v2 = p_pred[:,w:]
            
    v = torch.cat((v1, v2), 1)

    loss = nets.task_loss(v, y, params).mean()
    return loss.item()


DEVICE = "cpu"
if torch.cuda.is_available(): 
    DEVICE = "cuda"
print("DEVICE", DEVICE, "hi")

X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data(DEVICE) 
variables = {'X_train_': X_train_pt, 'Y_train_': Y_train_pt, 
        'X_test_': X_test_pt, 'Y_test_': Y_test_pt}

params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}

mask = get_decision_mask(Y_train_pt, DEVICE)

EPOCHS_rmse = 200

print("TRAINING 2-STAGE")
model_rmse = model_classes.Net(X_train[:,:-1], Y_train, [200, 200]).to(DEVICE)
model_rmse = nets.run_rmse_net(model_rmse, variables, X_train, Y_train, EPOCHS_rmse)

# p = model_rmse(X_test_pt)[0]
# solver = SolvePointQP(params, DEVICE=DEVICE)
# decision = solver(p)[:,:params["n"]]

EPOCHS_e2e = 200

print("TRAINING END-to-END")
end_to_end_net = nets.train_end_to_end(X_train[:,:-1], Y_train, variables, params, EPOCHS_e2e, DEVICE)

print("TRAIN P-MODEL")
p_net = nets.train_pnet(end_to_end_net, X_train_pt, Y_train_pt, params, EPOCHS_e2e, DEVICE)

print("TRAIN F-MODEL")
f_net = nets.train_fnet(p_net, X_train[:,:-1], Y_train, variables, params, EPOCHS_e2e, DEVICE)



end_to_end_net.eval()
p_net.eval()
f_net.eval()





# for indx in range(len(X_test_pt)):
for indx in [1]:
    X_t = X_test_pt[indx:indx+1,:]
    Y_t = Y_test_pt[indx:indx+1,:]

    print("end-to-end cost:", nets.task_loss(end_to_end_net(X_t), Y_t, params).mean().item())


    a = 0.01
    best_cost = 1e5 
    best_w = -1
    for w in range(24):
        c = pred_obj(X_t, 1) + a*w
        print(w, c)

        if c < best_cost: 
            best_cost = c
            best_w = w


#     print("best:", best_w, best_cost)
    print("true cost:", best_w, eval_decision(X_t, Y_t, best_w) + a*best_w)
    print()

# print("all costs:")
# for w in range(24):
#     print(w," ", eval_decision(X_t, Y_t, w) + a*w)
