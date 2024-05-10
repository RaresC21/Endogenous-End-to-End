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

from data import get_data, get_decision_mask


warnings.simplefilter("ignore")


DEVICE = "cpu"
if torch.cuda.is_available(): 
    DEVICE = "cuda"
print("DEVICE", DEVICE, "hi")

X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data(DEVICE) 
variables = {'X_train_': X_train_pt, 'Y_train_': Y_train_pt, 
        'X_test_': X_test_pt, 'Y_test_': Y_test_pt}

params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}

mask = get_decision_mask(Y_train_pt, DEVICE)

EPOCHS_rmse = 1000

print("TRAINING 2-STAGE")
model_rmse = model_classes.Net(X_train[:,:-1], Y_train, [200, 200]).to(DEVICE)
model_rmse = nets.run_rmse_net(model_rmse, variables, X_train, Y_train, EPOCHS_rmse)

# p = model_rmse(X_test_pt)[0]
# solver = SolvePointQP(params, DEVICE=DEVICE)
# decision = solver(p)[:,:params["n"]]


EPOCHS_e2e = 1000

print("TRAINING END-to-END")
end_to_end_net = nets.train_end_to_end(X_train[:,:-1], Y_train, variables, params, EPOCHS_e2e, DEVICE)

print("TRAIN P-MODEL")
nets.train_pnet(Y_train_pt, params, EPOCHS_e2e, DEVICE)