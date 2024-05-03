from Problem import ProblemGenerator
from models import * 

DEVICE = "cpu"
if torch.cuda.is_available(): 
    DEVICE = "cuda"

n_locations = 10 
holding_cost = 1
backorder_cost = 10

n_data = 100

problem = ProblemGenerator(n_locations, holding_cost, backorder_cost, DEVICE)
actions = problem.get_actions(n_data) 
z       = problem.get_demand(n_data)

EPOCHS = 10000

p = PNet(problem, DEVICE=DEVICE).to(DEVICE)
p = train_p(p, actions, z, problem, EPOCHS=EPOCHS)

f = FNet(problem, DEVICE=DEVICE).to(DEVICE) 
f = train_e2e(f, p, actions, z, problem, EPOCHS=EPOCHS)

