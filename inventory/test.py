from Problem import ProblemGenerator
from models import * 

def evaluate_decisions(p): 
    for w in problem.actions: 
        preds = p(torch.ones_like(actions) * w, z)
        print("decision", w.item(), "cost:", problem.get_objective(preds, z).mean().item())


DEVICE = "cpu"
if torch.cuda.is_available(): 
    DEVICE = "cuda"

n_locations = 3
holding_cost = 1
backorder_cost = 10

n_data = 100

problem = ProblemGenerator(n_locations, holding_cost, backorder_cost, DEVICE)
actions = problem.get_actions(n_data) 
z       = problem.get_demand(n_data)

EPOCHS = 10000

two_stage_model = PNet(problem, DEVICE=DEVICE).to(DEVICE)
two_stage_model = train_2stage(two_stage_model, actions, z, problem, EPOCHS=EPOCHS)
evaluate_decisions(two_stage_model)
    
p = PNet(problem, DEVICE=DEVICE).to(DEVICE)
p = train_p(p, actions, z, problem, EPOCHS=EPOCHS)

f = FNet(problem, DEVICE=DEVICE).to(DEVICE) 
f = train_e2e(f, p, actions, z, problem, EPOCHS=EPOCHS)

decision = decision_e2e(p, f, problem)
pred = p(torch.ones_like(actions) * decision, z)

print("end-to-end average cost:", problem.get_objective(pred, z).mean().item())
evaluate_decisions(p)
