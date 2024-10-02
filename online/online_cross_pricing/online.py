import numpy as np
from helper import *
from tqdm import tqdm
from methods import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRIALS = 20
    n_iter = 200
    n_items = 3
    n_start_data = 50
    prices = np.linspace(0, 1, 3)
    n_prices = prices.shape[0]

    all_opt = []
    all_lin = []
    all_squarecb = []
    all_robust = []
    all_random = []


    for i in tqdm(range(TRIALS)):
        # print("Trial: ", i+1)
        # np.random.seed(i+1)
        generator = ProblemGenerator(n_items, prices)

        price_data = generator.get_actions(n_start_data)
        demand = generator.get_demand(price_data)
        revenue = generator.get_objective(price_data)

        all_train_actions_lin = price_data.copy()
        all_demand_lin = demand.copy()
        all_revenue_lin = revenue.copy()

        all_train_actions_squarecb = price_data.copy()
        all_demand_squarecb = demand.copy()
        all_revenue_squarecb = revenue.copy()

        all_train_actions_robust = price_data.copy()
        all_demand_robust = demand.copy()
        all_revenue_robust = revenue.copy()

        opt_results = []
        lin_results = []
        squarecb_results = []
        robust_results = []
        random_results = []

        for n in tqdm(range(n_iter)):
            
            linear_predictor = LinearPredictor(prices, all_train_actions_lin, all_demand_lin)
            lin_pred = linear_predictor.predict(price_data)
            lin_dec = linear_predictor.decision()

            optimal_decision = pricing_opt(generator.alpha, generator.beta, generator.gamma, prices)

            opt_obj = generator.get_objective(np.expand_dims(optimal_decision, axis=0))[0]
            lin_obj = generator.get_objective(np.expand_dims(lin_dec, axis=0))[0]

            # squareCB
            square_cb = SquareCB(all_train_actions_squarecb, all_demand_squarecb, prices, n)
            _, w_square_cb = square_cb.choose()
            square_cb_obj = generator.get_objective(np.expand_dims(w_square_cb, axis=0))[0]


            # robust
            robust_2stage = Robust2Stage(prices, all_train_actions_robust, all_demand_robust)
            w_robust = robust_2stage.decision(0.5)
            robust_obj = generator.get_objective(np.expand_dims(w_robust, axis=0))[0]

            # random
            obj_random = generator.get_objective(np.expand_dims(random_decision(n_items), axis=0))[0]
            random_results.append(obj_random)
            
            # add one data point to the training set

            opt_results.append(opt_obj)
            lin_results.append(lin_obj)
            squarecb_results.append(square_cb_obj)
            robust_results.append(robust_obj)

            all_train_actions_lin = np.vstack((all_train_actions_lin, lin_dec))
            all_demand_lin = np.vstack((all_demand_lin, generator.get_demand(all_train_actions_lin[-1:])))
            all_revenue_lin = np.append(all_revenue_lin, generator.get_objective(np.expand_dims(all_train_actions_lin[-1], axis=0)))

            all_train_actions_squarecb = np.vstack((all_train_actions_squarecb, w_square_cb))
            all_demand_squarecb = np.vstack((all_demand_squarecb, generator.get_demand(all_train_actions_squarecb[-1:])))
            all_revenue_squarecb = np.append(all_revenue_squarecb, generator.get_objective(np.expand_dims(all_train_actions_squarecb[-1], axis=0)))

            all_train_actions_robust = np.vstack((all_train_actions_robust, w_robust))
            all_demand_robust = np.vstack((all_demand_robust, generator.get_demand(all_train_actions_robust[-1:])))
            all_revenue_robust = np.append(all_revenue_robust, generator.get_objective(np.expand_dims(all_train_actions_robust[-1], axis=0)))


        all_opt.append(opt_results)
        all_lin.append(lin_results)
        all_squarecb.append(squarecb_results)
        all_robust.append(robust_results)
        all_random.append(random_results)


# evolution of objective over time - means over trials
fig, ax = plt.subplots()
ax.plot(np.mean(all_opt, axis=0), label='Mean Predictor')
ax.plot(np.mean(all_lin, axis=0), label='Linear Predictor')
ax.plot(np.mean(all_squarecb, axis=0), label='SquareCB')
ax.plot(np.mean(all_robust, axis=0), label='Robust')
# ax.plot(np.mean(all_random, axis=0), label='Random')
ax.set_title('Evolution of objective over time')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Objective')
ax.legend()
plt.show()

        



