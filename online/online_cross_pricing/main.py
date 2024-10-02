import matplotlib.pyplot as plt 

from problem import * 
from helper import * 
from methods import * 

if __name__ == '__main__':

    TRIALS = 5

    all_opt = []
    all_lin = [] 
    all_rob = []
    all_e2e = []

    for T in range(TRIALS):
        print("TRIAL", T+1, "/", TRIALS)
        np.random.seed(T)

        n_items = 5
        prices = np.arange(0,1.01,0.25)
        n_prices = prices.shape[0]

        n_data = 50

        generator = ProblemGenerator(n_items, prices)
        price_data = generator.get_actions(n_data)
        demand = generator.get_demand(price_data)
        revenue = generator.get_objective(price_data)
        
        linear_predictor = LinearPredictor(prices, price_data, demand)
        lin_pred = linear_predictor.predict(price_data)
        lin_dec = linear_predictor.decision()

        optimal_decision = pricing_opt(generator.alpha, generator.beta, generator.gamma, prices)

        opt_obj = generator.get_objective(np.expand_dims(optimal_decision, axis=0))[0]
        lin_obj = generator.get_objective(np.expand_dims(lin_dec, axis=0))[0]
        print("optimal          decision:", opt_obj)
        print("linear predictor decision:", lin_obj)

        print("Robust two-stage ---------------------------------------------")

        robust_2stage = Robust2Stage(prices, price_data, demand)
        robust_objs = []
        epsilons = np.arange(0,2,0.1)
        for eps in epsilons:
            r = robust_2stage.decision(eps)
            ob = generator.get_objective(np.expand_dims(r, axis=0))[0]
            robust_objs.append(ob)
            print("robust           decision:", r, ob)

        # print("Robust end-to-end ---------------------------------------------")

        # robust_e2e = RobustE2E(prices, price_data, demand)
        # robust_e2e_objs = []
        # epsilons = np.arange(0,2,0.1)
        # for eps in epsilons:
        #     r = robust_e2e.decision(eps)
        #     ob = generator.get_objective(np.expand_dims(r, axis=0))[0]
        #     robust_e2e_objs.append(ob)
        #     print("robust           decision:", r, ob)

        all_opt.append(opt_obj)
        all_lin.append(lin_obj)
        all_rob.append(robust_objs)
        # all_e2e.append(robust_e2e_objs)

    opt_mean = np.mean(all_opt)
    lin_mean = np.mean(all_lin)

    print("Optimal", np.mean(all_opt, axis=0), '\pm', np.std(all_opt, axis=0))
    print("Linear two-stage", np.mean(all_lin, axis=0), '\pm', np.std(all_lin, axis=0))
    print("Robust", np.mean(all_rob, axis=0), '\pm', np.std(all_rob, axis=0))

    plt.plot([0,1], [opt_mean, opt_mean], label='Optimal', color = 'green')
    plt.plot([0,1], [lin_mean, lin_mean], label='Linear 2-stage', linestyle = ':', color = 'red')
    plt.plot(epsilons/np.max(epsilons), np.mean(all_rob, axis=0), label='Robust', linestyle = '-.', color = 'blue')
    # plt.plot(epsilons/np.max(epsilons), np.mean(all_rob, axis=0), label='Robust 2-stage', linestyle = '-.', color = 'yellow')
    # plt.plot(epsilons_2stage/np.max(epsilons_2stage), np.mean(all_e2e, axis=0), label='Robust end-to-end', color='blue')
    plt.xlabel('Robustness (eps)')
    plt.ylabel('Revenue')
    plt.title("Average Revenue vs. Robustness")
    plt.legend() 
    plt.show()
