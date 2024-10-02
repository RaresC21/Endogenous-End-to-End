import matplotlib.pyplot as plt 

from problem import * 
from helper import * 
from methods import * 

if __name__ == '__main__':

    TRIALS = 1
    all_opt = [] 
    all_lin = [] 
    all_e2e_1 = [] 
    all_e2e_2 = [] 
    all_sample_costs = []
    all_approx_gd_costs = []
    all_approx_obj = []
    all_kernel_costs = []

    capacity = 3

    n_items = 5

    increment = 2
    start_ = 10
    total_data = 500
    n_datas = range(start_, 100, increment)
    # n_data = 1000

    unit_cost = 0.7
    
    REG = 0.0

    all_in_sample_obj = []

    for i in range(TRIALS):
        # np.random.seed(i+1)
        np.random.seed(i+1)

        opt = []
        lin = [] 
        e2e_2 = []
        e2e_1 = []
        in_sample_obj = []
        sample_costs = []
        approx_gd_costs = []
        approx_obj_costs = []
        kernel_costs = []

        last_model = None

        generator = ProblemGenerator(n_items, unit_cost)
        all_train_actions = generator.get_actions(total_data)
        all_demand = generator.get_demand(all_train_actions)
        all_revenue = generator.objective(all_train_actions, all_demand, unit_cost)

        fast_predictor = ObjectivePredictor(n_items, generator.unit_cost, capacity, warm_start = last_model, REG=0)

        for n_data in n_datas:
            # if n_data > 100: break
            train_actions = all_train_actions[:n_data]
            demand = all_demand[:n_data]
            revenue = all_revenue[:n_data]

            train_actions_add = all_train_actions[n_data-increment:n_data]
            demand_add = all_demand[n_data-increment:n_data]
            revenue_add = all_revenue[n_data-increment:n_data]

            best_decision = np.argmin(revenue)
            in_sample_action = train_actions[best_decision]
            # in_sample_action = in_sample_action / np.sum(in_sample_action) * capacity
            v = generator.get_objective(np.expand_dims(in_sample_action, axis=0))[0]

            print("Trial", i, "n data:", len(train_actions))

            optimal_decision = assortment_opt(generator.alpha, generator.base_demand, unit_cost, capacity)
            opt_obj = generator.get_objective(np.expand_dims(optimal_decision, axis=0))[0]

            # print(optimal_decision)
            # print(in_sample_action)

            linear_predictor = LinearPredictor(train_actions, demand, generator.unit_cost, capacity)
            lin_pred = linear_predictor.predict(train_actions)
            lin_dec = linear_predictor.decision()
            lin_obj = generator.get_objective(np.expand_dims(lin_dec, axis=0))[0]
            if last_model is None: 
                last_model = linear_predictor

            print("optimal          decision:", opt_obj)
            print("Baseline         decision:", v)
            print("linear predictor decision:", lin_obj)


            sample_gd = SampleGD(train_actions, demand, unit_cost, n_samples = 20)
            sample_decision = sample_gd.decision(capacity) 
            print('sample decision:', sample_decision)
            sample_cost = generator.get_objective(np.expand_dims(sample_decision, axis=0))[0]

            print("sample approx   decision:", sample_cost)

            approx_gd = ApproximateGD(train_actions, demand, unit_cost)
            approx_decision = approx_gd.decision()
            print('approx decision:', approx_decision)
            approx_cost = generator.get_objective(np.expand_dims(approx_decision, axis=0))[0]

            print("approximate gd  decision:", approx_cost)

            # approx_obj = ApproximateObjective(train_actions, demand, unit_cost)
            # approx_obj_decision = approx_obj.decision()
            # approx_obj_cost = generator.get_objective(np.expand_dims(approx_obj_decision, axis=0))[0]

            # print("approx obj      decision:", approx_obj_cost)

            kernel_predictor = KernelPredictor(train_actions, demand, generator.unit_cost, capacity)
            kernel_dec = kernel_predictor.decision()
            kernel_obj = generator.get_objective(np.expand_dims(kernel_dec, axis=0))[0]
            print("kernel predictor decision:", kernel_obj)

            fast_predictor.add_data(train_actions_add, demand_add)
            obj_dec = fast_predictor.decision(capacity)
            obj_2 = generator.get_objective(np.expand_dims(obj_dec, axis=0))[0]            
            # obj_2 = 0
            print("objec  predictor decision 3:", obj_2)

            opt.append(opt_obj)
            e2e_2.append(obj_2)
            lin.append(lin_obj)
            in_sample_obj.append(v)
            sample_costs.append(sample_cost)
            kernel_costs.append(kernel_obj)
            approx_gd_costs.append(approx_cost)
            # approx_obj_costs.append(approx_obj_cost)

            print()

            # last_model = objective_predictor
            last_model = fast_predictor

        all_opt.append(opt) 
        all_lin.append(lin) 
        all_e2e_1.append(e2e_1)
        all_e2e_2.append(e2e_2)
        all_in_sample_obj.append(in_sample_obj)
        all_sample_costs.append(sample_costs)
        all_kernel_costs.append(kernel_costs)
        all_approx_gd_costs.append(approx_gd_costs)
        # all_approx_obj.append(approx_obj_costs)

        # all_opt = np.array(all_opt)
        # all_lin = np.array(all_lin)
        # all_e2e_1 = np.array(all_e2e_1)
        # all_e2e_2 = np.array(all_e2e_2)
        # all_in_sample_obj = np.array(all_in_sample_obj)

    # plt.plot(n_datas, np.mean(all_in_sample_obj, axis=0), label='Baseline')
    plt.plot(n_datas, np.mean(all_opt, axis=0), label='Mean Prediction', color='black')
    plt.plot(n_datas, np.mean(all_lin, axis=0), label='Two-Stage', color = 'red', linestyle = ':')
    # plt.plot(n_datas, np.mean(all_e2e_1, axis=0), label='e2e - reg=1')
    plt.plot(n_datas, np.mean(all_e2e_2, axis=0), label='End-to-End', color = 'blue', linestyle = '-.')
    plt.plot(n_datas, np.mean(all_sample_costs, axis=0), label='sample end-to-end', color = 'green', linestyle = '--')
    plt.plot(n_datas, np.mean(all_approx_gd_costs, axis=0), label='approximate GD', color = 'orange', linestyle = '-')
    plt.plot(n_datas, np.mean(all_kernel_costs, axis=0), label='kernel', color = 'aqua', linestyle = '-')
    # plt.plot(n_datas, np.mean(all_approx_obj, axis=0), label='approx obj', color = 'purple', linestyle = '-')
    plt.xlabel("# Data")
    plt.ylabel("Cost")
    plt.title("Cost vs. Data")
    plt.legend() 
    plt.show()