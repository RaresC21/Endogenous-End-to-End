import matplotlib.pyplot as plt
import numpy as np
from helper import *
from two_stage import *
from problem import *
from tqdm import tqdm
import nn_model


if __name__ == '__main__':
    TRIALS = 1
    capacity = 3
    n_items = 2
    n_start_data = 20
    n_iter = 100
    unit_cost = 0.7
    prices = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    all_opt = []
    all_two_stage = []
    all_joint = []
    all_kernel = []
    all_random = []
    all_sample_gd = []
    all_nn = []
    all_squarecb = []


    for i in tqdm(range(TRIALS)):
        # print("Trial: ", i+1)
        # np.random.seed(i+1)
        generator = ProblemGenerator(n_items, unit_cost)
        all_train_actions_two_stage = generator.get_actions(n_start_data)
        all_demand_two_stage = generator.get_demand(all_train_actions_two_stage)
        all_revenue_two_stage = generator.objective(all_train_actions_two_stage, all_demand_two_stage, unit_cost)

        all_train_actions_joint = all_train_actions_two_stage.copy()
        all_demand_joint = all_demand_two_stage.copy()
        all_revenue_joint = all_revenue_two_stage.copy()

        all_train_actions_kernel = all_train_actions_two_stage.copy()
        all_demand_kernel = all_demand_two_stage.copy()
        all_revenue_kernel = all_revenue_two_stage.copy()

        all_train_actions_gd = all_train_actions_two_stage.copy()
        all_demand_gd = all_demand_two_stage.copy()
        all_revenue_gd = all_revenue_two_stage.copy()

        all_train_actions_squarecb = all_train_actions_two_stage.copy()
        all_demand_squarecb = all_demand_two_stage.copy()
        all_revenue_squarecb = all_revenue_two_stage.copy()

        all_train_actions_nn = all_train_actions_two_stage.copy()
        all_demand_nn = all_demand_two_stage.copy()
        all_revenue_nn = all_revenue_two_stage.copy()

        # nominal_demands = generator.get_nominal_demand(all_train_actions)
        # # print("NOMINAL DEMANDS", nominal_demands)
        # # print("ALL TRAIN ACTIONS", all_train_actions)
        # # print(generator.get_nominal_demand([0.5, 0.8, 0.5, 0.7, 0.5]))
        # # print(generator.get_nominal_demand([0.5, 0.8, 0.5, 0.7, 0.5]))
        # opt_decisions = np.argmin(generator.objective(all_train_actions, nominal_demands, unit_cost))
        # print("OPTIMAL DECISIONS", opt_decisions)

        opt_results = []
        two_stage_results = []
        join_results = []
        kernel_results = []
        sample_gd_results = []
        nn_results = []
        random_results = []
        squarecb_results = []
        warm_start = None

        # net = nn_model.Net(num_features=n_items)
        # trained_net = nn_model.train(net, torch.tensor(all_train_actions_nn).float(), torch.tensor(all_demand_nn).float(), unit_cost, torch.tensor(all_revenue_nn).float(), num_epochs=20)

        for n in tqdm(range(n_iter)):

            #retrain the model every 10 iterations
            # if n % 20 == 0 and n != 0:
            #     trained_net = nn_model.train(net, torch.tensor(all_train_actions_nn).float(), torch.tensor(all_demand_nn).float(), unit_cost, torch.tensor(all_revenue_nn).float(), num_epochs=20)

            # random decisions
            obj_random = generator.get_objective(np.expand_dims(random_decision(n_items), axis=0))[0]
            random_results.append(obj_random)

            # optimal solution
            optimal_decision = assortment_opt(generator.alpha, generator.base_demand, unit_cost, capacity)
            obj_opt = generator.get_objective(np.expand_dims(optimal_decision, axis=0))[0]

            # train two-stage model
            # two_stage = TwoStage(all_train_actions_two_stage, all_demand_two_stage, unit_cost)
            # w_two_stage, _ = two_stage.decision()
            # obj = generator.get_objective(np.expand_dims(w_two_stage, axis=0))[0]

            # train joint optimization model
            joint = JointOptimization(all_train_actions_joint, all_demand_joint, unit_cost)
            w_joint, _, warm_start0, warm_start1 = joint.joint_optimization(warm_start=warm_start)
            obj_join = generator.get_objective(np.expand_dims(w_joint, axis=0))[0]
            warm_start = (warm_start0, warm_start1)

            # sample GD method
            sample_gd  = SampleGD(all_train_actions_gd, all_demand_gd, unit_cost, n_samples=2)
            w_sample_gd = sample_gd.decision(capacity)
            obj_sample_gd = generator.get_objective(np.expand_dims(w_sample_gd, axis=0))[0]

            # # SquareCB
            square_cb = SquareCB(all_train_actions_squarecb, all_demand_squarecb, prices, n)
            _, w_square_cb = square_cb.choose()
            square_cb_obj = generator.get_objective(np.expand_dims(w_square_cb, axis=0))[0]

            # NN model
            # nn = SampleGD_NN(all_train_actions_nn, all_demand_nn, unit_cost, trained_net, n_samples=50)
            # w_nn = nn.decision(capacity)
            # obj_nn = generator.get_objective(np.expand_dims(w_nn, axis=0))[0]

            opt_results.append(obj_opt)
            # two_stage_results.append(obj)
            join_results.append(obj_join)
            sample_gd_results.append(obj_sample_gd)
            squarecb_results.append(square_cb_obj)
            # nn_results.append(obj_nn)

            # train kernel-based model
            # kernel = KernelPredictor(all_train_actions_kernel, all_demand_kernel, unit_cost, capacity)
            # w_kernel = kernel.decision()
            # obj_kernel = generator.get_objective(np.expand_dims(w_kernel, axis=0))[0]
            # kernel_results.append(obj_kernel)

            # add one data point to the training set

            # all_train_actions_two_stage = np.vstack((all_train_actions_two_stage, w_two_stage))
            # all_demand_two_stage = np.vstack((all_demand_two_stage, generator.get_demand(all_train_actions_two_stage[-1:])))
            # all_revenue_two_stage = np.append(all_revenue_two_stage, generator.get_objective(np.expand_dims(all_train_actions_two_stage[-1], axis=0)))

            all_train_actions_squarecb = np.vstack((all_train_actions_squarecb, w_square_cb))
            all_demand_squarecb = np.vstack((all_demand_squarecb, generator.get_demand(all_train_actions_squarecb[-1:])))
            all_revenue_squarecb = np.append(all_revenue_squarecb, generator.get_objective(np.expand_dims(all_train_actions_squarecb[-1], axis=0)))


            all_train_actions_joint = np.vstack((all_train_actions_joint, w_joint))
            all_demand_joint = np.vstack((all_demand_joint, generator.get_demand(all_train_actions_joint[-1:])))
            all_revenue_joint = np.append(all_revenue_joint, generator.get_objective(np.expand_dims(all_train_actions_joint[-1], axis=0)))

            # all_train_actions_kernel = np.vstack((all_train_actions_kernel, w_kernel))
            # all_demand_kernel = np.vstack((all_demand_kernel, generator.get_demand(all_train_actions_kernel[-1:])))
            # all_revenue_kernel = np.append(all_revenue_kernel, generator.get_objective(np.expand_dims(all_train_actions_kernel[-1], axis=0)))

            all_train_actions_gd = np.vstack((all_train_actions_gd, w_sample_gd))
            all_demand_gd = np.vstack((all_demand_gd, generator.get_demand(all_train_actions_gd[-1:])))
            all_revenue_gd = np.append(all_revenue_gd, generator.get_objective(np.expand_dims(all_train_actions_gd[-1], axis=0)))

            # all_train_actions_nn = np.vstack((all_train_actions_nn, w_nn))
            # all_demand_nn = np.vstack((all_demand_nn, generator.get_demand(all_train_actions_nn[-1:])))
            # all_revenue_nn = np.append(all_revenue_nn, generator.get_objective(np.expand_dims(all_train_actions_nn[-1], axis=0)))



        all_opt.append(opt_results)
        all_two_stage.append(two_stage_results)
        all_random.append(random_results)
        all_joint.append(join_results)
        # all_kernel.append(kernel_results)
        all_sample_gd.append(sample_gd_results)
        all_squarecb.append(squarecb_results)
        all_nn.append(nn_results)


def running_mean(x):
    N = 10
    return np.convolve(x, np.ones(N)/N, mode='valid')
        
# evolution of objective over time - means over trials
fig, ax = plt.subplots()
ax.plot(running_mean(np.mean(all_opt, axis=0)), label='Mean Predictor')
# ax.plot(np.mean(all_two_stage, axis=0), label='Two-stage')
ax.plot(running_mean(np.mean(all_random, axis=0)), label='Random')
ax.plot(running_mean(np.mean(all_joint, axis=0)), label='Joint Optimization')
# ax.plot(np.mean(all_kernel, axis=0), label='Kernel')
ax.plot(running_mean(np.mean(all_sample_gd, axis=0)), label='Sample GD')
ax.plot(running_mean(np.mean(all_squarecb, axis=0)), label='SquareCB')
# ax.plot(np.mean(all_nn, axis=0), label='NN')
ax.set_title('Evolution of objective over time')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Objective')
ax.legend()
plt.show()
plt.savefig("new_pics/results.png")





