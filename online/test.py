import numpy as np



data = np.load('data/' + str('Adapted UCB') + '.npy')

print(np.unique(data[:, 0]))