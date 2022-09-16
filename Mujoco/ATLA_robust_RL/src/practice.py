a = [(1, 2)]
a = [(a[0][1], -a[0][1])]
print(a)
print(a[0][1])

# import matplotlib.pyplot as plt
# import numpy as np
# a = np.array([1,2])
# b = np.array([2,3])
# plt.figure()
# plt.plot(a, b, label='lalala')
# plt.legend(loc=3, bbox_to_anchor=(1.05, 0),borderaxespad=0.)
# plt.savefig('./a.png',bbox_inches='tight')

import torch
a = torch.tensor([1.,-2.])
print(a.abs())
print(a.abs().max())
a = a / a.abs().max()
print(a)
if a.abs().max() > 0.0:
    print("hello")

import random
for i in range(20):
    print(random.random())

a = "aaa\r"
print(a + "bbb")

a = torch.tensor([1.0, 2.0, 3.0])
print(torch.prod(a))
print(torch.log(torch.prod(a)).item())

import numpy as np
all_mean_reward = [0.1, 0.3, 0.2, 0.5, 0.4]
all_mean_std = [1, 2, 3, 4, 5]

# a = np.array([all_mean_reward, all_mean_std])
a = []
for i in range(len(all_mean_reward)):
    a.append((all_mean_reward[i], all_mean_std[i]))
a = np.array(a)
print(a)
b = a[:, 0]
index = np.lexsort((b,))
print(a[index])
print(a[index][2][0])
print(a[index][2][1])