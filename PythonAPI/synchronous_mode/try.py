# import os, sys 

# directory = '/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/data/dest_start'
# # os.walk(directory)
# sub_dirs = [x[0] for x in os.walk(directory)]
# print("sub_dirs")
# print(sub_dirs)


import autograd.numpy as np
from autograd import grad

def tanh(x):
	y = np.exp(-2.0*x)
	return (1.0-y)/(1.0+y)

def comb(x):
	print("in comb")
	print(np.sin(x[0]), np.cos(x[1]))
	return np.concatenate((np.sin(x[0]), np.cos(x[1])), axis=0)


grad_tanh = grad(tanh)
# print(type(tanh))
# print(grad_tanh(1.0))
grad_comb = grad(comb)
print(np.ones(2))
print(grad_comb(np.ones(2))) 