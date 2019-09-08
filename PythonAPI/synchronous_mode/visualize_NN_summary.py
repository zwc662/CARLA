import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from torchviz import make_dot
from torchsummary import summary

from NN_controller import mlp 

torch.set_default_dtype(torch.float32)

num_wps = 10
model = mlp(nx=(5+3*num_wps), ny = 2)

sample_state = np.array([0.00414619210919023, -0.15624896240234376, -0.19012335205078126, -0.5009140014648438, 0.75, -0.15629389953613282, -0.20678996276855469, 1.4991319444444444, -0.15633935546875, -0.22345657348632814, \
	1.4991319444444444, -0.15638481140136717, -0.24012318420410156, 1.4991319444444444, -0.15643026733398438, -0.25678976440429685, 1.4991319444444444, -0.15645565795898436, -0.2734342346191406, 1.5011489868164063, \
	-0.15625730895996093, -0.29004119873046874, 1.5064541286892361, -0.1557822265625, -0.3066424865722656, 1.511759270562066, -0.15503054809570313, -0.323233642578125, \
	1.5170644124348958, -0.1540864562988281, -0.33986141967773437, 1.5181676228841146, -0.152741455078125, -0.356009033203125, -0.44086286756727433]).astype(np.float32).reshape(1, -1)

print("sample_state", sample_state.shape)
state = torch.from_numpy(sample_state)

MLP_dict_path = "/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_dict_nx=8_wps10_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MLP_dict_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

state = state.to(device)
output = model(state)

print("output")
print(output)
# print("model")
# print(model)
'''
model
mlp(
  (fc1): Linear(in_features=35, out_features=70, bias=True)
  (fc2): Linear(in_features=70, out_features=140, bias=True)
  (fc3): Linear(in_features=140, out_features=105, bias=True)
  (fc4): Linear(in_features=105, out_features=2, bias=True)
  (sig): Sigmoid()
  (tanh): Tanh()
)
'''

shape = (1, 35)
print("summary")
summary(model, shape)

'''
summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 2, 70]           2,520
            Linear-2               [-1, 2, 140]           9,940
            Linear-3               [-1, 2, 105]          14,805
            Linear-4                 [-1, 2, 2]             212
           Sigmoid-5                    [-1, 2]               0
              Tanh-6                    [-1, 2]               0
================================================================
Total params: 27,477
Trainable params: 27,477
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.10
Estimated Total Size (MB): 0.11
----------------------------------------------------------------
'''
