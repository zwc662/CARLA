'''
Refer to scenario_runner/e2c_controller for parsing data
	For e2c, need x & x_next for encoder and decoder, 
	while for image-embedded controller only current frame is needed

Refer to coiltraine/coil_icra for network architecture and resnet34imnet.yaml for configurations

Refer to 
Compared to NN_controller.py, differ in:
	1. get_data_loaders -> get_data_loaders_img (add CarlaData_cur and CarlaDataPro_cur)
	2. model type: mlp -> MyCoil

'''

# %matplotlilb inline
import os, sys
import glob
import numpy as np
import pandas as pd
import csv
import pickle
import matplotlib.pyplot as plt
import random
from random import shuffle
from tqdm import trange, tqdm

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage

# for MyCoil
import importlib
import yaml

from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray

import cv2

from attribute_dict import AttributeDict
from models.building_blocks import FC
from models.building_blocks import Join

torch.set_default_dtype(torch.float32)

# create a class to combine image and measurement data
class CarlaData_cur(Dataset):
	'''
	retrieve data based on frame number: camera images (single central.png) & ctv info.npy
	the img size can be customized, check sensor/hud setting of collected data
	'''
	def __init__(self, ds_dir, img_width = 200, img_height=88, num_wps=0):
		# find all available data
		self.dir = ds_dir # parent dir for saving processed_.pkl and for searching image and measurement data by frame
		self.img_width = img_width
		self.img_height = img_height
		self.dim_img = self.img_width*self.img_height # *3 for RGB channels
		self.dim_img_lat = 100 # TODO: latent vector dimension
		self.dim_m = num_wps*3 + 5 # for concatenate loc_diff and v (3) # 6 for loc_diff t(3), v(3) # 9 for transform (6), velocity(9)
		self.dim_u = 2
		self.dim_z = self.dim_img_lat + self.dim_m  # done in E2C_cat __init__
		self.num_wps = num_wps
		self._process()

	def __len__(self):
		return len(self._processed)

	def __getitem__(self, index):
		return self._processed[index]

	@staticmethod
	def _process_img(img, img_width, img_height):
		'''
		convert color to gray scale
		(check img_size)
		convert image to tensor
		'''
		# use PIL
		# return ToTensor()((img.convert('L').
		# 					resize((img_width, img_height))))
		# use cv2
		pass


	@staticmethod
	def _process_ctv(ctv, ny=2, num_wps=1):
		# TODO: keep consistent with get_data_loaders
		max_pos_val = 500
		max_yaw_val = 180
		max_speed_val = 40
		# convert a numpy array to tensor: control(2), state(5+3*num_wps)
		# split the npy data into control (output) and measurement (input)
		if any(np.isnan(ctv)):
			print("ctv contain nan value, continue")
			return None, None

		action = ctv[:ny]
		# normalize the state
		state = [ctv[2]/max_speed_val,ctv[3]/max_pos_val, ctv[4]/max_pos_val, ctv[7]/max_yaw_val, ctv[9]/max_speed_val]
		for j in range(num_wps): # concatenate x, y, yaw of future_wps
			state = state + [ctv[10+j*6]/max_pos_val, ctv[11+j*6]/max_pos_val, ctv[14+j*6]/max_yaw_val]

		return torch.from_numpy(action).float(), torch.from_numpy(np.array(state)).float()

	def _process(self):
		frame_interval = 1
		preprocessed_file = os.path.join(self.dir, 'processed_{}.pkl'.format(str(frame_interval)))
		if not os.path.exists(preprocessed_file):
			print("writing {}".format(preprocessed_file))
			# create data and dump
			# recursive traverse sub-directories
			print("searching for imgs from", self.dir)
			# TODO check how to use glob.glob correctly
			# imgs = glob.glob('{}**.png'.format(self.dir), recursive=True)
			imgs = []
			for root, dirs, files in os.walk(self.dir):
				for name in files:
					if name.endswith((".png")):
						# append abs path
						# print("sample image path")
						# print("root", root)
						# print("dirs", dirs)
						# print("files", files)
						# print(os.path.join(root,name))
						imgs.append(os.path.join(root,name))
						# print("imgs", imgs)
						# raise ValueError("stop here")

			# Alternatively, use for loop to traverse each subdirectory to aviod redundant frame number
			# # single directory
			# imgs = sorted(glob(os.path.join(self.dir,"*.png"))) # sorted by frame numbers
			# shuffle(imgs) # if need randomness
			print("sample image files", imgs[:3])
			print("{} frames".format(len(imgs)))

			processed = []
			# for frame_number in frame_numbers: # TODO: add frame_interval if needed
			for img_dir in imgs:
				# use PIL
				# load image
				# img = Image.open(img_dir) # TODO: check image directory
				# # show the image for debugging
				# img.show()
				# use cv2
				img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
				# cv2.imshow("before transform", img)
				# cv2.waitKey(0)
				img = img.astype(np.float)
				img = torch.from_numpy(img).type(torch.FloatTensor)
				img = img / 255.
				# print("cv2 image", img.shape) # torch.Size([88, 200, 3])
				frame_number = img_dir.split('/')[-1].split('.')[0]
				# load ctv array, ctv: control transform, velocity <=> u: action, m: measurement
				ctv_path = img_dir.split('.png')[0]+'_ctv.npy'
				# print("frame", frame_number, ctv_path)
				ctv = np.load(ctv_path)
				# print(ctv)
				if any(np.isnan(ctv)):
					print("ctv contain nan value, continue")
					continue
				# process data
				u, m = self._process_ctv(ctv, ny=2, num_wps=num_wps)
				# print("parse u, m")
				# print(u,m)
				# processed.append([self._process_img(img, self.img_width, self.img_height), m, u])
				processed.append([img, m, u])
				# print("processing for frame ", int(frame_number))
				# print(processed[0])                
				# raise ValueError("stop here")

			with open(preprocessed_file, 'wb') as f:
				pickle.dump(processed, f)
			self._processed = processed
		else:
			# directly load the pickle file
			with open(preprocessed_file, 'rb') as f:
				print("directly load pickle file {}".format(preprocessed_file))
				self._processed = pickle.load(f)
		shuffle(self._processed)

	def query_data(self):
		if self._processed is None:
			raise ValueError("Dataset not loaded - call CarlaData._process() first.")
		print("_processed length", len(self._processed))
		return list(zip(*self._processed))[0], list(zip(*self._processed))[1], list(zip(*self._processed))[2]


class CarlaDataPro_cur(Dataset): 
	# compared to CarlaDataPro, no img_next and m_next needed
	# compared to ZUData, input is separated as img and m
	def __init__(self, img, m, u):
		self.img = img # list of tensors
		self.m = m
		self.u = u
		print("tensor type in CarlaDataPro_cur")
		print('img {}, m {}, u {}'.format(self.img[0].size(), self.m[0].size(), self.u[0].size()))

	def __len__(self):
		return len(self.img)

	def __getitem__(self, index):
		# return the item at certain index
		return self.img[index], self.m[index], self.u[index]


# create a class for the Dataset
class ZUData(Dataset):
	def __init__(self, z, u=None):
		self.z = z
		self.u = u

	def __len__(self):
		return len(self.z)

	def __getitem__(self, index):
		# return the item at certain index
		return self.z[index], self.u[index]

# create a class for multi-layer perceptron model
class mlp(nn.Module):
	def __init__(self, nx=8, ny=2):
		super(mlp, self).__init__()
		self.fc1 = nn.Linear(nx, 2 * nx)
		self.fc2 = nn.Linear(2 * nx, 4 * nx)
		self.fc3 = nn.Linear(4 * nx, 3 * nx)
		self.fc4 = nn.Linear(3 * nx, ny)
		
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		if x.size()[-1] == 2:
			x_0 = self.sig(x[:, 0]).unsqueeze(1)
			x_1 = self.tanh(x[:, 1]).unsqueeze(1)
			y = torch.cat((x_0, x_1), dim = 1)
		else:
			y = self.sig(x)
		return y


class MyCoil(nn.Module):
	# refer to coil_icra.py

	def __init__(self, dim_img, dim_m, dim_u, params): # dim_img_lat
		super(MyCoil, self).__init__()
		# self.dim_img = dim_img # TODO: check image size or resnet_module
		self.dim_m = dim_m
		# self.dim_img_lat = dim_img_lat # should specify in MyCoil_config ['res']['num_classes']
		# self.dim_z = self.dim_img_lat + self.dim_m
		self.dim_u = dim_u
		
		self.params = params

		# use resnet for perception
		resnet_module = importlib.import_module('models.building_blocks.resnet')
		resnet_module = getattr(resnet_module, params['perception']['res']['name'])
		self.perception = resnet_module(pretrained=params['perception']['res']['pretrained'], \
										num_classes=params['perception']['res']['num_classes'])
		self.dim_img_lat = params['perception']['res']['num_classes']
		
		# For FC, 'neurons' specifies the number of neurons per layer as a list
		self.measurement = FC(params={'neurons': [self.dim_m] + params['measurement']['fc']['neurons'],
									   'dropouts': params['measurement']['fc']['dropouts'],
									   'end_layer': False})
		self.join = Join(
			params={'after_process':
						FC(params={'neurons':
										[params['measurement']['fc']['neurons'][-1] + self.dim_img_lat] + 
										params['join']['fc']['neurons'],
									'dropouts': params['join']['fc']['dropouts'],
									'end_layer': False}),
					'mode': 'cat'})

		# TODO: stop at the Join layer and create the final controller layer
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()
		# TODO: modify the number of neurons in MyCoil_config if necessary
		self.controller = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] + 
												params['controller']['fc']['neurons'] + 
												[self.dim_u],
									 'dropouts': params['controller']['fc']['dropouts']+ [0.0],
									 'end_layer': True}) 
		# Note: For end layer, add one more dropout, otherwise "ValueError: Dropouts should be from the len of kernels minus 1"

	def forward(self, img, m):
		# Apply perception module
		img, inter = self.perception(img)
		# Apply measurement module
		m = self.measurement(m)
		# Join the perception and measurement
		j = self.join(img, m)
		# print("j", j.size())

		# my controller
		x = self.controller(j)

		if x.size()[-1] == 2:
			# separated activation for throttle and steer
			x_0 = self.sig(x[:, 0]).unsqueeze(1)
			x_1 = self.tanh(x[:, 1]).unsqueeze(1)
			y = torch.cat((x_0, x_1), dim = 1)
		else:
			y = self.sig(x)
		return y

def weighted_mse_loss(input,target):
	#alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
	weights = Variable(torch.Tensor([10000])) # .cuda()   # change [1, 1000, 0.1] to [1000, 0.1]
	pct_var = (input-target)**2
	out = pct_var * weights.expand_as(target)
	loss = out.mean() 
	return loss


def get_data_loaders(datafile,batch_size=128):
	# get data from csv file and process it
	max_pos_val = 500
	max_yaw_val = 180
	max_speed_val = 40

	z = []
	u = []
	line_count = 0
	if_print = True

	# parse the csv data
	# writer.writerow(["throttle", "steer", \
	#                  "cur_speed", "cur_x", "cur_y", "cur_z", "cur_pitch", "cur_yaw", "cur_roll", \
	#                  "target_speed", "target_x", "target_y", "target_z", "target_pitch", "target_yaw", "target_roll"])
	print("get_data_loaders", num_wps)
	with open(datafile) as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			try:
				row = [float(i) for i in row]
				if any(np.isnan(row)):
					print("row contain nan value, continue")
					continue
			except ValueError:
				print(row[0], "continue")
				continue
			action = row[0:2]
			# normalize the state
			state = [row[2]/max_speed_val,row[3]/max_pos_val, row[4]/max_pos_val, row[7]/max_yaw_val, row[9]/max_speed_val]
			for j in range(num_wps): # concatenate x, y, yaw of future_wps
				state = state + [row[10+j*6]/max_pos_val, row[11+j*6]/max_pos_val, row[14+j*6]/max_yaw_val]

			if if_print:
				print("sample state and action", state, action)
				if_print = False

			z.append(torch.from_numpy(np.array(state).astype(np.float32)))
			u.append(torch.from_numpy(np.array(action).astype(np.float32)))
			line_count += 1

	print("process {} lines".format(line_count))
	print("z[0] {}, u[0] {}".format(z[0].size(), u[0].size()))
	train_dataset = ZUData(z=z, u=u)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	return train_loader

def get_data_loaders_img(ds_dir, batch_size=128, img_width = 200, img_height=88, num_wps=1):
	# ds_dir: dir to get data from processed*.pkl.
	# 		  if it does not exist, CarlaData_cur will generate a new one
	print("in get_data_loaders_img")
	dataset = CarlaData_cur(ds_dir, img_width = img_width, img_height=img_height, num_wps=num_wps)
	img, m, u = dataset.query_data() # should parse measurement data in process_ctv based on num_wps
	print(img[0].shape, m[0].shape, u[0].shape)
	train_dataset = CarlaDataPro_cur(img, m, u)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	return train_loader


def train(model, optimizer, data_loader, device, lr, loss_fn=None, \
		  MLP_model_path=None, epochs=500, save_model=False, save_dict=False, loss_values=None, \
		  img_width=200, img_height=88, start_epoch=0): #epochs =500, set to 2 for a quick debugging
	
	'''
	Compared to train in NN_controller.py, way of loading data is different.
	output = model(data) => output = model(img, m)
	'''
	MLP_model_path_base = MLP_model_path
	print("iterate over epochs")
	loss_values = [] # save loss of each epoch for plot
	if_print = True
	model = model.to(device)
	print("optimizer", optimizer)
	# reset lr
	for g in optimizer.param_groups:
		g['lr'] = lr

	# print("model")
	# print(model)

	# loss_values: loss for each epoch
	# train_losses: loss for each batch
	if loss_values is None: # train_state == 1:
		loss_values = []
	for epoch in range(start_epoch, epochs): # adjust start_epoch for resume training
		train_losses = []
		model.train()
		# momdify the number of output of enumerate
		for i, (img, m, u) in enumerate(data_loader):
			img = img.view(img.shape[0], -1, img_height, img_width)
			m = m.view(m.shape[0], -1)
			u = u.view(m.shape[0], -1)
			# print("check input size")
			# print(img.size(), m.size(), u.size())
			img, m, target = img.to(device), m.to(device), u.to(device)

			optimizer.zero_grad()
			output = model(img, m)
			if loss_fn is not None:
				# use pre-defined loss in torch
				loss = loss_fn(output, target)
			else:
				func = torch.nn.MSELoss()
				# calculate your own loss
				# loss1 = -binary_crossentropy(target[:, 0], outputs[:, 0]).sum().mean()
				# loss2 = weighted_mse_loss(outputs[:, 1:], target[:, 1:])
				# test the scale of loss of throttle and steer
				loss1 = -binary_crossentropy(target[:, 0], output[:, 0]).sum().mean() # throttle ~30-40
				loss2 = func(output[:,1], target[:,1]) # ~0.1
				loss = loss1 + loss2
				# print("loss", loss1.data.cpu().numpy(), loss2.data.cpu().numpy(), loss.data.cpu().numpy())
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())

			# print for debugging
			if if_print:
				print("example target", target[0])
				print("example output", output[0])
				if_print = False
			
			train_losses.append(loss.item())
	
		loss_values.append(np.mean(train_losses))
				
		model.eval()

		print('epoch : {}, train loss : {:.4f},'\
		 .format(epoch+1, np.mean(train_losses)))

		if ((epoch+1)%100 == 0): # Don't wait until the end to save the model
			# save the model

			MLP_model_path = MLP_model_path_base[:-4] + '_ep{}'.format(str(epoch+1)) + '.pth' # otherwise ep number is concatenated
			MLP_dict_path = MLP_model_path.replace('_model_', '_dict_')
			
			if save_dict:
				print("save state_dict")
				torch.save({
							'epoch': epoch,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'loss_fn': loss_fn,
							'loss_values': loss_values,
							}, MLP_dict_path)
			if save_model:
				print("save the entire model")
				torch.save(model, MLP_model_path)

	return loss_values


if __name__ == '__main__':

	img_width = 200 
	img_height = 88
	# Device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# print("device", device)

	lr_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
	batch_range = [32, 64, 128, 256]
	num_wps_range = [1, 5, 10, 20, 30]
	# num_wps = 30

	params_product = [(lr, batch_size) for lr in lr_range for batch_size in batch_range]
	# params_product = [comb[0] for comb in params_product] #"squeeze"
	print("params_product example", params_product[0], "num_wps", params_product[0][0], "lr",  params_product[0][1])

	train_state = 1 # 1:train from scratch; 2: load state_dict and resume training 3: load and evaluate

	grid_search = True
	if grid_search:
		gs_log = "models/mlp/mlp_model_nx=8_gs_log.csv"
		if not os.path.exists(gs_log):
			# TODO: once create the file, command the following lines
			# TODO: adjust the header based on the params dict
			print("create gs_log file")
			with open(gs_log, 'w', newline='') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(["num_wps", "lr", "batch_size", "last_train_loss", "optimizer"])
	
	i = 0
	for params_list in params_product:
		# parse the params
		# num_wps = params_list[0]
		# lr = params_list[0]
		# batch_size = params_list[1]
		lr = 0.001
		batch_size = 64

		# model type
		# input: current and target states (speed, x, y, yaw)
		# output: throttle, steer
		# model = mlp(nx=8, ny=2)
		# TODO: Load configurations for MyCoil from MyCoil_config
		yaml_filename = 'MyCoil_config.yaml'
		with open(yaml_filename, 'r') as f:
			yaml_file = yaml.load(f, Loader=yaml.FullLoader)
			yaml_cfg = AttributeDict(yaml_file)
		
		for num_wps in num_wps_range:

			num_wps = 10  # try first
			
			params = yaml_cfg.MODEL_CONFIGURATION
			print("params")
			print(params)
			# Load a pre-trained model TODO: should put in "if train_state == 1"
			model = MyCoil(dim_img=(img_height*img_width), dim_m=(5+3*num_wps), dim_u=2, params=params)
			model_dict = model.state_dict()

			optimizer_range = [torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0, amsgrad=False), \
					   torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False), \
					   torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.01, dampening=0, weight_decay=0, nesterov=False)]
			
			optimizer = optimizer_range[0]
			# for optimizer in optimizer_range:
						

			optim_name = str(optimizer).split("(")[0][:-1]
			MLP_model_path = 'models/mlp/mlp_img_model_nx=8_wps{}_lr{}_bs{}_optim{}.pth'.format(num_wps, lr, batch_size, optim_name)
			MLP_dict_path = MLP_model_path.replace('_model_', '_dict_')
			# check if path exists: if os.path.exists(MLP_dict_path): 

			ds_parent_dir = '/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/data/dest_start'
			# os.walk(directory)
			ds_sub_dirs = [x[0] for x in os.walk(ds_parent_dir)]

			for i in range(1, len(ds_sub_dirs)):
				if i == 1:
					train_state = 1
				else: 
					train_state = 2

				print("{} load {}".format(train_state, ds_sub_dirs[i]))
				if train_state == 1:
					print("create a new model")
					# TODO: create a subfolder to differntiate with dest_start and start_dest
					# TODO: try with a small dataset first
					ds_dir = ds_sub_dirs[i]
					train_loader = get_data_loaders_img(ds_dir, batch_size=batch_size, img_width = img_width, img_height=img_height, num_wps=num_wps)
					
					# Partially load the pre-trianed model from COIL
					# for debugging
					# print("param_tensor of empty MyCoil model")
					# for param_tensor in model.state_dict():
					# 	print(param_tensor, "\t", model.state_dict()[param_tensor].size())
					# 	break
					# Load the pretrained_dict
					print("load pretrained model from CoIL")
					pretrained_coil_dict_path = '/home/ruihan/scenario_runner/models/CoIL/CoIL_180000.pth'
					pretrained_dict = torch.load(pretrained_coil_dict_path)
					pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
					# Overwrite entries in the existing state dict
					print("Update pretrained_dict")
					model_dict.update(pretrained_dict)
					# Load the new state dict
					model.load_state_dict(model_dict)
					model.eval()
					# for param_tensor in model.state_dict():
					# 	print(param_tensor, "\t", model.state_dict()[param_tensor])
					# 	break

					train_loss_ep = train(model, optimizer, train_loader, device, lr, loss_fn=torch.nn.MSELoss(), MLP_model_path=MLP_model_path,\
										  save_model=True, save_dict=True, epochs=1000, img_width=img_width, img_height=img_height)

				elif train_state == 2: 
					print("load state_dict")
					ds_dir = None # any retrain data dir
					train_loader = get_data_loaders_img(ds_dir, batch_size=batch_size, img_width = img_width, img_height=img_height, num_wps=num_wps)
					#load the existing model
					checkpoint = torch.load(MLP_dict_path)
					model.load_state_dict(checkpoint['model_state_dict'])
					model = model.to(device)
					optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
					epoch = checkpoint['epoch']
					print("load state_dict, epoch", epoch)
					loss_fn = checkpoint['loss_fn']
					train_losses = checkpoint['train_losses']
					# resume training
					# rename
					MLP_model_path = 'models/mlp/dest_start_merge_retrain_models/mlp_img_model_nx=8_Adam_lr{}_bs{}_optim{}.pth'.format(lr, batch_size, optim_name)
					train_loss_ep = train(model, optimizer, train_loader, device, lr, loss_fn=torch.nn.MSELoss(), MLP_model_path=MLP_model_path, \
						save_model=True, save_dict=True, epochs=1000, loss_values=loss_values, img_width=img_width, img_height=img_height, start_epoch=epoch)

				# print(model)
				plt.figure(i) # save as a separte graph
				i += 1 
				plt.plot(train_loss_ep)
				plt.xlabel('Epoch number')
				plt.ylabel('Train loss')
				plt.savefig('{}_loss_{}.png'.format(MLP_model_path[:-4], ds_dir.split('/')[-1][4:])) # exclude ".pth"
				# plt.show()

				if grid_search:
					# record in csv log
					# TODO: keep consistent with the headers
					row = [num_wps, lr, batch_size, np.mean(train_loss_ep[-10:]), optim_name]
					with open(gs_log, 'a+') as csvFile:
						writer = csv.writer(csvFile)
						writer.writerow(row)
						csvFile.close()

			break # for optimizer
			
		break # for params_list
