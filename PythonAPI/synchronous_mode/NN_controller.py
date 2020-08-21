# %matplotlilb inline
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import fnmatch
import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float32)

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


def weighted_mse_loss(input,target):
    #alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    weights = Variable(torch.Tensor([10000])) # .cuda()   # change [1, 1000, 0.1] to [1000, 0.1]
    pct_var = (input-target)**2
    out = pct_var * weights.expand_as(target)
    loss = out.mean() 
    return loss


def get_data_loaders(datafiles,batch_size=128, num_wps=1, full_state = False, normalize = False):
    # process data
    if normalize:
        max_pos_val = 500
        max_yaw_val = 180
        max_speed_val = 40
    else:
        max_pos_val = 1
        max_yaw_val = 1
        max_speed_val = 1


    z = []
    u = []
    line_count = 0
    if_print = True

    # parse the csv data
    # writer.writerow(["throttle", "steer", \
    #                  "cur_speed", "cur_x", "cur_y", "cur_z", "cur_pitch", "cur_yaw", "cur_roll", \
    #                  "target_speed", "target_x", "target_y", "target_z", "target_pitch", "target_yaw", "target_roll"])
    print("get_data_loaders", num_wps)
    for datafile in datafiles:
     print("Read from ", datafile)
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
             if full_state:
                state = [row[2]/max_speed_val,row[3]/max_pos_val, row[4]/max_pos_val, row[7]/max_yaw_val, row[9]/max_speed_val]
                for j in range(num_wps): # concatenate x, y, yaw of future_wps
                                state = state + [row[10+j*6]/max_pos_val, row[11+j*6]/max_pos_val, row[14+j*6]/max_yaw_val]
             else:
                             state = [row[2]/max_speed_val,row[3]/max_pos_val, row[4]/max_pos_val, row[5]/max_yaw_val, row[6]/max_speed_val]
                             for j in range(num_wps): # concatenate x, y, yaw of future_wps
                                 state = state + [row[7+j*3]/max_pos_val, row[8+j*3]/max_pos_val, row[9+j*3]/max_yaw_val]

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


def train(model, optimizer, data_loader, device, lr, loss_fn=None, \
          MLP_model_path=None, save_model=False, save_dict=False, epochs=500, loss_values=None, start_epoch=0): #epochs =500, set to 2 for a quick debugging
    
    MLP_model_path_base = MLP_model_path
    print("iterate over epochs on ", device)
    loss_values = [] # save loss of each epoch for plot
    if_print = False 
    model = model.to(device)
    print("optimizer", optimizer)
    # TODO: check whether can reset lr
    for g in optimizer.param_groups:
        g['lr'] = lr

    print("model")
    print(model)
    # loss_values: loss for each epoch
    # train_losses: loss for each batch
    if loss_values is None: # train_state == 1:
        loss_values = []
    for epoch in range(start_epoch, start_epoch + epochs):
        train_losses = []
        model.train()
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # if any(np.isnan(output)):
            #     print("output nan, continue")
            #     continue
            if loss_fn is not None:
                # use pre-defined loss in torch
                loss = loss_fn(output, target)
                # print("loss", loss, output, target)
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
    
        loss_values.append(np.mean(train_losses))
                
        model.eval()

        print('epoch : {}, train loss : {:.4f},'\
         .format(epoch+1, np.mean(train_losses)))


        if ((epoch+1)%1000 == 0): # Don't wait until the end to save the model
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


def grid_search_train():

    # For pure state input, load data from "long_states.csv"
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device", device)

    lr_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    batch_range = [32, 64, 128, 256]
    num_wps_range = [10, 20, 30] # later resume training with 5 (epoch 200)
    # num_wps = 30

    params_product = [(lr, batch_size) for lr in lr_range for batch_size in batch_range]
    # params_product = [comb[0] for comb in params_product] #"squeeze"
    # print("params_product example", params_product[0], "num_wps", params_product[0][0], "lr",  params_product[0][1])

    # TODO: retrain for num_wps == 20
    train_state = 1 # 1:train from scratch; 2: load state_dict and resume training 3: load and evaluate

    grid_search = True

    if grid_search:
        gs_log = "models/mlp/mlp_model_nx=8_SR0.5_gs_num_wps_log.csv" 
        # 1. search for hyper-params (lr, bs) 2. search for num_wps
        # TODO: once create the file, command the following lines
        # TODO: adjust the header based on the params dict
    
        with open(gs_log, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["num_wps", "lr", "batch_size", "last_train_loss", "optimizer"])
    
    i = 0
    for params_list in params_product:
        # parse the params
        # lr = params_list[0]
        # batch_size = params_list[1]
        lr = 0.001
        batch_size = 32
        
        for num_wps in num_wps_range:

            # num_wps = 20 # retrain
            # model type
            # input: current and target states (speed, x, y, yaw)
            # output: throttle, steer
            # TODO: modify nx based on num_wps
            # model = mlp(nx=8, ny=2)
            model = mlp(nx=(5+3*num_wps), ny = 2) #(cur_speed, cur_x, cur_y, cur_yaw, target_speed, (x, y, yaw)*num_wps)

            optimizer_range = [torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0, amsgrad=False), \
                       torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False), \
                       torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.01, dampening=0, weight_decay=0, nesterov=False)]
            
            optimizer = optimizer_range[2]
            # for optimizer in optimizer_range:
            #     optimizer = optimizer_range[2]

            optim_name = str(optimizer).split("(")[0][:-1]
            MLP_model_path = 'models/mlp/dest_start_SR0.5_models/mlp_model_nx=8_wps{}_lr{}_bs{}_optim{}.pth'.format(num_wps, lr, batch_size, optim_name)
            MLP_dict_path = MLP_model_path.replace('_model_', '_dict_')
            # check if path exists: if os.path.exists(MLP_dict_path): 
            if train_state == 1:
                print("create a new model")
                datafile="data/dest_start/merged.csv" # TODO: modify the file name
                train_loader = get_data_loaders(datafile, batch_size=batch_size, num_wps=num_wps)
                train_loss_ep = train(model, optimizer, train_loader, device, lr, loss_fn=torch.nn.MSELoss(), MLP_model_path=MLP_model_path,\
                                      save_model=True, save_dict=True, epochs=1000)

            elif train_state == 2: 
                print("load state_dict")
                retrain_datafile = "data/dest_start/merged.csv" # TODO: change the directory
                train_loader = get_data_loaders(retrain_datafile)
                #load the existing model
                checkpoint = torch.load(MLP_dict_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                print("load state_dict, epoch", epoch)
                loss_fn = checkpoint['loss_fn']
                loss_values = checkpoint['loss_values']
                # resume training
                # rename
                MLP_model_path = 'models/mlp/dest_start_merge_retrain_models/mlp_model_nx=8_wps{}_lr{}_bs{}_optim{}.pth'.format(num_wps, lr, batch_size, optim_name)
                train_loss_ep = train(model, optimizer, train_loader, device, lr, loss_fn=torch.nn.MSELoss(), MLP_model_path=MLP_model_path, \
                    save_model=True, save_dict=True, epochs=1000, loss_values=loss_values, start_epoch=epoch)

            # print(model)
            plt.figure(i) # save as a separte graph
            i += 1 
            plt.plot(train_loss_ep)
            plt.xlabel('Epoch number')
            plt.ylabel('Train loss')
            plt.savefig('{}_loss.png'.format(MLP_model_path[:-4])) # exclude ".pth"
            # plt.show()

            if grid_search:
                # record in csv log
                # TODO: keep consistent with the headers
                row = [num_wps, lr, batch_size, np.mean(train_loss_ep[-10:]), optim_name]
                with open(gs_log, 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                    csvFile.close()

            # break # for optimizer/num_wps

        break # for params_list

def one_train():

    # For pure state input, load data from "long_states.csv"
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device", device)

    lr = 0.0001
    batch_size = 32
    num_wps = 5

    target_speed = 30
        
    model = mlp(nx=(5+3*num_wps), ny = 2) #(cur_speed, cur_x, cur_y, cur_yaw, target_speed, (x, y, yaw)*num_wps)

    optimizer =torch.optim.SGD(model.parameters(), lr=lr, momentum=0.01, dampening=0, weight_decay=0, nesterov=False)
            

    optim_name = str(optimizer).split("(")[0][:-1]

    retrain_datafiles = []
    datafile_base = './datasets/'
    for retrain_datafile in os.listdir(datafile_base):
            if fnmatch.fnmatch(retrain_datafile, "nx_{}_ny_2_from_1_wpfile_spd_{}*70000*friction_safe_150cm_run*.csv".format(\
                    5 + 3 * num_wps, target_speed)):
                retrain_datafiles.append(datafile_base + retrain_datafile)
            elif fnmatch.fnmatch(retrain_datafile, "nx_{}_ny_2_from_1_wpfile_spd_{}*71000*friction_safe_150cm_run*.csv".format(\
                    5 + 3 * num_wps, target_speed)):
                retrain_datafiles.append(datafile_base + retrain_datafile)
                
        
    train_loader = get_data_loaders(retrain_datafiles, batch_size, num_wps)
    #MLP_model_path = "/home/depend/workspace/carla/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_dict_nx=8_wps5_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"
    load_MLP_model_path = './checkpoints/IJCAI/mlp_dict_nx={}_wps{}_spd_{}_lr{}_bs{}_optim{}_ep91999_friction_safe_150cm_train_ep96000.pth'.format(\
                5 + 3 * num_wps, num_wps, target_speed, lr, batch_size, optim_name)
    load_MLP_model_path = "./checkpoints/IJCAI/mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep65999_friction_safe_150cm_train_ep71000.pth"

    checkpoint = torch.load(load_MLP_model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("load state_dict, epoch", epoch)
    loss_fn = checkpoint['loss_fn']
    #loss_values = checkpoint['loss_values']
    loss_values = None
    save_MLP_model_path = './checkpoints/IJCAI/mlp_model_nx={}_wps{}_spd_{}_lr{}_bs{}_optim{}_ep{}_friction_safe_150cm_train.pth'.format(\
                5 + 3 * num_wps, num_wps, target_speed, lr, batch_size, optim_name, epoch)
    train_loss_ep = train(model, optimizer, train_loader, device, lr, loss_fn=torch.nn.MSELoss(), MLP_model_path=save_MLP_model_path, save_model=True, save_dict=True, epochs=5001, loss_values=loss_values, start_epoch=epoch)

    plt.figure(1) # save as a separte graph
    plt.plot(train_loss_ep)
    plt.xlabel('Epoch number')
    plt.ylabel('Train loss')
    plt.savefig('{}_loss.png'.format(save_MLP_model_path[:-4])) # exclude ".pth"

def enhance_train(prev_epoch, from_epoch, to_epoch):
    print("Enhance training begins")

    # For pure state input, load data from "long_states.csv"
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    lr = 0.0001
    batch_size = 32
    num_wps = 5

    target_speed = 30
        
    model = mlp(nx=(5+3*num_wps), ny = 2) #(cur_speed, cur_x, cur_y, cur_yaw, target_speed, (x, y, yaw)*num_wps)

    optimizer =torch.optim.SGD(model.parameters(), lr=lr, momentum=0.01, dampening=0, weight_decay=0, nesterov=False)
            

    optim_name = str(optimizer).split("(")[0][:-1]

    retrain_datafiles = []
    datafile_base = './datasets/'
    for retrain_datafile in os.listdir(datafile_base):
            if fnmatch.fnmatch(retrain_datafile, "nx_{}_ny_2_from_1_wpfile_spd_{}*ep_{}_friction_safe_150cm_enhance_lagrange_1.0_*.csv".format(\
                    5 + 3 * num_wps, target_speed, from_epoch)):
                retrain_datafiles.append(datafile_base + retrain_datafile)
                
        
    train_loader = get_data_loaders(retrain_datafiles, batch_size, num_wps)
    #MLP_model_path = "/home/depend/workspace/carla/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_dict_nx=8_wps5_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"
    load_MLP_model_path = './checkpoints/IJCAI/1/mlp_dict_nx={}_wps{}_spd_{}_lr{}_bs{}_optim{}_ep{}_friction_safe_150cm_train_ep{}.pth'.format(\
                5 + 3 * num_wps, num_wps, target_speed, lr, batch_size, optim_name, prev_epoch, from_epoch)
    checkpoint = torch.load(load_MLP_model_path, map_location = 'cpu')
    print("Checkpoint loaded: {}".format(load_MLP_model_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("load state_dict, epoch", epoch)
    loss_fn = checkpoint['loss_fn']
    #loss_values = checkpoint['loss_values']
    loss_values = None
    save_MLP_model_path = './checkpoints/IJCAI/1/mlp_model_nx={}_wps{}_spd_{}_lr{}_bs{}_optim{}_ep{}_friction_safe_150cm_train.pth'.format(\
                5 + 3 * num_wps, num_wps, target_speed, lr, batch_size, optim_name, epoch)
    train_loss_ep = train(model, optimizer, train_loader, device, lr, loss_fn=torch.nn.MSELoss(), MLP_model_path=save_MLP_model_path, save_model=True, save_dict=True, epochs=to_epoch - from_epoch + 1, loss_values=loss_values, start_epoch=epoch)

    plt.figure(1) # save as a separte graph
    plt.plot(train_loss_ep)
    plt.xlabel('Epoch number')
    plt.ylabel('Train loss')
    plt.savefig('{}_loss.png'.format(save_MLP_model_path[:-4])) # exclude ".pth"
if __name__ == "__main__":
    #one_train()
    enhance_train(63999, 66000, 70000)
    pass
