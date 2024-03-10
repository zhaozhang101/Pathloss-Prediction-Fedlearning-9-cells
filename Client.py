import copy
import sys
import torch.nn
import random
from model import DoraNet
from util import *
from dataset import DoraSet, DoraSetComb
import os
import pandas as pd

cudaIdx = "cuda:0"
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
self_loss = torch.nn.MSELoss(reduction='mean')


class Client:  # as a user
    def __init__(self, args, data_loader, user_idx, globel_parameters):
        self.args = args
        self.data_loader = data_loader
        self.user_idx = user_idx
        self.model = DoraNet()
        self.model.load_state_dict(globel_parameters)
        # self.median_model, self.old_model, self.penal_model = copy.deepcopy(self.model), copy.deepcopy(self.model), copy.deepcopy(
        #     self.model)
        self.previous_parameters = globel_parameters
        self.scale = 1
        self.key_size = 374 * self.scale

    def train(self):  # training locally
        self.model = self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        gt_pathloss = self.mk_plMat()
        for local_epoch in range(1, self.args.local_epochs + 1):
            LOS = []
            for i, (pos, pathloss) in enumerate(self.data_loader):
                pos = pos.float().to(device)
                pathloss = pathloss.float().to(device)
                pre_pathloss, pre_pl = self.model(pos)
                optimizer.zero_grad()
                # # 162.69553466165152 92.12944541001944
                #   167.50507994638315 91.20744704249563
                #   -83.63877986074748 14.356142334145595
                #   -81.95750666978137 18.060331759145292
                #   -84.59552483282998 20.14371746595895
                #   -82.58517819098684 21.263828579168226
                pre_pathloss = pre_pathloss * torch.tensor([14.35, 18.06, 20.14, 21.26],device=pre_pathloss.device) + torch.tensor(
                    [-83.63, -81.95, -84.59, -82.58],device=pre_pathloss.device)
                loss = self.model.loss(pre_pathloss, pathloss)  # 直接法
                # loss = self.model.loss(gt_pathloss, pre_pl) # 矩阵法
                loss.backward()
                LOS.append(loss.item())
                optimizer.step()
                # self.step_function(pos,pathloss)
            print("Local_epoch_{0:2d},loss_{1:.4e}".format(local_epoch, sum(LOS) / len(LOS)))
        self.model = self.model.to("cpu")
        return self.model.state_dict()

    def mk_plMat(self):
        plMat = torch.zeros(1, 4, self.key_size, self.key_size)
        # testMat = torch.zeros(1,4,5,5)
        for i, (pos, pathloss) in enumerate(self.data_loader):
            pos_t = np.round((pos.to('cpu') + 20) * self.scale).detach().numpy().astype(int)  # b , 2
            for idx in range(pos_t.shape[0]):
                # a = pathloss[i,:]
                # b = plMat[0,:,pos_t[i, 0],pos_t[i, 1]]
                plMat[0, :, pos_t[idx, 0], pos_t[idx, 1]] = pathloss[idx, :]
                # testMat[0,:,1,1] = pathloss[i,:]
                # print()
        return plMat
