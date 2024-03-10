import copy
import sys
import torch.nn
import random
from model import DoraNet
from util import *
from dataset import DoraSet, DoraSetComb
from Client import *
import os
import pandas as pd

class Link(object):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.size = np.zeros((1,), dtype=np.float64)

    def pass_link(self, pay_load):
        for k, v in pay_load.items():  # 计算总参数量
            self.size = self.size + np.sum(v.numel())
        return pay_load

class FedAvgServer:  # used as a center
    def __init__(self, args, global_model, train_dataloaders,  valid_loader):
        self.global_model = global_model
        self.global_parameters = self.global_model.state_dict()
        self.down_link = Link("down_link")
        self.up_link = Link("up_link")
        self.train_dataloaders = train_dataloaders
        self.valid_dataloaders = valid_loader
        self.args = args

    def download(self, user_idx):  # 计算下行到num_user总的参数量
        local_parameters = []
        for i in range(len(user_idx)):
            local_parameters.append(self.down_link.pass_link(copy.deepcopy(self.global_parameters)))
        return local_parameters

    def upload(self, local_parameters):  # 上传本地用户的 各模型参数
        for i, (k, v) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(v)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][k]
            tmp_v = tmp_v / len(local_parameters)  # FedAvg # 对每个层 网络参数 求平均
            self.global_parameters[k] = tmp_v*0.8 + v*0.2  # 更改 全局模型的参数

    def activateClient(self, train_dataloaders, user_idx, global_parameters):
        local_parameters = self.download(user_idx)
        clients = []
        for i in range(len(user_idx)):
            clients.append(Client(self.args,train_dataloaders[user_idx[i]], user_idx[i], global_parameters))
        return clients, local_parameters

    def train(self):
        pathloss_scores = []
        ul_commCost_scores = []
        dl_commCost_scores = []
        for epoch in range(1, self.args.epochs + 1):  ## start training
            # user_idx = np.random.choice(a=num_users, size=num_activate_users, replace=False, p=None).tolist()
            # user_idx = random.sample(np.arange(0,10).tolist(),5)
            # user_idx = [x for x in range(epoch % 10, 90, 20)]  # 选五个用户进行测试
            user_idx = np.random.choice(a=self.args.num_users, size=self.args.num_activate_users, replace=False, p=None).tolist()
            # user_idx = [5]
            # lr = (lr * 0.8) if epoch % 50 == 0 else lr
            clients, local_parameters = self.activateClient(self.train_dataloaders, user_idx, self.global_parameters)
            # 返回的是 num_user个 globalparameters （每个都是一样的）;server downlink size
            for i in range(len(user_idx)):
                # model = DoraNet()
                # model.load_state_dict(local_parameters[i])
                # model.train()
                model_state_dict = clients[i].train()
                local_parameters[i] = self.up_link.pass_link(model_state_dict)  # 更改 up_link 的size
            self.upload(local_parameters)  # 更改全局模型参数
            self.global_model.load_state_dict(self.global_parameters)

            test_model = copy.deepcopy(self.global_model)
            pathloss_score = self.valid(test_model, epoch, self.args.Result)
            pathloss_scores.append(pathloss_score)
            ul_commCost_scores.append(self.up_link.size)
            dl_commCost_scores.append(self.down_link.size)
            checkPoint(epoch, self.args.epochs, self.global_model, pathloss_scores, ul_commCost_scores, dl_commCost_scores, self.args.saveModelInterval,
                       self.args.saveLossInterval)


    def valid(self, model, epoch, Result):
        with torch.no_grad():
            model.eval().to(device)
            losses = Recoder()
            scores = Recoder()
            for i, (pos, pathloss) in enumerate(self.valid_dataloaders):
                pos = pos.float().to(device)
                pathloss = pathloss.float().to(device)
                p_pathloss,_ = model(pos)
                loss = torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0]))  ## unit in dB
                tmp1 = torch.sum(
                    torch.abs(10 ** (0.1 * p_pathloss[pathloss != 0]) - 10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
                tmp2 = torch.sum(torch.abs(10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
                score = tmp1 / tmp2
                if score > 1:
                    score = torch.tensor([1])
                losses.update(loss.item(), len(pos))
                scores.update(score.item(), len(pos))

            print(
                f"Global Epoch: {epoch}----loss:{losses.avg():.4f}----pathloss_score:{-10 * np.log10(scores.avg()):.4f}")
            list = np.array([epoch, losses.avg(), -10 * np.log10(scores.avg())]).tolist()
            data = pd.DataFrame([list])
            data.to_csv(os.path.join(Result, 'csvname.csv'), mode='a', header=None, index=False)
        del model
        return -10 * np.log10(scores.avg())