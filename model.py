import copy
import torch
from torch import nn
import numpy
import numpy as np

cudaIdx = "cuda:0"
pdist = nn.PairwiseDistance(p=2)
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")

scale = 1
key_size = 374 * scale


def extract_PL(premat_pathloss, pos):  # b 2
    pos_t = np.round((pos.to('cpu') + 20) * scale).detach().numpy().astype(int)  # b , 2
    pl = []
    for i in range(pos_t.shape[0]):
        pl.append(premat_pathloss[0, :, pos_t[i, 0], pos_t[i, 1]].view(1, 4))
    pathloss = torch.cat(pl, 0)
    return pathloss.to(pos.device)


class DoraNet(nn.Module):
    def __init__(self, test=False):
        super(DoraNet, self).__init__()
        self.test = test

        self.mcl = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=(3, 3), padding=(0, 0), stride=(4, 4), output_padding=(0, 0)),
            nn.Tanh(),
            nn.InstanceNorm2d(8),
            nn.Dropout(),
            nn.ConvTranspose2d(256, 64, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(32),
            nn.Dropout(),
            nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(0, 0)),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(64),
            nn.Dropout(),
            nn.ConvTranspose2d(16, 4, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), output_padding=(1, 1)),
            nn.Sigmoid()
        )
        self.input = torch.randn([1, 1024, 12, 12], requires_grad=False, dtype=torch.float32) * 0.1

    def forward(self, pos):

        self.pre_pl = self.mcl(self.input.to(pos.device))
        prepathloss = extract_PL(self.pre_pl, pos)

        return prepathloss, self.pre_pl
        # b 4 374 374

    def loss(self, gt_pathloss, pre_pl):
        gt_pathloss = gt_pathloss.to(pre_pl.device)
        # self.pre_pl = self.mcl(self.input.to(self.mcl.device))
        loss = torch.mean(torch.abs(gt_pathloss[gt_pathloss != 0] - pre_pl[gt_pathloss != 0]))
        return loss


def main():
    b = 10
    doraNet = DoraNet()
    pos = torch.zeros((b, 2))
    pathloss = torch.zeros(b, 4)
    p_pathloss = doraNet(pos)
    print(torch.mean(torch.abs(p_pathloss[0] - pathloss)))


if __name__ == '__main__':
    main()
