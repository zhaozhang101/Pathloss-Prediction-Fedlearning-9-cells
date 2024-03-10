import torch
from torch import nn
cudaIdx = "cuda:0"
pdist = nn.PairwiseDistance(p=2)
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")


class DoraNet_linear(nn.Module):
    def __init__(self, test=False):
        super(DoraNet_linear, self).__init__()
        self.test = test
        self.mlp = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),

            nn.Linear(256, 4)
        )

    def forward(self, pos):
        pos = pos.reshape(-1,2)
        pos = (pos + 20) / 370
        pathloss = self.mlp(pos)
        return pathloss



def main():
    b = 10
    doraNet = DoraNet()
    pos = torch.zeros((b, 2))
    pathloss = torch.zeros(b, 4)
    p_pathloss = doraNet(pos)
    print(torch.mean(torch.abs(p_pathloss - pathloss)))


if __name__ == '__main__':
    main()